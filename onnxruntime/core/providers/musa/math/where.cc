// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "where.h"
#include "where_simple_kernel.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/common.h"
#include "core/providers/musa/musa_fwd.h"
#include "core/providers/musa/musa_utils.h"
#include "core/providers/musa/musa_execution_provider.h"
#include "core/providers/cpu/tensor/utils.h"
#include <musa_runtime.h>
#include <mudnn.h>
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <mutex>

using onnxruntime::common::Status;

namespace onnxruntime {
namespace musa {

// Phase 2.1 Optimization: Shape Data Cache for GPU kernel to avoid frequent musaMalloc/musaFree
class WhereShapeCache {
private:
  struct CacheEntry {
    long long* gpu_ptr;
    size_t size;
    bool in_use;
    
    CacheEntry() : gpu_ptr(nullptr), size(0), in_use(false) {}
    CacheEntry(long long* ptr, size_t sz) : gpu_ptr(ptr), size(sz), in_use(true) {}
  };
  
  static constexpr size_t MAX_CACHE_ENTRIES = 4;  // Support 4 shape arrays (cond, x, y, output)
  
public:
  static long long* GetShapeBuffer(size_t required_size) {
    // Simplified approach: use static cache without thread_local for now 
    // to avoid compilation issues with MUSA compiler
    static std::vector<CacheEntry> cache_entries;
    static std::mutex cache_mutex;
    
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    // Look for available cached buffer of sufficient size
    for (auto& entry : cache_entries) {
      if (!entry.in_use && entry.gpu_ptr && entry.size >= required_size) {
        entry.in_use = true;
        return entry.gpu_ptr;
      }
    }
    
    // Find empty slot or add new entry
    for (auto& entry : cache_entries) {
      if (!entry.gpu_ptr) {
        long long* new_ptr = nullptr;
        auto status = musaMalloc(reinterpret_cast<void**>(&new_ptr), required_size);
        if (status == musaSuccess && new_ptr) {
          entry = CacheEntry(new_ptr, required_size);
          return new_ptr;
        }
        break;
      }
    }
    
    // If cache is full, add new entry
    if (cache_entries.size() < MAX_CACHE_ENTRIES) {
      long long* new_ptr = nullptr;
      auto status = musaMalloc(reinterpret_cast<void**>(&new_ptr), required_size);
      if (status == musaSuccess && new_ptr) {
        cache_entries.emplace_back(new_ptr, required_size);
        return new_ptr;
      }
    }
    
    // Fallback: allocate directly
    long long* ptr = nullptr;
    musaMalloc(reinterpret_cast<void**>(&ptr), required_size);
    return ptr;
  }
  
  static void ReleaseShapeBuffer(long long* ptr) {
    if (!ptr) return;
    
    // Use same static variables as GetShapeBuffer
    static std::vector<CacheEntry> cache_entries;
    static std::mutex cache_mutex;
    
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    // Mark as not in use in cache
    for (auto& entry : cache_entries) {
      if (entry.gpu_ptr == ptr) {
        entry.in_use = false;
        return;
      }
    }
    
    // If not in cache, free directly  
    musaFree(ptr);
  }
};

// Memory pool for Where operator to reduce dynamic allocation overhead
class WhereMemoryPool {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
        
        MemoryBlock(void* p, size_t s) : ptr(p), size(s), in_use(true) {}
    };
    
    std::vector<MemoryBlock> blocks_;
    std::mutex mutex_;

public:
    ~WhereMemoryPool() {
        // Clean up all allocated blocks
        for (auto& block : blocks_) {
            if (block.ptr) {
                musaFree(block.ptr);
            }
        }
    }
    
    void* Allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Look for available block with sufficient size
        for (auto& block : blocks_) {
            if (!block.in_use && block.size >= size) {
                block.in_use = true;
                return block.ptr;
            }
        }
        
        // Allocate new block
        void* ptr = nullptr;
        auto status = musaMalloc(&ptr, size);
        if (status == musaSuccess && ptr != nullptr) {
            blocks_.emplace_back(ptr, size);
            return ptr;
        }
        
        return nullptr;
    }
    
    void Free(void* ptr) {
        if (ptr == nullptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& block : blocks_) {
            if (block.ptr == ptr) {
                block.in_use = false;
                break;
            }
        }
    }
    
    // Get singleton instance
    static WhereMemoryPool& GetInstance() {
        static WhereMemoryPool instance;
        return instance;
    }
};

// Compute where operator output shape based upon three way broad-casting.
Status ComputeOutputShape(const std::string& node_name, const TensorShape& cond_shape,
                          const TensorShape& x_shape, const TensorShape& y_shape, TensorShape& out_shape) {
  size_t cond_rank = cond_shape.NumDimensions();
  size_t x_rank = x_shape.NumDimensions();
  size_t y_rank = y_shape.NumDimensions();
  size_t out_rank = std::max(std::max(cond_rank, x_rank), y_rank);

  std::vector<int64_t> output_dims(out_rank, 0);
  for (size_t i = 0; i < out_rank; ++i) {
    int64_t cond_dim = 1;
    if (i < cond_rank)
      cond_dim = cond_shape[cond_rank - 1 - i];

    int64_t x_dim = 1;
    if (i < x_rank)
      x_dim = x_shape[x_rank - 1 - i];

    int64_t y_dim = 1;
    if (i < y_rank)
      y_dim = y_shape[y_rank - 1 - i];

    int64_t out_dim = std::max(std::max(cond_dim, x_dim), y_dim);
    // special case to handle a dim of 0 which can be broadcast with a 1
    if (out_dim == 1)
      out_dim = std::min(std::min(cond_dim, x_dim), y_dim);

    if (cond_dim != out_dim && cond_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": condition operand cannot broadcast on dim ", cond_rank - 1 - i,
                             " Condition Shape: ", cond_shape.ToString(), ", X Shape: ", x_shape.ToString(), ", Y Shape: ", y_shape.ToString());
    if (x_dim != out_dim && x_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": X operand cannot broadcast on dim ", x_rank - 1 - i,
                             " Condition Shape: ", cond_shape.ToString(), ", X Shape: ", x_shape.ToString(), ", Y Shape: ", y_shape.ToString());
    if (y_dim != out_dim && y_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": Y operand cannot broadcast on dim ", y_rank - 1 - i,
                             " Condition Shape: ", cond_shape.ToString(), ", X Shape: ", x_shape.ToString(), ", Y Shape: ", y_shape.ToString());
    output_dims[out_rank - 1 - i] = out_dim;
  }

  out_shape = TensorShape(output_dims);
  return Status::OK();
}

// Check if tensors need broadcasting and handle them appropriately
Status HandleBroadcastingForTernary(const TensorShape& cond_shape, const TensorShape& x_shape, 
                                    const TensorShape& y_shape, const TensorShape& output_shape,
                                    bool& needs_broadcasting) {
  // Empty output tensors are now supported
  // Check if all tensors have the same shape (no broadcasting needed)
  bool same_shapes = (cond_shape == x_shape) && (x_shape == y_shape) && (y_shape == output_shape);
  needs_broadcasting = !same_shapes;
  
  if (needs_broadcasting) {
    // Broadcasting will be handled by manual tensor expansion in the Prepare stage
    // This includes scalar broadcasting, complex broadcasting, and empty tensor cases
    return Status::OK();  // Allow broadcasting to proceed with manual expansion
  }
  
  return Status::OK();
}

// Broadcast tensor to target shape using generic broadcasting logic
template<typename T>
Status BroadcastTensorToShape(const Tensor* input_tensor, const TensorShape& target_shape, 
                             void** expanded_data, size_t* expanded_bytes, 
                             musaStream_t stream = nullptr) {
  const TensorShape& input_shape = input_tensor->Shape();
  size_t target_elements = target_shape.Size();
  size_t element_size = sizeof(T);
  *expanded_bytes = target_elements * element_size;
  
  // Handle empty tensor case - allocate minimum 1 byte to avoid null pointer issues
  if (target_elements == 0) {
    *expanded_bytes = std::max(size_t(1), *expanded_bytes);  // Allocate at least 1 byte
  }
  
  // Allocate MUSA memory for expanded tensor using memory pool
  *expanded_data = WhereMemoryPool::GetInstance().Allocate(*expanded_bytes);
  if (*expanded_data == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, 
                           "Failed to allocate MUSA memory for tensor expansion");
  }
  
  // Create host buffer for expanded tensor - handle empty case
  std::vector<T> host_buffer;
  if (target_elements > 0) {
    host_buffer.resize(target_elements);
  }
  const T* input_data = input_tensor->Data<T>();
  size_t input_elements = input_shape.Size();
  
  // Handle empty tensor case first
  if (target_elements == 0) {
    // For empty tensors, no actual data copying is needed
    return Status::OK();
  }
  
  // Handle scalar case (most common)
  if (input_elements == 1) {
    T scalar_value = input_data[0];
    std::fill(host_buffer.begin(), host_buffer.end(), scalar_value);
  } else {
    // Handle complex broadcasting
    auto input_dims = input_shape.GetDims();
    auto target_dims = target_shape.GetDims();
    
    // For broadcasting: iterate through all target positions and map to input positions
    for (size_t i = 0; i < target_elements; ++i) {
      // Convert linear index i to multi-dimensional indices for target shape
      std::vector<size_t> target_indices(target_dims.size());
      size_t temp_i = i;
      for (int d = static_cast<int>(target_dims.size()) - 1; d >= 0; --d) {
        target_indices[d] = temp_i % target_dims[d];
        temp_i /= target_dims[d];
      }
      
      // Map target indices to input indices (broadcasting rules)
      std::vector<size_t> input_indices(input_dims.size());
      int input_dim_offset = static_cast<int>(target_dims.size()) - static_cast<int>(input_dims.size());
      for (size_t d = 0; d < input_dims.size(); ++d) {
        size_t target_d = d + input_dim_offset;
        if (input_dims[d] == 1) {
          input_indices[d] = 0;  // Broadcast dimension
        } else {
          input_indices[d] = target_indices[target_d];
        }
      }
      
      // Convert input indices back to linear index
      size_t input_idx = 0;
      for (size_t d = 0; d < input_dims.size(); ++d) {
        input_idx = input_idx * input_dims[d] + input_indices[d];
      }
      
      host_buffer[i] = input_data[input_idx];
    }
  }
  
  // Copy to device memory using async transfer
  musaError_t status;
  if (stream) {
    status = musaMemcpyAsync(*expanded_data, host_buffer.data(), *expanded_bytes, 
                             musaMemcpyHostToDevice, stream);
  } else {
    status = musaMemcpy(*expanded_data, host_buffer.data(), *expanded_bytes, musaMemcpyHostToDevice);
  }
  if (status != musaSuccess) {
    WhereMemoryPool::GetInstance().Free(*expanded_data);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to copy tensor expansion to device");
  }
  
  return Status::OK();
}

// MUSA device-based Where implementation using MusaPreparation and mudnn::Ternary
template <typename T>
Status SimpleMusaWhereOp(const MusaPreparation& prepare, const TensorShape& cond_shape,
                        const TensorShape& x_shape, const TensorShape& y_shape,
                        const TensorShape& output_shape) {
  // Handle empty output tensor case - mudnn doesn't support empty tensors
  if (output_shape.Size() == 0) {
    // For empty output tensors, no computation is needed, just return success
    return Status::OK();
  }
  
  if (prepare.inputTensors.size() < 3 || prepare.outputTensors.size() < 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid prepared tensor configuration");
  }

  // Check if broadcasting is needed and handle it
  bool needs_broadcasting = false;
  ORT_RETURN_IF_ERROR(HandleBroadcastingForTernary(cond_shape, x_shape, y_shape, output_shape, needs_broadcasting));
  
  // Check if we need broadcasting (for potential future SELECTV2 optimization)
  bool has_broadcasting = needs_broadcasting && (output_shape.Size() > 0);

  // Use mudnn Ternary class for device computation (after manual broadcasting)
  try {
    // Create mudnn Ternary operation
    ::musa::dnn::Ternary ternary_op;
    auto status = ternary_op.SetMode(::musa::dnn::Ternary::Mode::SELECTV2);
    
    if (status != ::musa::dnn::Status::SUCCESS) {
      // Fall back to SELECT mode if SELECTV2 is not available
      status = ternary_op.SetMode(::musa::dnn::Ternary::Mode::SELECT);
    }
    
    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set mudnn Ternary mode");
    }

    // Run the ternary operation on expanded tensors
    status = ternary_op.Run(prepare.GetHandle(),
                          const_cast<::musa::dnn::Tensor&>(prepare.outputTensors[0]),  // output tensor
                          prepare.inputTensors[0],   // condition tensor (expanded)
                          prepare.inputTensors[1],   // X tensor (expanded)
                          prepare.inputTensors[2]);  // Y tensor (expanded)

    if (status != ::musa::dnn::Status::SUCCESS) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "mudnn Ternary operation failed, status: " +
                             std::to_string(static_cast<int>(status)));
    }

  } catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Exception in mudnn Ternary operation: " +
                           std::string(e.what()));
  }

  return Status::OK();
}

template <typename T>
Status Where<T>::Prepare(OpKernelContext* ctx, MusaPreparation& prepare) const {
  const Tensor* condition = ctx->Input<Tensor>(0);  // condition (bool)
  const Tensor* X = ctx->Input<Tensor>(1);          // true values
  const Tensor* Y = ctx->Input<Tensor>(2);          // false values

  if (condition == nullptr || X == nullptr || Y == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input tensors are null");
  }

  // Compute output shape using three-way broadcasting
  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(ComputeOutputShape("Where", condition->Shape(), X->Shape(), Y->Shape(), output_shape));

  // Allocate output tensor
  Tensor* output = ctx->Output(0, output_shape);
  if (output == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Failed to allocate output tensor");
  }

  // Store shapes in existing fields for broadcasting info
  prepare.input_a_shape = condition->Shape();  // condition
  prepare.input_b_shape = X->Shape();          // X (true values)  
  prepare.output_shape = output_shape;

  // Check if broadcasting is needed (kept for potential future SELECTV2 optimization)
  bool need_broadcasting = (condition->Shape() != output_shape || X->Shape() != output_shape || Y->Shape() != output_shape);

  // Setup condition tensor (bool type mapped to UINT8) - handle broadcasting if needed
  ::musa::dnn::Tensor condition_tensor;
  
  // Check if condition needs broadcasting to match output shape
  if (condition->Shape() != output_shape) {
    // Expanding condition tensor for broadcasting
    void* expanded_cond_data = nullptr;
    size_t expanded_bytes = 0;
    
    // Use generic broadcasting for condition tensor (bool -> uint8_t)
    const bool* cond_data = condition->Data<bool>();
    size_t output_elements = output_shape.Size();
    expanded_bytes = output_elements * sizeof(uint8_t);
    
    // Handle empty tensor case - allocate minimum memory
    if (output_elements == 0) {
      expanded_bytes = std::max(size_t(1), expanded_bytes);
    }
    
    // Allocate MUSA memory for expanded condition using memory pool
    expanded_cond_data = WhereMemoryPool::GetInstance().Allocate(expanded_bytes);
    if (expanded_cond_data == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, 
                             "Failed to allocate MUSA memory for condition expansion");
    }
    
    // Handle empty tensor case - skip data processing
    if (output_elements == 0) {
      // Setup mudnn tensor with minimal allocated memory
      auto mudnn_status = condition_tensor.SetAddr(expanded_cond_data);
      if (mudnn_status != ::musa::dnn::Status::SUCCESS) {
        WhereMemoryPool::GetInstance().Free(expanded_cond_data);
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set mudnn condition tensor address");
      }
      
      mudnn_status = condition_tensor.SetType(::musa::dnn::Tensor::Type::UINT8);
      if (mudnn_status != ::musa::dnn::Status::SUCCESS) {
        WhereMemoryPool::GetInstance().Free(expanded_cond_data);
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set mudnn condition tensor type");
      }
      
      // Set the output shape (which is empty)
      auto dims_span = output_shape.GetDims();
      std::vector<int64_t> expanded_dims(dims_span.begin(), dims_span.end());
      mudnn_status = condition_tensor.SetNdInfo(static_cast<int>(expanded_dims.size()), expanded_dims.data());
      if (mudnn_status != ::musa::dnn::Status::SUCCESS) {
        WhereMemoryPool::GetInstance().Free(expanded_cond_data);
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set mudnn condition tensor dims");
      }
    } else {
      // Create host buffer for expanded condition
      std::vector<uint8_t> host_buffer(output_elements);
      
      // Handle bool -> uint8_t conversion and broadcasting manually
      auto cond_dims = condition->Shape().GetDims();
      auto target_dims = output_shape.GetDims();
      
      // For broadcasting: iterate through all target positions and map to input positions  
      for (size_t i = 0; i < output_elements; ++i) {
        // Convert linear index i to multi-dimensional indices for target shape
        std::vector<size_t> target_indices(target_dims.size());
        size_t temp_i = i;
        for (int d = static_cast<int>(target_dims.size()) - 1; d >= 0; --d) {
          target_indices[d] = temp_i % target_dims[d];
          temp_i /= target_dims[d];
        }
        
        // Map target indices to input indices (broadcasting rules)
        std::vector<size_t> input_indices(cond_dims.size());
        int input_dim_offset = static_cast<int>(target_dims.size()) - static_cast<int>(cond_dims.size());
        for (size_t d = 0; d < cond_dims.size(); ++d) {
          size_t target_d = d + input_dim_offset;
          if (cond_dims[d] == 1) {
            input_indices[d] = 0;  // Broadcast dimension
          } else {
            input_indices[d] = target_indices[target_d];
          }
        }
        
        // Convert input indices back to linear index
        size_t input_idx = 0;
        for (size_t d = 0; d < cond_dims.size(); ++d) {
          input_idx = input_idx * cond_dims[d] + input_indices[d];
        }
        
        // Convert bool to uint8_t for mudnn compatibility
        host_buffer[i] = cond_data[input_idx] ? 1 : 0;
      }
      
      // Copy to device memory using async transfer
      musaStream_t stream = Stream(ctx);
      musaError_t status;
      if (stream) {
        status = musaMemcpyAsync(expanded_cond_data, host_buffer.data(), expanded_bytes,
                                musaMemcpyHostToDevice, stream);
      } else {
        status = musaMemcpy(expanded_cond_data, host_buffer.data(), expanded_bytes, musaMemcpyHostToDevice);
      }
      if (status != musaSuccess) {
        WhereMemoryPool::GetInstance().Free(expanded_cond_data);
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Failed to copy condition expansion to device");
      }
      
      // Setup mudnn tensor with expanded memory
      auto mudnn_status = condition_tensor.SetAddr(expanded_cond_data);
      if (mudnn_status != ::musa::dnn::Status::SUCCESS) {
        WhereMemoryPool::GetInstance().Free(expanded_cond_data);
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set mudnn condition tensor address");
      }
      
      mudnn_status = condition_tensor.SetType(::musa::dnn::Tensor::Type::UINT8);
      if (mudnn_status != ::musa::dnn::Status::SUCCESS) {
        WhereMemoryPool::GetInstance().Free(expanded_cond_data);
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set mudnn condition tensor type");
      }
      
      // Set the expanded shape (same as output shape)
      auto dims_span = output_shape.GetDims();
      std::vector<int64_t> expanded_dims(dims_span.begin(), dims_span.end());
      mudnn_status = condition_tensor.SetNdInfo(static_cast<int>(expanded_dims.size()), expanded_dims.data());
      if (mudnn_status != ::musa::dnn::Status::SUCCESS) {
        WhereMemoryPool::GetInstance().Free(expanded_cond_data);
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set mudnn condition tensor dims");
      }
    }
    
  } else {
    // Normal case: setup condition tensor normally
    ORT_RETURN_IF_ERROR(SetupMusaTensor(condition_tensor, condition, ::musa::dnn::Tensor::Type::UINT8, &prepare));
  }
  
  prepare.inputTensors.push_back(condition_tensor);

  // Setup X tensor (T type) - handle broadcasting if needed
  ::musa::dnn::Tensor x_tensor;
  const auto musa_dtype = GetMusaDataType<T>();
  
  if (X->Shape() != output_shape) {
    // Expanding X tensor for broadcasting
    
    void* expanded_x_data = nullptr;
    size_t expanded_bytes = 0;
    musaStream_t stream = Stream(ctx);
    ORT_RETURN_IF_ERROR(BroadcastTensorToShape<T>(X, output_shape, &expanded_x_data, &expanded_bytes, stream));
    
    // Setup mudnn tensor with expanded memory
    auto mudnn_status = x_tensor.SetAddr(expanded_x_data);
    if (mudnn_status != ::musa::dnn::Status::SUCCESS) {
      WhereMemoryPool::GetInstance().Free(expanded_x_data);
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set mudnn X tensor address");
    }
    
    mudnn_status = x_tensor.SetType(musa_dtype);
    if (mudnn_status != ::musa::dnn::Status::SUCCESS) {
      WhereMemoryPool::GetInstance().Free(expanded_x_data);
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set mudnn X tensor type");
    }
    
    // Set the expanded shape (same as output shape)
    auto dims_span = output_shape.GetDims();
    std::vector<int64_t> expanded_dims(dims_span.begin(), dims_span.end());
    mudnn_status = x_tensor.SetNdInfo(static_cast<int>(expanded_dims.size()), expanded_dims.data());
    if (mudnn_status != ::musa::dnn::Status::SUCCESS) {
      WhereMemoryPool::GetInstance().Free(expanded_x_data);
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set mudnn X tensor dims");
    }
  } else {
    ORT_RETURN_IF_ERROR(SetupMusaTensor(x_tensor, X, musa_dtype, &prepare));
  }
  prepare.inputTensors.push_back(x_tensor);

  // Setup Y tensor (T type) - handle broadcasting if needed
  ::musa::dnn::Tensor y_tensor;
  
  if (Y->Shape() != output_shape) {
    // Expanding Y tensor for broadcasting
    
    void* expanded_y_data = nullptr;
    size_t expanded_bytes = 0;
    musaStream_t stream = Stream(ctx);
    ORT_RETURN_IF_ERROR(BroadcastTensorToShape<T>(Y, output_shape, &expanded_y_data, &expanded_bytes, stream));
    
    // Setup mudnn tensor with expanded memory
    auto mudnn_status = y_tensor.SetAddr(expanded_y_data);
    if (mudnn_status != ::musa::dnn::Status::SUCCESS) {
      WhereMemoryPool::GetInstance().Free(expanded_y_data);
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set mudnn Y tensor address");
    }
    
    mudnn_status = y_tensor.SetType(musa_dtype);
    if (mudnn_status != ::musa::dnn::Status::SUCCESS) {
      WhereMemoryPool::GetInstance().Free(expanded_y_data);
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set mudnn Y tensor type");
    }
    
    // Set the expanded shape (same as output shape)
    auto dims_span = output_shape.GetDims();
    std::vector<int64_t> expanded_dims(dims_span.begin(), dims_span.end());
    mudnn_status = y_tensor.SetNdInfo(static_cast<int>(expanded_dims.size()), expanded_dims.data());
    if (mudnn_status != ::musa::dnn::Status::SUCCESS) {
      WhereMemoryPool::GetInstance().Free(expanded_y_data);
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to set mudnn Y tensor dims");
    }
  } else {
    ORT_RETURN_IF_ERROR(SetupMusaTensor(y_tensor, Y, musa_dtype, &prepare));
  }
  prepare.inputTensors.push_back(y_tensor);

  // Setup output tensor (T type)
  ::musa::dnn::Tensor output_tensor;
  ORT_RETURN_IF_ERROR(SetupMusaTensor(output_tensor, output, musa_dtype, &prepare));
  prepare.outputTensors.push_back(output_tensor);

  return Status::OK();
}

// GPU kernel launch wrapper - compatibility with C++11
template<typename T>
Status LaunchSimpleWhereKernel(const Tensor* condition, const Tensor* X, const Tensor* Y, 
                                Tensor* output, musaStream_t stream) {
  if (output->Shape().Size() == 0) {
    return Status::OK();
  }
  
  auto total_elements = static_cast<long long>(output->Shape().Size());
  
  // Pad input shapes to output rank so broadcasted tensors can use the GPU path.
  int rank = std::max(1, static_cast<int>(output->Shape().NumDimensions()));
  if (rank > 8) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Where GPU kernel supports up to 8 dimensions, got ", rank);
  }

  std::vector<long long> cond_shape_host, x_shape_host, y_shape_host, output_shape_host;

  auto pad_shape = [rank](const TensorShape& shape) {
    std::vector<long long> padded(rank, 1);
    const auto dims = shape.GetDims();
    const int offset = rank - static_cast<int>(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
      padded[offset + static_cast<int>(i)] = static_cast<long long>(dims[i]);
    }
    return padded;
  };

  cond_shape_host = pad_shape(condition->Shape());
  x_shape_host = pad_shape(X->Shape());
  y_shape_host = pad_shape(Y->Shape());
  output_shape_host = pad_shape(output->Shape());
  
  // Allocate GPU memory for shapes using cache to avoid frequent allocation
  size_t shape_size = rank * sizeof(long long);
  
  long long *cond_shape_gpu = WhereShapeCache::GetShapeBuffer(shape_size);
  long long *x_shape_gpu = WhereShapeCache::GetShapeBuffer(shape_size); 
  long long *y_shape_gpu = WhereShapeCache::GetShapeBuffer(shape_size);
  long long *output_shape_gpu = WhereShapeCache::GetShapeBuffer(shape_size);
  
  if (!cond_shape_gpu || !x_shape_gpu || !y_shape_gpu || !output_shape_gpu) {
    // Cleanup any allocated buffers on failure
    WhereShapeCache::ReleaseShapeBuffer(cond_shape_gpu);
    WhereShapeCache::ReleaseShapeBuffer(x_shape_gpu);
    WhereShapeCache::ReleaseShapeBuffer(y_shape_gpu);
    WhereShapeCache::ReleaseShapeBuffer(output_shape_gpu);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to allocate GPU shape buffers");
  }
  
  musaMemcpy(cond_shape_gpu, cond_shape_host.data(), shape_size, musaMemcpyHostToDevice);
  musaMemcpy(x_shape_gpu, x_shape_host.data(), shape_size, musaMemcpyHostToDevice);
  musaMemcpy(y_shape_gpu, y_shape_host.data(), shape_size, musaMemcpyHostToDevice);
  musaMemcpy(output_shape_gpu, output_shape_host.data(), shape_size, musaMemcpyHostToDevice);
  
  // Launch kernel with proper broadcasting support
  Status result = Status::OK();
  if (std::is_same<T, float>::value) {
    launch_where_kernel_float(
        condition->Data<bool>(),
        X->Data<float>(),
        Y->Data<float>(),
        output->MutableData<float>(),
        total_elements,
        cond_shape_gpu, x_shape_gpu, y_shape_gpu, output_shape_gpu, rank,
        static_cast<void*>(stream));
  } else if (std::is_same<T, double>::value) {
    launch_where_kernel_double(
        condition->Data<bool>(),
        X->Data<double>(),
        Y->Data<double>(),
        output->MutableData<double>(),
        total_elements,
        cond_shape_gpu, x_shape_gpu, y_shape_gpu, output_shape_gpu, rank,
        static_cast<void*>(stream));
  } else if (std::is_same<T, int32_t>::value) {
    launch_where_kernel_int32(
        condition->Data<bool>(),
        reinterpret_cast<const int*>(X->Data<int32_t>()),
        reinterpret_cast<const int*>(Y->Data<int32_t>()),
        reinterpret_cast<int*>(output->MutableData<int32_t>()),
        total_elements,
        cond_shape_gpu, x_shape_gpu, y_shape_gpu, output_shape_gpu, rank,
        static_cast<void*>(stream));
  } else if (std::is_same<T, int64_t>::value) {
    launch_where_kernel_int64(
        condition->Data<bool>(),
        reinterpret_cast<const long long*>(X->Data<int64_t>()),
        reinterpret_cast<const long long*>(Y->Data<int64_t>()),
        reinterpret_cast<long long*>(output->MutableData<int64_t>()),
        total_elements,
        cond_shape_gpu, x_shape_gpu, y_shape_gpu, output_shape_gpu, rank,
        static_cast<void*>(stream));
  } else if (std::is_same<T, MLFloat16>::value) {
    launch_where_kernel_half(
        condition->Data<bool>(),
        reinterpret_cast<const void*>(X->Data<MLFloat16>()),
        reinterpret_cast<const void*>(Y->Data<MLFloat16>()),
        reinterpret_cast<void*>(output->MutableData<MLFloat16>()),
        total_elements,
        cond_shape_gpu, x_shape_gpu, y_shape_gpu, output_shape_gpu, rank,
        static_cast<void*>(stream));
  } else {
    result = ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, 
                            "GPU kernel not implemented for this data type");
  }
  
  // Release GPU shape buffers back to cache
  WhereShapeCache::ReleaseShapeBuffer(cond_shape_gpu);
  WhereShapeCache::ReleaseShapeBuffer(x_shape_gpu);
  WhereShapeCache::ReleaseShapeBuffer(y_shape_gpu);
  WhereShapeCache::ReleaseShapeBuffer(output_shape_gpu);
  
  if (!result.IsOK()) {
    return result;
  }
  
  // Check for kernel errors
  auto musa_status = musaGetLastError();
  if (musa_status != musaSuccess) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                          "Where GPU kernel failed: " + 
                          std::string(musaGetErrorString(musa_status)));
  }
  
  return Status::OK();
}

template <typename T>
Status Where<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* condition = ctx->Input<Tensor>(0);
  const Tensor* X = ctx->Input<Tensor>(1);
  const Tensor* Y = ctx->Input<Tensor>(2);
  
  if (condition == nullptr || X == nullptr || Y == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input tensors are null");
  }

  // Compute output shape for broadcasting
  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(ComputeOutputShape("Where", condition->Shape(), 
                                        X->Shape(), Y->Shape(), output_shape));

  Tensor* output = ctx->Output(0, output_shape);
  auto stream = Stream(ctx);
  return LaunchSimpleWhereKernel<T>(condition, X, Y, output, stream);
}

// Explicit template instantiations
template class Where<float>;
template class Where<double>;
template class Where<MLFloat16>;
template class Where<int32_t>;
template class Where<int64_t>;

// Registration macros for Where kernel
#define REGISTER_MUSA_WHERE_VERSIONED_TYPED_KERNEL(ver_start, ver_end, T) \
    ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX( \
        Where, kOnnxDomain, ver_start, ver_end, T, kMusaExecutionProvider, \
        (*KernelDefBuilder::Create()) \
            .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
            .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>()), \
        Where<T>);

#define REGISTER_MUSA_WHERE_TYPED_KERNEL(ver, T) \
    ONNX_OPERATOR_TYPED_KERNEL_EX( \
        Where, kOnnxDomain, ver, T, kMusaExecutionProvider, \
        (*KernelDefBuilder::Create()) \
            .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
            .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>()), \
        Where<T>);

// Register for ONNX v9-15 (versioned)
REGISTER_MUSA_WHERE_VERSIONED_TYPED_KERNEL(9, 15, float)
REGISTER_MUSA_WHERE_VERSIONED_TYPED_KERNEL(9, 15, MLFloat16)
REGISTER_MUSA_WHERE_VERSIONED_TYPED_KERNEL(9, 15, int32_t)
REGISTER_MUSA_WHERE_VERSIONED_TYPED_KERNEL(9, 15, int64_t)

// Register for ONNX v16+ (latest)
REGISTER_MUSA_WHERE_TYPED_KERNEL(16, float)
REGISTER_MUSA_WHERE_TYPED_KERNEL(16, MLFloat16)
REGISTER_MUSA_WHERE_TYPED_KERNEL(16, int32_t)
REGISTER_MUSA_WHERE_TYPED_KERNEL(16, int64_t)

#define REGISTER_MUSA_SELECT_TYPED_KERNEL(OpName, T) \
    ONNX_OPERATOR_TYPED_KERNEL_EX( \
        OpName, kOnnxDomain, 1, T, kMusaExecutionProvider, \
        (*KernelDefBuilder::Create()) \
            .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
            .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>()), \
        Where<T>);

#define REGISTER_MUSA_SELECT_KERNELS(OpName) \
    REGISTER_MUSA_SELECT_TYPED_KERNEL(OpName, float) \
    REGISTER_MUSA_SELECT_TYPED_KERNEL(OpName, double) \
    REGISTER_MUSA_SELECT_TYPED_KERNEL(OpName, MLFloat16) \
    REGISTER_MUSA_SELECT_TYPED_KERNEL(OpName, int32_t) \
    REGISTER_MUSA_SELECT_TYPED_KERNEL(OpName, int64_t)

REGISTER_MUSA_SELECT_KERNELS(Select)
REGISTER_MUSA_SELECT_KERNELS(SelectV2)

#undef REGISTER_MUSA_SELECT_KERNELS
#undef REGISTER_MUSA_SELECT_TYPED_KERNEL

}  // namespace musa
}  // namespace onnxruntime
