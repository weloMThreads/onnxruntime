// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "resize.h"
#include "core/providers/cpu/tensor/utils.h"
#include <cstring>
#include <musa_runtime.h>

using namespace onnxruntime::common;

namespace onnxruntime {
namespace musa {

// Implemented in resize_impl.mu — wraps ResizeImpl<half> for host code
// that cannot reference the 'half' type directly.
void ResizeImplHalf(
    musaStream_t stream, const onnxruntime::UpsampleMode mode, int rank,
    TArray<int64_t>& input_shape, TArray<int64_t>& output_shape,
    TArray<int64_t>& input_strides, TArray<fast_divmod>& output_div_pitches,
    TArray<float>& scales_vals, TArray<float, 10>& roi_vals,
    const void* input_data, void* output_data, size_t N,
    bool use_extrapolation, float extrapolation_value,
    float cubic_coeff_a, bool exclude_outside,
    ResizeCoordinateTransformationMode coord_mode, ResizeNearestMode nearest_mode,
    void* dims_mapping);

namespace {

// Helper to dispatch ResizeImpl with proper GPU type.
template <typename T>
Status CallResizeImpl(
    musaStream_t stream, const onnxruntime::UpsampleMode mode, int rank,
    TArray<int64_t>& input_shape, TArray<int64_t>& output_shape,
    TArray<int64_t>& input_strides, TArray<fast_divmod>& output_div_pitches,
    TArray<float>& scales_vals, TArray<float, 10>& roi_vals,
    const Tensor* X, Tensor* Y, size_t output_count,
    bool use_extrapolation, float extrapolation_value,
    float cubic_coeff_a, bool exclude_outside,
    ResizeCoordinateTransformationMode coord_mode, ResizeNearestMode nearest_mode,
    void* dims_mapping) {
  ResizeImpl<T>(stream, mode, rank, input_shape, output_shape,
                input_strides, output_div_pitches, scales_vals, roi_vals,
                X->Data<T>(), Y->MutableData<T>(), output_count,
                use_extrapolation, static_cast<T>(extrapolation_value),
                cubic_coeff_a, exclude_outside, coord_mode, nearest_mode, dims_mapping);
  return Status::OK();
}

template <>
Status CallResizeImpl<MLFloat16>(
    musaStream_t stream, const onnxruntime::UpsampleMode mode, int rank,
    TArray<int64_t>& input_shape, TArray<int64_t>& output_shape,
    TArray<int64_t>& input_strides, TArray<fast_divmod>& output_div_pitches,
    TArray<float>& scales_vals, TArray<float, 10>& roi_vals,
    const Tensor* X, Tensor* Y, size_t output_count,
    bool use_extrapolation, float extrapolation_value,
    float cubic_coeff_a, bool exclude_outside,
    ResizeCoordinateTransformationMode coord_mode, ResizeNearestMode nearest_mode,
    void* dims_mapping) {
  ResizeImplHalf(stream, mode, rank, input_shape, output_shape,
                 input_strides, output_div_pitches, scales_vals, roi_vals,
                 X->DataRaw(), Y->MutableDataRaw(), output_count,
                 use_extrapolation, extrapolation_value,
                 cubic_coeff_a, exclude_outside, coord_mode, nearest_mode, dims_mapping);
  return Status::OK();
}

}  // namespace

#define REGISTER_KERNEL_TYPED(T)                                   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                         \
      Resize,                                                      \
      kOnnxDomain,                                                 \
      10, 10,                                                      \
      T,                                                           \
      kMusaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                  \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),  \
      Resize<T>);                                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                         \
      Resize,                                                      \
      kOnnxDomain,                                                 \
      11, 12,                                                      \
      T,                                                           \
      kMusaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                  \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                  \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()), \
      Resize<T>);                                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                         \
      Resize,                                                      \
      kOnnxDomain,                                                 \
      13, 17,                                                      \
      T,                                                           \
      kMusaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                  \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                  \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()), \
      Resize<T>);                                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                   \
      Resize,                                                      \
      kOnnxDomain,                                                 \
      18,                                                          \
      T,                                                           \
      kMusaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                  \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                  \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()), \
      Resize<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
Resize<T>::Resize(const OpKernelInfo& info) : UpsampleBase(info), MusaKernel(info) {
  ORT_ENFORCE(!UpsampleBase::antialias_, "Resize: antialias is not supported in MusaEP.");
}

template <typename T>
Status Resize<T>::BaseCompute(OpKernelContext* context,
                              gsl::span<const float> roi,
                              gsl::span<const float> scales,
                              gsl::span<const int64_t> output_dims) const {
  const Tensor* X = context->Input<Tensor>(0);
  auto X_dims = X->Shape().GetDims();
  int32_t rank = static_cast<int32_t>(X_dims.size());

  ORT_ENFORCE(static_cast<int32_t>(output_dims.size()) == rank,
              "Rank of input and output tensor should be same.");
  if (rank == 0)
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Resize: input tensor cannot be scalar.");
  if (rank != static_cast<int32_t>(scales.size()))
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Resize: input tensor's dimension does not match the scales.");
  if (roi.size() != 2 * X_dims.size()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                  "Resize: size of roi array should be 2 * N where N is the rank of input tensor X.");
  }

  Tensor* Y = context->Output(0, output_dims);
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  TensorPitches input_pitches(X_dims);
  TArray<int64_t> input_strides(input_pitches);

  TensorPitches output_pitches(output_dims);
  TArray<fast_divmod> output_div_pitches(rank);
  for (int32_t i = 0; i < rank; ++i) {
    output_div_pitches[i] = fast_divmod(gsl::narrow_cast<int>(output_pitches[i]));
  }
  const size_t output_count = Y->Shape().Size();

  // V0.16: is_same memcpy short-circuit is the V0.15-and-earlier behavior; do
  // NOT gate it on MUSA_RESIZE_DISABLE_FASTPATH so that disabling the new C=5
  // vec kernel still leaves the original is_same path intact (the env switch
  // only exists for A/B testing the new fast paths added in V0.16).
  const bool is_same = std::all_of(scales.begin(), scales.end(), [](float v) { return v == 1.0f; }) &&
                       (coordinate_transform_mode_ != ResizeCoordinateTransformationMode::TF_CROP_AND_RESIZE);
  if (is_same) {
    if (Y->MutableDataRaw() != X->DataRaw()) {
      MUSA_RETURN_IF_ERROR(musaMemcpyAsync(Y->MutableDataRaw(), X->DataRaw(),
                                           output_count * sizeof(T), musaMemcpyDeviceToDevice, Stream(context)));
    }
    return Status::OK();
  }

  if (antialias_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Resize: antialias is not supported in MusaEP.");
  }

  TArray<int64_t> input_shape(X_dims);
  TArray<int64_t> output_shape(output_dims);
  TArray<float, 10> roi_vals(roi);
  TArray<float> scales_vals(scales);

  size_t temp_buffer_size = CalcResizeBufferSize(mode_, output_dims.data(), static_cast<int32_t>(output_dims.size()));
  auto dims_mapping_buffer = GetScratchBuffer<unsigned char>(temp_buffer_size, context->GetComputeStream());
  void* dims_mapping = reinterpret_cast<void*>(dims_mapping_buffer.get());

  return CallResizeImpl<T>(Stream(context), mode_, rank, input_shape, output_shape,
                           input_strides, output_div_pitches, scales_vals, roi_vals,
                           X, Y, output_count, use_extrapolation_, extrapolation_value_,
                           cubic_coeff_a_, exclude_outside_, coordinate_transform_mode_,
                           nearest_mode_, dims_mapping);
}

template <typename T>
Status Resize<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  ORT_ENFORCE(X != nullptr);
  auto input_dims = X->Shape().GetDims();

  TensorShapeVector output_dims(input_dims.size());
  InlinedVector<float> roi_array(input_dims.size() * 2, 0.0f);

  if (!roi_cached_) {
    bool use_default_roi = true;
    if (need_roi_input_) {
      ORT_ENFORCE(roi_input_idx_ > 0, "Invalid roi input index.");
      const auto* roi = context->Input<Tensor>(roi_input_idx_);
      if (roi != nullptr) {
        ParseRoiData(roi, roi_array);
        use_default_roi = false;
      }
    }
    if (use_default_roi) {
      size_t input_rank = input_dims.size();
      roi_array.resize(input_rank * 2);
      for (size_t i = 0; i < input_rank; ++i) {
        roi_array[i] = 0;
        roi_array[i + input_rank] = 1;
      }
    }
  }

  ComputeROIWithAxes(roi_array, input_dims.size());

  InlinedVector<float> scales_array(input_dims.size());

  if (OpKernel::Node().InputDefs().size() == 1) {
    scales_array = scales_;
    ComputeOutputShape(scales_array, input_dims, output_dims);
    return BaseCompute(context, roi_array, scales_, output_dims);
  }

  const Tensor* scales = context->Input<Tensor>(scales_input_idx_);
  const Tensor* sizes = context->Input<Tensor>(sizes_input_idx_);

  if (scales_cached_) {
    ORT_RETURN_IF_NOT(sizes == nullptr, "Only one of scales or sizes must be provided as input.");
    scales_array = scales_;
    ComputeOutputShape(scales_array, input_dims, output_dims);
    return BaseCompute(context, roi_array, scales_array, output_dims);
  }

  if (scales != nullptr && scales->Shape().Size() != 0) {
    ORT_ENFORCE(sizes == nullptr, "Only one of scales or sizes must be provided as input.");
    ORT_RETURN_IF_ERROR(ParseScalesData(scales, scales_array, input_dims.size()));
    ComputeOutputShape(scales_array, input_dims, output_dims);
  } else {
    ORT_ENFORCE(sizes != nullptr && sizes->Shape().Size() != 0,
                "Either scales or sizes MUST be provided as input.");
    ORT_RETURN_IF_ERROR(ParseSizesData(sizes, output_dims, input_dims));
    ORT_RETURN_IF_ERROR(ParseScalesDataAndAdjustOutputSize(output_dims, input_dims, scales_array));
  }

  return BaseCompute(context, roi_array, scales_array, output_dims);
}

}  // namespace musa
}  // namespace onnxruntime
