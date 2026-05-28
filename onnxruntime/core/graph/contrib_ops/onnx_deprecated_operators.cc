// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/defs/tensor_proto_util.h"

// Suppress a warning: global initializer calls a non-constexpr function 'symbol' which is from
// ONNX_OPERATOR_SET_SCHEMA_EX macro and only happens in debug build
#if defined(_WIN32) && !defined(NDEBUG)
#pragma warning(disable : 26426)
#endif

// Register removed experimental ops for backward compatibility.
// Experimental operators do not have version history. However, Windows 10 1809(RS5) takes bunch of experimental operators
// as production ops. In order to maintain backward compatibility when the experimental ops are removed from ONNX
// they need to be added in onnxruntime as contrib ops.
// ONNX exp ops(Affine, Crop, ParametricSoftplus, ImageScaler, ThresholdedRelu, DynamicSlice, ScaledTanh, MVN) old
// version history maintenance
// See: https://github.com/onnx/onnx/pull/1909

#include "core/graph/contrib_ops/contrib_defs.h"
using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace contrib {
constexpr const char* Affine_ver1_doc = R"DOC(
Affine takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the affine function, y = alpha * x + beta,
is applied to the tensor elementwise.
)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    Affine, 1,
    OpSchema()
        .SetDoc(Affine_ver1_doc)
        .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, 1.0f)
        .Attr("beta", "Value of beta", AttributeProto::FLOAT, 0.0f)
        .Input(0, "X", "1D input tensor", "T")
        .Output(0, "Y", "1D output tensor", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* ParametricSoftplus_ver1_doc = R"DOC(
ParametricSoftplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = alpha * ln(exp(beta * x) + 1), is applied to
the tensor elementwise.
)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    ParametricSoftplus, 1,
    OpSchema()
        .SetDoc(ParametricSoftplus_ver1_doc)
        .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("beta", "Value of beta", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Input(0, "X", "1D input tensor", "T")
        .Output(0, "Y", "1D input tensor", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* Log1p_ver1_doc = R"DOC(
Log1p takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where y = log(1 + x) is applied elementwise. This schema is kept
for TensorFlow-converted models that emit Log1p in the ONNX domain.
)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    Log1p, 1,
    OpSchema()
        .SetDoc(Log1p_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* Expm1_ver1_doc = R"DOC(
Expm1 takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where y = exp(x) - 1 is applied elementwise. This schema is kept
for TensorFlow-converted models that emit Expm1 in the ONNX domain.
)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    Expm1, 1,
    OpSchema()
        .SetDoc(Expm1_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* Square_ver1_doc = R"DOC(
Square takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where y = x * x is applied elementwise. This schema is kept
for TensorFlow-converted models that emit Square in the ONNX domain.
)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    Square, 1,
    OpSchema()
        .SetDoc(Square_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)",
                              "tensor(int32)", "tensor(int64)"},
                        "Constrain input and output tensor types.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* Rsqrt_ver1_doc = R"DOC(
Rsqrt takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where y = 1 / sqrt(x) is applied elementwise. This schema is kept
for TensorFlow-converted models that emit Rsqrt in the ONNX domain.
)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    Rsqrt, 1,
    OpSchema()
        .SetDoc(Rsqrt_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* Select_ver1_doc = R"DOC(
Select takes a bool condition tensor and two value tensors, and returns values
from X where condition is true and Y where condition is false. Numpy-style
broadcasting is supported. This schema is kept for TensorFlow-converted models
that emit Select in the ONNX domain.
)DOC";

constexpr const char* SelectV2_ver1_doc = R"DOC(
SelectV2 takes a bool condition tensor and two value tensors, and returns values
from X where condition is true and Y where condition is false. Numpy-style
broadcasting is supported. This schema is kept for TensorFlow-converted models
that emit SelectV2 in the ONNX domain.
)DOC";

#define ONNX_DEPRECATED_SELECT_SCHEMA(OpName, Doc)                                            \
  ONNX_CONTRIB_OPERATOR_SET_SCHEMA(                                                           \
      OpName, 1,                                                                               \
      OpSchema()                                                                               \
          .SetDoc(Doc)                                                                         \
          .Input(0, "condition", "Condition tensor", "B")                                \
          .Input(1, "X", "Tensor selected when condition is true", "T")                  \
          .Input(2, "Y", "Tensor selected when condition is false", "T")                 \
          .Output(0, "Z", "Output tensor", "T")                                          \
          .TypeConstraint("B", {"tensor(bool)"}, "Constrain condition to bool tensor.") \
          .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)",  \
                                "tensor(int32)", "tensor(int64)", "tensor(bool)"},       \
                          "Constrain value tensor types.")                                  \
          .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {           \
            ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 1, 0);                     \
            if (ONNX_NAMESPACE::hasNInputShapes(ctx, 3)) {                                     \
              ONNX_NAMESPACE::TensorShapeProto cond_x_shape;                                   \
              ONNX_NAMESPACE::bidirectionalBroadcastShapeInference(                            \
                  ctx.getInputType(0)->tensor_type().shape(),                                  \
                  ctx.getInputType(1)->tensor_type().shape(), cond_x_shape);                   \
              ONNX_NAMESPACE::bidirectionalBroadcastShapeInference(                            \
                  cond_x_shape, ctx.getInputType(2)->tensor_type().shape(),                    \
                  *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());               \
            }                                                                                  \
          }));

ONNX_DEPRECATED_SELECT_SCHEMA(Select, Select_ver1_doc)
ONNX_DEPRECATED_SELECT_SCHEMA(SelectV2, SelectV2_ver1_doc)
#undef ONNX_DEPRECATED_SELECT_SCHEMA

constexpr const char* AddV2_ver1_doc = R"DOC(
AddV2 takes two input tensors and produces one output tensor where
y = x1 + x2 elementwise with Numpy-style broadcasting. This schema is kept
for TensorFlow-converted models that emit AddV2 in the ONNX domain.
)DOC";

constexpr const char* BiasAdd_ver1_doc = R"DOC(
BiasAdd adds a 1D bias tensor to the channel dimension of value. The channel
axis is the last dimension for NHWC and dimension 1 for NCHW. This schema is
kept for TensorFlow-converted models that emit BiasAdd in the ONNX domain.
)DOC";

constexpr const char* BiasAddV1_ver1_doc = R"DOC(
BiasAddV1 adds a 1D bias tensor to the channel dimension of value. The channel
axis is the last dimension for NHWC and dimension 1 for NCHW. This schema is
kept for TensorFlow-converted models that emit BiasAddV1 in the ONNX domain.
)DOC";

constexpr const char* SubV2_ver1_doc = R"DOC(
SubV2 takes two input tensors and produces one output tensor where
y = x1 - x2 elementwise with Numpy-style broadcasting. This schema is kept
for TensorFlow-converted models that emit SubV2 in the ONNX domain.
)DOC";

constexpr const char* RealDiv_ver1_doc = R"DOC(
RealDiv takes two input tensors and produces one output tensor where
y = x1 / x2 elementwise with Numpy-style broadcasting. This schema is kept
for TensorFlow-converted models that emit RealDiv in the ONNX domain.
)DOC";

constexpr const char* AddN_ver1_doc = R"DOC(
AddN takes one or more input tensors and produces their elementwise sum. This
schema supports Numpy-style broadcasting and is kept for TensorFlow-converted
models that emit AddN in the ONNX domain.
)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    AddN, 1,
    OpSchema()
        .SetDoc(AddN_ver1_doc)
        .Input(0, "inputs", "Input tensors", "T", OpSchema::Variadic, false)
        .Output(0, "sum", "Output tensor", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)",
                              "tensor(int32)", "tensor(int64)"},
                        "Constrain input and output tensor types.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (ctx.getNumInputs() > 0 && ctx.getInputType(0) != nullptr &&
              ctx.getInputType(0)->has_tensor_type() &&
              ctx.getInputType(0)->tensor_type().has_shape()) {
            ONNX_NAMESPACE::TensorShapeProto accumulated_shape =
                ctx.getInputType(0)->tensor_type().shape();
            for (size_t i = 1; i < ctx.getNumInputs(); ++i) {
              if (ctx.getInputType(i) == nullptr ||
                  !ctx.getInputType(i)->has_tensor_type() ||
                  !ctx.getInputType(i)->tensor_type().has_shape()) {
                return;
              }
              ONNX_NAMESPACE::TensorShapeProto next_shape;
              ONNX_NAMESPACE::bidirectionalBroadcastShapeInference(
                  accumulated_shape, ctx.getInputType(i)->tensor_type().shape(), next_shape);
              accumulated_shape = std::move(next_shape);
            }
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() = accumulated_shape;
          }
        }));

constexpr const char* DivNoNan_ver1_doc = R"DOC(
DivNoNan takes two input tensors and produces one output tensor where
y = x1 / x2 if x2 is non-zero, and y = 0 if x2 is zero. This schema is kept
for TensorFlow-converted models that emit DivNoNan in the ONNX domain.
)DOC";

constexpr const char* SquaredDifference_ver1_doc = R"DOC(
SquaredDifference takes two input tensors and produces one output tensor where
y = (x1 - x2) * (x1 - x2) elementwise with Numpy-style broadcasting. This schema
is kept for TensorFlow-converted models that emit SquaredDifference in the ONNX domain.
)DOC";

constexpr const char* FloorDiv_ver1_doc = R"DOC(
FloorDiv takes two input tensors and produces one output tensor where
y = floor(x1 / x2) elementwise with Numpy-style broadcasting. This schema is
kept for TensorFlow-converted models that emit FloorDiv in the ONNX domain.
)DOC";

constexpr const char* FloorMod_ver1_doc = R"DOC(
FloorMod takes two input tensors and produces one output tensor where
y = x1 - floor(x1 / x2) * x2 elementwise with Numpy-style broadcasting. This
schema is kept for TensorFlow-converted models that emit FloorMod in the ONNX domain.
)DOC";

constexpr const char* Maximum_ver1_doc = R"DOC(
Maximum takes two input tensors and produces one output tensor where
y = max(x1, x2) elementwise with Numpy-style broadcasting. This schema is kept
for TensorFlow-converted models that emit Maximum in the ONNX domain.
)DOC";

constexpr const char* Minimum_ver1_doc = R"DOC(
Minimum takes two input tensors and produces one output tensor where
y = min(x1, x2) elementwise with Numpy-style broadcasting. This schema is kept
for TensorFlow-converted models that emit Minimum in the ONNX domain.
)DOC";

constexpr const char* GreaterEqual_ver1_doc = R"DOC(
GreaterEqual takes two input tensors and produces one bool output tensor where
y = x1 >= x2 elementwise with Numpy-style broadcasting. This schema is kept
for TensorFlow-converted models that emit GreaterEqual in the ONNX domain.
)DOC";

constexpr const char* LessEqual_ver1_doc = R"DOC(
LessEqual takes two input tensors and produces one bool output tensor where
y = x1 <= x2 elementwise with Numpy-style broadcasting. This schema is kept
for TensorFlow-converted models that emit LessEqual in the ONNX domain.
)DOC";

constexpr const char* NotEqual_ver1_doc = R"DOC(
NotEqual takes two input tensors and produces one bool output tensor where
y = x1 != x2 elementwise with Numpy-style broadcasting. This schema is kept
for TensorFlow-converted models that emit NotEqual in the ONNX domain.
)DOC";

constexpr const char* ZerosLike_ver1_doc = R"DOC(
ZerosLike takes one input tensor and produces an output tensor with the same
shape and type, filled with zero values. This schema is kept for
TensorFlow-converted models that emit ZerosLike in the ONNX domain.
)DOC";

constexpr const char* IdentityN_ver1_doc = R"DOC(
IdentityN forwards each input tensor to the matching output tensor. This schema
is kept for TensorFlow-converted models that emit IdentityN in the ONNX domain.
)DOC";

constexpr const char* ShapeN_ver1_doc = R"DOC(
ShapeN returns one shape tensor for each input tensor. The output element type
is selected by the out_type attribute. This schema is kept for
TensorFlow-converted models that emit ShapeN in the ONNX domain.
)DOC";

constexpr const char* ConcatV2_ver1_doc = R"DOC(
ConcatV2 concatenates one or more tensors along the scalar axis provided as the
last input. This schema is kept for TensorFlow-converted models that emit
ConcatV2 in the ONNX domain.
)DOC";

constexpr const char* SplitV_ver1_doc = R"DOC(
SplitV splits a tensor along split_dim using size_splits. One size may be -1,
in which case it is inferred from the input dimension. This schema is kept for
TensorFlow-converted models that emit SplitV in the ONNX domain.
)DOC";

constexpr const char* ReverseV2_ver1_doc = R"DOC(
ReverseV2 reverses a tensor along the axes provided by a 1D axis tensor. This
schema is kept for TensorFlow-converted models that emit ReverseV2 in the ONNX
domain.
)DOC";

constexpr const char* InvertPermutation_ver1_doc = R"DOC(
InvertPermutation computes the inverse of a 1D permutation tensor. This schema
is kept for TensorFlow-converted models that emit InvertPermutation in the ONNX
domain.
)DOC";

constexpr const char* BroadcastGradientArgs_ver1_doc = R"DOC(
BroadcastGradientArgs returns the reduction axes needed to map a broadcasted
gradient back to two input shapes. This schema is kept for TensorFlow-converted
models that emit BroadcastGradientArgs in the ONNX domain.
)DOC";

constexpr const char* ConcatOffset_ver1_doc = R"DOC(
ConcatOffset returns one offset vector per input shape for a TensorFlow-style
Concat along concat_dim. This schema is kept for TensorFlow-converted models
that emit ConcatOffset in the ONNX domain.
)DOC";

static const std::vector<std::string> tf_compat_float_types = {
    "tensor(float16)", "tensor(float)", "tensor(double)"};

static const std::vector<std::string> tf_compat_numeric_types = {
    "tensor(float16)", "tensor(float)", "tensor(double)",
    "tensor(int32)", "tensor(int64)"};

static const std::vector<std::string> tf_compat_compare_types = {
    "tensor(float16)", "tensor(float)", "tensor(double)",
    "tensor(int32)", "tensor(int64)"};

static const std::vector<std::string> tf_compat_not_equal_types = {
    "tensor(float16)", "tensor(float)", "tensor(double)",
    "tensor(int32)", "tensor(int64)", "tensor(bool)"};

static const std::vector<std::string> tf_compat_zeros_like_types = {
    "tensor(float16)", "tensor(float)", "tensor(double)",
    "tensor(int32)", "tensor(int64)", "tensor(bool)"};

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    IdentityN, 1,
    OpSchema()
        .SetDoc(IdentityN_ver1_doc)
        .Input(0, "inputs", "Input tensors", "T", OpSchema::Variadic, false, 1)
        .Output(0, "outputs", "Output tensors", "T", OpSchema::Variadic, false, 1)
        .TypeConstraint("T", OpSchema::all_tensor_types(),
                        "Constrain inputs and outputs to tensor types.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          if (ctx.getNumInputs() != ctx.getNumOutputs()) {
            fail_type_inference("IdentityN input and output counts must match.");
          }
          for (size_t i = 0; i < ctx.getNumInputs(); ++i) {
            ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, i, i);
            if (ctx.getInputType(i) != nullptr && ctx.getInputType(i)->has_tensor_type() &&
                ctx.getInputType(i)->tensor_type().has_shape()) {
              *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() =
                  ctx.getInputType(i)->tensor_type().shape();
            }
          }
        }));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    ShapeN, 1,
    OpSchema()
        .SetDoc(ShapeN_ver1_doc)
        .Attr("out_type", "Shape output tensor element type.",
              AttributeProto::INT, static_cast<int64_t>(TensorProto_DataType_INT32))
        .Input(0, "inputs", "Input tensors", "T", OpSchema::Variadic, false, 1)
        .Output(0, "outputs", "Shape tensors", "T1", OpSchema::Variadic, false, 1)
        .TypeConstraint("T", OpSchema::all_tensor_types(),
                        "Constrain inputs to tensor types.")
        .TypeConstraint("T1", {"tensor(int32)", "tensor(int64)", "tensor(float16)",
                                "tensor(float)", "tensor(double)"},
                        "Constrain shape output tensor types.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          if (ctx.getNumInputs() != ctx.getNumOutputs()) {
            fail_type_inference("ShapeN input and output counts must match.");
          }

          int64_t out_type = TensorProto_DataType_INT32;
          if (const auto* out_type_attr = ctx.getAttribute("out_type")) {
            out_type = out_type_attr->i();
          }

          for (size_t i = 0; i < ctx.getNumOutputs(); ++i) {
            ONNX_NAMESPACE::updateOutputElemType(ctx, i, static_cast<int32_t>(out_type));
            auto* output_shape = ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape();
            auto* rank_dim = output_shape->add_dim();
            if (i < ctx.getNumInputs() && ctx.getInputType(i) != nullptr &&
                ctx.getInputType(i)->has_tensor_type() &&
                ctx.getInputType(i)->tensor_type().has_shape()) {
              rank_dim->set_dim_value(ctx.getInputType(i)->tensor_type().shape().dim_size());
            }
          }
        }));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    ConcatV2, 1,
    OpSchema()
        .SetDoc(ConcatV2_ver1_doc)
        .Input(0, "inputs", "Input tensors followed by scalar axis tensor", "T",
               OpSchema::Variadic, false, 2)
        .Output(0, "concat", "Concatenated tensor", "T")
        .TypeConstraint("T", tf_compat_zeros_like_types,
                        "Constrain input and output tensor types.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          if (ctx.getNumInputs() < 2) {
            fail_type_inference("ConcatV2 requires at least one value input and one axis input.");
          }
          ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        }));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    SplitV, 1,
    OpSchema()
        .SetDoc(SplitV_ver1_doc)
        .Input(0, "value", "Input tensor", "T")
        .Input(1, "size_splits", "1D split size tensor", "Tlen")
        .Input(2, "split_dim", "Scalar split axis tensor", "Taxis")
        .Output(0, "outputs", "Split output tensors", "T", OpSchema::Variadic, false, 1)
        .TypeConstraint("T", tf_compat_zeros_like_types,
                        "Constrain input and output tensor types.")
        .TypeConstraint("Tlen", {"tensor(int32)", "tensor(int64)"},
                        "Constrain split sizes to int32 or int64.")
        .TypeConstraint("Taxis", {"tensor(int32)", "tensor(int64)"},
                        "Constrain split axis to int32 or int64.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          for (size_t i = 0; i < ctx.getNumOutputs(); ++i) {
            ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, i);
          }
        }));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    ReverseV2, 1,
    OpSchema()
        .SetDoc(ReverseV2_ver1_doc)
        .Input(0, "tensor", "Input tensor", "T")
        .Input(1, "axis", "1D axis tensor", "Tidx")
        .Output(0, "output", "Reversed tensor", "T")
        .TypeConstraint("T", tf_compat_zeros_like_types,
                        "Constrain input and output tensor types.")
        .TypeConstraint("Tidx", {"tensor(int32)", "tensor(int64)"},
                        "Constrain axis tensor to int32 or int64.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    InvertPermutation, 1,
    OpSchema()
        .SetDoc(InvertPermutation_ver1_doc)
        .Input(0, "x", "1D permutation tensor", "T")
        .Output(0, "y", "Inverse permutation tensor", "T")
        .TypeConstraint("T", {"tensor(int32)", "tensor(int64)"},
                        "Constrain input and output to int32 or int64.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    BroadcastGradientArgs, 1,
    OpSchema()
        .SetDoc(BroadcastGradientArgs_ver1_doc)
        .Input(0, "s0", "First input shape tensor", "T")
        .Input(1, "s1", "Second input shape tensor", "T")
        .Output(0, "r0", "Reduction axes for the first input", "T")
        .Output(1, "r1", "Reduction axes for the second input", "T")
        .TypeConstraint("T", {"tensor(int32)", "tensor(int64)"},
                        "Constrain shape and axes tensors to int32 or int64.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
          ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 1);
          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
          ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape()->add_dim();
        }));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    ConcatOffset, 1,
    OpSchema()
        .SetDoc(ConcatOffset_ver1_doc)
        .Input(0, "concat_dim", "Scalar concat axis", "T")
        .Input(1, "shape", "Input shape tensors", "T", OpSchema::Variadic, false, 1)
        .Output(0, "offset", "Offset tensors", "T", OpSchema::Variadic, false, 1)
        .TypeConstraint("T", {"tensor(int32)", "tensor(int64)"},
                        "Constrain concat_dim, shapes, and offsets to int32 or int64.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          for (size_t i = 0; i < ctx.getNumOutputs(); ++i) {
            ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, i);
            const size_t shape_input = i + 1;
            if (shape_input < ctx.getNumInputs() && ctx.getInputType(shape_input) != nullptr &&
                ctx.getInputType(shape_input)->has_tensor_type() &&
                ctx.getInputType(shape_input)->tensor_type().has_shape()) {
              *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() =
                  ctx.getInputType(shape_input)->tensor_type().shape();
            }
          }
        }));

#define ONNX_DEPRECATED_BINARY_BROADCAST_SCHEMA(OpName, Doc, TypeList)                         \
  ONNX_CONTRIB_OPERATOR_SET_SCHEMA(                                                            \
      OpName, 1,                                                                                \
      OpSchema()                                                                                \
          .SetDoc(Doc)                                                                          \
          .Input(0, "A", "First input tensor", "T")                                      \
          .Input(1, "B", "Second input tensor", "T")                                     \
          .Output(0, "C", "Output tensor", "T")                                          \
          .TypeConstraint("T", TypeList, "Constrain input and output tensor types.")         \
          .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {            \
            ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);                      \
            if (ONNX_NAMESPACE::hasNInputShapes(ctx, 2)) {                                      \
              ONNX_NAMESPACE::bidirectionalBroadcastShapeInference(                             \
                  ctx.getInputType(0)->tensor_type().shape(),                                   \
                  ctx.getInputType(1)->tensor_type().shape(),                                   \
                  *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());                \
            }                                                                                   \
          }));

ONNX_DEPRECATED_BINARY_BROADCAST_SCHEMA(AddV2, AddV2_ver1_doc, tf_compat_numeric_types)

#define ONNX_DEPRECATED_BIAS_ADD_SCHEMA(OpName, Doc)                                \
  ONNX_CONTRIB_OPERATOR_SET_SCHEMA(                                                 \
      OpName, 1,                                                                    \
      OpSchema()                                                                    \
          .SetDoc(Doc)                                                              \
          .Attr("data_format", "TensorFlow data format: NHWC or NCHW.",          \
                AttributeProto::STRING, std::string("NHWC"))                       \
          .Input(0, "value", "Input tensor", "T")                              \
          .Input(1, "bias", "1D bias tensor", "T")                             \
          .Output(0, "output", "Output tensor", "T")                           \
          .TypeConstraint("T", tf_compat_numeric_types,                            \
                          "Constrain input, bias, and output tensor types.")       \
          .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {\
            ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);          \
            if (ctx.getNumInputs() > 0 && ctx.getInputType(0) != nullptr &&         \
                ctx.getInputType(0)->has_tensor_type() &&                           \
                ctx.getInputType(0)->tensor_type().has_shape()) {                   \
              *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() =        \
                  ctx.getInputType(0)->tensor_type().shape();                       \
            }                                                                       \
          }));

ONNX_DEPRECATED_BIAS_ADD_SCHEMA(BiasAdd, BiasAdd_ver1_doc)
ONNX_DEPRECATED_BIAS_ADD_SCHEMA(BiasAddV1, BiasAddV1_ver1_doc)
#undef ONNX_DEPRECATED_BIAS_ADD_SCHEMA

ONNX_DEPRECATED_BINARY_BROADCAST_SCHEMA(SubV2, SubV2_ver1_doc, tf_compat_numeric_types)
ONNX_DEPRECATED_BINARY_BROADCAST_SCHEMA(RealDiv, RealDiv_ver1_doc, tf_compat_float_types)
ONNX_DEPRECATED_BINARY_BROADCAST_SCHEMA(DivNoNan, DivNoNan_ver1_doc, tf_compat_float_types)
ONNX_DEPRECATED_BINARY_BROADCAST_SCHEMA(SquaredDifference, SquaredDifference_ver1_doc, tf_compat_numeric_types)
ONNX_DEPRECATED_BINARY_BROADCAST_SCHEMA(FloorDiv, FloorDiv_ver1_doc, tf_compat_numeric_types)
ONNX_DEPRECATED_BINARY_BROADCAST_SCHEMA(FloorMod, FloorMod_ver1_doc, tf_compat_numeric_types)
ONNX_DEPRECATED_BINARY_BROADCAST_SCHEMA(Maximum, Maximum_ver1_doc, tf_compat_numeric_types)
ONNX_DEPRECATED_BINARY_BROADCAST_SCHEMA(Minimum, Minimum_ver1_doc, tf_compat_numeric_types)

#define ONNX_DEPRECATED_COMPARISON_BROADCAST_SCHEMA(OpName, Doc, TypeList)                         \
  ONNX_CONTRIB_OPERATOR_SET_SCHEMA(                                                                \
      OpName, 1,                                                                                    \
      OpSchema()                                                                                    \
          .SetDoc(Doc)                                                                              \
          .Input(0, "A", "First input tensor", "T")                                          \
          .Input(1, "B", "Second input tensor", "T")                                         \
          .Output(0, "C", "Output bool tensor", "T1")                                        \
          .TypeConstraint("T", TypeList, "Constrain input tensor types.")                       \
          .TypeConstraint("T1", {"tensor(bool)"}, "Constrain output tensor type to bool.")    \
          .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {                \
            ONNX_NAMESPACE::updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto_DataType_BOOL); \
            if (ONNX_NAMESPACE::hasNInputShapes(ctx, 2)) {                                          \
              ONNX_NAMESPACE::bidirectionalBroadcastShapeInference(                                 \
                  ctx.getInputType(0)->tensor_type().shape(),                                       \
                  ctx.getInputType(1)->tensor_type().shape(),                                       \
                  *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());                    \
            }                                                                                       \
          }));

ONNX_DEPRECATED_COMPARISON_BROADCAST_SCHEMA(GreaterEqual, GreaterEqual_ver1_doc, tf_compat_compare_types)
ONNX_DEPRECATED_COMPARISON_BROADCAST_SCHEMA(LessEqual, LessEqual_ver1_doc, tf_compat_compare_types)
ONNX_DEPRECATED_COMPARISON_BROADCAST_SCHEMA(NotEqual, NotEqual_ver1_doc, tf_compat_not_equal_types)

#undef ONNX_DEPRECATED_BINARY_BROADCAST_SCHEMA
#undef ONNX_DEPRECATED_COMPARISON_BROADCAST_SCHEMA

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    ZerosLike, 1,
    OpSchema()
        .SetDoc(ZerosLike_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T", tf_compat_zeros_like_types,
                        "Constrain input and output tensor types.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* Fill_ver1_doc = R"DOC(
Fill creates a tensor with shape from dims and fills it with a scalar value.
This schema is kept for TensorFlow-converted models that emit Fill in the ONNX
domain.
)DOC";

constexpr const char* BroadcastTo_ver1_doc = R"DOC(
BroadcastTo takes an input tensor and a target shape tensor, then broadcasts
the input to that shape. This schema is kept for TensorFlow-converted models
that emit BroadcastTo in the ONNX domain.
)DOC";

constexpr const char* GatherNd_ver1_doc = R"DOC(
GatherNd gathers slices from params according to indices. This schema is kept
for TensorFlow-converted models that emit GatherNd in the ONNX domain.
)DOC";

constexpr const char* GatherV2_ver1_doc = R"DOC(
GatherV2 gathers slices from params according to indices along a scalar axis
input, with TensorFlow batch_dims semantics. This schema is kept for
TensorFlow-converted models that emit GatherV2 in the ONNX domain.
)DOC";

constexpr const char* IsNan_ver1_doc = R"DOC(
IsNan takes a floating-point tensor and returns a bool tensor that marks NaN
elements. This schema is kept for TensorFlow-converted models that emit IsNan
in the ONNX domain.
)DOC";

constexpr const char* LogicalAnd_ver1_doc = R"DOC(
LogicalAnd takes two bool tensors and produces their elementwise logical AND
with Numpy-style broadcasting. This schema is kept for TensorFlow-converted
models that emit LogicalAnd in the ONNX domain.
)DOC";

constexpr const char* LogicalOr_ver1_doc = R"DOC(
LogicalOr takes two bool tensors and produces their elementwise logical OR
with Numpy-style broadcasting. This schema is kept for TensorFlow-converted
models that emit LogicalOr in the ONNX domain.
)DOC";

constexpr const char* LogicalNot_ver1_doc = R"DOC(
LogicalNot takes one bool tensor and produces its elementwise logical negation.
This schema is kept for TensorFlow-converted models that emit LogicalNot in the
ONNX domain.
)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    Fill, 1,
    OpSchema()
        .SetDoc(Fill_ver1_doc)
        .Input(0, "dims", "Output shape tensor", "index_type")
        .Input(1, "value", "Scalar value tensor", "T")
        .Output(0, "output", "Output tensor", "T")
        .TypeConstraint("T", tf_compat_zeros_like_types,
                        "Constrain value and output tensor types.")
        .TypeConstraint("index_type", {"tensor(int32)", "tensor(int64)"},
                        "Constrain dims tensor type to int32 or int64.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 1, 0);
        }));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    BroadcastTo, 1,
    OpSchema()
        .SetDoc(BroadcastTo_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Input(1, "shape", "Target shape tensor", "Tshape")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T", tf_compat_numeric_types,
                        "Constrain input and output tensor types.")
        .TypeConstraint("Tshape", {"tensor(int64)"},
                        "Constrain shape tensor type to int64.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        }));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    GatherNd, 1,
    OpSchema()
        .SetDoc(GatherNd_ver1_doc)
        .Input(0, "params", "Input tensor", "T")
        .Input(1, "indices", "Index tensor", "Tindices")
        .Output(0, "output", "Output tensor", "T")
        .TypeConstraint("T", tf_compat_zeros_like_types,
                        "Constrain input and output tensor types.")
        .TypeConstraint("Tindices", {"tensor(int64)"},
                        "Constrain indices tensor type to int64.")
        .Attr("batch_dims", "The number of batch dimensions.",
              AttributeProto::INT, static_cast<int64_t>(0))
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        }));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    GatherV2, 1,
    OpSchema()
        .SetDoc(GatherV2_ver1_doc)
        .Attr("batch_dims", "Number of leading batch dimensions.",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Input(0, "params", "Input tensor", "Tparams")
        .Input(1, "indices", "Index tensor", "Tindices")
        .Input(2, "axis", "Scalar axis tensor", "Taxis")
        .Output(0, "output", "Output tensor", "Tparams")
        .TypeConstraint("Tparams", tf_compat_zeros_like_types,
                        "Constrain params and output tensor types.")
        .TypeConstraint("Tindices", {"tensor(int32)", "tensor(int64)"},
                        "Constrain indices tensor type to int32 or int64.")
        .TypeConstraint("Taxis", {"tensor(int32)", "tensor(int64)"},
                        "Constrain axis tensor type to int32 or int64.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        }));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    IsNan, 1,
    OpSchema()
        .SetDoc(IsNan_ver1_doc)
        .Input(0, "X", "Input tensor", "T1")
        .Output(0, "Y", "Output bool tensor", "T2")
        .TypeConstraint("T1", tf_compat_float_types,
                        "Constrain input to floating-point tensors.")
        .TypeConstraint("T2", {"tensor(bool)"},
                        "Constrain output tensor type to bool.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          ONNX_NAMESPACE::updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto_DataType_BOOL);
          if (ctx.getInputType(0) != nullptr && ctx.getInputType(0)->has_tensor_type() &&
              ctx.getInputType(0)->tensor_type().has_shape()) {
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() =
                ctx.getInputType(0)->tensor_type().shape();
          }
        }));

#define ONNX_DEPRECATED_LOGICAL_BINARY_SCHEMA(OpName, Doc)                              \
  ONNX_CONTRIB_OPERATOR_SET_SCHEMA(                                                     \
      OpName, 1,                                                                         \
      OpSchema()                                                                         \
          .SetDoc(Doc)                                                                   \
          .Input(0, "A", "First bool tensor", "T")                          \
          .Input(1, "B", "Second bool tensor", "T")                         \
          .Output(0, "C", "Output bool tensor", "T")                        \
          .TypeConstraint("T", {"tensor(bool)"}, "Constrain tensors to bool.") \
          .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {     \
            ONNX_NAMESPACE::updateOutputElemType(ctx, 0, ONNX_NAMESPACE::TensorProto_DataType_BOOL); \
            if (ONNX_NAMESPACE::hasNInputShapes(ctx, 2)) {                                \
              ONNX_NAMESPACE::bidirectionalBroadcastShapeInference(                       \
                  ctx.getInputType(0)->tensor_type().shape(),                             \
                  ctx.getInputType(1)->tensor_type().shape(),                             \
                  *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());          \
            }                                                                             \
          }));

ONNX_DEPRECATED_LOGICAL_BINARY_SCHEMA(LogicalAnd, LogicalAnd_ver1_doc)
ONNX_DEPRECATED_LOGICAL_BINARY_SCHEMA(LogicalOr, LogicalOr_ver1_doc)
#undef ONNX_DEPRECATED_LOGICAL_BINARY_SCHEMA

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    LogicalNot, 1,
    OpSchema()
        .SetDoc(LogicalNot_ver1_doc)
        .Input(0, "X", "Input bool tensor", "T")
        .Output(0, "Y", "Output bool tensor", "T")
        .TypeConstraint("T", {"tensor(bool)"}, "Constrain tensors to bool.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* PadV2_ver11_doc = R"DOC(
PadV2 pads an input tensor using a pads tensor and scalar constant value. This
schema is kept for TensorFlow-converted models that emit PadV2 in the ONNX
domain. It uses the dynamic-input Pad signature introduced by ONNX Pad-11.
)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    PadV2, 11,
    OpSchema()
        .SetDoc(PadV2_ver11_doc)
        .Attr("mode", "Supported values are constant, reflect, edge and wrap.",
              AttributeProto::STRING, std::string("constant"))
        .Input(0, "data", "Input tensor", "T")
        .Input(1, "pads", "Padding for the beginning and ending of each axis.", "Tpads")
        .Input(2, "constant_value", "Scalar padding value.", "T", OpSchema::Optional)
        .Output(0, "output", "Padded tensor", "T")
        .TypeConstraint("T", tf_compat_zeros_like_types,
                        "Constrain input and output tensor types.")
        .TypeConstraint("Tpads", {"tensor(int64)"},
                        "Constrain pads tensor type to int64.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        }));

constexpr const char* ExpandDims_ver1_doc = R"DOC(
ExpandDims inserts a dimension of size 1 at the axis provided by the dim input.
This schema is kept for TensorFlow-converted models that emit ExpandDims in the
ONNX domain.
)DOC";

constexpr const char* StopGradient_ver1_doc = R"DOC(
StopGradient forwards its input tensor unchanged for inference. This schema is
kept for TensorFlow-converted models that emit StopGradient in the ONNX domain.
)DOC";

constexpr const char* Snapshot_ver1_doc = R"DOC(
Snapshot forwards its input tensor unchanged for inference. This schema is kept
for TensorFlow-converted models that emit Snapshot in the ONNX domain.
)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    ExpandDims, 1,
    OpSchema()
        .SetDoc(ExpandDims_ver1_doc)
        .Input(0, "input", "Input tensor", "T")
        .Input(1, "dim", "Scalar or 1-D axis tensor", "Tdim")
        .Output(0, "output", "Output tensor", "T")
        .TypeConstraint("T", tf_compat_zeros_like_types,
                        "Constrain input and output tensor types.")
        .TypeConstraint("Tdim", {"tensor(int64)"},
                        "Constrain axis tensor type to int64.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        }));

#define ONNX_DEPRECATED_IDENTITY_SCHEMA(OpName, Doc)                                    \
  ONNX_CONTRIB_OPERATOR_SET_SCHEMA(                                                     \
      OpName, 1,                                                                         \
      OpSchema()                                                                         \
          .SetDoc(Doc)                                                                   \
          .Input(0, "input", "Input tensor", "T")                             \
          .Output(0, "output", "Output tensor", "T")                          \
          .TypeConstraint("T", tf_compat_zeros_like_types,                           \
                          "Constrain input and output tensor types.")                 \
          .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_DEPRECATED_IDENTITY_SCHEMA(StopGradient, StopGradient_ver1_doc)
ONNX_DEPRECATED_IDENTITY_SCHEMA(Snapshot, Snapshot_ver1_doc)
#undef ONNX_DEPRECATED_IDENTITY_SCHEMA

constexpr const char* BitwiseBinary_ver1_doc = R"DOC(
Applies a bitwise binary operation elementwise with Numpy-style broadcasting.
This schema keeps TensorFlow-converted opset 17 models that emitted ONNX-domain
Bitwise* nodes loadable on ORT versions where the official schema starts later.
)DOC";

constexpr const char* BitwiseNot_ver1_doc = R"DOC(
Applies bitwise not elementwise. This schema keeps TensorFlow-converted opset
17 models that emitted ONNX-domain BitwiseNot loadable on ORT versions where
the official schema starts later.
)DOC";

static const std::vector<std::string> bitwise_integer_types = {
    "tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)",
    "tensor(uint8)", "tensor(uint16)", "tensor(uint32)", "tensor(uint64)"};

#define ONNX_DEPRECATED_BITWISE_BINARY_SCHEMA(OpName)                                      \
  ONNX_CONTRIB_OPERATOR_SET_SCHEMA(                                                        \
      OpName, 1,                                                                            \
      OpSchema()                                                                            \
          .SetDoc(BitwiseBinary_ver1_doc)                                                   \
          .Input(0, "A", "First input tensor", "T")                                  \
          .Input(1, "B", "Second input tensor", "T")                                 \
          .Output(0, "C", "Output tensor", "T")                                      \
          .TypeConstraint("T", bitwise_integer_types,                                      \
                          "Constrain input and output types to integer tensors.")          \
          .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {        \
            ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);                  \
            if (ONNX_NAMESPACE::hasNInputShapes(ctx, 2)) {                                  \
              ONNX_NAMESPACE::bidirectionalBroadcastShapeInference(                         \
                  ctx.getInputType(0)->tensor_type().shape(),                               \
                  ctx.getInputType(1)->tensor_type().shape(),                               \
                  *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());            \
            }                                                                                \
          }));

ONNX_DEPRECATED_BITWISE_BINARY_SCHEMA(BitwiseAnd)
ONNX_DEPRECATED_BITWISE_BINARY_SCHEMA(BitwiseOr)
ONNX_DEPRECATED_BITWISE_BINARY_SCHEMA(BitwiseXor)

#undef ONNX_DEPRECATED_BITWISE_BINARY_SCHEMA

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    BitwiseNot, 1,
    OpSchema()
        .SetDoc(BitwiseNot_ver1_doc)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T", bitwise_integer_types,
                        "Constrain input and output types to integer tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* ImageScaler_ver1_doc =
    R"DOC(Scale and bias the input image. Bias values are stored in
the same ordering as the image pixel format.)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    ImageScaler, 1,
    OpSchema()
        .SetDoc(ImageScaler_ver1_doc)
        .Attr("bias", "Bias applied to each channel, same size as C.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("scale", "The scale to apply.", AttributeProto::FLOAT, 1.0f)
        .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
        .Output(0, "output", "Result, has same shape and type as input", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* Crop_ver1_doc =
    R"DOC(Crop and image to the specified spatial dimensions. If scale is given,
then optionally start the crop offset by the left/top border amounts.
If scale is not provided, crop the borders as provided.)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    Crop, 1,
    OpSchema()
        .SetDoc(Crop_ver1_doc)
        .Attr("border", "A 1-D values of (leftBorder, topBorder, rightBorder, bottomBorder).", AttributeProto::INTS,
              OPTIONAL_VALUE)
        .Attr("scale", "A 1-D values of (height, width).", AttributeProto::INTS, OPTIONAL_VALUE)
        .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
        .Output(0, "output", "Result, has same type as input, with H and W dimensions reduced.", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors."));

constexpr const char* ThresholdedRelu_ver1_doc = R"DOC(
ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise. )DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    ThresholdedRelu, 1,
    OpSchema()
        .SetDoc(ThresholdedRelu_ver1_doc)
        .Attr("alpha", "Threshold value", AttributeProto::FLOAT, 1.0f)
        .Input(0, "X", "Input tensor", "T")
        .Output(0, "Y", "Output tensor", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* DynamicSlice_ver1_doc = R"DOC(
Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
Slices uses `axes`, `starts` and `ends` inputs to specify the start and end
dimension for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represent number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`.
If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
Example 1:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]
  result = [
      [5, 6, 7],
  ]
Example 2:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 1000]
  result = [
      [2, 3, 4],
  ]
)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    DynamicSlice, 1,
    OpSchema()
        .SetDoc(DynamicSlice_ver1_doc)
        .Input(0, "data", "Tensor of data to extract slices from.", "T")
        .Input(1, "starts", "1-D tensor of starting indices of corresponding axis in `axes`", "Tind")
        .Input(2, "ends", "1-D tensor of ending indices (exclusive) of corresponding axis in axes", "Tind")
        .Input(3, "axes", "1-D tensor of axes that `starts` and `ends` apply to.", "Tind", OpSchema::Optional)
        .Output(0, "output", "Sliced data tensor.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain indices to integer types"));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(GivenTensorFill, 1,
                                 OpSchema()
                                     .Input(0, "shape", "The shape of filled tensor", "T", OpSchema::Optional)
                                     .Output(0, "X", "The filled tensor", "T")
                                     .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                                                     "Constrain input and output types to float tensors.")
                                     .Attr("values", "", AttributeProto::FLOATS, OPTIONAL_VALUE)
                                     .Attr("shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
                                     .Attr("input_as_shape", "", AttributeProto::INT, OPTIONAL_VALUE)
                                     .Attr("extra_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
                                     .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                       ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
                                       if (ctx.getAttribute("shape") != nullptr) {
                                         propagateShapeFromAttributeToOutput(ctx, "shape", 0);
                                         return;
                                       }
                                       // The type constraints above do not allow for input_as_shape
                                       // and may need to be fixed.
                                       if (getAttribute(ctx, "input_as_shape", 0) != 0)  // dynamic shape
                                         return;
                                       std::vector<int64_t> extra_shape;
                                       getRepeatedAttribute(ctx, "extra_shape", extra_shape);
                                       if (hasInputShape(ctx, 0)) {
                                         ONNX_NAMESPACE::TensorShapeProto shape =
                                             ctx.getInputType(0)->tensor_type().shape();
                                         for (auto extra_dim_val : extra_shape) {
                                           if (extra_dim_val < 0)
                                             fail_shape_inference(
                                                 "Negative values are not allowed in a shape specification");
                                           shape.add_dim()->set_dim_value(extra_dim_val);
                                         }
                                         updateOutputShape(ctx, 0, shape);
                                       }
                                     }));

constexpr const char* Scale_ver1_doc = R"DOC(
Scale takes one input data (Tensor<float>) and produces one output data
(Tensor<float>) whose value is the input data tensor scaled element-wise.
)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    Scale, 1,
    OpSchema()
        .Input(0, "input", "Input data to be scaled", "T")
        .Output(0, "output", "Output data after scaling", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .SetDoc(Scale_ver1_doc)
        .Attr("scale", "The scale to apply.", AttributeProto::FLOAT, 1.0f)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* GRUUnit_ver1_doc = R"DOC(
GRUUnit computes the activations of a standard GRU,
in a sequence-length aware fashion.
Concretely, given the (fused) inputs X (TxNxD), the previous hidden
state (NxD), and the sequence lengths (N), computes the GRU
activations, avoiding computation if the input is invalid (as in, the
value at X[t][n] >= seqLengths[n].
)DOC";

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(GRUUnit, 1,
                                 OpSchema()
                                     .SetDoc(GRUUnit_ver1_doc)
                                     .Attr("drop_states",
                                           "Bool to determine if hidden state is zeroes or passed "
                                           "along for timesteps past the given sequence_length.",
                                           AttributeProto::INT, OPTIONAL_VALUE)
                                     .Input(0, "hidden_prev", "The previous GRU hidden state.", "T")
                                     .Input(1, "gates",
                                            "Unactivated gate outputs from forget, update, "
                                            "and output gates, pre-activation.",
                                            "T")
                                     .Input(2, "seq_lengths",
                                            "Array of sequence lengths.  "
                                            "len(seq_lengths) should equal batch size N.",
                                            "T")
                                     .Input(3, "t", "The timestep for this operation.", "T")
                                     .Output(0, "hidden", "The new GRU hidden state calculated by this op.", "T")
                                     .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                                                     "Constrain input and output types to float tensors."));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(GivenTensorFill, 10,
                                 OpSchema()
                                     .Deprecate()
                                     .Input(0, "shape", "The shape of filled tensor", "T", OpSchema::Optional)
                                     .Output(0, "X", "The filled tensor", "T")
                                     .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                                                     "Constrain input and output types to float tensors.")
                                     .Attr("values", "", AttributeProto::FLOATS, OPTIONAL_VALUE)
                                     .Attr("shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
                                     .Attr("input_as_shape", "", AttributeProto::INT, OPTIONAL_VALUE)
                                     .Attr("extra_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
                                     .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                       ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
                                       if (ctx.getAttribute("shape") != nullptr) {
                                         propagateShapeFromAttributeToOutput(ctx, "shape", 0);
                                         return;
                                       }
                                       // The type constraints above do not allow for input_as_shape
                                       // and may need to be fixed.
                                       if (getAttribute(ctx, "input_as_shape", 0) != 0)  // dynamic shape
                                         return;
                                       std::vector<int64_t> extra_shape;
                                       getRepeatedAttribute(ctx, "extra_shape", extra_shape);
                                       if (hasInputShape(ctx, 0)) {
                                         ONNX_NAMESPACE::TensorShapeProto shape =
                                             ctx.getInputType(0)->tensor_type().shape();
                                         for (auto extra_dim_val : extra_shape) {
                                           if (extra_dim_val < 0)
                                             fail_shape_inference(
                                                 "Negative values are not allowed in a shape specification");
                                           shape.add_dim()->set_dim_value(extra_dim_val);
                                         }
                                         updateOutputShape(ctx, 0, shape);
                                       }
                                     }));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    Scale, 10,
    OpSchema()
        .Deprecate()
        .Input(0, "input", "Input data to be scaled", "T")
        .Output(0, "output", "Output data after scaling", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .SetDoc(Scale_ver1_doc)
        .Attr("scale", "The scale to apply.", AttributeProto::FLOAT, 1.0f)
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(GRUUnit, 10,
                                 OpSchema()
                                     .Deprecate()
                                     .SetDoc(GRUUnit_ver1_doc)
                                     .Attr("drop_states",
                                           "Bool to determine if hidden state is zeroes or passed "
                                           "along for timesteps past the given sequence_length.",
                                           AttributeProto::INT, OPTIONAL_VALUE)
                                     .Input(0, "hidden_prev", "The previous GRU hidden state.", "T")
                                     .Input(1, "gates",
                                            "Unactivated gate outputs from forget, update, "
                                            "and output gates, pre-activation.",
                                            "T")
                                     .Input(2, "seq_lengths",
                                            "Array of sequence lengths.  "
                                            "len(seq_lengths) should equal batch size N.",
                                            "T")
                                     .Input(3, "t", "The timestep for this operation.", "T")
                                     .Output(0, "hidden", "The new GRU hidden state calculated by this op.", "T")
                                     .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                                                     "Constrain input and output types to float tensors."));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    MeanVarianceNormalization, 1,
    OpSchema()
        .SetDoc(R"DOC(Perform mean variance normalization.)DOC")
        .Attr("across_channels", "If 1, mean and variance are computed across channels. Default is 0.",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("normalize_variance", "If 0, normalize the mean only.  Default is 1.", AttributeProto::INT,
              static_cast<int64_t>(1))
        .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
        .Output(0, "output", "Result, has same shape and type as input", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    ScaledTanh, 1,
    OpSchema()
        .Attr("alpha", "Scaling value", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("beta", "Scaling value", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Input(0, "input", "Input tensor", "T")
        .Output(0, "output",
                "The scaled hyperbolic tangent values of the input tensor "
                "computed element-wise",
                "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    Affine, 10,
    OpSchema()
        .Deprecate()
        .SetDoc(Affine_ver1_doc)
        .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, 1.0f)
        .Attr("beta", "Value of beta", AttributeProto::FLOAT, 0.0f)
        .Input(0, "X", "1D input tensor", "T")
        .Output(0, "Y", "1D output tensor", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    ParametricSoftplus, 10,
    OpSchema()
        .Deprecate()
        .SetDoc(ParametricSoftplus_ver1_doc)
        .Attr("alpha", "Value of alpha", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("beta", "Value of beta", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Input(0, "X", "1D input tensor", "T")
        .Output(0, "Y", "1D input tensor", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    ImageScaler, 10,
    OpSchema()
        .Deprecate()
        .SetDoc(ImageScaler_ver1_doc)
        .Attr("bias", "Bias applied to each channel, same size as C.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("scale", "The scale to apply.", AttributeProto::FLOAT, 1.0f)
        .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
        .Output(0, "output", "Result, has same shape and type as input", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    Crop, 10,
    OpSchema()
        .Deprecate()
        .SetDoc(Crop_ver1_doc)
        .Attr("border", "A 1-D values of (leftBorder, topBorder, rightBorder, bottomBorder).", AttributeProto::INTS)
        .Attr("scale", "A 1-D values of (height, width).", AttributeProto::INTS, OPTIONAL_VALUE)
        .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
        .Output(0, "output", "Result, has same type as input, with H and W dimensions reduced.", "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
          // Type inference
          ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Shape inference
          auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

          if (ONNX_NAMESPACE::hasNInputShapes(ctx, 1)) {
            const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
            const auto input_rank = input_shape.dim_size();
            if (input_rank != 4) fail_shape_inference("Input's shape must be 4-D");

            // parse necessary attributes for further processing
            std::vector<int64_t> border;
            bool border_present = getRepeatedAttribute(ctx, "border", border);
            if (!border_present || border.size() != 4)
              fail_shape_inference(
                  "'Border' attribute must be present and must contain exactly 4 values - "
                  "(left_border, top_border, right_border, bottom_border)");

            std::vector<int64_t> scale;
            bool scale_present = getRepeatedAttribute(ctx, "scale", scale);
            if (scale_present && scale.size() != 2)
              fail_shape_inference("'Scale' must contain exactly 2 values - (height, width)");

            // actual shape inference processing
            // [N, C] can be copied over from the input as is
            *output_shape->mutable_dim(static_cast<int>(0)) = input_shape.dim(static_cast<int>(0));
            *output_shape->mutable_dim(static_cast<int>(1)) = input_shape.dim(static_cast<int>(1));

            // process 'H' and 'W'
            if (!utils::HasDimValue(input_shape.dim(static_cast<int>(2))) ||
                !utils::HasDimValue(input_shape.dim(static_cast<int>(3)))) {
              // either height and width input has symbolic dims, so can't proceed further
              // add two dims as placeholders for output_H and output_W and return
              output_shape->add_dim();
              output_shape->add_dim();
              return;
            }

            int64_t H = input_shape.dim(static_cast<int>(2)).dim_value();
            int64_t W = input_shape.dim(static_cast<int>(3)).dim_value();

            int64_t left_border = border[0], top_border = border[1], right_border = border[2],
                    bottom_border = border[3];

            if (H < top_border + bottom_border)
              fail_shape_inference("Input's height (", H,
                                   ") needs to be greater than or equal to "
                                   "the top_border (",
                                   top_border, ") + bottom_border (", bottom_border, ")");

            if (W < left_border + right_border)
              fail_shape_inference("Input's width (", W,
                                   ") needs to be greater than or equal to "
                                   "the left_border (",
                                   left_border, ") + right_border (", right_border, ")");

            int64_t bottom_limit = H - bottom_border;
            int64_t right_limit = W - right_border;

            // scale = (height, width)
            if (!scale.empty()) {
              bottom_limit = top_border + scale[0];
              right_limit = left_border + scale[1];

              if (H < bottom_limit)
                fail_shape_inference("Input's height (", H, ") needs to be greater than or equal to the top_border (",
                                     top_border, ") + scale[0] (", scale[0], ")");

              if (W < right_limit)
                fail_shape_inference("Input's width (", W, ") needs to be greater than or equal to the left_border (",
                                     left_border, ") + scale[1] (", scale[1], ")");
            }

            auto* h_output_dim = output_shape->add_dim();
            h_output_dim->set_dim_value(bottom_limit - top_border);

            auto* w_output_dim = output_shape->add_dim();
            w_output_dim->set_dim_value(right_limit - left_border);
          } else {
            // Rank Inference at the very least
            // (We know that the output is going to be 4-D)
            for (int i = 0; i < 4; ++i) {
              output_shape->add_dim();
            }
          }
        }));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    DynamicSlice, 10,
    OpSchema()
        .Deprecate()
        .SetDoc(DynamicSlice_ver1_doc)
        .Input(0, "data", "Tensor of data to extract slices from.", "T")
        .Input(1, "starts", "1-D tensor of starting indices of corresponding axis in `axes`", "Tind")
        .Input(2, "ends", "1-D tensor of ending indices (exclusive) of corresponding axis in axes", "Tind")
        .Input(3, "axes", "1-D tensor of axes that `starts` and `ends` apply to.", "Tind", OpSchema::Optional)
        .Output(0, "output", "Sliced data tensor.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "Constrain indices to integer types"));

ONNX_CONTRIB_OPERATOR_SET_SCHEMA(
    ScaledTanh, 10,
    OpSchema()
        .Deprecate()
        .Attr("alpha", "Scaling value", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("beta", "Scaling value", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Input(0, "input", "Input tensor", "T")
        .Output(0, "output",
                "The scaled hyperbolic tangent values of the input tensor "
                "computed element-wise",
                "T")
        .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"},
                        "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

// End of ONNX exp ops(Affine, Crop, ParametricSoftplus, ImageScaler, ThresholdedRelu, DynamicSlice, ScaledTanh, MVN)
// old version history maintenance
}  // namespace contrib
}  // namespace onnxruntime
