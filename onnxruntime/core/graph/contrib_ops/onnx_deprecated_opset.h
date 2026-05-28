// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/onnx_protobuf.h"
#include "core/graph/contrib_ops/ms_schema.h"

// This file contains deprecated ONNX operators that have been removed from ONNX spec, but we still need to keep them
// to maintain backward compatibility. Strictly speaking, this file doesn't define an opset. It only contains a group
// of operators.

namespace onnxruntime {
namespace contrib {
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Affine);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ParametricSoftplus);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Log1p);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Expm1);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Square);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Rsqrt);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Select);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, SelectV2);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, AddV2);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, BiasAdd);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, BiasAddV1);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, SubV2);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, RealDiv);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, AddN);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, DivNoNan);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, SquaredDifference);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, FloorDiv);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, FloorMod);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Maximum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Minimum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GreaterEqual);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LessEqual);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, NotEqual);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ZerosLike);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, IdentityN);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ShapeN);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ConcatV2);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, SplitV);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReverseV2);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, InvertPermutation);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, BroadcastGradientArgs);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ConcatOffset);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Fill);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, BroadcastTo);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GatherNd);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GatherV2);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, IsNan);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LogicalAnd);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LogicalOr);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LogicalNot);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, PadV2);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ExpandDims);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, StopGradient);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Snapshot);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, BitwiseAnd);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, BitwiseOr);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, BitwiseXor);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, BitwiseNot);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ImageScaler);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Crop);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ThresholdedRelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, DynamicSlice);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GivenTensorFill);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Scale);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GRUUnit);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, GivenTensorFill);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Scale);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, GRUUnit);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, MeanVarianceNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ScaledTanh);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Affine);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ParametricSoftplus);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ImageScaler);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Crop);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, DynamicSlice);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ScaledTanh);

class OpSet_ONNX_Deprecated {
 public:
  static void ForEachSchema(std::function<void(ONNX_NAMESPACE::OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Affine)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ParametricSoftplus)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Log1p)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Expm1)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Square)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Rsqrt)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Select)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, SelectV2)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, AddV2)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, BiasAdd)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, BiasAddV1)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, SubV2)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, RealDiv)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, AddN)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, DivNoNan)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, SquaredDifference)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, FloorDiv)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, FloorMod)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Maximum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Minimum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GreaterEqual)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LessEqual)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, NotEqual)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ZerosLike)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, IdentityN)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ShapeN)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ConcatV2)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, SplitV)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReverseV2)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, InvertPermutation)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, BroadcastGradientArgs)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ConcatOffset)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Fill)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, BroadcastTo)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GatherNd)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GatherV2)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, IsNan)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LogicalAnd)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LogicalOr)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LogicalNot)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, PadV2)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ExpandDims)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, StopGradient)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Snapshot)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, BitwiseAnd)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, BitwiseOr)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, BitwiseXor)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, BitwiseNot)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ImageScaler)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Crop)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ThresholdedRelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, DynamicSlice)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GivenTensorFill)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Scale)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GRUUnit)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, GivenTensorFill)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Scale)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, GRUUnit)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, MeanVarianceNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ScaledTanh)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Affine)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ParametricSoftplus)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ImageScaler)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Crop)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, DynamicSlice)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ScaledTanh)>());
  }
};
}  // namespace contrib
}  // namespace onnxruntime
