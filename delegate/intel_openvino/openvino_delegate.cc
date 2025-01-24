/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/openvino_delegate.h"

#include <memory>
#include <vector>

#include "delegate/intel_openvino/openvino_delegate_kernel.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"

namespace tflite {
namespace openvinodelegate {
bool OpenVINODelegate::CheckInputType(TfLiteType tensor_type,
                                      TfLiteType expected_type) const {
  return expected_type == tensor_type;
}

bool OpenVINODelegate::CheckDataTypeSupported(
    const TfLiteOpaqueContext *context, const TfLiteOpaqueNode *node,
    const std::vector<std::vector<TfLiteType>> supported_types) const {
  const int *inputs;
  int num_inputs;
  if (TfLiteOpaqueNodeInputs(node, &inputs, &num_inputs) != kTfLiteOk)
    return false;

  if (num_inputs < supported_types.size()) return false;

  for (int i = 0; i < supported_types.size(); i++) {
    int tensor_id = inputs[i];
    bool supported = false;
    const TfLiteOpaqueTensor *opaque_tensor =
        TfLiteOpaqueContextGetOpaqueTensor(context, tensor_id);
    TfLiteType type = TfLiteOpaqueTensorType(opaque_tensor);
    for (TfLiteType supported_type : supported_types[i])
      supported |= CheckInputType(type, supported_type);
    if (!supported) return false;
  }

  return true;
}

bool OpenVINODelegate::CheckDims(
    const TfLiteOpaqueContext *context, const TfLiteOpaqueNode *node,
    const std::vector<std::vector<int>> &dims_size) const {
  const int *inputs;
  int num_inputs;
  if (TfLiteOpaqueNodeInputs(node, &inputs, &num_inputs) != kTfLiteOk)
    return false;
  if (num_inputs < dims_size.size()) return false;

  for (int i = 0; i < dims_size.size(); i++) {
    bool supported = false;
    const TfLiteOpaqueTensor *opaque_tensor =
        TfLiteOpaqueContextGetOpaqueTensor(context, inputs[i]);
    for (int j = 0; j < dims_size[i].size(); j++) {
      if (TfLiteOpaqueTensorNumDims(opaque_tensor) == dims_size[i][j]) {
        supported |= true;
        for (int k = 0; k < dims_size[i][j]; k++)
          if (TfLiteOpaqueTensorDim(opaque_tensor, k) == 0) return false;
      }
    }
    if (!supported) return false;
  }

  return true;
}

bool OpenVINODelegate::CheckNodeSupportByOpenVINO(
    const TfLiteRegistrationExternal *registration,
    const TfLiteOpaqueNode *node, const TfLiteOpaqueContext *context) const {
  const int *inputs;
  int num_inputs;
  if (TfLiteOpaqueNodeInputs(node, &inputs, &num_inputs) != kTfLiteOk)
    return false;
  switch (TfLiteOperatorGetBuiltInCode(registration)) {
    case kTfLiteBuiltinAdd: {
      return CheckDataTypeSupported(context, node,
                                    {{kTfLiteFloat32}, {kTfLiteFloat32}}) &&
             CheckDims(context, node, {{1, 2, 3, 4}, {1, 2, 3, 4}});
    }
    case kTfLiteBuiltinAveragePool2d: {
      return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}}) &&
             CheckDims(context, node, {{4}});
    }
    case kTfLiteBuiltinConv2d: {
      if (num_inputs == 2) {
        return CheckDataTypeSupported(context, node,
                                      {{kTfLiteFloat32}, {kTfLiteFloat32}}) &&
               CheckDims(context, node, {{4}, {4}});
      } else if (num_inputs == 3) {
        return CheckDataTypeSupported(
                   context, node,
                   {{kTfLiteFloat32}, {kTfLiteFloat32}, {kTfLiteFloat32}}) &&
               CheckDims(context, node, {{4}, {4}, {1}});
      } else {
        return false;
      }
    }
    case kTfLiteBuiltinConcatenation: {
      // NOTE: Concatenation is allowed to have variadic input tensors , but we
      // check type for 2 input tensors. Rest are assumed to have same type if
      // present
      return CheckDataTypeSupported(context, node,
                                    {{kTfLiteFloat32}, {kTfLiteFloat32}}) &&
             CheckDims(context, node, {{1, 2, 3, 4}, {1, 2, 3, 4}});
    }
    case kTfLiteBuiltinDepthwiseConv2d: {
      if (num_inputs == 2) {
        return CheckDataTypeSupported(context, node,
                                      {{kTfLiteFloat32}, {kTfLiteFloat32}}) &&
               CheckDims(context, node, {{4}, {4}});
      } else if (num_inputs == 3) {
        return CheckDataTypeSupported(
                   context, node,
                   {{kTfLiteFloat32}, {kTfLiteFloat32}, {kTfLiteFloat32}}) &&
               CheckDims(context, node, {{4}, {4}, {1}});
      } else {
        return false;
      }
    }
    case kTfLiteBuiltinDequantize: {
      return CheckDataTypeSupported(context, node, {{kTfLiteFloat16}});
    }
    case kTfLiteBuiltinResizeBilinear: {
      return CheckDataTypeSupported(context, node,
                                    {{kTfLiteFloat32}, {kTfLiteInt32}});
    }
    case kTfLiteBuiltinRelu: {
      return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}});
    }
    case kTfLiteBuiltinRelu6: {
      return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}});
    }
    case kTfLiteBuiltinLogistic: {
      return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}});
    }
    case kTfLiteBuiltinHardSwish: {
      return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}});
    }
    case kTfLiteBuiltinMul: {
      return CheckDataTypeSupported(context, node,
                                    {{kTfLiteFloat32}, {kTfLiteFloat32}}) &&
             CheckDims(context, node, {{1, 2, 3, 4}, {1, 2, 3, 4}});
    }
    case kTfLiteBuiltinSoftmax: {
      auto *softmax_params = reinterpret_cast<TfLiteSoftmaxParams *>(
          TfLiteOpaqueNodeGetBuiltinData(node));
      if (softmax_params->beta != 1.0f) {
        TFLITE_LOG(INFO) << "Unsupported Softmax op, beta value is not 1.0";
        return false;
      }
      return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}});
    }
    case kTfLiteBuiltinTanh: {
      return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}});
    }
    case kTfLiteBuiltinReshape: {
      return CheckDataTypeSupported(context, node,
                                    {{kTfLiteFloat32}, {kTfLiteInt32}}) &&
             CheckDims(context, node, {{1, 2, 3, 4}, {1}});
    }
    case kTfLiteBuiltinMaxPool2d: {
      return CheckDataTypeSupported(context, node, {{kTfLiteFloat32}}) &&
             CheckDims(context, node, {{4}});
    }
    case kTfLiteBuiltinMean: {
      return CheckDataTypeSupported(context, node,
                                    {{kTfLiteFloat32}, {kTfLiteInt32}}) &&
             CheckDims(context, node, {{4}, {1}});
    }
    case kTfLiteBuiltinTransposeConv: {
      if (num_inputs == 3) {
        return CheckDataTypeSupported(
                   context, node,
                   {{kTfLiteInt32}, {kTfLiteFloat32}, {kTfLiteFloat32}}) &&
               CheckDims(context, node, {{1}, {4}, {4}});
      } else if (num_inputs == 4) {
        return CheckDataTypeSupported(context, node,
                                      {{kTfLiteInt32},
                                       {kTfLiteFloat32},
                                       {kTfLiteFloat32},
                                       {kTfLiteFloat32}}) &&
               CheckDims(context, node, {{1}, {4}, {4}, {1}});
      } else {
        return false;
      }
    }
    case kTfLiteBuiltinPad: {
      return CheckDataTypeSupported(
                 context, node,
                 {{kTfLiteFloat32}, {kTfLiteInt32, kTfLiteInt64}}) &&
             CheckDims(context, node, {{1, 2, 3, 4}, {2}});
    }
    case kTfLiteBuiltinSub: {
      return CheckDataTypeSupported(context, node,
                                    {{kTfLiteFloat32}, {kTfLiteFloat32}}) &&
             CheckDims(context, node, {{1, 2, 3, 4}, {1, 2, 3, 4}});
    }
    case kTfLiteBuiltinPrelu: {
      return CheckDataTypeSupported(context, node,
                                    {{kTfLiteFloat32}, {kTfLiteFloat32}}) &&
             CheckDims(context, node, {{4}, {1, 2, 3}});
    }
    default:
      return false;
  }
}

bool OpenVINODelegate::IsNodeSupportedByDelegate(
    const TfLiteRegistrationExternal *registration,
    const TfLiteOpaqueNode *node, TfLiteOpaqueContext *context) const {
  if (registration == nullptr || node == nullptr || context == nullptr)
    return false;
  return CheckNodeSupportByOpenVINO(registration, node, context);
}

TfLiteStatus OpenVINODelegate::Initialize(TfLiteOpaqueContext *context) {
  return kTfLiteOk;
}

const char *OpenVINODelegate::Name() const {
  return "OpenVINO SimpleOpaqueDelegate";
}

std::unique_ptr<cros::SimpleAsyncDelegateKernelInterface>
OpenVINODelegate::CreateDelegateKernelInterface() {
  return std::unique_ptr<tflite::openvinodelegate::OpenVINOAsyncDelegateKernel>(
      new tflite::openvinodelegate::OpenVINOAsyncDelegateKernel(options_));
}

}  // namespace openvinodelegate
}  // namespace tflite
