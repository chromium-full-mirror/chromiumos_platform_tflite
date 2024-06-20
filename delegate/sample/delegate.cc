/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "delegate/sample/delegate.h"

#include <memory>

#include "common/simple_async_delegate.h"
#include "delegate/sample/kernel.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/core/c/c_api_opaque.h"
#include "tensorflow/lite/core/c/common.h"

namespace tflite::cros {

bool CrosSampleDelegate::IsNodeSupportedByDelegate(
    const TfLiteRegistrationExternal* registration_external,
    const TfLiteOpaqueNode* node,
    TfLiteOpaqueContext* context) const {
  TfLiteBuiltinOperator builtin_operator =
      TfLiteOperatorGetBuiltInCode(registration_external);
  void* builtin_data = TfLiteOpaqueNodeGetBuiltinData(node);

  // The delegate only supports addition and subtraction operations without
  // fused activation.
  switch (builtin_operator) {
    case kTfLiteBuiltinAdd: {
      auto* params = reinterpret_cast<TfLiteAddParams*>(builtin_data);
      if (params == nullptr || params->activation != kTfLiteActNone) {
        return false;
      }
      break;
    }
    case kTfLiteBuiltinSub: {
      auto* params = reinterpret_cast<TfLiteSubParams*>(builtin_data);
      if (params == nullptr || params->activation != kTfLiteActNone) {
        return false;
      }
      break;
    }
    default: {
      return false;
    }
  }

  // The delegate only accepts two inputs with float32 type.
  if (TfLiteOpaqueNodeNumberOfInputs(node) != 2) {
    return false;
  }
  const auto* input1 = TfLiteOpaqueNodeGetInput(context, node, 0);
  const auto* input2 = TfLiteOpaqueNodeGetInput(context, node, 1);
  if (input1 == nullptr || TfLiteOpaqueTensorType(input1) != kTfLiteFloat32 ||
      input2 == nullptr || TfLiteOpaqueTensorType(input2) != kTfLiteFloat32) {
    return false;
  }

  // The delegate doesn't support broadcasting; it requires both inputs to have
  // the same shape.
  int num_dims = TfLiteOpaqueTensorNumDims(input1);
  if (num_dims != TfLiteOpaqueTensorNumDims(input2)) {
    return false;
  }
  for (int i = 0; i < num_dims; ++i) {
    if (TfLiteOpaqueTensorDim(input1, i) != TfLiteOpaqueTensorDim(input2, i)) {
      return false;
    }
  }

  return true;
}

TfLiteStatus CrosSampleDelegate::Initialize(TfLiteOpaqueContext* context) {
  return kTfLiteOk;
}

const char* CrosSampleDelegate::Name() const {
  return kName;
}

std::unique_ptr<SimpleAsyncDelegateKernelInterface>
CrosSampleDelegate::CreateDelegateKernelInterface() {
  return std::make_unique<CrosSampleDelegateKernel>();
}

}  // namespace tflite::cros
