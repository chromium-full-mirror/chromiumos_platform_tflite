/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "delegate/sample/kernel.h"

#include "tensorflow/lite/core/c/c_api_opaque.h"

namespace tflite::cros {

TfLiteStatus CrosSampleDelegateKernel::Init(
    TfLiteOpaqueContext* context,
    const TfLiteOpaqueDelegateParams* params) {
  return core_.Init(context, params);
}

TfLiteStatus CrosSampleDelegateKernel::Prepare(TfLiteOpaqueContext* context,
                                               TfLiteOpaqueNode* node) {
  return core_.Prepare();
}

TfLiteStatus CrosSampleDelegateKernel::Eval(TfLiteOpaqueContext* context,
                                            TfLiteOpaqueNode* node) {
  int num_inputs = TfLiteOpaqueNodeNumberOfInputs(node);
  for (int i = 0; i < num_inputs; ++i) {
    auto tensor = TfLiteOpaqueNodeGetInput(context, node, i);
    auto data = TfLiteOpaqueTensorData(tensor);
    core_.SetExternalTensorMemory(tensor, data);
  }

  int num_outputs = TfLiteOpaqueNodeNumberOfOutputs(node);
  for (int i = 0; i < num_outputs; ++i) {
    auto tensor = TfLiteOpaqueNodeGetOutput(context, node, i);
    auto data = TfLiteOpaqueTensorData(tensor);
    core_.SetExternalTensorMemory(tensor, data);
  }

  return core_.Eval();
}

TfLiteAsyncKernel* CrosSampleDelegateKernel::AsyncKernel(
    TfLiteOpaqueContext* context,
    TfLiteOpaqueNode* node) {
  return nullptr;
}

}  // namespace tflite::cros
