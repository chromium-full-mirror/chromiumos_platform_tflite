/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/openvino_delegate_kernel.h"

#include <openvino/runtime/core.hpp>

#include <vector>

#include "delegate/intel_openvino/log.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus OpenVINODelegateKernel::Init(
    TfLiteOpaqueContext *context, const TfLiteOpaqueDelegateParams *params) {
  TfLiteStatus init_status = ov_delegate_core_->Init();
  if (init_status != kTfLiteOk) return init_status;

  TfLiteStatus set_status =
      ov_delegate_core_->CreateGraphfromTfLite(context, params);
  if (set_status != kTfLiteOk) return set_status;

  return kTfLiteOk;
}

TfLiteStatus OpenVINODelegateKernel::Prepare(TfLiteOpaqueContext *context,
                                             TfLiteOpaqueNode *node) {
  TFLITE_LOG(INFO) << "inside Prepare";
  return kTfLiteOk;
}

TfLiteStatus OpenVINODelegateKernel::Eval(TfLiteOpaqueContext *context,
                                          TfLiteOpaqueNode *node) {
  std::vector<int> compute_inputs = ov_delegate_core_->getComputeInputs();
  for (int i = 0; i < compute_inputs.size(); i++) {
    int t = compute_inputs[i];
    ov::Tensor inputBlob =
        ov_delegate_core_->getInferRequest().get_input_tensor(i);
    void *dest = inputBlob.data();

    const TfLiteOpaqueTensor *opaque_input_tensor =
        TfLiteOpaqueContextGetOpaqueTensor(context, t);
    auto len = TfLiteOpaqueTensorByteSize(opaque_input_tensor);
    void *src = TfLiteOpaqueTensorData(opaque_input_tensor);

    std::memcpy(dest, src, len);
  }

  ov_delegate_core_->getInferRequest().start_async();
  if (!ov_delegate_core_->getInferRequest().wait_for(
          std::chrono::milliseconds(10000))) {
    TFLITE_LOG(ERROR) << "Infer request failed";
    return kTfLiteError;
  }

  std::vector<int> outputs = ov_delegate_core_->getOutputs();
  for (int o = 0; o < outputs.size(); o++) {
    int t = outputs[o];
    ov::Tensor outputBlob =
        ov_delegate_core_->getInferRequest().get_output_tensor(o);
    const TfLiteOpaqueTensor *opaque_output_tensor =
        TfLiteOpaqueContextGetOpaqueTensor(context, t);
    void *dest = TfLiteOpaqueTensorData(opaque_output_tensor);
    void *src = outputBlob.data();
    auto len = TfLiteOpaqueTensorByteSize(opaque_output_tensor);
    std::memcpy(dest, src, len);
  }

  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
