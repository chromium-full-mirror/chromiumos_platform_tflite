/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_CORE_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_CORE_H_

#include <openvino/openvino.hpp>
#include <vector>

#include "openvino_graph_builder.h"

namespace tflite {
namespace openvinodelegate {
class OpenVINODelegateCore {
 public:
  OpenVINODelegateCore(std::string plugins_path)
      : ov_core_(ov::Core(plugins_path)) {}
  TfLiteStatus Init();

  std::vector<int> getComputeInputs() { return compute_inputs_; }

  std::vector<int> getOutputs() { return outputs_; }

  ov::InferRequest getInferRequest() const { return infer_request_; }

  TfLiteStatus CreateGraphfromTfLite(TfLiteOpaqueContext *context,
                                     const TfLiteOpaqueDelegateParams *params);

 private:
  std::unique_ptr<OpenVINOGraphBuilder> openvino_graph_builder_;
  ov::Core ov_core_;
  std::shared_ptr<ov::Model> model_;
  ov::CompiledModel compiled_model_;
  std::vector<int> compute_inputs_;
  std::vector<int> outputs_;
  ov::InferRequest infer_request_;
};

}  // namespace openvinodelegate
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_KERNEL_H_
