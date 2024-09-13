/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DELEGATE_INTEL_OPENVINO_OPENVINO_DELEGATE_CORE_H_
#define DELEGATE_INTEL_OPENVINO_OPENVINO_DELEGATE_CORE_H_

#include <openvino/openvino.hpp>
#include <openvino/runtime/remote_context.hpp>

#include <memory>
#include <string>
#include <vector>

#include "delegate/intel_openvino/openvino_delegate.h"
#include "delegate/intel_openvino/openvino_graph_builder.h"

namespace tflite {
namespace openvinodelegate {
class OpenVINODelegateCore {
 public:
  explicit OpenVINODelegateCore(std::string plugins_path)
      : ov_core_(ov::Core(plugins_path)) {}
  TfLiteStatus Init();

  std::vector<int> getComputeInputs() { return compute_inputs_; }

  std::vector<int> getOutputs() { return outputs_; }

  ov::InferRequest getInferRequest() const { return infer_request_; }

  TfLiteStatus CreateModel(TfLiteOpaqueContext *context,
                           const TfLiteOpaqueDelegateParams *params,
                           const TfLiteOpenVINODelegateOptions *options);
  TfLiteStatus CompileAndInfer();

  ov::RemoteContext get_context() {
    return ov_core_.get_default_context("NPU");
  }

 private:
  TfLiteStatus InitializeModel(TfLiteOpaqueContext *context,
                               const TfLiteOpaqueDelegateParams *params,
                               bool caching_enabled,
                               const std::string &cache_file_name = "");
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

#endif  // DELEGATE_INTEL_OPENVINO_OPENVINO_DELEGATE_CORE_H_
