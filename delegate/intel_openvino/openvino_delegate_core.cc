/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/openvino_delegate_core.h"

#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_set>

#include "delegate/intel_openvino/operations/openvino_node_manager.h"
#include "tensorflow/lite/c/c_api_opaque.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus OpenVINODelegateCore::Init() {
  std::vector<std::string> ov_devices = ov_core_.get_available_devices();
  if (std::find(ov_devices.begin(), ov_devices.end(), "NPU") ==
      ov_devices.end()) {
    TFLITE_LOG(ERROR) << "Could not find plugin for NPU";
    return kTfLiteDelegateError;
  } else {
    return kTfLiteOk;
  }
}

TfLiteStatus OpenVINODelegateCore::InitializeModel(
    TfLiteOpaqueContext *context, const TfLiteOpaqueDelegateParams *params,
    bool caching_enabled, const std::string &cache_file_name) {
  if (context == nullptr || params == nullptr) return kTfLiteError;
  const std::unordered_set<int> inputs(
      &params->input_tensors->data[0],
      &params->input_tensors->data[params->input_tensors->size]);

  for (int o = 0; o < params->output_tensors->size; o++) {
    const int output_tensor_idx = params->output_tensors->data[o];
    outputs_.push_back(output_tensor_idx);
  }

  if (!caching_enabled)
    openvino_graph_builder_ =
        std::make_unique<OpenVINOGraphBuilder>(std::make_unique<NodeManager>());

  for (int i = 0; i < params->nodes_to_replace->size; i++) {
    const int delegate_node_id = params->nodes_to_replace->data[i];
    TfLiteOpaqueNode *delegate_node;
    TfLiteRegistrationExternal *delegate_node_registration;
    if (TfLiteOpaqueContextGetNodeAndRegistration(
            context, delegate_node_id, &delegate_node,
            &delegate_node_registration) != kTfLiteOk)
      return kTfLiteError;

    const int *inputs_data = nullptr;
    int num_inputs = 0;
    if (TfLiteOpaqueNodeInputs(delegate_node, &inputs_data, &num_inputs) !=
        kTfLiteOk)
      return kTfLiteError;
    for (int k = 0; k < num_inputs; k++) {
      const int t = inputs_data[k];
      if (t == kTfLiteOptionalTensor) continue;
      const void *data = nullptr;
      auto opaque_tensor = TfLiteOpaqueContextGetOpaqueTensor(context, t);
      auto allocation_type = TfLiteOpaqueTensorGetAllocationType(opaque_tensor);
      if (allocation_type == kTfLiteMmapRo ||
          allocation_type == kTfLitePersistentRo) {
        data = TfLiteOpaqueTensorData(opaque_tensor);
        if (!caching_enabled)
          if (openvino_graph_builder_->CreateConstNode(context, t) != kTfLiteOk)
            return kTfLiteError;
      }
      if (inputs.contains(t)) {
        if (data == nullptr) {
          if (!caching_enabled)
            if (openvino_graph_builder_->AddInputParams(opaque_tensor, t) !=
                kTfLiteOk)
              return kTfLiteError;
          compute_inputs_.push_back(t);
        }
      }
    }
    if (!caching_enabled) {
      if (openvino_graph_builder_->CreateNodeFromTfLiteOp(
              delegate_node_registration, delegate_node, context) != kTfLiteOk)
        return kTfLiteError;
    }
  }

  if (!caching_enabled) {
    if (openvino_graph_builder_->UpdateResultNodes(context, outputs_) !=
        kTfLiteOk)
      return kTfLiteError;
    model_ =
        std::make_shared<ov::Model>(openvino_graph_builder_->getResultNodes(),
                                    openvino_graph_builder_->getInputParams());
  } else {
    model_ = ov_core_.read_model(cache_file_name);
    if (!model_) TFLITE_LOG(WARNING) << "Read model from disk failed";
  }
  if (!model_) return kTfLiteError;
  return kTfLiteOk;
}

TfLiteStatus OpenVINODelegateCore::CompileAndInfer() {
  // TODO: get device string from flags
  std::string deviceStr = "NPU";
  ov::AnyMap config;
  // Below param helps accelerate inference on NPU device. It helps in HW
  // acceleration.
  config["NPU_COMPILATION_MODE_PARAMS"] = "enable-se-ptrs-operations=true";

  compiled_model_ = ov_core_.compile_model(model_, deviceStr, config);
  infer_request_ = compiled_model_.create_infer_request();
  return kTfLiteOk;
}

TfLiteStatus OpenVINODelegateCore::CreateModel(
    TfLiteOpaqueContext *context, const TfLiteOpaqueDelegateParams *params,
    const TfLiteOpenVINODelegateOptions *delegate_options) {
  // If cache_dir is set, and
  //    if cached model exists BuildModelFromCache
  //    else initialize and build model from tflite runtime
  std::filesystem::path cache_file_path;
  if (!delegate_options->cache_dir.empty() &&
      !delegate_options->model_token.empty()) {
    cache_file_path = delegate_options->cache_dir;
    cache_file_path /= delegate_options->model_token + ".xml";
    ov_core_.set_property(ov::cache_dir(delegate_options->cache_dir));
    if (std::filesystem::exists(std::filesystem::status(cache_file_path))) {
      auto status = InitializeModel(context, params, true /*caching_enabled*/,
                                    cache_file_path.string());
      if (status == kTfLiteOk) return status;
    }
  }

  // If cache file is absent or caching is not enabled
  // Initialize model from TFLite runtime
  auto status = InitializeModel(context, params, false /*caching_not_enabled*/);
  if (status != kTfLiteOk) return status;
  if (!delegate_options->cache_dir.empty() &&
      !delegate_options->model_token.empty()) {
    if (access(delegate_options->cache_dir.c_str(), W_OK) == 0) {
      ov::serialize(model_, cache_file_path.string());
    } else {
      TFLITE_LOG(WARNING) << "Serialization failed - Continue from built model";
      return kTfLiteOk;
    }
  }
  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
