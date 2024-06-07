/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "openvino_delegate_core.h"

#include "operations/openvino_node_manager.h"
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

TfLiteStatus OpenVINODelegateCore::CreateGraphfromTfLite(
    TfLiteOpaqueContext *context, const TfLiteOpaqueDelegateParams *params) {
  if (context == nullptr || params == nullptr) return kTfLiteError;
  const std::unordered_set<int> inputs(
      &params->input_tensors->data[0],
      &params->input_tensors->data[params->input_tensors->size]);

  openvino_graph_builder_ =
      std::make_unique<OpenVINOGraphBuilder>(std::make_unique<NodeManager>());

  for (int o = 0; o < params->output_tensors->size; o++) {
    const int output_tensor_idx = params->output_tensors->data[o];
    outputs_.push_back(output_tensor_idx);
  }

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
      if (allocation_type == kTfLiteMmapRo) {
        data = TfLiteOpaqueTensorData(opaque_tensor);
        if (openvino_graph_builder_->CreateConstNode(context, t) != kTfLiteOk)
          return kTfLiteError;
      }
      if (inputs.contains(t) != 0) {
        if (data == nullptr) {
          if (openvino_graph_builder_->AddInputParams(opaque_tensor, t) !=
              kTfLiteOk)
            return kTfLiteError;
          compute_inputs_.push_back(t);
        }
      }
    }
    if (openvino_graph_builder_->CreateNodeFromTfLiteOp(
            delegate_node_registration, delegate_node, context) != kTfLiteOk)
      return kTfLiteError;
  }

  openvino_graph_builder_->UpdateResultNodes(context, outputs_);
  std::shared_ptr<ov::Model> model =
      std::make_shared<ov::Model>(openvino_graph_builder_->getResultNodes(),
                                  openvino_graph_builder_->getInputParams());
  // TODO: get device string from flags
  std::string deviceStr = "NPU";
  ov::AnyMap config;
  // Below param helps accelerate inference on NPU device. It helps in HW
  // acceleration.
  config["NPU_COMPILATION_MODE_PARAMS"] = "enable-se-ptrs-operations=true";
  compiled_model_ = ov_core_.compile_model(model, deviceStr, config);

  infer_request_ = compiled_model_.create_infer_request();
  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
