/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/operations/include/sub.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Sub::CreateNode() {
  auto *sub_params = GetBuiltinData<TfLiteSubParams>();
  auto input_node_1 = getInputNode(tensor_indices_[INPUT_NODE_1]);
  if (input_node_1 == nullptr) {
    TFLITE_LOG(INFO) << "input node 1 is null";
    return kTfLiteError;
  }
  auto input_node_2 = getInputNode(tensor_indices_[INPUT_NODE_2]);
  if (input_node_2 == nullptr) {
    TFLITE_LOG(INFO) << "input Node 2 is null";
    return kTfLiteError;
  }

  auto sub_node = std::make_shared<ov::opset8::Subtract>(
      input_node_1, input_node_2, ov::op::AutoBroadcastType::NUMPY);
  output_node_ = ApplyActivation(sub_node, sub_params->activation);
  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
