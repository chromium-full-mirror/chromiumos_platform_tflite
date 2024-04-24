/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/operations/include/mul.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Mul::CreateNode() {
  auto *mul_params = GetBuiltinData<TfLiteMulParams>();
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

  auto mul_node = std::make_shared<ov::opset3::Multiply>(
      input_node_1, input_node_2, ov::op::AutoBroadcastType::NUMPY);
  output_node_ = ApplyActivation(mul_node, mul_params->activation);
  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
