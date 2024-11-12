/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/operations/include/softmax.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Softmax::CreateNode() {
  auto *softmax_params = GetBuiltinData<TfLiteSoftmaxParams>();

  auto input_node_1 = getInputNode(tensor_indices_[INPUT_NODE_1]);
  if (input_node_1 == nullptr) {
    TFLITE_LOG(INFO) << "input node 1 is null";
    return kTfLiteError;
  }

  auto beta_node = CreateConstNode(ov::element::f32, ov::Shape{1},
                                   std::vector<float>{softmax_params->beta});
  auto mul_node = std::make_shared<ov::opset3::Multiply>(
      input_node_1, beta_node, ov::op::AutoBroadcastType::NUMPY);

  // NOTE: assumption here is: Tensorflow always computes softmax along
  // channel(last) dimesnsion. After transpose, our channel shifts to dim 1,
  // which is default axis attribute for Softmax.
  output_node_ = std::make_shared<ov::opset8::Softmax>(mul_node);
  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
