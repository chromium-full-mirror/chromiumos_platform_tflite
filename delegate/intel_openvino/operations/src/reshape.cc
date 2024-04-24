/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/operations/include/reshape.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Reshape::CreateNode() {
  auto input_node = getInputNode(tensor_indices_[INPUT_NODE_1]);
  if (input_node == nullptr) {
    TFLITE_LOG(ERROR) << "input node is null";
    return kTfLiteError;
  }

  ov::Output<ov::Node> shape_node = getInputNode(tensor_indices_[SHAPE_NODE]);

  output_node_ = std::make_shared<ov::opset3::Reshape>(input_node, shape_node,
                                                       /*special_zero=*/false);

  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
