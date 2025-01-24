/*
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/operations/include/prelu.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus PRelu::CreateNode() {
  auto input_node_1 = getInputNode(tensor_indices_[INPUT_NODE_1]);
  if (input_node_1 == nullptr) {
    TFLITE_LOG(INFO) << "input node 1 is null";
    return kTfLiteError;
  }
  auto input_node_2 = getInputNode(tensor_indices_[INPUT_NODE_2]);
  if (input_node_2 == nullptr) {
    TFLITE_LOG(INFO) << "input node 2 is null";
    return kTfLiteError;
  }

  auto dims_node_1 = GetDims(tensor_indices_[INPUT_NODE_1]);
  auto dims_node_2 = GetDims(tensor_indices_[INPUT_NODE_2]);
  // Note: Unsqueeze the slope to make it a 4D tensor and then transposed,
  // since OpenVINO expects NCHW format and channels dimensions are expected
  // on the 1st index.
  int i = dims_node_1.size() - dims_node_2.size();
  if (i > 0) {
    std::vector<int64_t> axes_to_unsqueeze(i);
    std::iota(axes_to_unsqueeze.begin(), axes_to_unsqueeze.end(), 0);

    auto scalar = std::make_shared<ov::op::v0::Constant>(
        ov::element::i32, ov::Shape{axes_to_unsqueeze.size()}, 1);
    input_node_2 =
        std::make_shared<ov::op::v0::Unsqueeze>(input_node_2, scalar);
  }
  std::shared_ptr<ov::Node> transposed_input_node_1, transposed_input_node_2;
  if (Transpose(NHWC_NCHW, input_node_1, transposed_input_node_1) != kTfLiteOk)
    return kTfLiteError;
  if (Transpose(NHWC_NCHW, input_node_2, transposed_input_node_2) != kTfLiteOk)
    return kTfLiteError;

  auto output_node = std::make_shared<ov::op::v0::PRelu>(
      transposed_input_node_1, transposed_input_node_2);
  if (Transpose(NCHW_NHWC, output_node, output_node_) != kTfLiteOk)
    return kTfLiteError;
  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
