/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/operations/include/pad.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Pad::CreateNode() {
  auto input_node = getInputNode(tensor_indices_[INPUT_NODE_1]);
  if (input_node == nullptr) {
    TFLITE_LOG(INFO) << "input node is null";
    return kTfLiteError;
  }
  auto padding_node = getInputNode(tensor_indices_[INPUT_NODE_2]);
  if (padding_node == nullptr) {
    TFLITE_LOG(INFO) << "padding_node Node 2 is null";
    return kTfLiteError;
  }

  // Fetch the 2D paddings as a 1D vector, and then split it into 2
  auto [paddings_2d, size] =
      GetTensorDataPtrAndCount(tensor_indices_[INPUT_NODE_2]);
  auto half_size = size / 2;
  auto pad_dtype = GetTensorType(tensor_indices_[INPUT_NODE_2]);

  std::vector<int64_t> paddings_0(half_size);
  std::vector<int64_t> paddings_1(half_size);
  if (pad_dtype == ov::element::i32) {
    auto* data = reinterpret_cast<int32_t*>(paddings_2d);
    for (size_t i = 0; i < half_size; i++) {
      paddings_0[i] = data[2 * i];
      paddings_1[i] = data[2 * i + 1];
    }
  } else {
    auto* data = reinterpret_cast<int64_t*>(paddings_2d);
    for (size_t i = 0; i < half_size; i++) {
      paddings_0[i] = data[2 * i];
      paddings_1[i] = data[2 * i + 1];
    }
  }

  auto pads_begin =
      CreateConstNode(pad_dtype, {(unsigned int)half_size}, paddings_0);
  auto pads_end =
      CreateConstNode(pad_dtype, {(unsigned int)half_size}, paddings_1);
  if (pads_begin == nullptr || pads_end == nullptr) {
    TFLITE_LOG(INFO) << "Failed to create const node for padding";
    return kTfLiteError;
  }

  output_node_ = std::make_shared<ov::opset8::Pad>(
      input_node, pads_begin, pads_end, ov::op::PadMode::CONSTANT);
  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
