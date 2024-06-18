/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/operations/include/mean.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Mean::CreateNode() {
  auto *reducer_params = GetBuiltinData<TfLiteReducerParams>();

  auto input_node = getInputNode(tensor_indices_[INPUT_NODE_1]);
  if (input_node == nullptr) {
    TFLITE_LOG(INFO) << "input node is null";
    return kTfLiteError;
  }

  auto reduction_axes = getInputNode(tensor_indices_[INPUT_NODE_2]);
  if (reduction_axes == nullptr) {
    TFLITE_LOG(INFO) << "reduction_axes is null";
    return kTfLiteError;
  }

  auto [data, count] = GetTensorDataPtrAndCount(tensor_indices_[1]);
  if (count == 0 || data == nullptr) {
    TFLITE_LOG(INFO) << "Failed to get reduction_axes data";
    return kTfLiteError;
  }
  auto *axes_ptr = reinterpret_cast<int32_t *>(data);
  std::vector<int32_t> axes_vec(axes_ptr, axes_ptr + count);

  auto axes_node =
      CreateConstNode(ov::element::i32, {(unsigned int)count}, axes_vec);
  if (axes_node == nullptr) {
    TFLITE_LOG(INFO) << "Failed to create const node for axes";
    return kTfLiteError;
  }

  bool keep_dims = reducer_params->keep_dims;
  output_node_ = std::make_shared<ov::op::v1::ReduceMean>(input_node, axes_node,
                                                          keep_dims);

  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
