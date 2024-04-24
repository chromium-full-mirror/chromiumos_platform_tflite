/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/operations/include/max_pool_2d.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus MaxPool2D::CreateNode() {
  auto *max_pool_params = GetBuiltinData<TfLitePoolParams>();

  auto input_node = getInputNode(tensor_indices_[INPUT_NODE_1]);
  if (input_node == nullptr) {
    TFLITE_LOG(INFO) << "input node is null";
    return kTfLiteError;
  }

  ov::Strides strides{(size_t)max_pool_params->stride_height,
                      (size_t)max_pool_params->stride_width};

  // will be ignored since auto_pad is specified
  ov::Shape padding_begin = {0, 0};
  ov::Shape padding_end = {0, 0};

  ov::Shape kernel{(size_t)max_pool_params->filter_height,
                   (size_t)max_pool_params->filter_width};

  ov::op::PadType auto_pad;
  TfLiteStatus tf_status = CalculatePadding(max_pool_params->padding, auto_pad);

  auto max_pool_node = std::make_shared<ov::opset3::MaxPool>(
      input_node, strides, padding_begin, padding_end, kernel,
      ov::op::RoundingType::FLOOR, auto_pad);

  output_node_ = ApplyActivation(max_pool_node, max_pool_params->activation);
  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
