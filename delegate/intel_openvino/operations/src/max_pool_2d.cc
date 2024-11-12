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
  if (CalculatePadding(max_pool_params->padding, auto_pad) != kTfLiteOk)
    return kTfLiteError;

  std::shared_ptr<ov::Node> transposed_input_node;
  if (Transpose(NHWC_NCHW, input_node, transposed_input_node) != kTfLiteOk)
    return kTfLiteError;

  auto max_pool_node = std::make_shared<ov::opset3::MaxPool>(
      transposed_input_node, strides, padding_begin, padding_end, kernel,
      ov::op::RoundingType::FLOOR, auto_pad);

  if (Transpose(NCHW_NHWC, max_pool_node, output_node_) != kTfLiteOk)
    return kTfLiteError;

  output_node_ = ApplyActivation(output_node_, max_pool_params->activation);
  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
