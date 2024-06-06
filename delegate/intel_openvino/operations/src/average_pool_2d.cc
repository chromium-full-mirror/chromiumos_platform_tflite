/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/operations/include/average_pool_2d.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus AveragePool2D::CreateNode() {
  auto *avg_pool_params = GetBuiltinData<TfLitePoolParams>();
  auto input_node = getInputNode(tensor_indices_[INPUT_NODE_1]);
  if (input_node == nullptr) {
    TFLITE_LOG(ERROR) << "input node is null";
    return kTfLiteError;
  }

  ov::Strides strides = {(size_t)avg_pool_params->stride_height,
                         (size_t)avg_pool_params->stride_width};
  ov::Shape kernel = {(size_t)avg_pool_params->filter_height,
                      (size_t)avg_pool_params->filter_width};
  ov::Shape padding_begin = {0, 0};
  ov::Shape padding_end = {0, 0};
  ov::op::PadType auto_pad;

  TfLiteStatus tf_status = CalculatePadding(avg_pool_params->padding, auto_pad);
  if (tf_status == kTfLiteError) {
    TFLITE_LOG(ERROR) << "Invalid Padding";
    return kTfLiteError;
  }

  std::shared_ptr<ov::Node> transposed_input_node;
  if (Transpose(NHWC_NCHW, input_node, transposed_input_node) != kTfLiteOk)
    return kTfLiteError;

  auto average_pool_2d_node = std::make_shared<ov::opset8::AvgPool>(
      transposed_input_node, strides, padding_begin, padding_end, kernel,
      /*exclude_pad=*/true, ov::op::RoundingType::FLOOR, auto_pad);

  if (Transpose(NCHW_NHWC, average_pool_2d_node, output_node_) != kTfLiteOk)
    return kTfLiteError;

  output_node_ = ApplyActivation(output_node_, avg_pool_params->activation);

  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
