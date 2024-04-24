/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/operations/include/conv_2d.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Conv2D::CreateNode() {
  auto *conv2d_params = GetBuiltinData<TfLiteConvParams>();
  ov::Strides strides = {(size_t)conv2d_params->stride_height,
                         (size_t)conv2d_params->stride_width};
  ov::Strides dilations = {(size_t)conv2d_params->dilation_height_factor,
                           (size_t)conv2d_params->dilation_width_factor};
  ov::CoordinateDiff padding_begin = {0, 0};
  ov::CoordinateDiff padding_end = {0, 0};
  ov::op::PadType auto_pad;

  TfLiteStatus status = CalculatePadding(conv2d_params->padding, auto_pad);
  if (status != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Invalid padding type in conv2d";
    return kTfLiteError;
  }

  auto input_node = getInputNode(tensor_indices_[INPUT_NODE_1]);
  auto filter_node = getInputNode(tensor_indices_[FILTER_NODE]);
  auto bias_node = getInputNode(tensor_indices_[BIAS_NODE]);

  if (!GetGraphNodeManager()->isIndexAParam(tensor_indices_[FILTER_NODE])) {
    ov::AxisVector order = {0, 3, 1, 2};
    const auto order_node = ov::opset3::Constant::create(
        ov::element::i64, ov::Shape{order.size()}, order);
    filter_node =
        std::make_shared<ov::opset3::Transpose>(filter_node, order_node);
  }

  auto conv_node = std::make_shared<ov::opset8::Convolution>(
      input_node, filter_node, strides, padding_begin, padding_end, dilations,
      auto_pad);
  auto bias_dims = GetDims(tensor_indices_[BIAS_NODE]);
  std::vector<uint32_t> shape(conv_node->get_shape().size(), 1);
  shape[1] = bias_dims[0];
  auto shape_node =
      CreateConstNode(ov::element::i32, ov::Shape{shape.size()}, shape);

  bias_node = std::make_shared<ov::opset3::Reshape>(bias_node, shape_node,
                                                    /*special_zero=*/true);

  output_node_ = std::make_shared<ov::opset3::Add>(
      conv_node, bias_node, ov::op::AutoBroadcastType::NUMPY);

  output_node_ = ApplyActivation(output_node_, conv2d_params->activation);
  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
