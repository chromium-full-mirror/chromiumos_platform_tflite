/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/operations/include/depthwise_conv_2d.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus DepthwiseConv2D::CreateNode() {
  auto *depth_conv2dParams = GetBuiltinData<TfLiteDepthwiseConvParams>();
  // TODO: check for datatypes, tensor shapes, and non dynamic allocation
  auto input_node = getInputNode(tensor_indices_[INPUT_NODE_1]);
  auto filter_node = getInputNode(tensor_indices_[FILTER_NODE]);
  bool has_bias = false;
  ov::Output<ov::Node> bias_node;
  std::vector<size_t> strides = {(size_t)depth_conv2dParams->stride_height,
                                 (size_t)depth_conv2dParams->stride_width};
  std::vector<size_t> dilations = {
      (size_t)depth_conv2dParams->dilation_height_factor,
      (size_t)depth_conv2dParams->dilation_width_factor};
  if (tensor_indices_size_ < 3) {
    has_bias = false;
  } else if (tensor_indices_size_ == 3) {
    bias_node = getInputNode(tensor_indices_[BIAS_NODE]);
    has_bias = true;
  }

  ov::op::PadType auto_pad;
  auto input_dims = GetDims(tensor_indices_[INPUT_NODE_1]);

  TfLiteStatus status = CalculatePadding(depth_conv2dParams->padding, auto_pad);
  if (status != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Invalid padding type in depthwise conv2d";
    return kTfLiteError;
  }

  std::shared_ptr<ov::Node> transposed_filter_node, transposed_input_node;
  if (Transpose(IHWO_OIHW, filter_node, transposed_filter_node) != kTfLiteOk)
    return kTfLiteError;
  if (Transpose(NHWC_NCHW, input_node, transposed_input_node) != kTfLiteOk)
    return kTfLiteError;

  std::vector<size_t> shape(&transposed_filter_node->get_shape()[0],
                            &transposed_filter_node->get_shape()[0] + 4);
  auto num_groups = input_dims[3] / transposed_filter_node->get_shape()[1];
  shape.insert(shape.begin(), num_groups);
  shape[1] = transposed_filter_node->get_shape()[0] / num_groups;
  auto shape_node =
      CreateConstNode(ov::element::i32, ov::Shape{shape.size()}, shape);

  transposed_filter_node = std::make_shared<ov::opset3::Reshape>(
      transposed_filter_node, shape_node, true);

  auto depthwise_conv_node = std::make_shared<ov::opset3::GroupConvolution>(
      transposed_input_node, transposed_filter_node, ov::Strides(strides),
      ov::CoordinateDiff(0, 0), ov::CoordinateDiff(0, 0),
      ov::Strides(dilations), auto_pad);

  if (Transpose(NCHW_NHWC, depthwise_conv_node, output_node_) != kTfLiteOk)
    return kTfLiteError;

  if (has_bias) {
    output_node_ = std::make_shared<ov::opset3::Add>(
        output_node_, bias_node, ov::op::AutoBroadcastType::NUMPY);
  }

  output_node_ = ApplyActivation(output_node_, depth_conv2dParams->activation);
  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
