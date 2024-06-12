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
  bool has_bias = false;
  ov::Output<ov::Node> bias_node;

  if (tensor_indices_size_ < 3) {
    has_bias = false;
  } else if (tensor_indices_size_ == 3) {
    bias_node = getInputNode(tensor_indices_[BIAS_NODE]);
    has_bias = true;
  }

  TfLiteStatus status = CalculatePadding(conv2d_params->padding, auto_pad);
  if (status != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Invalid padding type in conv2d";
    return kTfLiteError;
  }

  auto input_node = getInputNode(tensor_indices_[INPUT_NODE_1]);
  auto filter_node = getInputNode(tensor_indices_[FILTER_NODE]);

  std::shared_ptr<ov::Node> transposed_filter_node, transposed_input_node;
  if (Transpose(OHWI_OIHW, filter_node, transposed_filter_node) != kTfLiteOk)
    return kTfLiteError;
  if (Transpose(NHWC_NCHW, input_node, transposed_input_node) != kTfLiteOk)
    return kTfLiteError;

  auto input_shape = transposed_input_node->get_shape();    // NCHW
  auto filter_shape = transposed_filter_node->get_shape();  // G*Cout,Cin,H,W
  auto num_groups = input_shape[1] / filter_shape[1];       // G = C/Cin
  std::shared_ptr<ov::Node> conv_node;
  if (num_groups > 1) {
    filter_shape[0] /= num_groups;  // G*Cout/G
    filter_shape.insert(filter_shape.begin(), num_groups);
    auto shape_node = CreateConstNode(
        ov::element::i32, ov::Shape{filter_shape.size()}, filter_shape);
    transposed_filter_node = std::make_shared<ov::opset3::Reshape>(
        transposed_filter_node, shape_node, /*special_zero=*/false);
    // Perform group convolution since no. of groups > 1
    conv_node = std::make_shared<ov::opset3::GroupConvolution>(
        transposed_input_node, transposed_filter_node, strides, padding_begin,
        padding_end, dilations, auto_pad);
  } else {
    conv_node = std::make_shared<ov::opset8::Convolution>(
        transposed_input_node, transposed_filter_node, strides, padding_begin,
        padding_end, dilations, auto_pad);
  }

  if (Transpose(NCHW_NHWC, conv_node, output_node_) != kTfLiteOk)
    return kTfLiteError;

  if (has_bias) {
    output_node_ = std::make_shared<ov::opset3::Add>(
        output_node_, bias_node, ov::op::AutoBroadcastType::NUMPY);
  }

  output_node_ = ApplyActivation(output_node_, conv2d_params->activation);
  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
