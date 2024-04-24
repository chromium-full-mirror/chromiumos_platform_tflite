/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/operations/include/transpose_conv.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus TransposeConv::CreateNode() {
  auto *transpose_conv_params = GetBuiltinData<TfLiteTransposeConvParams>();
  std::shared_ptr<ov::Node> weights_node = nullptr;
  std::shared_ptr<ov::Node> input_node = nullptr;
  weights_node = getInputNode(tensor_indices_[TRANSPOSE_CONV_WEIGHTS]);
  input_node = getInputNode(tensor_indices_[TRANSPOSE_CONV_INPUT]);
  bool has_bias = false;
  std::shared_ptr<ov::Node> bias_node = nullptr;
  if (tensor_indices_size_ >= 4) {
    bias_node = getInputNode(tensor_indices_[TRANSPOSE_CONV_BIAS]);
    has_bias = true;
  }
  ov::Strides strides = {(size_t)transpose_conv_params->stride_height,
                         (size_t)transpose_conv_params->stride_width};
  size_t dilation_width_factor = 1.0, dilation_height_factor = 1.0;
  ov::Strides dilations = {dilation_height_factor, dilation_width_factor};
  ov::op::PadType auto_pad;

  TfLiteStatus status =
      CalculatePadding(transpose_conv_params->padding, auto_pad);
  if (status != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Invalid padding type in transpose convolution";
    return kTfLiteError;
  }

  ov::CoordinateDiff padding_begin = {0, 0};
  ov::CoordinateDiff padding_end = {0, 0};

  // TODO: Possibly add a test compilation flag to change the order for test
  // mode
  //  Comment below transpose lines for test mode
  ov::AxisVector order = {3, 0, 1, 2};
  const auto order_node = ov::opset3::Constant::create(
      ov::element::i64, ov::Shape{order.size()}, order);
  weights_node =
      std::make_shared<ov::opset3::Transpose>(weights_node, order_node);

  std::shared_ptr<ov::Node> transpose_conv_node = nullptr;
  size_t spatial_dimensions_size = 2;
  int32_t output_shape[4];
  std::vector<int32_t> spatial_dimensions(spatial_dimensions_size);
  GetTensorData(tensor_indices_[TRANSPOSE_CONV_OUTPUT_SHAPE], &output_shape);
  spatial_dimensions[0] = output_shape[1];
  spatial_dimensions[1] = output_shape[2];
  auto output_shape_node = std::make_shared<ov::opset8::Constant>(
      ov::element::i32, ov::Shape{spatial_dimensions_size}, spatial_dimensions);

  transpose_conv_node = std::make_shared<ov::opset3::ConvolutionBackpropData>(
      input_node, weights_node, output_shape_node, strides, padding_begin,
      padding_end, dilations, auto_pad);

  if (has_bias) {
    std::vector<int> bias_dims;
    bias_dims = GetDims(tensor_indices_[TRANSPOSE_CONV_BIAS]);
    std::vector<uint32_t> shape(transpose_conv_node->get_shape().size(), 1);
    shape[1] = bias_dims[0];
    auto shape_node =
        CreateConstNode(ov::element::i32, ov::Shape{shape.size()}, shape);
    bias_node = std::make_shared<ov::opset3::Reshape>(bias_node, shape_node,
                                                      /*special_zero=*/true);
    output_node_ = std::make_shared<ov::opset3::Add>(
        transpose_conv_node, bias_node, ov::op::AutoBroadcastType::NUMPY);
  } else {
    output_node_ = transpose_conv_node;
  }
  output_node_ =
      ApplyActivation(output_node_, transpose_conv_params->activation);
  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
