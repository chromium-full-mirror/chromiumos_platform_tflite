/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/operations/include/resize_bilinear.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus ResizeBilinear::CreateNode() {
  auto *resize_bilinearParams = GetBuiltinData<TfLiteResizeBilinearParams>();
  auto input_node = getInputNode(tensor_indices_[INPUT_NODE_1]);
  auto shape_node = getInputNode(tensor_indices_[INPUT_NODE_2]);
  struct ov::op::v11::Interpolate::InterpolateAttrs attrs;

  attrs.mode = ov::op::v11::Interpolate::InterpolateMode::LINEAR_ONNX;
  attrs.shape_calculation_mode = ov::op::v11::Interpolate::ShapeCalcMode::SIZES;

  if (resize_bilinearParams->align_corners) {
    attrs.coordinate_transformation_mode =
        ov::op::v11::Interpolate::CoordinateTransformMode::ALIGN_CORNERS;
  } else if (resize_bilinearParams->half_pixel_centers) {
    attrs.coordinate_transformation_mode =
        ov::op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL;
  } else {
    attrs.coordinate_transformation_mode =
        ov::op::v11::Interpolate::CoordinateTransformMode::ASYMMETRIC;
  }

  std::shared_ptr<ov::Node> transposed_input_node;
  if (Transpose(NHWC_NCHW, input_node, transposed_input_node) != kTfLiteOk)
    return kTfLiteError;

  std::vector<int32_t> axes_vec = {2, 3};
  auto axes_node = CreateConstNode(ov::element::i32, /*size=*/{2}, axes_vec);

  output_node_ = std::make_shared<ov::op::v11::Interpolate>(
      transposed_input_node, shape_node, axes_node, attrs);

  if (Transpose(NCHW_NHWC, output_node_, output_node_) != kTfLiteOk)
    return kTfLiteError;

  return kTfLiteOk;
}
}  // namespace openvinodelegate
}  // namespace tflite
