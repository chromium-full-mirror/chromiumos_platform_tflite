/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/operations/operations_base.h"

#include <memory>
#include <utility>
#include <vector>

#include "delegate/intel_openvino/operations/utility.h"

namespace tflite {
namespace openvinodelegate {

void OperationsBase::UpdateNodeInfo(const int *tensor_indices, int size,
                                    void *builtin_data) {
  tensor_indices_ = tensor_indices;
  tensor_indices_size_ = size;
  SetBuiltinData(builtin_data);
}

void OperationsBase::SetGraphData(const TfLiteOpaqueContext *context,
                                  NodeManager *node_manager) {
  context_ = context;
  node_manager_ = node_manager;
}

TfLiteStatus OperationsBase::Transpose(
    LayoutConversion type, std::shared_ptr<ov::Node> input,
    std::shared_ptr<ov::Node> &transposed_node) {
  ov::AxisVector order;
  switch (type) {
    case NHWC_NCHW:
      order = {0, 3, 1, 2};
      break;
    case NCHW_NHWC:
      order = {0, 2, 3, 1};
      break;
    case IHWO_OIHW:
      order = {3, 0, 1, 2};
      break;
    case OHWI_OIHW:
      order = {0, 3, 1, 2};
      break;
    default:
      TFLITE_LOG(ERROR) << "Invalid layout conversion type";
      return kTfLiteError;
  }
  const auto order_node = ov::opset3::Constant::create(
      ov::element::i32, ov::Shape{order.size()}, order);
  transposed_node = std::make_shared<ov::opset3::Transpose>(input, order_node);
  return kTfLiteOk;
}

TfLiteStatus OperationsBase::CalculatePadding(TfLitePadding padding,
                                              ov::op::PadType &auto_pad) {
  switch (padding) {
    case kTfLitePaddingSame: {
      auto_pad = ov::op::PadType::SAME_UPPER;
      return kTfLiteOk;
    }
    case kTfLitePaddingValid: {
      auto_pad = ov::op::PadType::VALID;
      return kTfLiteOk;
    }
    default:
      return kTfLiteError;
  }
}

std::shared_ptr<ov::Node> OperationsBase::ApplyActivation(
    std::shared_ptr<ov::Node> input, TfLiteFusedActivation activation) {
  switch (activation) {
    case kTfLiteActNone:
      return input;
    case kTfLiteActRelu:
      return std::make_shared<ov::opset8::Relu>(input);
    case kTfLiteActReluN1To1:
      return std::make_shared<ov::opset8::Clamp>(input, -1, 1);
    case kTfLiteActRelu6:
      return std::make_shared<ov::opset8::Clamp>(input, 0, 6);
    case kTfLiteActTanh:
      return std::make_shared<ov::opset8::Tanh>(input);
      // TODO: add support for kTfLiteActSignBit
    case kTfLiteActSigmoid:
      return std::make_shared<ov::opset8::Sigmoid>(input);
    default:
      return nullptr;
  }
}

std::vector<int> OperationsBase::GetDims(int index) {
  auto t = TfLiteOpaqueContextGetOpaqueTensor(context_, index);
  int32_t num_dims = TfLiteOpaqueTensorNumDims(t);
  std::vector<int> dims(num_dims);
  for (int i = 0; i < num_dims; i++) {
    dims[i] = TfLiteOpaqueTensorDim(t, i);
  }
  return dims;
}

void OperationsBase::GetTensorData(int index, void *data) {
  auto opaque_tensor = TfLiteOpaqueContextGetOpaqueTensor(context_, index);
  void *tensor_data = TfLiteOpaqueTensorData(opaque_tensor);
  auto size = TfLiteOpaqueTensorByteSize(opaque_tensor);
  std::memcpy(data, tensor_data, size);
}

ov::element::Type OperationsBase::GetTensorType(int index) {
  auto opaque_tensor = TfLiteOpaqueContextGetOpaqueTensor(context_, index);
  TfLiteType tensor_type = TfLiteOpaqueTensorType(opaque_tensor);
  return GetOVElementType(tensor_type);
}

std::pair<void *, int> OperationsBase::GetTensorDataPtrAndCount(int index) {
  auto opaque_tensor = TfLiteOpaqueContextGetOpaqueTensor(context_, index);
  void *tensor_data = TfLiteOpaqueTensorData(opaque_tensor);

  TfLiteType tensor_type = TfLiteOpaqueTensorType(opaque_tensor);
  ov::element::Type ov_element_type = GetOVElementType(tensor_type);
  if (ov_element_type == ov::element::undefined) {
    return std::make_pair(nullptr, 0);
  }
  int count =
      TfLiteOpaqueTensorByteSize(opaque_tensor) / ov_element_type.size();
  return std::make_pair(tensor_data, count);
}

}  // namespace openvinodelegate
}  // namespace tflite
