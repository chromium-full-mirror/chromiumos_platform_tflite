/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "openvino_graph_builder.h"

#include "delegate/intel_openvino/operations/include/add.h"
#include "delegate/intel_openvino/operations/include/average_pool_2d.h"
#include "delegate/intel_openvino/operations/include/concat.h"
#include "delegate/intel_openvino/operations/include/conv_2d.h"
#include "delegate/intel_openvino/operations/include/depthwise_conv_2d.h"
#include "delegate/intel_openvino/operations/include/dequantize.h"
#include "delegate/intel_openvino/operations/include/hardswish.h"
#include "delegate/intel_openvino/operations/include/logistic.h"
#include "delegate/intel_openvino/operations/include/max_pool_2d.h"
#include "delegate/intel_openvino/operations/include/mean.h"
#include "delegate/intel_openvino/operations/include/mul.h"
#include "delegate/intel_openvino/operations/include/relu.h"
#include "delegate/intel_openvino/operations/include/relu6.h"
#include "delegate/intel_openvino/operations/include/reshape.h"
#include "delegate/intel_openvino/operations/include/resize_bilinear.h"
#include "delegate/intel_openvino/operations/include/softmax.h"
#include "delegate/intel_openvino/operations/include/tanh.h"
#include "delegate/intel_openvino/operations/include/transpose_conv.h"
#include "delegate/intel_openvino/operations/utility.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus OpenVINOGraphBuilder::convertNHWCtoNCHW(
    std::vector<int> node_dims, std::shared_ptr<ov::Node> input,
    std::shared_ptr<ov::Node> &transposed_node) {
  if (input == nullptr) return kTfLiteError;

  // TODO: 0d(scalar) tensors are not supported in this version of delegate.
  // Support to be added in next release.
  if (node_dims.size() <= 0) return kTfLiteError;

  ov::AxisVector order = {0, 3, 1, 2};
  const auto order_node = std::make_shared<ov::opset8::Constant>(
      ov::element::i32, ov::Shape{order.size()}, order);

  // TODO: Generic NHWC to NCHW conversions to be moved to op specific
  // implementation in next version of delegate.
  if (node_dims.size() < 4) {
    auto size = node_dims.size();
    for (int i = 0; i < 4 - size; i++) {
      node_dims.insert(node_dims.begin(), 1);
    }
    auto new_size = std::make_shared<ov::opset8::Constant>(
        ov::element::i32, ov::Shape{4}, node_dims);
    input = std::make_shared<ov::opset8::Reshape>(input, new_size,
                                                  /*special_zero=*/false);
  }

  if (node_dims.size() >= 5) {
    TFLITE_LOG(ERROR) << "5D or greater than 5D tensors are not supported";
    return kTfLiteError;
  }
  transposed_node = std::make_shared<ov::opset3::Transpose>(input, order_node);

  return kTfLiteOk;
}

TfLiteStatus OpenVINOGraphBuilder::CreateConstNode(
    const TfLiteOpaqueContext *context, const int index) {
  if (context == nullptr) return kTfLiteError;
  const TfLiteOpaqueTensor *t =
      TfLiteOpaqueContextGetOpaqueTensor(context, index);

  std::vector<int> dims = GetDims(t);

  if (dims.size() <= 0) return kTfLiteError;

  const void *data = TfLiteOpaqueTensorData(t);
  if (data == NULL) {
    return kTfLiteError;
  }

  TfLiteType tensor_type = TfLiteOpaqueTensorType(t);
  ov::element::Type ov_element_type = GetOVElementType(tensor_type);
  if (ov_element_type == ov::element::undefined) {
    TFLITE_LOG(ERROR) << "Element type " << tensor_type << " not supported";
    return kTfLiteError;
  }

  auto const_node = std::make_shared<ov::opset8::Constant>(
      ov_element_type, ov::Shape(dims.begin(), dims.end()), data);
  node_manager_->setOutputAtOperandIndex(index, const_node);

  return kTfLiteOk;
}

TfLiteStatus OpenVINOGraphBuilder::UpdateResultNodes(
    const TfLiteOpaqueContext *context, std::vector<int> outputs) {
  if (context == nullptr) return kTfLiteError;
  if (outputs.size() < 1) return kTfLiteError;

  for (auto o : outputs) {
    auto out_node = node_manager_->getInterimNodeOutput(o);
    auto dims = out_node->get_shape();
    if (dims.size() == 4) {
      ov::AxisVector order = {0, 2, 3, 1};
      const auto order_node = std::make_shared<ov::opset8::Constant>(
          ov::element::i64, ov::Shape{order.size()}, order);
      out_node = std::make_shared<ov::opset3::Transpose>(out_node, order_node);
    }
    result_nodes_.push_back(out_node);
  }

  return kTfLiteOk;
}

TfLiteStatus OpenVINOGraphBuilder::AddInputParams(const TfLiteOpaqueTensor *t,
                                                  const int index) {
  if (t == nullptr) return kTfLiteError;
  if (index < 0) return kTfLiteError;

  std::vector<int> dims = GetDims(t);

  if (dims.size() <= 0) return kTfLiteError;

  auto input = std::make_shared<ov::opset3::Parameter>(
      ov::element::f32, ov::Shape(dims.begin(), dims.end()));
  input_params_.push_back(input);

  std::shared_ptr<ov::Node> interim;
  if (convertNHWCtoNCHW(dims, input, interim) != kTfLiteOk) return kTfLiteError;
  if (interim == nullptr) return kTfLiteError;
  node_manager_->setOutputAtOperandIndex(index, interim);
  node_manager_->insertIndexParameters(index);

  return kTfLiteOk;
}

TfLiteStatus OpenVINOGraphBuilder::CreateNodeFromTfLiteOp(
    TfLiteRegistrationExternal *registration, TfLiteOpaqueNode *node,
    TfLiteOpaqueContext *context) {
  if (registration == nullptr || node == nullptr || context == nullptr)
    return kTfLiteError;

  std::shared_ptr<OperationsBase> operation_node;
  if (CreateOpClass(registration, operation_node) != kTfLiteOk)
    return kTfLiteError;
  operation_node->SetGraphData(context, node_manager_.get());

  const int *inputs_data;
  int num_inputs;
  if (TfLiteOpaqueNodeInputs(node, &inputs_data, &num_inputs) != kTfLiteOk)
    return kTfLiteError;
  operation_node->UpdateNodeInfo((void *)inputs_data, num_inputs,
                                 TfLiteOpaqueNodeGetBuiltinData(node));
  if (operation_node->CreateNode() != kTfLiteOk) return kTfLiteError;

  std::shared_ptr<ov::Node> result_node = operation_node->GetOpResultNode();
  if (result_node == nullptr) return kTfLiteError;

  const int *outputs;
  int num_outputs;
  if (TfLiteOpaqueNodeOutputs(node, &outputs, &num_outputs) != kTfLiteOk)
    return kTfLiteError;
  // TODO: Assume there is only one output from the op. Handle multiple outputs
  // from one op in next version.
  if (num_outputs != 1) return kTfLiteError;
  node_manager_->setOutputAtOperandIndex(outputs[0], result_node);

  return kTfLiteOk;
}

std::shared_ptr<OperationsBase> OpenVINOGraphBuilder::CreateOpForCode(
    TfLiteBuiltinOperator builtin_code) {
  switch (builtin_code) {
    case kTfLiteBuiltinAdd: {
      return std::make_shared<Add>();
    }
    case kTfLiteBuiltinAveragePool2d: {
      return std::make_shared<AveragePool2D>();
    }
    case kTfLiteBuiltinConv2d: {
      return std::make_shared<Conv2D>();
    }
    case kTfLiteBuiltinConcatenation: {
      return std::make_shared<Concat>();
    }
    case kTfLiteBuiltinDepthwiseConv2d: {
      return std::make_shared<DepthwiseConv2D>();
    }
    case kTfLiteBuiltinDequantize: {
      return std::make_shared<Dequantize>();
    }
    case kTfLiteBuiltinMul: {
      return std::make_shared<Mul>();
    }
    case kTfLiteBuiltinResizeBilinear: {
      return std::make_shared<ResizeBilinear>();
    }
    case kTfLiteBuiltinRelu: {
      return std::make_shared<Relu>();
    }
    case kTfLiteBuiltinRelu6: {
      return std::make_shared<Relu6>();
    }
    case kTfLiteBuiltinLogistic: {
      return std::make_shared<Logistic>();
    }
    case kTfLiteBuiltinHardSwish: {
      return std::make_shared<HardSwish>();
    }
    case kTfLiteBuiltinSoftmax: {
      return std::make_shared<Softmax>();
    }
    case kTfLiteBuiltinTanh: {
      return std::make_shared<Tanh>();
    }
    case kTfLiteBuiltinReshape: {
      return std::make_shared<Reshape>();
    }
    case kTfLiteBuiltinMaxPool2d: {
      return std::make_shared<MaxPool2D>();
    }
    case kTfLiteBuiltinMean: {
      return std::make_shared<Mean>();
    }
    case kTfLiteBuiltinTransposeConv: {
      return std::make_shared<TransposeConv>();
    }
    default:
      return nullptr;
  }
}

TfLiteStatus OpenVINOGraphBuilder::CreateOpClass(
    TfLiteRegistrationExternal *registration,
    std::shared_ptr<OperationsBase> &op_base) {
  if (registration == nullptr) return kTfLiteError;
  auto builtin_code = TfLiteRegistrationExternalGetBuiltInCode(registration);
  op_base = CreateOpForCode(builtin_code);
  return op_base != nullptr ? kTfLiteOk : kTfLiteError;
}

}  // namespace openvinodelegate
}  // namespace tflite
