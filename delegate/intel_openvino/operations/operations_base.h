/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DELEGATE_INTEL_OPENVINO_OPERATIONS_OPERATIONS_BASE_H_
#define DELEGATE_INTEL_OPENVINO_OPERATIONS_OPERATIONS_BASE_H_

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset8.hpp>

#include <memory>
#include <utility>
#include <vector>

#include "delegate/intel_openvino/log.h"
#include "delegate/intel_openvino/operations/openvino_node_manager.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/kernels/padding.h"

typedef enum {
  INPUT_NODE_1 = 0,
  INPUT_NODE_2 = 1,
  FILTER_NODE = 1,
  BIAS_NODE = 2,
  SHAPE_NODE = 1,
  TRANSPOSE_CONV_OUTPUT_SHAPE = 0,
  TRANSPOSE_CONV_WEIGHTS = 1,
  TRANSPOSE_CONV_INPUT = 2,
  TRANSPOSE_CONV_BIAS = 3,
} Index;

namespace tflite {
namespace openvinodelegate {

class OperationsBase {
 public:
  void UpdateNodeInfo(const int *tensor_indices, int tensor_indices_size,
                      void *builtin_data);
  void SetGraphData(const TfLiteOpaqueContext *context,
                    NodeManager *node_manager);

  std::shared_ptr<ov::Node> GetOpResultNode() { return output_node_; }
  virtual TfLiteStatus CreateNode() = 0;
  virtual ~OperationsBase() {}

 protected:
  typedef enum {
    NHWC_NCHW,
    NCHW_NHWC,
    IHWO_OIHW,
    OHWI_OIHW,
  } LayoutConversion;

  std::shared_ptr<ov::Node> output_node_;
  template <typename T>
  T *GetBuiltinData() {
    return reinterpret_cast<T *>(builtin_data_);
  }
  void SetBuiltinData(void *builtin_data) { builtin_data_ = builtin_data; }
  std::shared_ptr<ov::Node> getInputNode(int index) {
    return node_manager_->getInterimNodeOutput(index);
  }
  NodeManager *GetGraphNodeManager() { return node_manager_; }

  template <typename T>
  std::shared_ptr<ov::Node> CreateConstNode(ov::element::Type elementType,
                                            ov::Shape shape,
                                            std::vector<T> data) {
    return std::make_shared<ov::opset8::Constant>(elementType, shape, data);
  }

  TfLiteStatus CalculatePadding(TfLitePadding padding,
                                ov::op::PadType &auto_pad);

  TfLiteStatus Transpose(LayoutConversion type, std::shared_ptr<ov::Node> input,
                         std::shared_ptr<ov::Node> &transposed_node);

  std::shared_ptr<ov::Node> ApplyActivation(std::shared_ptr<ov::Node> input,
                                            TfLiteFusedActivation activation);

  std::vector<int> GetDims(int index);

  void GetTensorData(int index, void *data);

  ov::element::Type GetTensorType(int index);

  std::pair<void *, int> GetTensorDataPtrAndCount(int index);

  const int *tensor_indices_;
  int tensor_indices_size_;

 private:
  void *builtin_data_ = nullptr;
  NodeManager *node_manager_;
  const TfLiteOpaqueContext *context_;
};

}  // namespace openvinodelegate
}  // namespace tflite

#endif  // DELEGATE_INTEL_OPENVINO_OPERATIONS_OPERATIONS_BASE_H_
