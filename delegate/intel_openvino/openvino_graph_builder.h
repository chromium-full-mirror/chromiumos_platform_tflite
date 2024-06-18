/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DELEGATE_INTEL_OPENVINO_OPENVINO_GRAPH_BUILDER_H_
#define DELEGATE_INTEL_OPENVINO_OPENVINO_GRAPH_BUILDER_H_

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset8.hpp>

#include <memory>
#include <utility>
#include <vector>

#include "delegate/intel_openvino/log.h"
#include "delegate/intel_openvino/operations/openvino_node_manager.h"
#include "delegate/intel_openvino/operations/operations_base.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"

namespace tflite {
namespace openvinodelegate {

class OpenVINOGraphBuilder {
 public:
  explicit OpenVINOGraphBuilder(std::unique_ptr<NodeManager> node_manager) {
    node_manager_ = std::move(node_manager);
  }

  std::vector<int> GetDims(const TfLiteOpaqueTensor *t) {
    int32_t num_dims = TfLiteOpaqueTensorNumDims(t);
    std::vector<int> dims(num_dims);
    for (int i = 0; i < num_dims; i++) {
      dims[i] = TfLiteOpaqueTensorDim(t, i);
    }
    return dims;
  }

  TfLiteStatus AddInputParams(const TfLiteOpaqueTensor *t, const int index);

  TfLiteStatus CreateConstNode(const TfLiteOpaqueContext *context,
                               const int index);

  TfLiteStatus UpdateResultNodes(const TfLiteOpaqueContext *context,
                                 std::vector<int> outputs);

  std::vector<std::shared_ptr<ov::Node>> getResultNodes() {
    return result_nodes_;
  }

  std::vector<std::shared_ptr<ov::opset3::Parameter>> getInputParams() {
    return input_params_;
  }

  size_t getNodeManagerSize() const { return node_manager_->getNodeCount(); }

  TfLiteStatus CreateNodeFromTfLiteOp(TfLiteRegistrationExternal *registration,
                                      TfLiteOpaqueNode *node,
                                      TfLiteOpaqueContext *context);
  std::shared_ptr<OperationsBase> CreateOpForCode(
      TfLiteBuiltinOperator builtin_code);
  TfLiteStatus CreateOpClass(TfLiteRegistrationExternal *registration,
                             std::shared_ptr<OperationsBase> &op_base);

 private:
  std::shared_ptr<NodeManager> node_manager_;
  std::vector<std::shared_ptr<ov::opset3::Parameter>> input_params_;
  std::vector<std::shared_ptr<ov::Node>> result_nodes_;
};

}  // namespace openvinodelegate
}  // namespace tflite

#endif  // DELEGATE_INTEL_OPENVINO_OPENVINO_GRAPH_BUILDER_H_
