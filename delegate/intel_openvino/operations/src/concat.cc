/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/operations/include/concat.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Concat::CreateNode() {
  auto *concat_params = GetBuiltinData<TfLiteConcatenationParams>();

  size_t n = tensor_indices_size_;
  std::vector<ov::Output<ov::Node>> inputs;
  for (size_t i = 0; i < n; i++) {
    auto inputOp = getInputNode(tensor_indices_[i]);
    inputs.push_back(inputOp);
  }

  auto concatNode =
      std::make_shared<ov::opset8::Concat>(inputs, concat_params->axis);
  output_node_ = ApplyActivation(concatNode, concat_params->activation);

  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
