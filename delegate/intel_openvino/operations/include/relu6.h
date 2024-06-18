/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_RELU6_H_
#define DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_RELU6_H_

#include "delegate/intel_openvino/operations/operations_base.h"

namespace tflite {
namespace openvinodelegate {

class Relu6 final : public OperationsBase {
 public:
  Relu6() {}
  TfLiteStatus CreateNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_RELU6_H_
