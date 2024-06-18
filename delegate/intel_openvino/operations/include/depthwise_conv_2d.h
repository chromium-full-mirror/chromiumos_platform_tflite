/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_DEPTHWISE_CONV_2D_H_
#define DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_DEPTHWISE_CONV_2D_H_

#include "delegate/intel_openvino/operations/operations_base.h"

namespace tflite {
namespace openvinodelegate {

class DepthwiseConv2D final : public OperationsBase {
 public:
  DepthwiseConv2D() {}
  TfLiteStatus CreateNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_DEPTHWISE_CONV_2D_H_
