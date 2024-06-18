/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DELEGATE_INTEL_OPENVINO_OPENVINO_DELEGATE_H_
#define DELEGATE_INTEL_OPENVINO_OPENVINO_DELEGATE_H_

#include <memory>
#include <vector>

#include "delegate/intel_openvino/openvino_delegate_kernel.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"

static const char kOpenVINOStableDelegateName[] = "intel_openvino_delegate";
static const char kOpenVINOStableDelegateVersion[] = "1.0.0";

struct TfLiteOpenVINODelegateOptions {
  /* debug_level for the OpenVINO delegate */
  int debug_level = 0;
};

namespace tflite {
namespace openvinodelegate {

// forward declaration
class OpenVINODelegateTestPeer;

class OpenVINODelegate : public SimpleOpaqueDelegateInterface {
 public:
  explicit OpenVINODelegate(const TfLiteOpenVINODelegateOptions *options) {
    if (options == nullptr) {
      TfLiteOpenVINODelegateOptions default_opt;
      options_ = default_opt;
    } else
      options_ = *options;
  }

  bool IsNodeSupportedByDelegate(const TfLiteRegistrationExternal *registration,
                                 const TfLiteOpaqueNode *node,
                                 TfLiteOpaqueContext *context) const override;

  TfLiteStatus Initialize(TfLiteOpaqueContext *context) override;

  const char *Name() const override;

  std::unique_ptr<SimpleOpaqueDelegateKernelInterface>
  CreateDelegateKernelInterface() override;

 private:
  TfLiteOpenVINODelegateOptions options_;
  friend class OpenVINODelegateTestPeer;
  bool CheckInputType(TfLiteType tensor_type, TfLiteType expected_type) const;
  bool CheckDataTypeSupported(
      const TfLiteOpaqueContext *context, const TfLiteOpaqueNode *node,
      const std::vector<std::vector<TfLiteType>> supported_types) const;
  bool CheckDims(const TfLiteOpaqueContext *context,
                 const TfLiteOpaqueNode *node,
                 const std::vector<std::vector<int>> &dims_size) const;
  bool CheckNodeSupportByOpenVINO(
      const TfLiteRegistrationExternal *registration,
      const TfLiteOpaqueNode *node, const TfLiteOpaqueContext *context) const;
};

}  // namespace openvinodelegate
}  // namespace tflite

#endif  // DELEGATE_INTEL_OPENVINO_OPENVINO_DELEGATE_H_
