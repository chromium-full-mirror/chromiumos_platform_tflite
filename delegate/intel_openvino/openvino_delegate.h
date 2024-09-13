/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DELEGATE_INTEL_OPENVINO_OPENVINO_DELEGATE_H_
#define DELEGATE_INTEL_OPENVINO_OPENVINO_DELEGATE_H_

#include <memory>
#include <string>
#include <vector>

#include "common/simple_async_delegate.h"
#include "delegate/intel_openvino/log.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"

static const char kOpenVINOStableDelegateName[] = "intel_openvino_delegate";
static const char kOpenVINOStableDelegateVersion[] = "1.0.0";

struct TfLiteOpenVINODelegateOptions {
  // Directory to store compilation cache.
  // TODO(b/344503269): Integrate this with OpenVINO.
  std::string cache_dir;

  // Unique token identifying the model that will run on this delegate instance.
  // TODO(b/344503269): Integrate this with OpenVINO.
  std::string model_token;
};

namespace tflite {
namespace openvinodelegate {

// forward declaration
class OpenVINODelegateTestPeer;

class OpenVINODelegate : public cros::SimpleAsyncDelegateInterface {
 public:
  explicit OpenVINODelegate(const TfLiteOpenVINODelegateOptions *options) {
    if (options == nullptr) {
      TfLiteOpenVINODelegateOptions default_opt;
      options_ = default_opt;
    } else {
      options_ = *options;
    }
    VLOGF(1) << DUMP(options_);
  }

  bool IsNodeSupportedByDelegate(const TfLiteRegistrationExternal *registration,
                                 const TfLiteOpaqueNode *node,
                                 TfLiteOpaqueContext *context) const override;

  TfLiteStatus Initialize(TfLiteOpaqueContext *context) override;

  const char *Name() const override;

  std::unique_ptr<cros::SimpleAsyncDelegateKernelInterface>
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
