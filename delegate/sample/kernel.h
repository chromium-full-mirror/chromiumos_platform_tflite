/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef DELEGATE_SAMPLE_KERNEL_H_
#define DELEGATE_SAMPLE_KERNEL_H_

#include <memory>

#include "common/simple_async_delegate.h"
#include "delegate/sample/async_kernel.h"
#include "delegate/sample/core.h"
#include "tensorflow/lite/core/c/c_api_opaque.h"

namespace tflite::cros {

// A simple delegate kernel that implement the computation of addition and
// subtraction operations. This class is thread-compatible.
class CrosSampleDelegateKernel : public SimpleAsyncDelegateKernelInterface {
 public:
  CrosSampleDelegateKernel();

  // Move-only.
  CrosSampleDelegateKernel(CrosSampleDelegateKernel&& other) = default;
  CrosSampleDelegateKernel& operator=(CrosSampleDelegateKernel&& other) =
      default;
  CrosSampleDelegateKernel(const CrosSampleDelegateKernel&) = delete;
  CrosSampleDelegateKernel& operator=(const CrosSampleDelegateKernel&) = delete;

  // Implementation of SimpleAsyncDelegateKernelInterface.
  TfLiteStatus Init(TfLiteOpaqueContext* context,
                    const TfLiteOpaqueDelegateParams* params) override;
  TfLiteStatus Prepare(TfLiteOpaqueContext* context,
                       TfLiteOpaqueNode* node) override;
  TfLiteStatus Eval(TfLiteOpaqueContext* context,
                    TfLiteOpaqueNode* node) override;
  TfLiteAsyncKernel* AsyncKernel(TfLiteOpaqueContext* context,
                                 TfLiteOpaqueNode* node) override;

 private:
  CrosSampleDelegateCore core_;
  std::unique_ptr<CrosSampleDelegateAsyncKernel> async_kernel_;
};

}  // namespace tflite::cros

#endif  // DELEGATE_SAMPLE_KERNEL_H_
