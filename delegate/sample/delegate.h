/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef DELEGATE_SAMPLE_DELEGATE_H_
#define DELEGATE_SAMPLE_DELEGATE_H_

#include <memory>

#include "common/simple_async_delegate.h"
#include "tensorflow/lite/core/c/common.h"

namespace tflite::cros {

// A simple delegate that supports only addition and subtraction operations.
// Implements SimpleAsyncDelegateInterface, and therefore the delegate can be
// easily be adapted to work with the stable TFLite delegate API via
// TfLiteOpaqueDelegateFactory.
class CrosSampleDelegate : public SimpleAsyncDelegateInterface {
 public:
  static constexpr char kName[] = "cros_sample_delegate";
  static constexpr char kVersion[] = "0.1.2";

  // CrosSampleDelegate supports float32 input type only. Returns true if the
  // inputs of 'node' are two tensors of float32 with the same shape and the
  // operation is addition or subtraction without fused activation.
  bool IsNodeSupportedByDelegate(
      const TfLiteRegistrationExternal* registration_external,
      const TfLiteOpaqueNode* node,
      TfLiteOpaqueContext* context) const override;

  // No-op. The delegate doesn't have extra steps to perform during
  // initialization.
  TfLiteStatus Initialize(TfLiteOpaqueContext* context) override;

  // Returns a name that identifies the delegate.
  const char* Name() const override;

  // Returns an instance of CrosSampleDelegateKernel that implements
  // SimpleAsyncDelegateKernelInterface. CrosSampleDelegateKernel describes
  // how a subgraph is delegated and the concrete evaluation of both addition
  // and subtraction operations to be performed by the delegate.
  std::unique_ptr<SimpleAsyncDelegateKernelInterface>
  CreateDelegateKernelInterface() override;
};

}  // namespace tflite::cros

#endif  // DELEGATE_SAMPLE_DELEGATE_H_
