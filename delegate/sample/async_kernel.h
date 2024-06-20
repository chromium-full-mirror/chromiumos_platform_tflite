/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef DELEGATE_SAMPLE_ASYNC_KERNEL_H_
#define DELEGATE_SAMPLE_ASYNC_KERNEL_H_

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "android/hardware_buffer.h"
#include "delegate/sample/core.h"
#include "tensorflow/lite/async/backend_async_kernel_interface.h"
#include "tensorflow/lite/core/async/interop/c/constants.h"
#include "tensorflow/lite/delegates/utils/async_type_helpers.h"

namespace tflite::cros {

using delegates::utils::kBufferTypeAHardwareBufferBlob;
using BackendAsyncKernelInterface = delegates::BackendAsyncKernelInterface;

// A helper generic map type where the key is an input/output tensor name.
template <typename T>
using IoTensorMap = std::map<std::pair<TfLiteIoType, std::string>, T>;

// The implementation class of async kernel API. Currently the model inference
// is still synchronous.
// This class is thread-safe.
class CrosSampleDelegateAsyncKernel : public BackendAsyncKernelInterface {
 public:
  explicit CrosSampleDelegateAsyncKernel(CrosSampleDelegateCore* core);

  // Neither copyable nor movable.
  CrosSampleDelegateAsyncKernel(CrosSampleDelegateAsyncKernel&& other) = delete;
  CrosSampleDelegateAsyncKernel& operator=(
      CrosSampleDelegateAsyncKernel&& other) = delete;
  CrosSampleDelegateAsyncKernel(const CrosSampleDelegateAsyncKernel&) = delete;
  CrosSampleDelegateAsyncKernel& operator=(
      const CrosSampleDelegateAsyncKernel&) = delete;

  // Implementation of BackendAsyncKernelInterface.
  TfLiteStatus RegisterBuffer(TfLiteOpaqueContext* context,
                              TfLiteIoType io_type,
                              const TfLiteBackendBuffer* buffer,
                              const TfLiteAttributeMap* attrs,
                              TfLiteBufferHandle handle) override;

  TfLiteStatus RegisterBufferSlice(TfLiteOpaqueContext* context,
                                   TfLiteBufferHandle buffer_pool,
                                   const TfLiteAttributeMap* attrs,
                                   TfLiteBufferHandle handle) override;

  TfLiteStatus UnregisterBuffer(TfLiteOpaqueContext* context,
                                TfLiteBufferHandle handle) override;

  const std::vector<const char*>& SupportedBufferTypes(
      TfLiteIoType io_type) const override;

  const std::vector<const char*>& SupportedSynchronizations(
      TfLiteIoType io_type) const override;

  bool ReconcileRestrictions(const TfLiteOpaqueContext* context,
                             const TfLiteOpaqueNode* node,
                             int tensor_index,
                             const TfLiteAttributeMap* user_provided_attributes,
                             TfLiteAttributeMap* merged,
                             TfLiteAttributeMap* conflict) const override;

  TfLiteStatus SetAttributes(TfLiteOpaqueContext* context,
                             TfLiteOpaqueNode* node,
                             int tensor_index,
                             const TfLiteAttributeMap* attrs) override;

  TfLiteStatus SetBufferAttributes(const TfLiteBackendBuffer* buffer,
                                   const TfLiteAttributeMap* attrs) override;

  TfLiteStatus GetBufferAttributes(const TfLiteBackendBuffer* buffer,
                                   TfLiteAttributeMap* attrs) override;

  TfLiteStatus Prepare(TfLiteOpaqueContext* context,
                       TfLiteOpaqueNode* node) override;

  TfLiteStatus Eval(TfLiteOpaqueContext* context,
                    TfLiteOpaqueNode* node,
                    TfLiteExecutionTask* task) override;

  TfLiteStatus Wait(TfLiteOpaqueContext* context,
                    TfLiteExecutionTask* task) override;

  TfLiteStatus Finish(TfLiteOpaqueContext* context,
                      TfLiteExecutionTask* task) override;

 private:
  const std::vector<const char*> supported_buffer_types_ = {
      kBufferTypeAHardwareBufferBlob};

  // TODO(shik): Support fence.
  const std::vector<const char*> supported_sync_types_ = {
      kTfLiteSyncTypeNoSyncObj};

  mutable absl::Mutex mutex_;

  CrosSampleDelegateCore& core_ ABSL_GUARDED_BY(mutex_);

  std::map<TfLiteBufferHandle, AHardwareBuffer*> registered_buffers_
      ABSL_GUARDED_BY(mutex_);

  std::map<TfLiteExecutionTask*, TfLiteStatus> task_status_map_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace tflite::cros

#endif  // DELEGATE_SAMPLE_ASYNC_KERNEL_H_
