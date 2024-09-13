/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DELEGATE_OPENVINO_ASYNC_KERNEL_H_
#define DELEGATE_OPENVINO_ASYNC_KERNEL_H_

#include <algorithm>
#include <map>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "android/hardware_buffer.h"
#include "delegate/intel_openvino/log.h"
#include "delegate/intel_openvino/openvino_delegate_kernel.h"
#include "tensorflow/lite/async/backend_async_kernel_interface.h"
#include "tensorflow/lite/core/async/interop/c/constants.h"
#include "tensorflow/lite/delegates/utils/async_type_helpers.h"

namespace tflite::openvinodelegate {

using tflite::delegates::utils::BufferAttributes;
using tflite::delegates::utils::BufferType;
using tflite::delegates::utils::kBufferTypeAHardwareBufferBlob;
using tflite::delegates::utils::SyncAttributes;
using BackendAsyncKernelInterface =
    ::tflite::delegates::BackendAsyncKernelInterface;

class OpenVINOAsyncDelegateKernel;
class OVDelegateAsyncKernel : public BackendAsyncKernelInterface {
 public:
  explicit OVDelegateAsyncKernel(OpenVINOAsyncDelegateKernel* core);
  ~OVDelegateAsyncKernel() override = default;

  // Buffer operations
  TfLiteStatus RegisterBuffer(TfLiteOpaqueContext* opaque_context,
                              TfLiteIoType io_type,
                              const TfLiteBackendBuffer* buffer,
                              const TfLiteAttributeMap* attrs,
                              TfLiteBufferHandle handle) override;
  TfLiteStatus RegisterBufferSlice(TfLiteOpaqueContext* context,
                                   TfLiteBufferHandle buffer_pool,
                                   const TfLiteAttributeMap* attrs,
                                   TfLiteBufferHandle handle) override {
    // TODO: implement the interface when required later
    TFLITE_LOG(INFO)
        << "OVDelegateAsyncKernel::RegisterBufferSlice is not implemented";
    return kTfLiteError;
  }
  TfLiteStatus UnregisterBuffer(TfLiteOpaqueContext* opaque_context,
                                TfLiteBufferHandle handle) override;

  // Reconciliations
  const std::vector<const char*>& SupportedBufferTypes(
      TfLiteIoType io_type) const override {
    return supported_buffer_types_;
  }
  const std::vector<const char*>& SupportedSynchronizations(
      TfLiteIoType io_type) const override {
    return supported_synchronizations_;
  }
  bool ReconcileRestrictions(const TfLiteOpaqueContext* opaque_context,
                             const TfLiteOpaqueNode* opaque_node,
                             int tensor_index,
                             const TfLiteAttributeMap* user_provided_attributes,
                             TfLiteAttributeMap* merged,
                             TfLiteAttributeMap* conflict) const override;
  TfLiteStatus SetAttributes(TfLiteOpaqueContext* context,
                             TfLiteOpaqueNode* node, int tensor_index,
                             const TfLiteAttributeMap* attrs) override;
  TfLiteStatus SetBufferAttributes(const TfLiteBackendBuffer* buffer,
                                   const TfLiteAttributeMap* attrs) override {
    // TODO: implement the interface when required later
    TFLITE_LOG(INFO)
        << "OVDelegateAsyncKernel::SetBufferAttributes is not supported";
    return kTfLiteError;
  }

  TfLiteStatus GetBufferAttributes(const TfLiteBackendBuffer* buffer,
                                   TfLiteAttributeMap* attrs) override {
    // TODO: implement the interface when required later
    TFLITE_LOG(INFO)
        << "OVDelegateAsyncKernel::GetBufferAttributes is not supported";
    return kTfLiteError;
  }
  TfLiteStatus Prepare(TfLiteOpaqueContext* context,
                       TfLiteOpaqueNode* node) override;

  // Execution methods
  TfLiteStatus Eval(TfLiteOpaqueContext* opaque_context,
                    TfLiteOpaqueNode* opaque_node,
                    TfLiteExecutionTask* task) override;
  TfLiteStatus Wait(TfLiteOpaqueContext* opaque_context,
                    TfLiteExecutionTask* task) override;
  TfLiteStatus Finish(TfLiteOpaqueContext* opaque_context,
                      TfLiteExecutionTask* task) override;

 private:
  mutable absl::Mutex mutex_;
  OpenVINOAsyncDelegateKernel& core_ ABSL_GUARDED_BY(mutex_);
  // For SupportedBufferTypes and SupportedSynchronizations
  const std::vector<const char*> supported_buffer_types_ = {
      kBufferTypeAHardwareBufferBlob};
  const std::vector<const char*> supported_synchronizations_ = {
      kTfLiteSyncTypeNoSyncObj,
      ::tflite::delegates::utils::kSyncTypeSyncFenceFd};
  std::map<TfLiteExecutionTask*, TfLiteStatus> task_status_map_
      ABSL_GUARDED_BY(mutex_);
  std::map<TfLiteBufferHandle, AHardwareBuffer*> ahwb_buffer_map_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace tflite::openvinodelegate

#endif  // DELEGATE_OPENVINO_ASYNC_KERNEL_H_
