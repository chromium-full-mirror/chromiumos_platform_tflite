/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef DELEGATE_MTK_NEURON_NEURON_ASYNC_KERNEL_H_
#define DELEGATE_MTK_NEURON_NEURON_ASYNC_KERNEL_H_

#include <map>
#include <vector>

#include "android/hardware_buffer.h"
#include "delegate/mtk_neuron/neuron_delegate_kernel.h"
#include "tensorflow/lite/async/backend_async_kernel_interface.h"
#include "tensorflow/lite/core/async/interop/c/constants.h"
#include "tensorflow/lite/delegates/utils/async_type_helpers.h"

namespace tflite::neuron {

using delegates::utils::kBufferTypeAHardwareBufferBlob;
using BackendAsyncKernelInterface =
    ::tflite::delegates::BackendAsyncKernelInterface;

class NeuronStableDelegateKernel;
class DelegateAsyncKernel : public BackendAsyncKernelInterface {
 public:
  explicit DelegateAsyncKernel(NeuronStableDelegateKernel* core);
  ~DelegateAsyncKernel() override = default;

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
    printf("DelegateAsyncKernel::RegisterBufferSlice unimplemented\n");
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
                                   const TfLiteAttributeMap* attrs) override;
  TfLiteStatus GetBufferAttributes(const TfLiteBackendBuffer* buffer,
                                   TfLiteAttributeMap* attrs) override;
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
  NeuronStableDelegateKernel& core_;
  TfLiteStatus RegisterBufferImpl(TfLiteContext* context, TfLiteIoType io_type,
                                  const TfLiteBackendBuffer* buffer,
                                  const TfLiteAttributeMap* attrs,
                                  TfLiteBufferHandle handle);
  TfLiteStatus UnregisterBufferImpl(TfLiteContext* context,
                                    TfLiteBufferHandle handle);
  // For SupportedBufferTypes and SupportedSynchronizations
  const std::vector<const char*> supported_buffer_types_ = {
      kBufferTypeAHardwareBufferBlob};
  const std::vector<const char*> supported_synchronizations_ = {
      kTfLiteSyncTypeNoSyncObj};
  std::map<TfLiteBufferHandle, AHardwareBuffer*> registered_buffers_;
  std::map<TfLiteExecutionTask*, TfLiteStatus> task_status_map_;
};

}  // namespace tflite::neuron

#endif  // DELEGATE_MTK_NEURON_NEURON_ASYNC_KERNEL_H_
