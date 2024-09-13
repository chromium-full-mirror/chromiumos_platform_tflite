/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/openvino_async_kernel.h"

#include <sys/mman.h>

#include "delegate/intel_openvino/openvino_delegate_kernel.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/core/async/c/task.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/util.h"

namespace tflite::openvinodelegate {

using tflite::delegates::utils::ReadBufferAttrs;
using tflite::delegates::utils::ReadSyncAttrs;

OVDelegateAsyncKernel::OVDelegateAsyncKernel(OpenVINOAsyncDelegateKernel* core)
    : core_(*core) {}

TfLiteStatus OVDelegateAsyncKernel::RegisterBuffer(
    TfLiteOpaqueContext* opaque_context, TfLiteIoType io_type,
    const TfLiteBackendBuffer* buffer, const TfLiteAttributeMap* attrs,
    TfLiteBufferHandle handle) {
  absl::MutexLock lock(&mutex_);
  if (TfLiteAttributeMapIsBufferAttributeMap(attrs)) {
    auto buffer_attrs = ReadBufferAttrs(attrs);
    size_t buffer_size = buffer_attrs.size.value();
    if (buffer_attrs.buffer_type != BufferType::kAHardwareBufferBlob)
      return kTfLiteError;

    auto* ptr =
        static_cast<AHardwareBuffer*>(TfLiteBackendBufferGetPtr(buffer));

    if (core_.RegisterBuffer(handle, ptr, buffer_size) != kTfLiteOk)
      return kTfLiteError;
    return kTfLiteOk;
  }
  return kTfLiteError;
}

TfLiteStatus OVDelegateAsyncKernel::UnregisterBuffer(
    TfLiteOpaqueContext* opaque_context, TfLiteBufferHandle handle) {
  absl::MutexLock lock(&mutex_);
  if (core_.UnregisterBuffer(handle) != kTfLiteOk) return kTfLiteError;
  return kTfLiteOk;
}

bool OVDelegateAsyncKernel::ReconcileRestrictions(
    const TfLiteOpaqueContext* opaque_context,
    const TfLiteOpaqueNode* opaque_node, int tensor_index,
    const TfLiteAttributeMap* user_provided_attributes,
    TfLiteAttributeMap* merged, TfLiteAttributeMap* conflict) const {
  absl::MutexLock lock(&mutex_);
  TfLiteOpaqueTensor* tensor =
      TfLiteOpaqueContextGetOpaqueTensor(opaque_context, tensor_index);
  if (TfLiteAttributeMapIsBufferAttributeMap(user_provided_attributes)) {
    BufferAttributes conflict_attrs{};
    auto buffer_attrs = ReadBufferAttrs(user_provided_attributes);

    auto buffer_type =
        buffer_attrs.buffer_type.value_or(BufferType::kAHardwareBufferBlob);
    if (buffer_type != BufferType::kAHardwareBufferBlob) {
      conflict_attrs.buffer_type = BufferType::kAHardwareBufferBlob;
      delegates::utils::WriteBufferAttrs(conflict_attrs, conflict);
      return false;
    }
    size_t tensor_size = TfLiteOpaqueTensorByteSize(tensor);
    buffer_attrs.size = std::max(buffer_attrs.size.value_or(0), tensor_size);
    delegates::utils::WriteBufferAttrs(buffer_attrs, merged);
    return true;
  } else if (TfLiteAttributeMapIsSyncAttributeMap(user_provided_attributes)) {
    SyncAttributes conflict_attrs;
    SyncAttributes merged_attrs;
    auto sync_attrs = delegates::utils::ReadSyncAttrs(user_provided_attributes);
    auto sync_type =
        sync_attrs.sync_type.value_or(delegates::utils::SyncType::kNoSyncObj);
    if (sync_type == delegates::utils::SyncType::kUnknown) {
      conflict_attrs.sync_type = delegates::utils::SyncType::kNoSyncObj;
      delegates::utils::WriteSyncAttrs(conflict_attrs, conflict);
      return false;
    }
    merged_attrs.sync_type = sync_type;
    delegates::utils::WriteSyncAttrs(merged_attrs, merged);
  } else {
    return false;
  }
  return true;
}

TfLiteStatus OVDelegateAsyncKernel::SetAttributes(
    TfLiteOpaqueContext* opaque_context, TfLiteOpaqueNode* opaque_node,
    int tensor_index, const TfLiteAttributeMap* attrs) {
  // TODO: This function is a no-op for now
  return kTfLiteOk;
}

TfLiteStatus OVDelegateAsyncKernel::Prepare(TfLiteOpaqueContext* opaque_context,
                                            TfLiteOpaqueNode* opaque_node) {
  // TODO: This function is a no-op for now
  return kTfLiteOk;
}

TfLiteStatus OVDelegateAsyncKernel::Eval(TfLiteOpaqueContext* opaque_context,
                                         TfLiteOpaqueNode* opaque_node,
                                         TfLiteExecutionTask* task) {
  absl::MutexLock lock(&mutex_);
  TfLiteStatus status = core_.EvalAsyncImpl(opaque_context, opaque_node, task);
  task_status_map_.insert_or_assign(task, status);
  return status;
}

TfLiteStatus OVDelegateAsyncKernel::Wait(TfLiteOpaqueContext* opaque_context,
                                         TfLiteExecutionTask* task) {
  absl::MutexLock lock(&mutex_);
  auto it = task_status_map_.find(task);
  if (it == task_status_map_.end()) {
    return kTfLiteError;
  }

  return it->second;
}

TfLiteStatus OVDelegateAsyncKernel::Finish(TfLiteOpaqueContext* opaque_context,
                                           TfLiteExecutionTask* task) {
  absl::MutexLock lock(&mutex_);
  size_t ret = task_status_map_.erase(task);
  return ret == 1 ? kTfLiteOk : kTfLiteError;
}

}  // namespace tflite::openvinodelegate
