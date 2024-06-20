/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "delegate/mtk_neuron/neuron_async_kernel.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "delegate/mtk_neuron/neuron_delegate_kernel.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/core/async/c/task.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/util.h"

namespace tflite::neuron {
using tflite::delegates::utils::ReadBufferAttrs;

DelegateAsyncKernel::DelegateAsyncKernel(NeuronStableDelegateKernel* core)
    : core_(*core) {}

constexpr size_t kDefaultByteAlignmentForNeuron = 128;

static size_t alignToLCM(size_t provided_size, size_t alignment_size) {
  return std::lcm(provided_size, alignment_size);
}

bool DelegateAsyncKernel::ReconcileRestrictions(
    const TfLiteOpaqueContext* opaque_context,
    const TfLiteOpaqueNode* opaque_node, int tensor_index,
    const TfLiteAttributeMap* user_provided_attributes,
    TfLiteAttributeMap* merged, TfLiteAttributeMap* conflict) const {
  TfLiteOpaqueTensor* tensor =
      TfLiteOpaqueContextGetOpaqueTensor(opaque_context, tensor_index);
  if (TfLiteAttributeMapIsBufferAttributeMap(user_provided_attributes)) {
    auto attrs = ReadBufferAttrs(user_provided_attributes);
    size_t tensor_size = TfLiteOpaqueTensorByteSize(tensor);
    size_t padding =
        alignToLCM(attrs.padding.value_or(1), kDefaultByteAlignmentForNeuron);
    attrs.size = std::max(attrs.size.value_or(0), tensor_size);
    attrs.padding = padding;
    delegates::utils::WriteBufferAttrs(attrs, merged);
    return true;
  } else if (TfLiteAttributeMapIsSyncAttributeMap(user_provided_attributes)) {
    auto attrs = delegates::utils::ReadSyncAttrs(user_provided_attributes);
    if (attrs.sync_type.value_or(delegates::utils::SyncType::kNoSyncObj) !=
        delegates::utils::SyncType::kNoSyncObj) {
      return false;
    }
  } else {
    return false;
  }
  return true;
}

TfLiteStatus DelegateAsyncKernel::SetAttributes(
    TfLiteOpaqueContext* opaque_context, TfLiteOpaqueNode* opaque_node,
    int tensor_index, const TfLiteAttributeMap* attrs) {
  return kTfLiteOk;
}

TfLiteStatus DelegateAsyncKernel::SetBufferAttributes(
    const TfLiteBackendBuffer* buffer, const TfLiteAttributeMap* attrs) {
  // TODO(b/348328994): Implement this. This is a no-op function for now to
  // unblock TensorFlow uprev.
  return kTfLiteDelegateError;
}

TfLiteStatus DelegateAsyncKernel::GetBufferAttributes(
    const TfLiteBackendBuffer* buffer, TfLiteAttributeMap* attrs) {
  // TODO(b/348328994): Implement this. This is a no-op function for now to
  // unblock TensorFlow uprev.
  return kTfLiteDelegateError;
}

TfLiteStatus DelegateAsyncKernel::Prepare(TfLiteOpaqueContext* opaque_context,
                                          TfLiteOpaqueNode* opaque_node) {
  return core_.Prepare(opaque_context, opaque_node);
}

TfLiteStatus DelegateAsyncKernel::RegisterBuffer(
    TfLiteOpaqueContext* opaque_context, TfLiteIoType io_type,
    const TfLiteBackendBuffer* buffer, const TfLiteAttributeMap* attrs,
    TfLiteBufferHandle handle) {
  auto buffer_attrs = ReadBufferAttrs(attrs);
  size_t buffer_size = buffer_attrs.size.value();

  auto* ptr = static_cast<AHardwareBuffer*>(TfLiteBackendBufferGetPtr(buffer));
  AHardwareBuffer_acquire(ptr);
  const native_handle_t* buffer_handle = AHardwareBuffer_getNativeHandle(ptr);
  if (buffer_handle == nullptr) {
    return kTfLiteError;
  }
  if (buffer_handle->numFds != 1) {
    return kTfLiteError;
  }
  // TODO: Add a proper way to query the FD to use
  int fd = buffer_handle->data[0];
  if (fd == -1) {
    return kTfLiteError;
  }
  registered_buffers_.emplace(handle, ptr);
  return core_.RegisterBuffer(handle, fd, buffer_size);
}

TfLiteStatus DelegateAsyncKernel::UnregisterBuffer(
    TfLiteOpaqueContext* opaque_context, TfLiteBufferHandle handle) {
  auto it = registered_buffers_.find(handle);
  if (it == registered_buffers_.end()) {
    return kTfLiteError;
  }
  TF_LITE_OPAQUE_ENSURE_STATUS(core_.UnregisterBuffer(handle));
  AHardwareBuffer_release(it->second);
  registered_buffers_.erase(it);
  return kTfLiteOk;
}

TfLiteStatus DelegateAsyncKernel::Eval(TfLiteOpaqueContext* opaque_context,
                                       TfLiteOpaqueNode* opaque_node,
                                       TfLiteExecutionTask* task) {
  TfLiteStatus status = core_.Eval(opaque_context, opaque_node, task);
  task_status_map_.insert_or_assign(task, status);
  return status;
}

TfLiteStatus DelegateAsyncKernel::Wait(TfLiteOpaqueContext* opaque_context,
                                       TfLiteExecutionTask* task) {
  auto it = task_status_map_.find(task);
  if (it == task_status_map_.end()) {
    return kTfLiteError;
  }
  return it->second;
}

TfLiteStatus DelegateAsyncKernel::Finish(TfLiteOpaqueContext* opaque_context,
                                         TfLiteExecutionTask* task) {
  size_t erased = task_status_map_.erase(task);
  return erased == 1 ? kTfLiteOk : kTfLiteError;
}

}  // namespace tflite::neuron
