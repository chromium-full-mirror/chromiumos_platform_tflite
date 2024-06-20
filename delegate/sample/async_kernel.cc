/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "delegate/sample/async_kernel.h"

#include <algorithm>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/synchronization/mutex.h"
#include "common/log.h"
#include "tensorflow/lite/core/async/c/task.h"
#include "tensorflow/lite/core/c/c_api_opaque.h"
#include "tensorflow/lite/delegates/utils/async_type_helpers.h"

namespace tflite::cros {

namespace {

using delegates::utils::ReadBufferAttrs;
using delegates::utils::ReadSyncAttrs;
using delegates::utils::ScopedTfLiteAttrMap;
using delegates::utils::SyncType;
using delegates::utils::WriteBufferAttrs;

}  // namespace

CrosSampleDelegateAsyncKernel::CrosSampleDelegateAsyncKernel(
    CrosSampleDelegateCore* core)
    : core_(*core) {}

TfLiteStatus CrosSampleDelegateAsyncKernel::RegisterBuffer(
    TfLiteOpaqueContext* context,
    TfLiteIoType io_type,
    const TfLiteBackendBuffer* buffer,
    const TfLiteAttributeMap* attrs,
    TfLiteBufferHandle handle) {
  absl::MutexLock lock(&mutex_);
  auto ahwb = static_cast<AHardwareBuffer*>(TfLiteBackendBufferGetPtr(buffer));
  if (ahwb == nullptr) {
    LOGF(ERROR) << "Got null AHardwareBuffer";
    return kTfLiteError;
  }
  AHardwareBuffer_acquire(ahwb);
  registered_buffers_.emplace(handle, ahwb);
  return kTfLiteOk;
}

TfLiteStatus CrosSampleDelegateAsyncKernel::RegisterBufferSlice(
    TfLiteOpaqueContext* context,
    TfLiteBufferHandle buffer_pool,
    const TfLiteAttributeMap* attrs,
    TfLiteBufferHandle handle) {
  // TODO(shik): Not supported yet.
  return kTfLiteError;
}

TfLiteStatus CrosSampleDelegateAsyncKernel::UnregisterBuffer(
    TfLiteOpaqueContext* context,
    TfLiteBufferHandle handle) {
  absl::MutexLock lock(&mutex_);
  auto it = registered_buffers_.find(handle);
  if (it == registered_buffers_.end()) {
    LOGF(ERROR) << "Unknown handle " << handle;
    return kTfLiteError;
  }
  AHardwareBuffer_release(it->second);
  registered_buffers_.erase(it);
  return kTfLiteOk;
}

const std::vector<const char*>&
CrosSampleDelegateAsyncKernel::SupportedBufferTypes(
    TfLiteIoType io_type) const {
  return supported_buffer_types_;
}

const std::vector<const char*>&
CrosSampleDelegateAsyncKernel::SupportedSynchronizations(
    TfLiteIoType io_type) const {
  return supported_sync_types_;
}

bool CrosSampleDelegateAsyncKernel::ReconcileRestrictions(
    const TfLiteOpaqueContext* context,
    const TfLiteOpaqueNode* node,
    int tensor_index,
    const TfLiteAttributeMap* user_provided_attributes,
    TfLiteAttributeMap* merged,
    TfLiteAttributeMap* conflict) const {
  if (TfLiteAttributeMapIsBufferAttributeMap(user_provided_attributes)) {
    auto attrs = ReadBufferAttrs(user_provided_attributes);
    size_t tensor_byte_size = TfLiteOpaqueTensorByteSize(
        TfLiteOpaqueContextGetOpaqueTensor(context, tensor_index));
    attrs.size = std::max(attrs.size.value_or(0), tensor_byte_size);
    WriteBufferAttrs(attrs, merged);
  } else if (TfLiteAttributeMapIsSyncAttributeMap(user_provided_attributes)) {
    auto attrs = ReadSyncAttrs(user_provided_attributes);
    if (attrs.sync_type.value_or(SyncType::kNoSyncObj) !=
        SyncType::kNoSyncObj) {
      return false;
    }
  } else {
    return false;
  }
  return true;
}

TfLiteStatus CrosSampleDelegateAsyncKernel::SetAttributes(
    TfLiteOpaqueContext* context,
    TfLiteOpaqueNode* node,
    int tensor_index,
    const TfLiteAttributeMap* attrs) {
  // No-op for now. We need to store sync information once we support fence.
  return kTfLiteOk;
}

TfLiteStatus CrosSampleDelegateAsyncKernel::SetBufferAttributes(
    const TfLiteBackendBuffer* buffer,
    const TfLiteAttributeMap* attrs) {
  // TODO(b/348328994): Implement this. This is a no-op function for now to
  // unblock TensorFlow uprev.
  return kTfLiteDelegateError;
}

TfLiteStatus CrosSampleDelegateAsyncKernel::GetBufferAttributes(
    const TfLiteBackendBuffer* buffer,
    TfLiteAttributeMap* attrs) {
  // TODO(b/348328994): Implement this. This is a no-op function for now to
  // unblock TensorFlow uprev.
  return kTfLiteDelegateError;
}

TfLiteStatus CrosSampleDelegateAsyncKernel::Prepare(
    TfLiteOpaqueContext* context,
    TfLiteOpaqueNode* node) {
  absl::MutexLock lock(&mutex_);
  return core_.Prepare();
}

TfLiteStatus CrosSampleDelegateAsyncKernel::Eval(TfLiteOpaqueContext* context,
                                                 TfLiteOpaqueNode* node,
                                                 TfLiteExecutionTask* task) {
  absl::MutexLock lock(&mutex_);

  std::vector<AHardwareBuffer*> locked_buffers;
  absl::Cleanup unlock_buffers = [&] {
    for (auto* buffer : locked_buffers) {
      AHardwareBuffer_unlock(buffer, nullptr);
    }
  };

  int num_inputs = 0;
  const int* inputs = nullptr;
  if (TfLiteOpaqueNodeInputs(node, &inputs, &num_inputs) != kTfLiteOk) {
    return kTfLiteError;
  }
  for (int i = 0; i < num_inputs; ++i) {
    auto tensor = TfLiteOpaqueNodeGetInput(context, node, i);
    TfLiteBufferHandle handle =
        TfLiteExecutionTaskGetBufferByIndex(task, inputs[i]);
    AHardwareBuffer* buffer = registered_buffers_.at(handle);

    void* addr = nullptr;
    if (AHardwareBuffer_lock(buffer,
                             AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                                 AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
                             /*fence=*/-1, /*rect=*/nullptr, &addr) != 0) {
      LOGF(ERROR) << "Failed to lock AHardwareBuffer " << buffer;
      return kTfLiteError;
    }
    locked_buffers.push_back(buffer);

    core_.SetExternalTensorMemory(tensor, addr);
  }

  int num_outputs = 0;
  const int* outputs = nullptr;
  if (TfLiteOpaqueNodeOutputs(node, &outputs, &num_outputs) != kTfLiteOk) {
    return kTfLiteError;
  }
  for (int i = 0; i < num_outputs; ++i) {
    auto tensor = TfLiteOpaqueNodeGetOutput(context, node, i);
    TfLiteBufferHandle handle =
        TfLiteExecutionTaskGetBufferByIndex(task, outputs[i]);
    AHardwareBuffer* buffer = registered_buffers_.at(handle);

    void* addr = nullptr;
    if (AHardwareBuffer_lock(buffer,
                             AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                                 AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
                             /*fence=*/-1, /*rect=*/nullptr, &addr) != 0) {
      LOGF(ERROR) << "Failed to lock AHardwareBuffer " << buffer;
      return kTfLiteError;
    }
    locked_buffers.push_back(buffer);

    core_.SetExternalTensorMemory(tensor, addr);
  }

  TfLiteStatus status = core_.Eval();
  task_status_map_.insert_or_assign(task, status);
  return status;
}

TfLiteStatus CrosSampleDelegateAsyncKernel::Wait(TfLiteOpaqueContext* context,
                                                 TfLiteExecutionTask* task) {
  absl::MutexLock lock(&mutex_);
  auto it = task_status_map_.find(task);
  if (it == task_status_map_.end()) {
    LOGF(ERROR) << "Unknown task " << task;
    return kTfLiteError;
  }
  return it->second;
}

TfLiteStatus CrosSampleDelegateAsyncKernel::Finish(TfLiteOpaqueContext* context,
                                                   TfLiteExecutionTask* task) {
  absl::MutexLock lock(&mutex_);
  size_t erased = task_status_map_.erase(task);
  return erased == 1 ? kTfLiteOk : kTfLiteError;
}

}  // namespace tflite::cros
