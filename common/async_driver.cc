/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/async_driver.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <span>  // NOLINT(build/include_order) - C++20 header is not recognized yet
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_format.h"
#include "android/hardware_buffer.h"
#include "common/scoped_ahwb.h"
#include "tensorflow/lite/core/async/c/task.h"
#include "tensorflow/lite/core/async/interop/c/constants.h"
#include "tensorflow/lite/delegates/utils/async_type_helpers.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"

namespace tflite::cros {

namespace {

constexpr char kSignatureKey[] = "serving_default";

using delegates::utils::BufferType;
using delegates::utils::CreateScopedTfLiteAttrMap;
using delegates::utils::CreateScopedTfLiteBackendBuffer;
using delegates::utils::CreateScopedTfLiteSynchronization;
using delegates::utils::kBufferTypeAHardwareBufferBlob;
using delegates::utils::ReadBufferAttrs;
using delegates::utils::ScopedTfLiteAttrMap;
using delegates::utils::ScopedTfLiteSynchronization;
using delegates::utils::SyncType;
using delegates::utils::WriteBufferAttrs;
using delegates::utils::WriteSyncAttrs;

bool ContainsString(const std::vector<const char*>& container,
                    const char* needle) {
  return any_of(container.begin(), container.end(),
                [&](const char* s) { return strcmp(s, needle) == 0; });
}

}  // namespace

std::unique_ptr<AsyncDriver> AsyncDriver::Create(
    TfLiteDelegatePtr delegate,
    std::unique_ptr<FlatBufferModel> model) {
  ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);
  builder.AddDelegate(delegate.get());
  std::unique_ptr<Interpreter> interpreter;
  if (builder(&interpreter) != kTfLiteOk) {
    return nullptr;
  }

  async::AsyncSignatureRunner* runner =
      interpreter->GetAsyncSignatureRunner(kSignatureKey);
  if (runner == nullptr) {
    return nullptr;
  }

  return std::unique_ptr<AsyncDriver>(new AsyncDriver(
      std::move(delegate), std::move(model), std::move(interpreter), runner));
}

TfLiteStatus AsyncDriver::Prepare() {
  if (ReconcileBufferAttributes() != kTfLiteOk) {
    return kTfLiteError;
  }
  if (ReconcileSyncAttributes() != kTfLiteOk) {
    return kTfLiteError;
  }
  if (runner_->PrepareBackends() != kTfLiteOk) {
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus AsyncDriver::SetInputTensorData(const std::string& name,
                                             std::span<const uint8_t> data) {
  auto it = tensor_buffer_ahwb_map_.find({kTfLiteIoTypeInput, name});
  if (it == tensor_buffer_ahwb_map_.end()) {
    return kTfLiteError;
  }

  auto& buffer = it->second;
  void* addr = nullptr;
  if (AHardwareBuffer_lock(buffer.get(),
                           AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                               AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
                           /*fence=*/-1, /*rect=*/nullptr, &addr) != 0) {
    return kTfLiteError;
  }
  memcpy(addr, data.data(), data.size());
  if (AHardwareBuffer_unlock(buffer.get(), /*fence=*/nullptr) != 0) {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

void AsyncDriver::SetInputTensorBuffer(const std::string& name,
                                       ScopedAHardwareBuffer&& buffer) {
  tensor_buffer_ahwb_map_.insert_or_assign({kTfLiteIoTypeInput, name},
                                           std::move(buffer));
}

void AsyncDriver::SetOutputTensorBuffer(const std::string& name,
                                        ScopedAHardwareBuffer&& buffer) {
  tensor_buffer_ahwb_map_.insert_or_assign({kTfLiteIoTypeOutput, name},
                                           std::move(buffer));
}

TfLiteStatus AsyncDriver::Invoke() {
  TfLiteExecutionTask* task = runner_->CreateTask();
  absl::Cleanup finish_task = [&] { runner_->Finish(task); };

  std::vector<TfLiteBufferHandle> registered_handles;
  absl::Cleanup unregister_buffers = [&] {
    for (auto handle : registered_handles) {
      runner_->UnregisterBuffer(handle);
    }
  };

  std::vector<ScopedTfLiteSynchronization> syncs;
  syncs.reserve(tensor_buffer_ahwb_map_.size());
  for (const auto& [key, ahwb] : tensor_buffer_ahwb_map_) {
    const auto& [io_type, name] = key;

    auto buffer = CreateScopedTfLiteBackendBuffer();
    TfLiteBackendBufferSetPtr(buffer.get(), ahwb.get());

    const auto attrs = WriteBufferAttrs({
        .buffer_type = BufferType::kAHardwareBufferBlob,
        .size = *tensor_buffer_attrs_map_.at(key).size,
    });
    TfLiteBufferHandle handle = kTfLiteNullBufferHandle;
    if (runner_->RegisterBuffer(io_type, buffer.get(), attrs.get(), &handle) !=
        kTfLiteOk) {
      return kTfLiteError;
    }
    registered_handles.push_back(handle);

    if (TfLiteExecutionTaskSetBuffer(task, io_type, name.c_str(), handle) !=
        kTfLiteOk) {
      return kTfLiteError;
    }

    syncs.emplace_back(CreateScopedTfLiteSynchronization());
    auto& sync = syncs.back();
    TfLiteSynchronizationSetPtr(sync.get(), nullptr);
    if (TfLiteExecutionTaskSetSync(task, io_type, name.c_str(), sync.get()) !=
        kTfLiteOk) {
      return kTfLiteError;
    }
  }

  if (runner_->InvokeAsync(task) != kTfLiteOk) {
    return kTfLiteError;
  }

  if (runner_->Wait(task) != kTfLiteOk) {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

std::vector<uint8_t> AsyncDriver::GetOutputTensorData(const std::string& name) {
  auto it = tensor_buffer_ahwb_map_.find({kTfLiteIoTypeOutput, name});
  if (it == tensor_buffer_ahwb_map_.end()) {
    return {};
  }

  auto& buffer = it->second;
  void* addr = nullptr;
  if (AHardwareBuffer_lock(buffer.get(),
                           AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                               AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
                           /*fence=*/-1, /*rect=*/nullptr, &addr) != 0) {
    return {};
  }

  size_t size = *tensor_buffer_attrs_map_.at({kTfLiteIoTypeOutput, name}).size;
  std::vector<uint8_t> data(size);
  memcpy(data.data(), addr, data.size());

  if (AHardwareBuffer_unlock(buffer.get(), /*fence=*/nullptr) != 0) {
    return {};
  }

  return data;
}

AsyncDriver::AsyncDriver(TfLiteDelegatePtr delegate,
                         std::unique_ptr<FlatBufferModel> model,
                         std::unique_ptr<Interpreter> interpreter,
                         async::AsyncSignatureRunner* runner)
    : delegate_(std::move(delegate)),
      model_(std::move(model)),
      interpreter_(std::move(interpreter)),
      runner_(runner) {}

TfLiteStatus AsyncDriver::ReconcileBufferAttributes() {
  const char* buffer_type_str = kBufferTypeAHardwareBufferBlob;
  const ScopedTfLiteAttrMap attrs = WriteBufferAttrs({
      .buffer_type = BufferType::kAHardwareBufferBlob,
  });

  for (TfLiteIoType io_type : {kTfLiteIoTypeInput, kTfLiteIoTypeOutput}) {
    const auto& supported = runner_->SupportedBufferTypes(io_type);
    if (!ContainsString(supported, buffer_type_str)) {
      return kTfLiteError;
    }

    auto& names = io_type == kTfLiteIoTypeInput ? runner_->input_names()
                                                : runner_->output_names();
    for (const char* name : names) {
      auto merged = CreateScopedTfLiteAttrMap(kTfLiteAttrMapTypeBuffer);
      if (!runner_->ReconcileRestrictions(io_type, name, attrs.get(),
                                          merged.get(), /*conflict=*/nullptr)) {
        return kTfLiteError;
      }
      if (runner_->SetAttributes(io_type, name, merged.get()) != kTfLiteOk) {
        return kTfLiteError;
      }

      const BufferAttributes attributes = ReadBufferAttrs(merged);
      if (!attributes.size.has_value() ||
          *attributes.size > std::numeric_limits<uint32_t>::max()) {
        return kTfLiteError;
      }
      tensor_buffer_attrs_map_.emplace(
          std::make_pair(io_type, std::string(name)), attributes);
      // TODO(shik): Support other attributes such as alignment.
    }
  }

  return kTfLiteOk;
}

TfLiteStatus AsyncDriver::ReconcileSyncAttributes() {
  // TODO(shik): Support fence fd and make this configurable.
  const char* sync_type_str = kTfLiteSyncTypeNoSyncObj;
  const ScopedTfLiteAttrMap attrs = WriteSyncAttrs({
      .sync_type = SyncType::kNoSyncObj,
  });

  for (TfLiteIoType io_type : {kTfLiteIoTypeInput, kTfLiteIoTypeOutput}) {
    const auto& supported = runner_->SupportedSynchronizations(io_type);
    if (!ContainsString(supported, sync_type_str)) {
      return kTfLiteError;
    }

    const auto& names = io_type == kTfLiteIoTypeInput ? runner_->input_names()
                                                      : runner_->output_names();
    for (const char* name : names) {
      auto merged = CreateScopedTfLiteAttrMap(kTfLiteAttrMapTypeSync);
      if (!runner_->ReconcileRestrictions(io_type, name, attrs.get(),
                                          merged.get(), /*conflict=*/nullptr)) {
        return kTfLiteError;
      }
      if (runner_->SetAttributes(io_type, name, merged.get()) != kTfLiteOk) {
        return kTfLiteError;
      }
    }
  }

  return kTfLiteOk;
}

absl::StatusOr<BufferAttributes> AsyncDriver::GetInputBufferAttributes(
    const std::string& name) {
  auto it = tensor_buffer_attrs_map_.find({kTfLiteIoTypeInput, name});
  if (it == tensor_buffer_attrs_map_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Cannot find input tensor with name = %s", name));
  }
  return it->second;
}

absl::StatusOr<BufferAttributes> AsyncDriver::GetOutputBufferAttributes(
    const std::string& name) {
  auto it = tensor_buffer_attrs_map_.find({kTfLiteIoTypeOutput, name});
  if (it == tensor_buffer_attrs_map_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Cannot find output tensor with name = %s", name));
  }
  return it->second;
}

TfLiteStatus AsyncDriver::AllocateBuffers() {
  for (const auto& [key, attrs] : tensor_buffer_attrs_map_) {
    if (tensor_buffer_ahwb_map_.contains(key)) {
      continue;
    }
    const uint32_t size = *attrs.size;
    ScopedAHardwareBuffer buffer(AHardwareBuffer_Desc{
        .width = size,
        .height = 1,
        .layers = 1,
        .format = AHARDWAREBUFFER_FORMAT_BLOB,
        .usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                 AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
        .stride = size,
    });
    if (!buffer) {
      return kTfLiteError;
    }
    tensor_buffer_ahwb_map_.emplace(key, std::move(buffer));
  }
  return kTfLiteOk;
}

}  // namespace tflite::cros
