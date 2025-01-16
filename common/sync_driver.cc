/*
 * Copyright 2025 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/sync_driver.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <span>  // NOLINT(build/include_order) - C++20 header is not recognized yet
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"

namespace tflite::cros {

namespace {

constexpr char kSignatureKey[] = "serving_default";

}  // namespace

std::unique_ptr<SyncDriver> SyncDriver::Create(
    TfLiteDelegatePtr delegate,
    std::unique_ptr<FlatBufferModel> model) {
  ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);
  builder.AddDelegate(delegate.get());
  std::unique_ptr<Interpreter> interpreter;
  if (builder(&interpreter) != kTfLiteOk) {
    return nullptr;
  }

  SignatureRunner* runner = interpreter->GetSignatureRunner(kSignatureKey);
  if (runner == nullptr) {
    return nullptr;
  }

  return std::unique_ptr<SyncDriver>(new SyncDriver(
      std::move(delegate), std::move(model), std::move(interpreter), runner));
}

TfLiteStatus SyncDriver::SetInputTensorData(const std::string& name,
                                            std::span<const uint8_t> data) {
  auto it = input_buffer_map_.find(name);
  if (it == input_buffer_map_.end()) {
    return kTfLiteError;
  }
  auto& buffer = it->second;
  memcpy(buffer.get(), data.data(), data.size());
  return kTfLiteOk;
}

void SyncDriver::SetInputTensorBuffer(const std::string& name,
                                      std::shared_ptr<uint8_t[]> buffer) {
  input_buffer_map_.insert_or_assign(name, std::move(buffer));
}

void SyncDriver::SetOutputTensorBuffer(const std::string& name,
                                       std::shared_ptr<uint8_t[]> buffer) {
  output_buffer_map_.insert_or_assign(name, std::move(buffer));
}

TfLiteStatus SyncDriver::Invoke() {
  for (const auto& [name, buffer] : input_buffer_map_) {
    const TfLiteTensor* tensor = runner_->input_tensor(name.c_str());
    if (tensor == nullptr) {
      return kTfLiteError;
    }
    TfLiteCustomAllocation allocation = {.data = buffer.get(),
                                         .bytes = tensor->bytes};
    TfLiteStatus status =
        runner_->SetCustomAllocationForInputTensor(name.c_str(), allocation);
    if (status != kTfLiteOk) {
      return kTfLiteError;
    }
  }
  for (const auto& [name, buffer] : output_buffer_map_) {
    const TfLiteTensor* tensor = runner_->output_tensor(name.c_str());
    if (tensor == nullptr) {
      return kTfLiteError;
    }
    TfLiteCustomAllocation allocation = {.data = buffer.get(),
                                         .bytes = tensor->bytes};
    TfLiteStatus status =
        runner_->SetCustomAllocationForOutputTensor(name.c_str(), allocation);
    if (status != kTfLiteOk) {
      return kTfLiteError;
    }
  }

  return runner_->Invoke();
}

std::vector<uint8_t> SyncDriver::GetOutputTensorData(const std::string& name) {
  const TfLiteTensor* output = runner_->output_tensor(name.c_str());
  if (output == nullptr) {
    return {};
  }

  std::vector<uint8_t> data(output->bytes);
  memcpy(data.data(), output->data.data, output->bytes);

  return data;
}

SyncDriver::SyncDriver(TfLiteDelegatePtr delegate,
                       std::unique_ptr<FlatBufferModel> model,
                       std::unique_ptr<Interpreter> interpreter,
                       tflite::SignatureRunner* runner)
    : delegate_(std::move(delegate)),
      model_(std::move(model)),
      interpreter_(std::move(interpreter)),
      runner_(runner) {}

TfLiteStatus SyncDriver::AllocateBuffers() {
  return runner_->AllocateTensors();
}

}  // namespace tflite::cros
