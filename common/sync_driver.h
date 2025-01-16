/*
 * Copyright 2025 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef COMMON_SYNC_DRIVER_H_
#define COMMON_SYNC_DRIVER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <span>  // NOLINT(build/include_order) - C++20 header is not recognized yet
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/signature_runner.h"

namespace tflite::cros {

using TfLiteDelegatePtr = Interpreter::TfLiteDelegatePtr;

// A driver to drive a delegate to run the model in a synchronous way.
// The typical flow would be:
// 1. Create an SyncDriver with the factory function Create().
// 2. Provide Input/Output buffers via Set{Input,Output}TensorBuffer(). Note
//    that the buffer should be aligned with tflite::kDefaultTensorAlignment.
// 3. Allocate buffers that are not provided through AllocateBuffers().
// 4. Set the data for input tensors with SetInputTensorData() if the input data
//    is not already in the buffer.
// 5. Run the model inference with Invoke().
// 6. Retrieve the model output with GetOutputTensorData().
//
// The step 2~6 can be performed multiple times.
//
// This class is *not* thread-safe.
class SyncDriver {
 public:
  // Factory function. Returns nullptr if there is any error.
  static std::unique_ptr<SyncDriver> Create(
      TfLiteDelegatePtr delegate,
      std::unique_ptr<FlatBufferModel> model);

  // Move-only.
  SyncDriver(SyncDriver&& other) = default;
  SyncDriver& operator=(SyncDriver&& other) = default;
  SyncDriver(const SyncDriver&) = delete;
  SyncDriver& operator=(const SyncDriver&) = delete;

  // Updates allocations for all tensors.
  TfLiteStatus AllocateBuffers();

  // Copies the provided data to the input tensor buffer.
  TfLiteStatus SetInputTensorData(const std::string& name,
                                  std::span<const uint8_t> data);

  template <typename T>
  TfLiteStatus SetInputTensorData(const std::string& name,
                                  const std::vector<T>& data) {
    auto begin = reinterpret_cast<const uint8_t*>(data.data());
    auto end = reinterpret_cast<const uint8_t*>(data.data() + data.size());
    return SetInputTensorData(name, std::span(begin, end));
  }

  // Set the input tensor buffer. The caller of this function should ensure the
  // description, i.e. name and size, of the buffer is correct.
  void SetInputTensorBuffer(const std::string& name,
                            std::shared_ptr<uint8_t[]> buffer);

  // Set the output tensor buffer. The caller of this function should ensure the
  // description of the buffer, i.e. name and size, is correct.
  void SetOutputTensorBuffer(const std::string& name,
                             std::shared_ptr<uint8_t[]> buffer);

  // Runs model inference and wait until it's finished.
  TfLiteStatus Invoke();

  // Copies the data from the output tensor buffer. Returns an empty vector if
  // there is any error.
  // TODO(ototot): Consider using absl::StatusOr to signal error in a less
  // error-prone way.
  std::vector<uint8_t> GetOutputTensorData(const std::string& name);

  template <typename T>
  std::vector<T> GetOutputTensorData(const std::string& name) {
    auto raw_data = GetOutputTensorData(name);
    std::vector<T> data(raw_data.size() / sizeof(T));
    memcpy(data.data(), raw_data.data(), raw_data.size());
    return data;
  }

 private:
  // The private constructor used in the factory function.
  SyncDriver(TfLiteDelegatePtr delegate,
             std::unique_ptr<FlatBufferModel> model,
             std::unique_ptr<Interpreter> interpreter,
             tflite::SignatureRunner* runner);

  TfLiteDelegatePtr delegate_;
  std::unique_ptr<FlatBufferModel> model_;
  std::unique_ptr<Interpreter> interpreter_;
  tflite::SignatureRunner* runner_;

  // The buffer for every input/output tensors.
  std::map<std::string, std::shared_ptr<uint8_t[]>> input_buffer_map_;
  std::map<std::string, std::shared_ptr<uint8_t[]>> output_buffer_map_;
};

};  // namespace tflite::cros

#endif  // COMMON_SYNC_DRIVER_H_
