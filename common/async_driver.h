/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef COMMON_ASYNC_DRIVER_H_
#define COMMON_ASYNC_DRIVER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <span>  // NOLINT(build/include_order) - C++20 header is not recognized yet
#include <string>
#include <utility>
#include <vector>

#include "common/android_hardware_buffer.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/model_builder.h"

namespace tflite::cros {

using TfLiteDelegatePtr = Interpreter::TfLiteDelegatePtr;

// A helper generic map type where the key is an input/output tensor name.
template <typename T>
using IoTensorMap = std::map<std::pair<TfLiteIoType, std::string>, T>;

// A driver to drive a delegate to run the model with async kernel API in a
// synchronous way. The typical flow would be:
// 1. Create an AsyncDriver with the factory function Create().
// 2. Prepare the hardware buffers with Prepare().
// 3. Set the data for input tensors with SetInputTensor().
// 4. Run the model inference with Invoke().
// 5. Retrive the model output with GetOutputTensor().
//
// The step 3~5 can be performed multiple times. It's ok to skip
// SetInputTensor() if the input data is same as the previous run, or to skip
// GetOutputTensor() if you don't care about the output.
//
// This class is thread-compatible.
class AsyncDriver {
 public:
  // Factory function. Returns nullptr if there is any error.
  static std::unique_ptr<AsyncDriver> Create(
      TfLiteDelegatePtr delegate,
      std::unique_ptr<FlatBufferModel> model);

  ~AsyncDriver();

  // Move-only.
  AsyncDriver(AsyncDriver&& other) = default;
  AsyncDriver& operator=(AsyncDriver&& other) = default;
  AsyncDriver(const AsyncDriver&) = delete;
  AsyncDriver& operator=(const AsyncDriver&) = delete;

  // Reconciles with the delegate to decide the buffer/sync attributes, and
  // allocates the buffers accordingly.
  TfLiteStatus Prepare();

  // Copies the provided data to the input tensor buffer.
  // TODO(shik): Add an API to allow providing AHardwareBuffer directly.
  TfLiteStatus SetInputTensor(const std::string& name,
                              std::span<const uint8_t> data);

  template <typename T>
  TfLiteStatus SetInputTensor(const std::string& name,
                              const std::vector<T>& data) {
    auto begin = reinterpret_cast<const uint8_t*>(data.data());
    auto end = reinterpret_cast<const uint8_t*>(data.data() + data.size());
    return SetInputTensor(name, std::span(begin, end));
  }

  // Runs model inference and wait until it's finished.
  TfLiteStatus Invoke();

  // Copies the data from the output tensor buffer. Returns an empty vector if
  // there is any error.
  // TODO(shik): Consider using absl::StatusOr to signal error in a less
  // error-prone way.
  std::vector<uint8_t> GetOutputTensor(const std::string& name);

  template <typename T>
  std::vector<T> GetOutputTensor(const std::string& name) {
    auto raw_data = GetOutputTensor(name);
    std::vector<T> data(raw_data.size() / sizeof(T));
    memcpy(data.data(), raw_data.data(), raw_data.size());
    return data;
  }

 private:
  // The private constructor used in the factory function.
  AsyncDriver(TfLiteDelegatePtr delegate,
              std::unique_ptr<FlatBufferModel> model,
              std::unique_ptr<Interpreter> interpreter,
              async::AsyncSignatureRunner* runner);

  TfLiteStatus ReconcileBufferAttributes();
  TfLiteStatus ReconcileSyncAttributes();
  TfLiteStatus AllocateBuffers();

  TfLiteDelegatePtr delegate_;
  std::unique_ptr<FlatBufferModel> model_;
  std::unique_ptr<Interpreter> interpreter_;
  async::AsyncSignatureRunner* runner_;

  // The sizes fro every input/output tensors. Populated in Prepare() ->
  // ReconcileBufferAttributes().
  IoTensorMap<size_t> tensor_buffer_size_map_;

  // TODO(shik): Create a ScopedAHardwareBuffer type to simplify buffer
  // management.
  // The AHardwareBuffer for every input/output tensors. The reference is
  // released in the destructor.
  IoTensorMap<AHardwareBuffer*> tensor_buffer_ahwb_map_;
};

};  // namespace tflite::cros

#endif  // COMMON_ASYNC_DRIVER_H_
