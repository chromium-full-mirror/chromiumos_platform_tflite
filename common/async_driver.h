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

#include "absl/status/statusor.h"
#include "android/hardware_buffer.h"
#include "common/scoped_ahwb.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/delegates/utils/async_type_helpers.h"

namespace tflite::cros {

using BufferAttributes = delegates::utils::BufferAttributes;
using TfLiteDelegatePtr = Interpreter::TfLiteDelegatePtr;

// A helper generic map type where the key is an input/output tensor name.
template <typename T>
using IoTensorMap = std::map<std::pair<TfLiteIoType, std::string>, T>;

// A driver to drive a delegate to run the model with async kernel API in a
// synchronous way. The typical flow would be:
// 1. Create an AsyncDriver with the factory function Create().
// 2. Prepare the hardware buffers with Prepare().
// 3. Provide Input/Output buffers via Set{Input,Output}TensorBuffer().
// 4. Allocate buffers that are not provided through AllocateBuffers().
// 5. Set the data for input tensors with SetInputTensorData() if the input data
//    is not already in the buffer.
// 6. Run the model inference with Invoke().
// 7. Retrieve the model output with GetOutputTensorData().
//
// The step 3~7 can be performed multiple times. After the first run,
// AllocateBuffers() is no longer required as every buffers are provided (by the
// caller or by the driver itself). It's also ok to skip AllocateBuffers() if
// the caller have provided every input and output buffer. Furthermore, it's ok
// to skip SetInputTensorData() if the input data is the same as the previous
// run, or to skip GetOutputTensorData() if you don't care about the output.
//
// This class is thread-compatible.
class AsyncDriver {
 public:
  // Factory function. Returns nullptr if there is any error.
  static std::unique_ptr<AsyncDriver> Create(
      TfLiteDelegatePtr delegate,
      std::unique_ptr<FlatBufferModel> model);

  // Move-only.
  AsyncDriver(AsyncDriver&& other) = default;
  AsyncDriver& operator=(AsyncDriver&& other) = default;
  AsyncDriver(const AsyncDriver&) = delete;
  AsyncDriver& operator=(const AsyncDriver&) = delete;

  // Reconciles with the delegate to decide the buffer/sync attributes, and
  // allocates the buffers accordingly.
  TfLiteStatus Prepare();

  // Allocate buffers that are not specified via Set{Input,Output}TensorBuffer.
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

  // Set the input tensor buffer. This will free the existing buffer. The caller
  // of this function should ensure the description of the buffer is correct.
  void SetInputTensorBuffer(const std::string& name,
                            ScopedAHardwareBuffer&& buffer);

  // Set the output tensor buffer. This will free the existing buffer. The
  // caller of this function should ensure the description of the buffer is
  // correct.
  void SetOutputTensorBuffer(const std::string& name,
                             ScopedAHardwareBuffer&& buffer);

  // Get the reconciled input buffer attribute. The buffer provider can
  // configure the buffer accordingly.
  absl::StatusOr<BufferAttributes> GetInputBufferAttributes(
      const std::string& name);

  // Get the reconciled Output buffer attribute. The buffer provider can
  // configure the buffer accordingly.
  absl::StatusOr<BufferAttributes> GetOutputBufferAttributes(
      const std::string& name);

  // Runs model inference and wait until it's finished.
  TfLiteStatus Invoke();

  // Copies the data from the output tensor buffer. Returns an empty vector if
  // there is any error.
  // TODO(shik): Consider using absl::StatusOr to signal error in a less
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
  AsyncDriver(TfLiteDelegatePtr delegate,
              std::unique_ptr<FlatBufferModel> model,
              std::unique_ptr<Interpreter> interpreter,
              async::AsyncSignatureRunner* runner);

  TfLiteStatus ReconcileBufferAttributes();
  TfLiteStatus ReconcileSyncAttributes();

  TfLiteDelegatePtr delegate_;
  std::unique_ptr<FlatBufferModel> model_;
  std::unique_ptr<Interpreter> interpreter_;
  async::AsyncSignatureRunner* runner_;

  // The sizes fro every input/output tensors. Populated in Prepare() ->
  // ReconcileBufferAttributes().
  IoTensorMap<BufferAttributes> tensor_buffer_attrs_map_;

  // The AHardwareBuffer for every input/output tensors. The reference is
  // released in the destructor.
  IoTensorMap<ScopedAHardwareBuffer> tensor_buffer_ahwb_map_;
};

};  // namespace tflite::cros

#endif  // COMMON_ASYNC_DRIVER_H_
