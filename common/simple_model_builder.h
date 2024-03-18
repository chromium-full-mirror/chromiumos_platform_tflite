/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef COMMON_SIMPLE_MODEL_BUILDER_H_
#define COMMON_SIMPLE_MODEL_BUILDER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/model_builder.h"

namespace tflite::cros {

// A helper class to simplify the model building process so we don't need to
// manipulate raw FlatBuffer types. This class is thread-compatible.
class SimpleModelBuilder {
 public:
  struct TensorArgs {
    // TODO(shik): Generate SignatureDef using the tensor names. It's not
    // supported by the ModelWriter we are using now so we have to do it
    // ourselves.
    std::string name;

    TfLiteType type = kTfLiteNoType;
    std::vector<int> shape;

    // Index of buffer returned by AddBuffer() for constant data.
    // The 0-th buffer is a sentinel empty buffer per TFLite schema, which means
    // no buffer.
    int buffer = 0;

    // TODO(shik): Support quantizations.
  };

  template <typename T>
  struct OperatorArgs {
    TfLiteBuiltinOperator op;

    // Input/Output tensor indices.
    // TODO(shik): Support intermediates tensors. It's not exposed by
    // Interpreter API we have to use Subgraph API directly in implementation.
    std::vector<int> inputs;
    std::vector<int> outputs;

    // The corresponding params for the given operator. For example, it should
    // be TfLiteAddParams for kTfLiteBuiltinAdd. See
    // tensorflow/lite/core/c/builtin_op_data.h for their names.
    T params;
  };

  SimpleModelBuilder();

  // Move-only.
  SimpleModelBuilder(SimpleModelBuilder&& other) = default;
  SimpleModelBuilder& operator=(SimpleModelBuilder&& other) = default;
  SimpleModelBuilder(const SimpleModelBuilder&) = delete;
  SimpleModelBuilder& operator=(const SimpleModelBuilder&) = delete;

  // Adds an input tensor. Returns the tensor index.
  int AddInput(const TensorArgs& args);

  // Adds an output tensor. Returns the tensor index.
  int AddOutput(const TensorArgs& args);

  // Adds an internal tensor. Returns the tensor index.
  int AddInternalTensor(const TensorArgs& args);

  // Adds a buffer with data. Returns the buffer index.
  int AddBuffer(std::vector<uint8_t> data);

  // Adds an operator node in the graph.
  template <typename T>
  void AddOperator(const OperatorArgs<T>& args) {
    // We have to use malloc here since this will be freed by Interpreter.
    void* builtin_data = malloc(sizeof(T));
    *static_cast<T*>(builtin_data) = args.params;
    AddOperatorImpl({
        .op = args.op,
        .inputs = args.inputs,
        .outputs = args.outputs,
        .params = ScopedBuiltinData(builtin_data),
    });
  }

  // Builds the model. This can be called multiple times.
  // TODO(shik): Support saving to a file directly.
  std::unique_ptr<FlatBufferModel> Build();

 private:
  struct FreeDeleter {
    void operator()(void* ptr) { free(ptr); }
  };
  using ScopedBuiltinData = std::unique_ptr<void, FreeDeleter>;

  void AddOperatorImpl(OperatorArgs<ScopedBuiltinData> args);

  int next_tensor_index_ = 0;

  // The added tensors stored as vectors of (tensor_index, tensor_args) pair.
  std::vector<std::pair<int, TensorArgs>> inputs_;
  std::vector<std::pair<int, TensorArgs>> outputs_;
  std::vector<std::pair<int, TensorArgs>> internal_tensors_;

  // The added buffers. The first one will be an empty vector to match TFLite
  // schema.
  std::vector<std::vector<uint8_t>> buffers_;

  // The added operators with params converted to owned void data pointers.
  std::vector<OperatorArgs<ScopedBuiltinData>> operators_;
};

}  // namespace tflite::cros

#endif  // COMMON_SIMPLE_MODEL_BUILDER_H_
