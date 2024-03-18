/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/simple_model_builder.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/tools/serialization/writer_lib.h"

namespace tflite::cros {

namespace {

class OwnedMemoryAllocation : public Allocation {
 public:
  OwnedMemoryAllocation(std::unique_ptr<uint8_t[]> data, size_t size)
      : Allocation(DefaultErrorReporter(), tflite::Allocation::Type::kMemory),
        data_(std::move(data)),
        size_(size) {}

  ~OwnedMemoryAllocation() override = default;
  const void* base() const override { return data_.get(); }
  size_t bytes() const override { return size_; }
  bool valid() const override { return true; }

 private:
  std::unique_ptr<uint8_t[]> data_;
  size_t size_;
};

}  // namespace

SimpleModelBuilder::SimpleModelBuilder() {
  buffers_.push_back({});
}

int SimpleModelBuilder::AddInput(const TensorArgs& args) {
  int idx = next_tensor_index_++;
  inputs_.push_back({idx, args});
  return idx;
}

int SimpleModelBuilder::AddOutput(const TensorArgs& args) {
  int idx = next_tensor_index_++;
  outputs_.push_back({idx, args});
  return idx;
}

int SimpleModelBuilder::AddInternalTensor(const TensorArgs& args) {
  int idx = next_tensor_index_++;
  internal_tensors_.push_back({idx, args});
  return idx;
}

int SimpleModelBuilder::AddBuffer(std::vector<uint8_t> data) {
  int idx = buffers_.size();
  buffers_.push_back(std::move(data));
  return idx;
}

std::unique_ptr<FlatBufferModel> SimpleModelBuilder::Build() {
  Interpreter interpreter;

  // Add tensors.
  interpreter.AddTensors(next_tensor_index_);

  std::vector<int> inputs;
  std::transform(inputs_.begin(), inputs_.end(), std::back_inserter(inputs),
                 [](auto x) { return x.first; });
  interpreter.SetInputs(inputs);

  std::vector<int> outputs;
  std::transform(outputs_.begin(), outputs_.end(), std::back_inserter(outputs),
                 [](auto x) { return x.first; });
  interpreter.SetOutputs(outputs);

  // Set tensor parameters.
  // Note that Initializer list are copy-initialized, so we have to use pointer
  // here to eliminate that copy and ensure the lifetime of tensor.name.c_str()
  // outlives the interpreter.
  for (const auto* tensors : {&inputs_, &outputs_, &internal_tensors_}) {
    for (const auto& [index, tensor] : *tensors) {
      if (tensor.buffer == 0) {
        if (interpreter.SetTensorParametersReadWrite(
                index, tensor.type, tensor.name.c_str(), tensor.shape,
                TfLiteQuantization()) != kTfLiteOk) {
          return nullptr;
        }
      } else {
        const std::vector<uint8_t>& buffer = buffers_[tensor.buffer];
        if (interpreter.SetTensorParametersReadOnly(
                index, tensor.type, tensor.name.c_str(), tensor.shape,
                TfLiteQuantization(),
                reinterpret_cast<const char*>(buffer.data()),
                buffer.size()) != kTfLiteOk) {
          return nullptr;
        }
      }
    }
  }

  // Add operators.
  ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  for (auto& op : operators_) {
    const TfLiteRegistration* reg =
        resolver.FindOp(static_cast<BuiltinOperator>(op.op), /*version=*/1);
    if (reg == nullptr) {
      return nullptr;
    }
    if (interpreter.AddNodeWithParameters(op.inputs, op.outputs,
                                          /*init_data=*/nullptr,
                                          /*init_data_size=*/0, op.params.get(),
                                          reg) != kTfLiteOk) {
      return nullptr;
    }
    // If AddNodeWithParameters() succeed, the pointer ownership is taken by
    // interpreter.
    (void)op.params.release();
  }

  std::unique_ptr<uint8_t[]> buffer;
  size_t size = 0;
  ModelWriter(&interpreter).GetBuffer(&buffer, &size);

  auto allocation =
      std::make_unique<OwnedMemoryAllocation>(std::move(buffer), size);
  return FlatBufferModel::VerifyAndBuildFromAllocation(std::move(allocation));
}

void SimpleModelBuilder::AddOperatorImpl(OperatorArgs<ScopedBuiltinData> args) {
  operators_.push_back(std::move(args));
}

}  // namespace tflite::cros
