/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/simple_model_builder.h"

#include <algorithm>
#include <fstream>
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

std::unique_ptr<FlatBufferModel> FixupSignatureDef(
    std::unique_ptr<FlatBufferModel> fb_model) {
  std::unique_ptr<ModelT> model(fb_model->GetModel()->UnPack());

  auto def = std::make_unique<SignatureDefT>();
  def->subgraph_index = 0;
  def->signature_key = SimpleModelBuilder::kSignatureKey;

  auto& graph = model->subgraphs[0];
  for (int i : graph->inputs) {
    auto map = std::make_unique<TensorMapT>();
    map->name = graph->tensors[i]->name;
    map->tensor_index = i;
    def->inputs.push_back(std::move(map));
  }
  for (int i : graph->outputs) {
    auto map = std::make_unique<TensorMapT>();
    map->name = graph->tensors[i]->name;
    map->tensor_index = i;
    def->outputs.push_back(std::move(map));
  }

  model->signature_defs.push_back(std::move(def));

  flatbuffers::FlatBufferBuilder fbb;
  flatbuffers::Offset<Model> packed_model = Model::Pack(fbb, model.get());
  FinishModelBuffer(fbb, packed_model);
  auto data = std::make_unique<uint8_t[]>(fbb.GetSize());
  memcpy(data.get(), fbb.GetBufferPointer(), fbb.GetSize());
  auto allocation =
      std::make_unique<OwnedMemoryAllocation>(std::move(data), fbb.GetSize());
  return FlatBufferModel::VerifyAndBuildFromAllocation(std::move(allocation));
}

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

std::unique_ptr<FlatBufferModel> SimpleModelBuilder::Build() const {
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
    // We have to use malloc here since this will be freed by Interpreter.
    void* builtin_data = malloc(op.params.size());
    memcpy(builtin_data, op.params.data(), op.params.size());
    if (interpreter.AddNodeWithParameters(op.inputs, op.outputs,
                                          /*init_data=*/nullptr,
                                          /*init_data_size=*/0, builtin_data,
                                          reg) != kTfLiteOk) {
      free(builtin_data);
      return nullptr;
    }
    // If AddNodeWithParameters() succeed, the ownership of builtin_data is
    // taken by the interpreter, so we don't need to free it here.
  }

  std::unique_ptr<uint8_t[]> buffer;
  size_t size = 0;
  ModelWriter(&interpreter).GetBuffer(&buffer, &size);

  auto allocation =
      std::make_unique<OwnedMemoryAllocation>(std::move(buffer), size);
  std::unique_ptr<FlatBufferModel> model =
      FlatBufferModel::VerifyAndBuildFromAllocation(std::move(allocation));
  if (model == nullptr) {
    return nullptr;
  }

  return FixupSignatureDef(std::move(model));
}

void SimpleModelBuilder::SaveTo(const std::string& path) const {
  std::unique_ptr<FlatBufferModel> model = Build();
  const Allocation* allocation = model->allocation();

  std::ofstream output_file(path);
  output_file.write(reinterpret_cast<const char*>(allocation->base()),
                    allocation->bytes());
}

}  // namespace tflite::cros
