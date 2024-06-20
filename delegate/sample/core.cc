/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "delegate/sample/core.h"

#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "absl/time/clock.h"
#include "common/log.h"
#include "tensorflow/lite/core/c/c_api_opaque.h"

namespace tflite::cros {

namespace {

float AddImpl(float a, float b) {
  return a + b;
}

float SubImpl(float a, float b) {
  return a - b;
}

int CalculateNumElements(const TfLiteOpaqueTensor* tensor) {
  int num_elements = 1;
  int num_dims = TfLiteOpaqueTensorNumDims(tensor);
  for (int i = 0; i < num_dims; ++i) {
    num_elements *= TfLiteOpaqueTensorDim(tensor, i);
  }
  return num_elements;
}

}  // namespace

TfLiteStatus CrosSampleDelegateCore::Init(
    TfLiteOpaqueContext* context,
    const TfLiteOpaqueDelegateParams* params) {
  int num_nodes = params->nodes_to_replace->size;
  node_infos_.reserve(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    TfLiteOpaqueNode* node = nullptr;
    TfLiteRegistrationExternal* registration = nullptr;
    int node_index = params->nodes_to_replace->data[i];
    TfLiteOpaqueContextGetNodeAndRegistration(context, node_index, &node,
                                              &registration);

    NodeInfo info = {
        .op = TfLiteOperatorGetBuiltInCode(registration),
        .input1 = TfLiteOpaqueNodeGetInput(context, node, 0),
        .input2 = TfLiteOpaqueNodeGetInput(context, node, 1),
        .output = TfLiteOpaqueNodeGetOutput(context, node, 0),
    };
    node_infos_.push_back(info);
  }

  using TensorSet = std::set<const TfLiteOpaqueTensor*>;
  TensorSet all_inputs;
  TensorSet all_outputs;
  for (const auto& info : node_infos_) {
    all_inputs.insert(info.input1);
    all_inputs.insert(info.input2);
    all_outputs.insert(info.output);
  }
  // If the input of some node is an output of some node in the same delegated
  // subgraph, it's an internal tensor for us.
  std::set_intersection(all_inputs.begin(), all_inputs.end(),
                        all_outputs.begin(), all_outputs.end(),
                        std::back_inserter(internal_tensors_));

  return kTfLiteOk;
}

TfLiteStatus CrosSampleDelegateCore::Prepare() {
  // Allocate memory for internal tensors. For external tensors, the memory will
  // be provided with SetExternalTensorMemory().
  for (const auto& tensor : internal_tensors_) {
    int size = CalculateNumElements(tensor);
    internal_tensors_memory_[tensor].resize(size);
  }

  return kTfLiteOk;
}

float* CrosSampleDelegateCore::GetRawDataSource(
    const TfLiteOpaqueTensor* tensor) {
  if (auto it = internal_tensors_memory_.find(tensor);
      it != internal_tensors_memory_.end()) {
    return it->second.data();
  }

  if (auto it = external_tensors_memory_.find(tensor);
      it != external_tensors_memory_.end()) {
    return static_cast<float*>(it->second);
  }

  // Fall back to use data pointer inside the tensor. Normally this should not
  // happen.
  return reinterpret_cast<float*>(TfLiteOpaqueTensorData(tensor));
}

void CrosSampleDelegateCore::SetExternalTensorMemory(
    const TfLiteOpaqueTensor* tensor,
    void* memory) {
  external_tensors_memory_.insert_or_assign(tensor, memory);
}

TfLiteStatus CrosSampleDelegateCore::Eval() {
  // TODO(shik): Build a common tracing utility that can measure elapsed time
  // and integrate with Perfetto.
  absl::Time start = absl::Now();
  for (const auto& info : node_infos_) {
    float* input1 = GetRawDataSource(info.input1);
    float* input2 = GetRawDataSource(info.input2);
    float* output = GetRawDataSource(info.output);

    // The input/output tensors have the same size.
    int size = CalculateNumElements(info.output);
    std::transform(input1, input1 + size, input2, output,
                   info.op == kTfLiteBuiltinAdd ? AddImpl : SubImpl);
  }
  absl::Duration elapsed = absl::Now() - start;
  VLOGF(1) << "Finished in " << elapsed;
  return kTfLiteOk;
}

}  // namespace tflite::cros
