/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef DELEGATE_SAMPLE_CORE_H_
#define DELEGATE_SAMPLE_CORE_H_

// TODO(shik): Note that the upstream sample delegate use flat_hash_{map,set}
// from absl/container. We haven't decided whether we would like to allow
// absl/libchrome usage here.
#include <map>
#include <vector>

#include "tensorflow/lite/core/c/c_api_opaque.h"

namespace tflite::cros {

// The core of delegate kernel that implement the computation of addition and
// subtraction operations.
// The typical flow would be:
// 1. Initialize the core with Init().
// 2. Prepare the memory for internal tensors with Prepare().
// 3. Provide the memory for external tensors with SetExternalTensorMemory().
// 4. Run the model inference with Eval().
//
// The steps 3 and 4 can be performed multiple times. It's ok to skip step 3 if
// the backing memories are the same as the previous run.
//
// This class is thread-compatible.
class CrosSampleDelegateCore {
 public:
  // Move-only.
  CrosSampleDelegateCore() = default;
  CrosSampleDelegateCore(CrosSampleDelegateCore&& other) = default;
  CrosSampleDelegateCore& operator=(CrosSampleDelegateCore&& other) = default;
  CrosSampleDelegateCore(const CrosSampleDelegateCore&) = delete;
  CrosSampleDelegateCore& operator=(const CrosSampleDelegateCore&) = delete;

  // Initializes the kernel by extracting necessary information into
  // `node_infos_` and deriving internal tensors.
  TfLiteStatus Init(TfLiteOpaqueContext* context,
                    const TfLiteOpaqueDelegateParams* params);

  // Prepares the memory for internal tensors. Note that this need be called
  // everytime tensors are resized so the memory could grow/shrink accordingly.
  TfLiteStatus Prepare();

  // Sets the memory pointer for an external tensor. The memory must be
  // cpu-accessible and synchronized.
  // TODO(shik): Expose an API for fine-grained synchronizations.
  void SetExternalTensorMemory(const TfLiteOpaqueTensor* tensor, void* memory);

  // Computes the inference result. Note that the data pointer of external
  // tensors may change for every inference.
  TfLiteStatus Eval();

 private:
  struct NodeInfo {
    TfLiteBuiltinOperator op;
    const TfLiteOpaqueTensor* input1;
    const TfLiteOpaqueTensor* input2;
    const TfLiteOpaqueTensor* output;
  };

  float* GetRawDataSource(const TfLiteOpaqueTensor* tensor);

  std::vector<NodeInfo> node_infos_;
  std::vector<const TfLiteOpaqueTensor*> internal_tensors_;

  // Owned memory for internal tensors.
  std::map<const TfLiteOpaqueTensor*, std::vector<float>>
      internal_tensors_memory_;

  // Non-owned memory for external tensors provided by
  // SetExternalTensorMemory().
  std::map<const TfLiteOpaqueTensor*, void*> external_tensors_memory_;
};

}  // namespace tflite::cros

#endif  // DELEGATE_SAMPLE_CORE_H_
