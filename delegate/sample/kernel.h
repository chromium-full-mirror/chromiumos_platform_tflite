/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef DELEGATE_SAMPLE_KERNEL_H_
#define DELEGATE_SAMPLE_KERNEL_H_

// TODO(shik): Note that the upstream sample delegate use flat_hash_{map,set}
// from absl/container. We haven't decided whether we would like to allow
// absl/libchrome usage here.
#include <map>
#include <vector>

#include "common/simple_async_delegate.h"
#include "tensorflow/lite/core/c/c_api_opaque.h"

namespace tflite::cros {

// A simple delegate kernel that implement the computation of addition and
// subtraction operations.
class CrosSampleDelegateKernel : public SimpleAsyncDelegateKernelInterface {
 public:
  // Initializes the kernel by extracting necessary information into
  // `node_infos_` and deriving internal tensors.
  TfLiteStatus Init(TfLiteOpaqueContext* context,
                    const TfLiteOpaqueDelegateParams* params) override;

  // Prepares the memory for internal tensors. Note that this will be called
  // everytime tensors are resized.
  TfLiteStatus Prepare(TfLiteOpaqueContext* context,
                       TfLiteOpaqueNode* node) override;

  // Computes the inference result. Note that the data pointer of external
  // tensors may change for every inference.
  TfLiteStatus Eval(TfLiteOpaqueContext* context,
                    TfLiteOpaqueNode* node) override;

  // Retrieves the async kernel.
  // Returns nullptr for now, since the delegate does not support asynchronous
  // execution yet.
  virtual TfLiteAsyncKernel* AsyncKernel(TfLiteOpaqueContext* context,
                                         TfLiteOpaqueNode* node) override;

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
  std::map<const TfLiteOpaqueTensor*, std::vector<float>>
      internal_tensors_memory_;
};

}  // namespace tflite::cros

#endif  // DELEGATE_SAMPLE_KERNEL_H_
