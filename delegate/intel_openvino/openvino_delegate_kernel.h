/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DELEGATE_INTEL_OPENVINO_OPENVINO_DELEGATE_KERNEL_H_
#define DELEGATE_INTEL_OPENVINO_OPENVINO_DELEGATE_KERNEL_H_

#include <openvino/openvino.hpp>

#include <map>
#include <memory>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "common/simple_async_delegate.h"
#include "delegate/intel_openvino/openvino_async_kernel.h"
#include "delegate/intel_openvino/openvino_delegate.h"
#include "delegate/intel_openvino/openvino_delegate_core.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"

namespace tflite {
namespace openvinodelegate {

class OVDelegateAsyncKernel;

class OpenVINOAsyncDelegateKernel
    : public cros::SimpleAsyncDelegateKernelInterface {
 public:
  explicit OpenVINOAsyncDelegateKernel(TfLiteOpenVINODelegateOptions options)
      : ov_delegate_core_(std::make_unique<OpenVINODelegateCore>(
            "/etc/openvino/plugins.xml")),
        async_kernel_(std::make_unique<OVDelegateAsyncKernel>(this)) {
    options_ = options;
  }

  TfLiteStatus Init(TfLiteOpaqueContext *context,
                    const TfLiteOpaqueDelegateParams *params) override;

  TfLiteStatus Prepare(TfLiteOpaqueContext *context,
                       TfLiteOpaqueNode *node) override;

  TfLiteStatus Eval(TfLiteOpaqueContext *context,
                    TfLiteOpaqueNode *node) override;

  TfLiteStatus EvalAsyncImpl(TfLiteOpaqueContext *context,
                             TfLiteOpaqueNode *node, TfLiteExecutionTask *task);

  TfLiteStatus RegisterBuffer(TfLiteBufferHandle handle, AHardwareBuffer *ptr,
                              size_t buffer_size);
  TfLiteStatus UnregisterBuffer(TfLiteBufferHandle handle);
  TfLiteAsyncKernel *AsyncKernel(TfLiteOpaqueContext *context,
                                 TfLiteOpaqueNode *node) override;

 private:
  static constexpr int kInferRequestTimeout = 10000;
  std::unique_ptr<OpenVINODelegateCore> ov_delegate_core_;
  std::unique_ptr<OVDelegateAsyncKernel> async_kernel_;
  TfLiteOpenVINODelegateOptions options_;
  std::vector<int> input_sync_fence_fds_ ABSL_GUARDED_BY(eval_mutex_);
  mutable absl::Mutex eval_mutex_;
  mutable absl::Mutex prep_mutex_;
  std::map<TfLiteBufferHandle, ov::RemoteTensor> buffer_map_;
  std::map<TfLiteBufferHandle, AHardwareBuffer *> ahwb_buffer_map_;
  std::map<TfLiteBufferHandle, void *> mmap_ret_map_;
  std::map<TfLiteBufferHandle, size_t> buffer_size_map_;
};

}  // namespace openvinodelegate
}  // namespace tflite

#endif  // DELEGATE_INTEL_OPENVINO_OPENVINO_DELEGATE_KERNEL_H_
