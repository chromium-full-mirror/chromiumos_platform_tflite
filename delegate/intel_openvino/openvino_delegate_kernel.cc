/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/openvino_delegate_kernel.h"

#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#include <openvino/runtime/remote_context.hpp>
#include <sys/mman.h>

#include <cerrno>
#include <vector>

#include "delegate/intel_openvino/log.h"
#include "tensorflow/lite/core/async/c/task.h"
#include "tensorflow/lite/core/async/interop/c/attribute_map.h"
#include "tensorflow/lite/delegates/utils/sync_fence.h"

using tflite::delegates::utils::WaitForAllFds;

namespace tflite {
namespace openvinodelegate {

TfLiteStatus OpenVINOAsyncDelegateKernel::Init(
    TfLiteOpaqueContext *context, const TfLiteOpaqueDelegateParams *params) {
  TfLiteStatus init_status = ov_delegate_core_->Init();
  if (init_status != kTfLiteOk) return init_status;

  TfLiteStatus set_status =
      ov_delegate_core_->CreateModel(context, params, &options_);
  if (set_status != kTfLiteOk) return set_status;

  return kTfLiteOk;
}

TfLiteAsyncKernel *OpenVINOAsyncDelegateKernel::AsyncKernel(
    TfLiteOpaqueContext *context, TfLiteOpaqueNode *node) {
  return async_kernel_->kernel();
}

TfLiteStatus OpenVINOAsyncDelegateKernel::Prepare(TfLiteOpaqueContext *context,
                                                  TfLiteOpaqueNode *node) {
  absl::MutexLock lock(&prep_mutex_);
  TfLiteStatus set_status = ov_delegate_core_->CompileAndInfer();
  if (set_status != kTfLiteOk) return set_status;
  return kTfLiteOk;
}

TfLiteStatus OpenVINOAsyncDelegateKernel::Eval(TfLiteOpaqueContext *context,
                                               TfLiteOpaqueNode *node) {
  std::vector<int> compute_inputs = ov_delegate_core_->getComputeInputs();
  for (int i = 0; i < compute_inputs.size(); i++) {
    int t = compute_inputs[i];
    ov::Tensor inputBlob =
        ov_delegate_core_->getInferRequest().get_input_tensor(i);
    void *dest = inputBlob.data();

    const TfLiteOpaqueTensor *opaque_input_tensor =
        TfLiteOpaqueContextGetOpaqueTensor(context, t);
    auto len = TfLiteOpaqueTensorByteSize(opaque_input_tensor);
    void *src = TfLiteOpaqueTensorData(opaque_input_tensor);

    std::memcpy(dest, src, len);
  }

  ov_delegate_core_->getInferRequest().start_async();
  if (!ov_delegate_core_->getInferRequest().wait_for(
          std::chrono::milliseconds(kInferRequestTimeout))) {
    TFLITE_LOG(ERROR) << "Infer request failed";
    return kTfLiteError;
  }

  std::vector<int> outputs = ov_delegate_core_->getOutputs();
  for (int o = 0; o < outputs.size(); o++) {
    int t = outputs[o];
    ov::Tensor outputBlob =
        ov_delegate_core_->getInferRequest().get_output_tensor(o);
    const TfLiteOpaqueTensor *opaque_output_tensor =
        TfLiteOpaqueContextGetOpaqueTensor(context, t);
    void *dest = TfLiteOpaqueTensorData(opaque_output_tensor);
    void *src = outputBlob.data();
    auto len = TfLiteOpaqueTensorByteSize(opaque_output_tensor);
    std::memcpy(dest, src, len);
  }

  return kTfLiteOk;
}

TfLiteStatus OpenVINOAsyncDelegateKernel::RegisterBuffer(
    TfLiteBufferHandle handle, AHardwareBuffer *ptr, size_t buffer_size) {
  size_t numElements = (buffer_size / 4);
  auto context = ov_delegate_core_->get_context()
                     .as<ov::intel_npu::level_zero::ZeroContext>();
  AHardwareBuffer_acquire(ptr);
  if (ptr == NULL) return kTfLiteError;

  const native_handle_t *buffer_handle = AHardwareBuffer_getNativeHandle(ptr);
  if (buffer_handle == nullptr) return kTfLiteError;
  if (buffer_handle->numFds != 1) return kTfLiteError;
  int fd = buffer_handle->data[0];
  if (fd == -1) return kTfLiteError;
  auto mmap_ret =
      mmap(NULL, buffer_size, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
  if (mmap_ret == MAP_FAILED) {
    return kTfLiteError;
  }
  mmap_ret_map_.emplace(handle, mmap_ret);
  buffer_size_map_.emplace(handle, buffer_size);
  auto remote_tensor =
      context.create_tensor(ov::element::f32, ov::Shape{1, numElements}, fd);
  buffer_map_.emplace(handle, remote_tensor);
  ahwb_buffer_map_.emplace(handle, ptr);
  return kTfLiteOk;
}

TfLiteStatus OpenVINOAsyncDelegateKernel::UnregisterBuffer(
    TfLiteBufferHandle handle) {
  auto tensor = buffer_map_.find(handle);
  if (tensor == buffer_map_.end()) return kTfLiteError;
  auto it = ahwb_buffer_map_.find(handle);
  if (it == ahwb_buffer_map_.end()) return kTfLiteError;
  auto mmap_it = mmap_ret_map_.find(handle);
  if (mmap_it == mmap_ret_map_.end()) return kTfLiteError;
  auto buffer_size_it = buffer_size_map_.find(handle);
  if (buffer_size_it == buffer_size_map_.end()) return kTfLiteError;

  munmap(mmap_it->second, buffer_size_it->second);
  tensor->second = {};
  AHardwareBuffer_release(it->second);
  ahwb_buffer_map_.erase(handle);
  buffer_map_.erase(handle);
  mmap_ret_map_.erase(handle);
  buffer_size_map_.erase(handle);
  return kTfLiteOk;
}

TfLiteStatus OpenVINOAsyncDelegateKernel::EvalAsyncImpl(
    TfLiteOpaqueContext *context, TfLiteOpaqueNode *node,
    TfLiteExecutionTask *task) {
  absl::MutexLock lock(&eval_mutex_);
  std::vector<int> compute_inputs = ov_delegate_core_->getComputeInputs();
  for (int i = 0; i < compute_inputs.size(); i++) {
    TfLiteSynchronization *sync_ptr =
        TfLiteExecutionTaskGetSyncByIndex(task, compute_inputs[i]);
    if (sync_ptr == nullptr) continue;
    auto sync_obj = TfLiteSynchronizationGetPtr(sync_ptr);
    if (sync_obj == nullptr) continue;

    input_sync_fence_fds_.push_back(*(reinterpret_cast<int *>(sync_obj)));
  }

  auto wait_resp = WaitForAllFds(input_sync_fence_fds_);
  if (!wait_resp.has_value()) return kTfLiteError;

  for (int i = 0; i < compute_inputs.size(); i++) {
    TfLiteBufferHandle buffer_handle =
        TfLiteExecutionTaskGetBufferByIndex(task, compute_inputs[i]);
    const TfLiteOpaqueTensor *tfl_tensor =
        TfLiteOpaqueNodeGetInput(context, node, i);
    int32_t num_dims = TfLiteOpaqueTensorNumDims(tfl_tensor);
    std::vector<int> dims(num_dims);
    for (int j = 0; j < num_dims; j++) {
      dims[j] = TfLiteOpaqueTensorDim(tfl_tensor, j);
    }
    ov::Tensor remote_tensor = buffer_map_.at(buffer_handle);
    remote_tensor.set_shape(ov::Shape(dims.begin(), dims.end()));
    ov_delegate_core_->getInferRequest().set_input_tensor(compute_inputs[i],
                                                          remote_tensor);
  }

  std::vector<int> outputs = ov_delegate_core_->getOutputs();
  for (int o = 0; o < outputs.size(); o++) {
    TfLiteBufferHandle buffer_handle =
        TfLiteExecutionTaskGetBufferByIndex(task, outputs[o]);
    const TfLiteOpaqueTensor *tfl_tensor =
        TfLiteOpaqueNodeGetOutput(context, node, o);
    int32_t num_dims = TfLiteOpaqueTensorNumDims(tfl_tensor);
    std::vector<int> dims(num_dims);
    for (int j = 0; j < num_dims; j++) {
      dims[j] = TfLiteOpaqueTensorDim(tfl_tensor, j);
    }

    ov::Tensor remote_tensor = buffer_map_.at(buffer_handle);
    remote_tensor.set_shape(ov::Shape(dims.begin(), dims.end()));
    ov_delegate_core_->getInferRequest().set_output_tensor(o, remote_tensor);
  }

  ov_delegate_core_->getInferRequest().start_async();
  if (!ov_delegate_core_->getInferRequest().wait_for(
          std::chrono::milliseconds(kInferRequestTimeout))) {
    TFLITE_LOG(ERROR) << "Infer request failed";
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
