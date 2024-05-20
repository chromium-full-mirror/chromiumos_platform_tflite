/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorflow/lite/core/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/core/acceleration/configuration/c/stable_delegate.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/stable_delegate_interface.h"

namespace {

// TODO(b/336951015): Replace with real implementation.

TfLiteOpaqueDelegate* OpenVINOStableDelegateCreateFunc(
    const void* tflite_settings) {
  return nullptr;
}

void OpenVINOStableDelegateDestroyFunc(
    TfLiteOpaqueDelegate* openvino_stable_delegate) {
  return;
}

int OpenVINOStableDelegateErrnoFunc(
    TfLiteOpaqueDelegate* openvino_stable_delegate) {
  return 0;
}

constexpr TfLiteOpaqueDelegatePlugin openvino_stable_delegate_plugin = {
    .create = OpenVINOStableDelegateCreateFunc,
    .destroy = OpenVINOStableDelegateDestroyFunc,
    .get_delegate_errno = OpenVINOStableDelegateErrnoFunc,
};

constexpr TfLiteStableDelegate openvino_stable_delegate = {
    .delegate_abi_version = TFL_STABLE_DELEGATE_ABI_VERSION,
    .delegate_name = "Stub OpenVINO Delegate",
    .delegate_version = "0.0.0",
    .delegate_plugin = &openvino_stable_delegate_plugin,
};

}  // namespace

extern "C" const TfLiteStableDelegate TFL_TheStableDelegate =
    openvino_stable_delegate;
