/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <utility>

#include "delegate/intel_openvino/log.h"
#include "delegate/intel_openvino/openvino_delegate.h"
#include "tensorflow/lite/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/acceleration/configuration/c/stable_delegate.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/stable_delegate_interface.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"

namespace {

TfLiteOpaqueDelegate *OpenVINOStableDelegateCreateFunc(
    const void *tflite_settings) {
  TfLiteOpenVINODelegateOptions options;

  auto *settings =
      static_cast<const ::tflite::TFLiteSettings *>(tflite_settings);
  if (const auto *caching_settings = settings->compilation_caching_settings();
      caching_settings != nullptr) {
    if (const auto *dir = caching_settings->cache_dir();
        dir != nullptr && dir->size() != 0) {
      options.cache_dir = dir->str();
    }
    if (const auto *token = caching_settings->model_token();
        token != nullptr && token->size() != 0) {
      options.model_token = token->str();
    }
  }

  auto delegate =
      std::make_unique<tflite::openvinodelegate::OpenVINODelegate>(&options);
  return tflite::cros::SimpleAsyncDelegateFactory::CreateAsyncDelegate(
      std::move(delegate));
}

void OpenVINOStableDelegateDestroyFunc(
    TfLiteOpaqueDelegate *openvino_stable_delegate) {
  tflite::cros::SimpleAsyncDelegateFactory::DeleteAsyncDelegate(
      openvino_stable_delegate);
}

int OpenVINOStableDelegateErrnoFunc(
    TfLiteOpaqueDelegate *openvino_stable_delegate) {
  // no-op
  return 0;
}

const TfLiteOpaqueDelegatePlugin openvino_stable_delegate_plugin = {
    OpenVINOStableDelegateCreateFunc, OpenVINOStableDelegateDestroyFunc,
    OpenVINOStableDelegateErrnoFunc};

const TfLiteStableDelegate openvino_stable_delegate = {
    TFL_STABLE_DELEGATE_ABI_VERSION, kOpenVINOStableDelegateName,
    kOpenVINOStableDelegateVersion, &openvino_stable_delegate_plugin};

}  // namespace

extern "C" const TfLiteStableDelegate TFL_TheStableDelegate =
    openvino_stable_delegate;
