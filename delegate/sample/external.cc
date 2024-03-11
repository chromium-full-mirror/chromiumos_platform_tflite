/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <memory>

#include "delegate/sample/delegate.h"

#include "tensorflow/lite/core/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/core/acceleration/configuration/c/stable_delegate.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/stable_delegate_interface.h"

namespace tflite::cros {

namespace {

TfLiteOpaqueDelegate* CreateFunc(const void* tflite_settings) {
  auto delegate = std::make_unique<CrosSampleDelegate>();
  return SimpleAsyncDelegateFactory::CreateAsyncDelegate(std::move(delegate));
}

void DestroyFunc(TfLiteOpaqueDelegate* delegate) {
  SimpleAsyncDelegateFactory::DeleteAsyncDelegate(delegate);
}

int ErrnoFunc(TfLiteOpaqueDelegate* delegate) {
  // TODO(shik): Implement this.
  return 0;
}

constexpr TfLiteOpaqueDelegatePlugin plugin = {
    .create = CreateFunc,
    .destroy = DestroyFunc,
    .get_delegate_errno = ErrnoFunc,
};

constexpr TfLiteStableDelegate cros_sample_delegate = {
    .delegate_abi_version = TFL_STABLE_DELEGATE_ABI_VERSION,
    .delegate_name = CrosSampleDelegate::kName,
    .delegate_version = CrosSampleDelegate::kVersion,
    .delegate_plugin = &plugin,
};

}  // namespace

}  // namespace tflite::cros

extern "C" constexpr TfLiteStableDelegate TFL_TheStableDelegate =
    tflite::cros::cros_sample_delegate;
