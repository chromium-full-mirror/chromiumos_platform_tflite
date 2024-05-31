/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "tensorflow/lite/core/acceleration/configuration/c/gpu_plugin.h"
#include "tensorflow/lite/core/acceleration/configuration/c/stable_delegate.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/stable_delegate_interface.h"

extern "C" const TfLiteStableDelegate TFL_TheStableDelegate = {
    .delegate_abi_version = TFL_STABLE_DELEGATE_ABI_VERSION,
    .delegate_name = "StableGPUDelegate",
    .delegate_version = "0.1.0",

    // GPU delegate is NOT actually ABI stable since it's using non-opaque API
    // internally. We wrap it as a stable delegate for testing-only.
    .delegate_plugin = reinterpret_cast<const TfLiteOpaqueDelegatePlugin*>(
        TfLiteGpuDelegatePluginCApi()),
};
