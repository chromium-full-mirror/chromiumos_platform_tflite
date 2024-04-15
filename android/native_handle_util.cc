/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "android/native_handle_util.h"

#include <memory>

namespace tflite::cros {

std::unique_ptr<native_handle_t> CreateNativeHandle(int fd) {
  size_t size = sizeof(native_handle_t) + sizeof(int);
  auto* handle = static_cast<native_handle_t*>(operator new(size));
  *handle = {
      .version = sizeof(native_handle_t),
      .numFds = 1,
      .numInts = 0,
  };
  handle->data[0] = fd;
  return std::unique_ptr<native_handle_t>(handle);
}

std::unique_ptr<native_handle_t> CloneNativeHandle(
    const native_handle_t* handle) {
  if (handle == nullptr) {
    return nullptr;
  }

  size_t size = sizeof(native_handle_t) +
                sizeof(int) * (handle->numFds + handle->numInts);
  auto* cloned = static_cast<native_handle_t*>(operator new(size));

  *cloned = {
      .version = sizeof(native_handle_t),
      .numFds = handle->numFds,
      .numInts = handle->numInts,
  };
  for (int i = 0; i < handle->numFds + handle->numInts; ++i) {
    cloned->data[i] = handle->data[i];
  }

  return std::unique_ptr<native_handle_t>(cloned);
}

}  // namespace tflite::cros
