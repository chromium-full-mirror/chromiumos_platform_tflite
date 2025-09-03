/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "android/native_handle_util.h"

#include <cstdlib>

namespace tflite::cros {

void NativeHandleDeleter::operator()(native_handle_t* handle) {
  std::free(handle);
}

OwnedNativeHandle CreateNativeHandle(int fd) {
  size_t size = sizeof(native_handle_t) + sizeof(int);
  auto* handle = static_cast<native_handle_t*>(std::malloc(size));
  *handle = {
      .version = sizeof(native_handle_t),
      .numFds = 1,
      .numInts = 0,
  };
  handle->data[0] = fd;
  return OwnedNativeHandle(handle);
}

OwnedNativeHandle CloneNativeHandle(const native_handle_t* handle) {
  if (handle == nullptr) {
    return nullptr;
  }

  size_t size = sizeof(native_handle_t) +
                sizeof(int) * (handle->numFds + handle->numInts);
  auto* cloned = static_cast<native_handle_t*>(std::malloc(size));

  *cloned = {
      .version = sizeof(native_handle_t),
      .numFds = handle->numFds,
      .numInts = handle->numInts,
  };
  for (int i = 0; i < handle->numFds + handle->numInts; ++i) {
    cloned->data[i] = handle->data[i];
  }

  return OwnedNativeHandle(cloned);
}

}  // namespace tflite::cros
