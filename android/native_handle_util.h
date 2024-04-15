/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef ANDROID_NATIVE_HANDLE_UTIL_H_
#define ANDROID_NATIVE_HANDLE_UTIL_H_

#include <memory>

#include "android/native_handle.h"

namespace tflite::cros {

// Creates a new native handle from the given fd.
std::unique_ptr<native_handle_t> CreateNativeHandle(int fd);

// Clones a native handle. The file descriptor(s) are copied without dup().
std::unique_ptr<native_handle_t> CloneNativeHandle(
    const native_handle_t* handle);

// TODO(shik): Add another copy helper that will dup() the file descriptors.

}  // namespace tflite::cros

#endif  // ANDROID_NATIVE_HANDLE_UTIL_H_
