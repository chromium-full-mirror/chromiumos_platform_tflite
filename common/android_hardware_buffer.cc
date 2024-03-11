/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/android_hardware_buffer.h"

#include <cerrno>

// TODO(shik): Implement the AHardwareBuffer shim API.

int AHardwareBuffer_allocate(const AHardwareBuffer_Desc* _Nonnull desc,
                             AHardwareBuffer* _Nullable* _Nonnull outBuffer) {
  return -ENOSYS;
}

void AHardwareBuffer_acquire(AHardwareBuffer* _Nonnull buffer) {}

void AHardwareBuffer_release(AHardwareBuffer* _Nonnull buffer) {}

void AHardwareBuffer_describe(const AHardwareBuffer* _Nonnull buffer,
                              AHardwareBuffer_Desc* _Nonnull outDesc) {}

int AHardwareBuffer_lock(AHardwareBuffer* _Nonnull buffer,
                         uint64_t usage,
                         int32_t fence,
                         const ARect* _Nullable rect,
                         void* _Nullable* _Nonnull outVirtualAddress) {
  return -ENOSYS;
}

int AHardwareBuffer_unlock(AHardwareBuffer* _Nonnull buffer,
                           int32_t* _Nullable fence) {
  return -ENOSYS;
}

int AHardwareBuffer_isSupported(const AHardwareBuffer_Desc* _Nonnull desc) {
  return 0;
}

const native_handle_t* AHardwareBuffer_getNativeHandle(
    const AHardwareBuffer* _Nonnull buffer) {
  return nullptr;
}

int AHardwareBuffer_createFromHandle(
    const AHardwareBuffer_Desc* _Nonnull desc,
    const native_handle_t* _Nonnull handle,
    int32_t method,
    AHardwareBuffer* _Nullable* _Nonnull outBuffer) {
  return -ENOSYS;
}
