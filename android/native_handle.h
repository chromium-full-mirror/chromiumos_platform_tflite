/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef ANDROID_NATIVE_HANDLE_H_
#define ANDROID_NATIVE_HANDLE_H_

/**
 * Copied from system/core/libcutils/include/cutils/native_handle.h in Android.
 */
typedef struct native_handle {
  int version;  // sizeof(native_handle_t)
  int numFds;   // number of file-descriptors at &data[0]
  int numInts;  // number of ints at &data[numFds]
  int data[0];  // numFds + numInts ints
} native_handle_t;

#endif  // ANDROID_NATIVE_HANDLE_H_
