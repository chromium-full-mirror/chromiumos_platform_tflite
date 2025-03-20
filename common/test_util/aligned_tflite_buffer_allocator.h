/*
 * Copyright 2025 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef COMMON_TEST_UTIL_ALIGNED_TFLITE_BUFFER_ALLOCATOR_H_
#define COMMON_TEST_UTIL_ALIGNED_TFLITE_BUFFER_ALLOCATOR_H_

#include <cstdint>
#include <memory>

// Allocate a buffer with the size of |bytes| and aligned to the alignment
// requirement of tflite.
std::shared_ptr<uint8_t[]> AllocateTfLiteAlignedTensorBuffer(size_t bytes);

#endif  // COMMON_TEST_UTIL_ALIGNED_TFLITE_BUFFER_ALLOCATOR_H_
