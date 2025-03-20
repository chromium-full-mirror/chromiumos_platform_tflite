/*
 * Copyright 2025 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/test_util/aligned_tflite_buffer_allocator.h"

#include <cstdint>
#include <memory>

#include "tensorflow/lite/util.h"

namespace {

void DeallocateTfLiteAlignedTensorBufferHelper(uint8_t* ptr) {
  operator delete[](ptr, std::align_val_t{tflite::kDefaultTensorAlignment});
}

uint8_t* AllocateTfLiteAlignedTensorBufferHelper(int bytes) {
  return new (std::align_val_t{tflite::kDefaultTensorAlignment}) uint8_t[bytes];
}

}  // namespace

std::shared_ptr<uint8_t[]> AllocateTfLiteAlignedTensorBuffer(size_t bytes) {
  return std::shared_ptr<uint8_t[]>(
      AllocateTfLiteAlignedTensorBufferHelper(bytes),
      DeallocateTfLiteAlignedTensorBufferHelper);
}
