/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/scoped_ahwb.h"

#include <utility>

#include "android/hardware_buffer.h"

namespace tflite::cros {

ScopedAHardwareBuffer::ScopedAHardwareBuffer() : ahwb_(nullptr) {}

ScopedAHardwareBuffer::ScopedAHardwareBuffer(AHardwareBuffer* ahwb)
    : ahwb_(ahwb) {}

ScopedAHardwareBuffer::ScopedAHardwareBuffer(const AHardwareBuffer_Desc& desc)
    : ahwb_(nullptr) {
  if (AHardwareBuffer_allocate(&desc, &ahwb_) != 0) {
    ahwb_ = nullptr;
  }
}

ScopedAHardwareBuffer::ScopedAHardwareBuffer(ScopedAHardwareBuffer&& other) {
  *this = std::move(other);
}

ScopedAHardwareBuffer& ScopedAHardwareBuffer::operator=(
    ScopedAHardwareBuffer&& other) {
  reset(other.ahwb_);
  other.ahwb_ = nullptr;
  return *this;
}

ScopedAHardwareBuffer::~ScopedAHardwareBuffer() {
  reset();
}

AHardwareBuffer* ScopedAHardwareBuffer::get() const {
  return ahwb_;
}

void ScopedAHardwareBuffer::reset(AHardwareBuffer* ahwb) {
  if (ahwb_) {
    AHardwareBuffer_release(ahwb_);
  }
  ahwb_ = ahwb;
}

AHardwareBuffer* ScopedAHardwareBuffer::release() {
  auto* ret = ahwb_;
  reset(nullptr);
  return ret;
}

void ScopedAHardwareBuffer::swap(ScopedAHardwareBuffer& other) {
  std::swap(ahwb_, other.ahwb_);
}

ScopedAHardwareBuffer::operator bool() const {
  return get() != nullptr;
}

}  // namespace tflite::cros
