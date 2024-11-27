/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef COMMON_SCOPED_AHWB_H_
#define COMMON_SCOPED_AHWB_H_

#include <utility>

#include "android/hardware_buffer.h"

namespace tflite::cros {

// ScopedAHardwareBuffer is a RAII wrapper for AHardwareBuffer. It owns and
// manages AHardwareBuffer via a pointer and would automatically deallocate the
// buffer when this object goes out of scope.
class ScopedAHardwareBuffer {
 public:
  // Default constructor. Creates a ScopedAHardwareBuffer with a null pointer.
  ScopedAHardwareBuffer();

  // Constructor that takes a AHardwareBuffer pointer.
  // The ScopedAHardwareBuffer takes ownership of the underlying
  // AHardwareBuffer.
  explicit ScopedAHardwareBuffer(AHardwareBuffer* ahwb);

  // Construct the AHardwareBuffer based on the provided description.
  explicit ScopedAHardwareBuffer(const AHardwareBuffer_Desc& desc);

  // Destructor. Deallocate the buffer.
  ~ScopedAHardwareBuffer();

  // Move constructor and move assignment constructor. Transfers ownership of
  // the AHardwareBuffer from another ScopedAHardwareBuffer.
  ScopedAHardwareBuffer(ScopedAHardwareBuffer&& other);
  ScopedAHardwareBuffer& operator=(ScopedAHardwareBuffer&& other);

  // ScopedAHardwareBuffer is move-only. Copying is not allowed.
  ScopedAHardwareBuffer(const ScopedAHardwareBuffer&) = delete;
  ScopedAHardwareBuffer& operator=(const ScopedAHardwareBuffer&) = delete;

  // Returns the underlying AHardwareBuffer pointer.
  // The caller should not free the pointer, as ScopedAHardwareBuffer retains
  // ownership.
  AHardwareBuffer* get() const;

  // Resets the ScopedAHardwareBuffer to own a new AHardwareBuffer pointer.
  // Releases the previously owned AHardwareBuffer if it was valid.
  void reset(AHardwareBuffer* ahwb = nullptr);

  // Releases the underlying AHardwareBuffer pointer.
  // The caller is responsible for freeing the pointer, as ScopedAHardwareBuffer
  // no longer retains ownership.
  AHardwareBuffer* release();

  // Swaps the underlying AHardwareBuffer.
  void swap(ScopedAHardwareBuffer& other);

  // Checks whether get() is nullptr;
  operator bool() const;

 private:
  AHardwareBuffer* ahwb_ = nullptr;
};

}  // namespace tflite::cros

#endif  // COMMON_SCOPED_AHWB_H_
