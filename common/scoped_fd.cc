/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/scoped_fd.h"

#include <unistd.h>
#include <utility>

namespace tflite::cros {

ScopedFd::ScopedFd() : ScopedFd(kInvalidFd) {}

ScopedFd::ScopedFd(int fd) : fd_(fd) {}

ScopedFd::ScopedFd(ScopedFd&& other) {
  *this = std::move(other);
}

ScopedFd& ScopedFd::operator=(ScopedFd&& other) {
  reset(other.fd_);
  other.fd_ = kInvalidFd;
  return *this;
}

ScopedFd::~ScopedFd() {
  reset();
}

void ScopedFd::reset(int fd) {
  if (fd_ >= 0) {
    close(fd_);
  }
  fd_ = fd;
}

int ScopedFd::get() const {
  return fd_;
}

ScopedFd::operator int() const {
  return get();
}

bool ScopedFd::is_valid() {
  return fd_ != kInvalidFd;
}

}  // namespace tflite::cros
