/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef COMMON_SCOPED_FD_H_
#define COMMON_SCOPED_FD_H_

namespace tflite::cros {

// ScopedFd is a RAII wrapper for file descriptors. It automatically closes the
// file descriptor when the object goes out of scope.
class ScopedFd {
 public:
  static const int kInvalidFd = -1;

  // Default constructor. Creates a ScopedFd with an invalid file descriptor.
  ScopedFd();

  // Constructor that takes a file descriptor.
  // The ScopedFd takes ownership of the provided file descriptor.
  explicit ScopedFd(int fd);

  // Destructor. Closes the owned file descriptor if it is valid.
  ~ScopedFd();

  // Move constructor and move assignment constructor. Transfers ownership of
  // the file descriptor from another ScopedFd.
  ScopedFd(ScopedFd&& other);
  ScopedFd& operator=(ScopedFd&& other);

  // ScopedFd is move-only. Copying is not allowed.
  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;

  // Resets the ScopedFd to own a new file descriptor.
  // Closes the previously owned file descriptor if it was valid.
  void reset(int fd = kInvalidFd);

  // Returns the underlying file descriptor.
  // The caller should not close the returned file descriptor, as ScopedFd
  // retains ownership.
  int get() const;

  // Implicit conversion to int.
  // Allows using ScopedFd directly as an integer representing the file
  // descriptor, which is convenient for passing as an argument to functions
  // that accept non-owning file descriptors.
  operator int() const;

  // Checks if the owned file descriptor is valid.
  bool is_valid();

 private:
  int fd_ = kInvalidFd;
};

}  // namespace tflite::cros

#endif  // COMMON_SCOPED_FD_H_
