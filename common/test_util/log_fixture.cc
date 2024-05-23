/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/test_util/log_fixture.h"

#include <sys/mman.h>

#include <string>

void LogTest::SetUp() {
  memfd_ = memfd_create("captured_stderr", MFD_CLOEXEC);
  ASSERT_NE(memfd_, -1);

  original_stderr_ = dup(STDERR_FILENO);
  ASSERT_NE(original_stderr_, -1);

  ASSERT_NE(dup2(memfd_, STDERR_FILENO), -1);
}

void LogTest::TearDown() {
  EXPECT_EQ(close(memfd_), 0);

  ASSERT_NE(dup2(original_stderr_, STDERR_FILENO), -1);
  EXPECT_EQ(close(original_stderr_), 0);
}

std::string LogTest::ConsumeOutput() {
  off_t size = lseek(memfd_, 0, SEEK_END);
  if (size == -1) {
    ADD_FAILURE() << "Failed to get memfd size";
    return "";
  }

  // Read all output.
  std::string output(size, '\0');
  EXPECT_EQ(lseek(memfd_, 0, SEEK_SET), 0);
  EXPECT_EQ(read(memfd_, output.data(), size), size);

  // Clear the file.
  EXPECT_EQ(ftruncate(memfd_, 0), 0);
  EXPECT_EQ(lseek(memfd_, 0, SEEK_SET), 0);

  return output;
}
