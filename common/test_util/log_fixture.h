/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef COMMON_TEST_UTIL_LOG_FIXTURE_H_
#define COMMON_TEST_UTIL_LOG_FIXTURE_H_

#include <gtest/gtest.h>

#include <string>

// A fixture that redirects stderr to a memfd for log testing.
class LogTest : public testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

  // Consumes and return the captured output from stderr.
  std::string ConsumeOutput();

 private:
  int memfd_ = -1;
  int original_stderr_ = -1;
};

#endif  // COMMON_TEST_UTIL_LOG_FIXTURE_H_
