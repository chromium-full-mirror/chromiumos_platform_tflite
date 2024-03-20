/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/log.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/mman.h>

#include <string>

namespace {

using ::testing::AllOf;
using ::testing::EndsWith;
using ::testing::HasSubstr;

// A fixture to redirect stderr to a memfd for testing.
class LogTest : public testing::Test {
 protected:
  void SetUp() override {
    memfd_ = memfd_create("captured_stderr", MFD_CLOEXEC);
    ASSERT_NE(memfd_, -1);

    original_stderr_ = dup(STDERR_FILENO);
    ASSERT_NE(original_stderr_, -1);

    ASSERT_NE(dup2(memfd_, STDERR_FILENO), -1);
  }

  void TearDown() override {
    EXPECT_EQ(close(memfd_), 0);

    ASSERT_NE(dup2(original_stderr_, STDERR_FILENO), -1);
    EXPECT_EQ(close(original_stderr_), 0);
  }

  // Consumes and return the captured output from stderr.
  std::string ConsumeOutput() {
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

 private:
  int memfd_ = -1;
  int original_stderr_ = -1;
};

TEST_F(LogTest, LOG) {
  LOG(INFO) << "apple";
  EXPECT_THAT(ConsumeOutput(),
              AllOf(HasSubstr("INFO"), HasSubstr("apple"), EndsWith("\n")));

  LOG(WARNING) << "banana";
  EXPECT_THAT(ConsumeOutput(),
              AllOf(HasSubstr("WARNING"), HasSubstr("banana"), EndsWith("\n")));

  LOG(ERROR) << "cat";
  EXPECT_THAT(ConsumeOutput(),
              AllOf(HasSubstr("ERROR"), HasSubstr("cat"), EndsWith("\n")));

  EXPECT_DEATH({ LOG(FATAL) << "oops"; }, "oops");
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
