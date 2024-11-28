/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/log.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>

#include "common/test_util/log_fixture.h"

namespace {

using ::testing::AllOf;
using ::testing::EndsWith;
using ::testing::HasSubstr;

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

TEST_F(LogTest, PLOG) {
  errno = EINVAL;
  PLOG(INFO) << "Failed to do something";
  int after_errno = errno;
  errno = 0;

  EXPECT_THAT(ConsumeOutput(),
              HasSubstr("Failed to do something: Invalid argument [22]\n"));
  EXPECT_EQ(after_errno, EINVAL);
}

TEST_F(LogTest, VLOG) {
  // We set `--v=2` when running log_test.
  VLOG(1) << "Lv1";
  EXPECT_THAT(ConsumeOutput(), HasSubstr("Lv1"));
  VLOG(2) << "Lv2";
  EXPECT_THAT(ConsumeOutput(), HasSubstr("Lv2"));
  VLOG(3) << "Lv3";
  EXPECT_EQ(ConsumeOutput(), "");
}

TEST_F(LogTest, FunctionName) {
  const char* fn_name = __FUNCTION__;
  EXPECT_STREQ(fn_name, "TestBody");

  LOGF(INFO) << "dog";
  EXPECT_THAT(ConsumeOutput(), HasSubstr("TestBody: dog"));

  PLOGF(INFO) << "egg";
  EXPECT_THAT(ConsumeOutput(), HasSubstr("TestBody: egg: Success [0]"));

  VLOGF(1) << "fox";
  EXPECT_THAT(ConsumeOutput(), HasSubstr("TestBody: fox"));
}

TEST_F(LogTest, DUMP) {
  std::string name = "gold";
  LOG(INFO) << DUMP(1 + 2, name);
  EXPECT_THAT(ConsumeOutput(), HasSubstr("1 + 2 = 3, name = \"gold\""));
}

TEST_F(LogTest, CHECK) {
  CHECK(2 > 1) << "2 should be larger than 1";
  EXPECT_DEATH({ CHECK(1 > 2) << "oops"; }, "Check failed: 1 > 2 oops");

  CHECK_EQ(1, 1);
  CHECK_NE(1, 2);
  CHECK_LT(1, 2);
  CHECK_LE(1, 1);
  CHECK_GT(2, 1);
  CHECK_GE(2, 2);
}

TEST_F(LogTest, PCHECK) {
  errno = EINVAL;
  EXPECT_DEATH(
      { PCHECK(1 > 2) << "oops"; },
      "Check failed: 1 > 2 oops: Invalid argument");
  int after_errno = errno;
  errno = 0;

  EXPECT_EQ(after_errno, EINVAL);
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
