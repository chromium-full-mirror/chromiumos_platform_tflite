/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/minimal_logging.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/test_util/log_fixture.h"

namespace tflite {

using ::testing::AllOf;
using ::testing::EndsWith;
using ::testing::HasSubstr;

TEST_F(LogTest, Severity) {
  // No `--v=` is specified when running this test.
  TFLITE_LOG(TFLITE_LOG_VERBOSE, "This should not output anything");
  EXPECT_EQ(ConsumeOutput(), "");

  TFLITE_LOG(TFLITE_LOG_INFO, "apple");
  EXPECT_THAT(ConsumeOutput(),
              AllOf(HasSubstr("INFO"), HasSubstr("apple"), EndsWith("\n")));

  TFLITE_LOG(TFLITE_LOG_WARNING, "banana");
  EXPECT_THAT(ConsumeOutput(),
              AllOf(HasSubstr("WARN"), HasSubstr("banana"), EndsWith("\n")));

  TFLITE_LOG(TFLITE_LOG_ERROR, "cat");
  EXPECT_THAT(ConsumeOutput(),
              AllOf(HasSubstr("ERROR"), HasSubstr("cat"), EndsWith("\n")));
}

TEST_F(LogTest, MacroVariant) {
  auto test_once = [&](const auto& log_once_fn) {
    log_once_fn("dog");
    EXPECT_THAT(ConsumeOutput(), HasSubstr("dog"));
    log_once_fn("egg");
    EXPECT_EQ(ConsumeOutput(), "");
  };
  auto log_once = [](const char* msg) {
    TFLITE_LOG_ONCE(TFLITE_LOG_INFO, "%s", msg);
  };
  auto log_prod_once = [](const char* msg) {
    TFLITE_LOG_PROD_ONCE(TFLITE_LOG_INFO, "%s", msg);
  };
  test_once(log_once);
  test_once(log_prod_once);

  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "fox");
  EXPECT_THAT(ConsumeOutput(), HasSubstr("fox"));
}

TEST_F(LogTest, FormatString) {
  TFLITE_LOG(TFLITE_LOG_INFO, "num = %d", 42);
  EXPECT_THAT(ConsumeOutput(), HasSubstr("num = 42"));

  TFLITE_LOG(TFLITE_LOG_INFO, "str = %s", "dog");
  EXPECT_THAT(ConsumeOutput(), HasSubstr("str = dog"));
}

}  // namespace tflite

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
