/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/dbg.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ::testing::AllOf;
using ::testing::HasSubstr;

// A fixture to redirect std::cerr to a std::stringstream for testing.
class DbgTest : public testing::Test {
 protected:
  void SetUp() override {
    original_cerr_buf_ = std::cerr.rdbuf();
    std::cerr.rdbuf(stream_.rdbuf());
  }

  void TearDown() override { std::cerr.rdbuf(original_cerr_buf_); }

  // Consumes and return the captured output from std::cerr.
  std::string ConsumeOutput() {
    std::string out = stream_.str();
    stream_.str("");
    return out;
  }

 private:
  std::streambuf* original_cerr_buf_;
  std::stringstream stream_;
};

TEST_F(DbgTest, Location) {
  DBG("here");
  EXPECT_THAT(ConsumeOutput(), AllOf(HasSubstr(__FILE__),
                                     HasSubstr(std::to_string(__LINE__ - 2)),
                                     HasSubstr(__FUNCTION__)));
}

TEST_F(DbgTest, BasicTypes) {
  const int num = 42;
  const bool ok = true;
  const char ch = 'A';
  const std::string str = "foo";
  const char* c_str = "hey\napple";
  const std::string_view sv = "string_view";

  DBG(num);
  EXPECT_THAT(ConsumeOutput(), HasSubstr("num = 42"));

  DBG(ok);
  EXPECT_THAT(ConsumeOutput(), HasSubstr("ok = true"));

  DBG(ch);
  EXPECT_THAT(ConsumeOutput(), HasSubstr("ch = 'A'"));

  DBG(str);
  EXPECT_THAT(ConsumeOutput(), HasSubstr("str = \"foo\""));

  DBG(c_str);
  EXPECT_THAT(ConsumeOutput(), HasSubstr("c_str = \"hey\\napple\""));

  DBG(sv);
  EXPECT_THAT(ConsumeOutput(), HasSubstr("sv = \"string_view\""));
}

TEST_F(DbgTest, MultipleArgs) {
  const int num = 42;
  const std::string str = "foo";

  DBG(num, str);
  EXPECT_THAT(ConsumeOutput(),
              AllOf(HasSubstr("num = 42"), HasSubstr("str = \"foo\"")));

  // Should support at least 16 arguments.
  DBG(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
  EXPECT_THAT(ConsumeOutput(), AllOf(HasSubstr("1 = 1"), HasSubstr("16 = 16")));
}

TEST_F(DbgTest, Expression) {
  DBG(1 + 2);
  EXPECT_THAT(ConsumeOutput(), HasSubstr("1 + 2 = 3"));
}

TEST_F(DbgTest, Containers) {
  std::vector<int> vec = {1, 2, 3};
  std::set<std::string> set = {"a", "b", "c"};

  DBG(vec);
  EXPECT_THAT(ConsumeOutput(), HasSubstr("vec = [1, 2, 3]"));

  DBG(set);
  EXPECT_THAT(ConsumeOutput(), HasSubstr("set = [\"a\", \"b\", \"c\"]"));
}

// The example used in the file comment of dbg.h.
std::string GetNameById(int id) {
  std::string name = (id == 42 ? "apple" : "banana");
  DBG(id, name);
  return name;
}

TEST_F(DbgTest, Example) {
  GetNameById(42);
  EXPECT_THAT(ConsumeOutput(),
              AllOf(HasSubstr("id = 42"), HasSubstr("name = \"apple\"")));
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
