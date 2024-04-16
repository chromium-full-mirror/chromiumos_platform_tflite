/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/dump.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
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

TEST(Dump, BasicTypes) {
  int num = 42;
  bool ok = true;
  char ch = 'A';

  std::string str = "foo";
  const char* c_str = "hey\napple";
  std::string_view sv = "string_view";

  std::pair<int, int> pair = {5, 14};
  std::tuple<int, bool, char> tuple = {42, true, 'A'};

  EXPECT_EQ(DUMP(num), "num = 42");
  EXPECT_EQ(DUMP(ok), "ok = true");
  EXPECT_EQ(DUMP(ch), "ch = 'A'");
  EXPECT_EQ(DUMP(str), "str = \"foo\"");
  EXPECT_EQ(DUMP(c_str), R"(c_str = "hey\napple")");
  EXPECT_EQ(DUMP(sv), "sv = \"string_view\"");
  EXPECT_EQ(DUMP(pair), "pair = (5, 14)");
  EXPECT_EQ(DUMP(tuple), "tuple = (42, true, 'A')");
}

TEST(Dump, MultipleArgs) {
  int num = 42;
  std::string str = "foo";

  EXPECT_EQ(DUMP(num, str), "num = 42, str = \"foo\"");

  // Should support at least 16 arguments.
  EXPECT_THAT(DUMP(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
              AllOf(HasSubstr("1 = 1"), HasSubstr("16 = 16")));
}

TEST(Dump, Expression) {
  EXPECT_EQ(DUMP(1 + 2), "1 + 2 = 3");
}

TEST(Dump, Containers) {
  int c_arr[] = {1, 2, 3};
  std::array<int, 3> arr = {1, 2, 3};
  std::vector<int> vec = {1, 2, 3};
  std::set<std::string> set = {"a", "b", "c"};
  std::map<int, std::string> map = {{1, "a"}, {2, "b"}};

  EXPECT_EQ(DUMP(c_arr), "c_arr = [1, 2, 3]");
  EXPECT_EQ(DUMP(arr), "arr = [1, 2, 3]");
  EXPECT_EQ(DUMP(vec), "vec = [1, 2, 3]");
  EXPECT_EQ(DUMP(set), R"(set = ["a", "b", "c"])");
  EXPECT_EQ(DUMP(map), R"(map = [(1, "a"), (2, "b")])");
}

struct Student {
  int id = 42;
  std::string name = "shik";
};

struct Course {
  int id = 514;
  std::vector<Student> students = {Student()};
};

struct HasArr {
  int arr[3] = {1, 2, 3};
};

struct HasRef {
  int& lref;
  int&& rref;
  const int& clref;
  const int&& crref;
};

class MoveOnly {
 public:
  MoveOnly() : value(42) {}
  MoveOnly(MoveOnly&& other) = default;
  MoveOnly& operator=(MoveOnly&& other) = default;
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;

  int value;
};

struct Empty {};

TEST(Dump, Struct) {
  Student student;
  Course course;
  HasArr has_arr;

  int val = 1;
  HasRef has_ref = {val, 2, 3, 4};

  MoveOnly move_only;
  Empty empty;

  EXPECT_EQ(DUMP(student), R"(student = {42, "shik"})");
  EXPECT_EQ(DUMP(course), R"(course = {514, [{42, "shik"}]})");
  EXPECT_EQ(DUMP(has_arr), "has_arr = {[1, 2, 3]}");
  EXPECT_EQ(DUMP(has_ref), "has_ref = {1, 2, 3, 4}");
  EXPECT_EQ(DUMP(move_only), "move_only = {42}");
  EXPECT_EQ(DUMP(empty), "empty = {}");
}

TEST_F(DbgTest, Location) {
  DBG("here");
  EXPECT_THAT(ConsumeOutput(), AllOf(HasSubstr(__FILE__),
                                     HasSubstr(std::to_string(__LINE__ - 2)),
                                     HasSubstr(__FUNCTION__)));
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
