/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef COMMON_DBG_H_
#define COMMON_DBG_H_

// DBG() is a convenience macro for printf-style debugging, which prints all the
// arguments as key-value pairs with location information into std::cerr.
//
// Example:
// ```cpp
// std::string GetNameById(int id) {
//   std::string name = (id == 42 ? "apple" : "banana");
//   DBG(id, name);
//   return name;
// }
// GetNameById(42);
// ```
//
// The above code will print (assuming DBG() is located at line 10):
// [example.cc:10:GetNameById] id = 42
// [example.cc:10:GetNameById] name = apple
//
// This macro MUST NOT be used in production. Uncomment the `defines` line in
// BUILD.bazel to enable it for local debugging.

#if defined(NDEBUG) && !defined(DBG_FORCE_ENABLE)
#error "dbg.h MUST NOT be used in production"
#endif

#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace dbg {

template <typename... T>
void DebugPrint(std::string_view file,
                int line,
                std::string_view func,
                std::initializer_list<std::string_view> names,
                T&&... values) {
  // TODO(shik): Support more types automatically, such as containers.
  auto value_strings = std::initializer_list<std::string>{
      ((std::stringstream() << values).str())...};
  auto name_it = names.begin();
  auto value_it = value_strings.begin();
  for (size_t i = 0; i < names.size(); ++i) {
    std::cerr << "[" << file << ":" << line << ":" << func << "] " << *name_it++
              << " = " << *value_it++ << std::endl;
  }
}

}  // namespace dbg

// Recursive macro tricks with C++20 __VA_OPT__. See
// https://www.scs.stanford.edu/~dm/blog/va-opt.html for more details.
#define DBG_PARENS ()

// Evaluates (re-scan) the macros at least 4^4 = 256 times.
#define DBG_EVAL(...) DBG_EVAL4(DBG_EVAL4(DBG_EVAL4(DBG_EVAL4(__VA_ARGS__))))
#define DBG_EVAL4(...) DBG_EVAL3(DBG_EVAL3(DBG_EVAL3(DBG_EVAL3(__VA_ARGS__))))
#define DBG_EVAL3(...) DBG_EVAL2(DBG_EVAL2(DBG_EVAL2(DBG_EVAL2(__VA_ARGS__))))
#define DBG_EVAL2(...) DBG_EVAL1(DBG_EVAL1(DBG_EVAL1(DBG_EVAL1(__VA_ARGS__))))
#define DBG_EVAL1(...) __VA_ARGS__

// DBG_MAP(fn, a1, a2, a3, ...) => fn(a1), fn(a2), fn(a3), ...
#define DBG_MAP(fn, ...) __VA_OPT__(DBG_EVAL(DBG_MAP_IMPL(fn, __VA_ARGS__)))
#define DBG_MAP_IMPL(fn, a1, ...) \
  fn(a1), __VA_OPT__(DBG_MAP_AGAIN DBG_PARENS(fn, __VA_ARGS__))
#define DBG_MAP_AGAIN() DBG_MAP_IMPL

// DBG_MAP(foo) => "foo"
#define DBG_STRINGIFY(x) #x

#define DBG(...)                                    \
  dbg::DebugPrint(__FILE__, __LINE__, __FUNCTION__, \
                  {DBG_MAP(DBG_STRINGIFY, __VA_ARGS__)}, __VA_ARGS__)

#endif  // COMMON_DBG_H_
