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
// [example.cc:10:GetNameById] name = "apple"
//
// This macro MUST NOT be used in production. Uncomment the `defines` line in
// BUILD.bazel to enable it for local debugging.

#if defined(NDEBUG) && !defined(DBG_FORCE_ENABLE)
#error "dbg.h MUST NOT be used in production"
#endif

#include <iostream>
#include <ranges>  // NOLINT(build/include_order) - C++20 header is not recognized yet
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "absl/strings/escaping.h"

namespace dbg {

namespace impl {

template <int N>
struct priority_tag : priority_tag<N - 1> {};

template <>
struct priority_tag<0> {};

class DebugPrinter {
 public:
  template <typename T>
  static std::string Stringify(T&& value) {
    DebugPrinter printer;
    printer.Print(std::forward<T>(value));
    return printer.GetString();
  }

 private:
  std::string GetString() { return stream_.str(); }

  template <typename T>
  void Print(T&& value) {
    Print(std::forward<T>(value), priority_tag<5>());
  }

  // TODO(shik): Support pointers.
  // TODO(shik): Support bytes with hex output.
  // TODO(shik): Be more careful with char*, which may not have null terminator
  // and we should limit the printing length.

  // Boolean value is printed as "true" or "false".
  template <typename T>
    requires std::same_as<std::decay_t<T>, bool>
  void Print(T&& value, priority_tag<5>) {
    stream_ << (value ? "true" : "false");
  }

  // Char-like value is quoted and escaped as 'x' and '\n'.
  template <typename T>
    requires std::same_as<std::decay_t<T>, char>
  void Print(T&& value, priority_tag<4>) {
    std::decay_t<T> ch = value;
    std::string_view view(&ch, 1);
    stream_ << "'" << absl::Utf8SafeCEscape(view) << "'";
  }

  // String-like value is quoted and escaped as "foo\nbar".
  template <typename T>
    requires std::convertible_to<T, std::string_view>
  void Print(T&& str, priority_tag<3>) {
    auto view = static_cast<std::string_view>(str);
    stream_ << '"' << absl::Utf8SafeCEscape(view) << '"';
  }

  // Container-like range value is printed as [v1, v2, ...].
  template <typename T>
    requires std::ranges::range<T>
  void Print(T&& range, priority_tag<2>) {
    auto begin = std::ranges::begin(range);
    auto end = std::ranges::end(range);
    stream_ << "[";
    for (auto it = begin; it != end; ++it) {
      if (it != begin) {
        stream_ << ", ";
      }
      Print(*it);
    }
    stream_ << "]";
  }

  // Tuple-like value is printed as (v1, v2, ...).
  // This is put after container overload to match std::array as a container.
  template <typename T>
    requires requires { typename std::tuple_size<std::decay_t<T>>::type; }
  void Print(T&& tuple, priority_tag<1>) {
    stream_ << "(";
    std::apply(
        [&](const auto&... element) {
          size_t idx = 0;
          ((stream_ << (idx++ > 0 ? ", " : ""), Print(element)), ...);
        },
        std::forward<T>(tuple));
    stream_ << ")";
  }

  // Use operator<< if it's not specialized above. The requires statement is
  // added to make the error message more readable, otherwise compiler will list
  // all possible candidate overloads of operator<<.
  template <typename T>
    requires requires(std::ostream& os, T value) {
      { os << value } -> std::convertible_to<std::ostream&>;
    }
  void Print(T&& value, priority_tag<0>) {
    stream_ << value;
  }

  std::stringstream stream_;
};

};  // namespace impl

template <typename T>
std::string Stringify(T&& value) {
  return impl::DebugPrinter::Stringify(std::forward<T>(value));
}

template <typename... T>
void DebugPrint(std::string_view file,
                int line,
                std::string_view func,
                std::initializer_list<std::string_view> names,
                T&&... values) {
  auto value_strings = std::initializer_list<std::string>{Stringify(values)...};
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
