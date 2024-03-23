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

// A helper type to utilize overload resolution with lambdas.
template <class... Ts>
struct Overloads : Ts... {
  using Ts::operator()...;
};

template <typename... Ts>
Overloads(Ts...) -> Overloads<Ts...>;

// A compile-time boolean constant macro to check whether value is
// structured-bindable. Uses Statement Expression extension to create SFINAE
// with structured binding.
#define DBG_IS_BINDABLE(...)                      \
  decltype((Overloads{                            \
      [](auto&& x, int) -> decltype(({            \
        [[maybe_unused]] auto& [__VA_ARGS__] = x; \
        std::true_type{};                         \
      })) {},                                     \
      [](auto&& x, bool) -> std::false_type {},   \
  })(value, 0))::value

// A constexpr if branch used in AsTuple(). Note that "else" is required to
// correctly discarded the unwanted branches.
#define DBG_AS_TUPLE(...)                            \
  /* NOLINTNEXTLINE(readability/braces) */           \
  else if constexpr (DBG_IS_BINDABLE(__VA_ARGS__)) { \
    auto& [__VA_ARGS__] = value;                     \
    return std::tie(__VA_ARGS__);                    \
  }

// Converts a struct value to a tuple. Need to manually expand the members
// because structured bindings cannot introduce a pack yet. See
// https://wg21.link/P1061 for the proposal to improve this in C++.
template <typename T>
constexpr auto AsTuple(const T& value) {
  if constexpr (std::is_empty_v<T>) {
    return std::tie();
  }

  // Supports up to 15 elements.
  DBG_AS_TUPLE(e1)
  DBG_AS_TUPLE(e1, e2)
  DBG_AS_TUPLE(e1, e2, e3)
  DBG_AS_TUPLE(e1, e2, e3, e4)
  DBG_AS_TUPLE(e1, e2, e3, e4, e5)
  DBG_AS_TUPLE(e1, e2, e3, e4, e5, e6)
  DBG_AS_TUPLE(e1, e2, e3, e4, e5, e6, e7)
  DBG_AS_TUPLE(e1, e2, e3, e4, e5, e6, e7, e8)
  DBG_AS_TUPLE(e1, e2, e3, e4, e5, e6, e7, e8, e9)
  DBG_AS_TUPLE(e1, e2, e3, e4, e5, e6, e7, e8, e9, e10)
  DBG_AS_TUPLE(e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11)
  DBG_AS_TUPLE(e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12)
  DBG_AS_TUPLE(e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13)
  DBG_AS_TUPLE(e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14)
  DBG_AS_TUPLE(e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15)
}

#undef DBG_AS_TUPLE_IMPL
#undef DBG_IS_BINDABLE

// A helper type for templated overloads resolution.
template <int N>
struct PriorityTag : PriorityTag<N - 1> {};

template <>
struct PriorityTag<0> {};

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
    Print(std::forward<T>(value), PriorityTag<6>());
  }

  // TODO(shik): Support pointers.
  // TODO(shik): Support bytes with hex output.
  // TODO(shik): Be more careful with char*, which may not have null terminator
  // and we should limit the printing length.

  // Boolean value is printed as "true" or "false".
  template <typename T>
    requires std::same_as<std::decay_t<T>, bool>
  void Print(T&& value, PriorityTag<6>) {
    stream_ << (value ? "true" : "false");
  }

  // Char-like value is quoted and escaped as 'x' and '\n'.
  template <typename T>
    requires std::same_as<std::decay_t<T>, char>
  void Print(T&& value, PriorityTag<5>) {
    std::decay_t<T> ch = value;
    std::string_view view(&ch, 1);
    stream_ << "'" << absl::Utf8SafeCEscape(view) << "'";
  }

  // String-like value is quoted and escaped as "foo\nbar".
  template <typename T>
    requires std::convertible_to<T, std::string_view>
  void Print(T&& str, PriorityTag<4>) {
    auto view = static_cast<std::string_view>(str);
    stream_ << '"' << absl::Utf8SafeCEscape(view) << '"';
  }

  // Container-like range value is printed as [v1, v2, ...].
  template <typename T>
    requires std::ranges::range<T>
  void Print(T&& range, PriorityTag<3>) {
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
  void Print(T&& tuple, PriorityTag<2>) {
    stream_ << "(";
    std::apply(
        [&](const auto&... element) {
          size_t idx = 0;
          ((stream_ << (idx++ > 0 ? ", " : ""), Print(element)), ...);
        },
        std::forward<T>(tuple));
    stream_ << ")";
  }

  // Structure-bindable class value is printed as {e1, e2, ...}.
  template <typename T>
    requires(std::is_class_v<std::decay_t<T>> &&
             !std::is_same_v<void, decltype(AsTuple(std::declval<T>()))>)
  void Print(T&& value, PriorityTag<1>) {
    stream_ << "{";
    std::apply(
        [&](const auto&... element) {
          size_t idx = 0;
          ((stream_ << (idx++ > 0 ? ", " : ""), Print(element)), ...);
        },
        AsTuple(std::forward<T>(value)));
    stream_ << "}";
  }

  // Use operator<< if it's not specialized above. The requires statement is
  // added to make the error message more readable, otherwise compiler will list
  // all possible candidate overloads of operator<<.
  template <typename T>
    requires requires(std::ostream& os, T value) {
      { os << value } -> std::convertible_to<std::ostream&>;
    }
  void Print(T&& value, PriorityTag<0>) {
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
