/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef COMMON_DUMP_H_
#define COMMON_DUMP_H_

// This header provides 2 convenience macros for writing variables and/or
// expressions to logs: DUMP() and DBG().
//
// DUMP() converts all the arguments as a string with key-value pairs.
//
// Example:
// ```cpp
// std::string GetNameById(int id) {
//   std::string name = (id == 42 ? "apple" : "banana");
//   LOG(INFO) << DUMP(id, name);
//   return name;
// }
// GetNameById(42);
// ```
//
// The above code will print `id = 42, name = "apple"` into log.
//
// DBG() is a shortcut for printf-style debugging, which prints all the
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
// [example.cc:10:GetNameById] id = 42, name = "apple"
//
// The DBG() macro MUST NOT be used in production. Uncomment the `defines` line
// in BUILD.bazel to enable it for local debugging.

#include <unistd.h>

#include <cstdio>
#include <iostream>
#include <ranges>  // NOLINT(build/include_order) - C++20 header is not recognized yet
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "absl/strings/escaping.h"

namespace dbg {

namespace impl {

template <typename T>
consteval std::string_view PrettyFunctionName() {
  return __PRETTY_FUNCTION__;
}

// A helper function for retrieving type name from templated type argument.
template <typename T>
consteval auto TypeName() {
  // Using compile-time constant evaluation instead of a hard-coded value here
  // for robustness and portability. On clang, this looks like:
  // std::string_view PrettyFunctionName() [T = void]
  std::string_view kVoidName = PrettyFunctionName<void>();
  size_t kPrefix = kVoidName.find("void");
  size_t kSuffix = kVoidName.size() - kPrefix - 4;

  std::string_view name = PrettyFunctionName<T>();
  return name.substr(kPrefix, name.size() - kPrefix - kSuffix);
}

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

#undef DBG_AS_TUPLE
#undef DBG_IS_BINDABLE

// A helper type for templated overloads resolution.
template <int N>
struct PriorityTag : PriorityTag<N - 1> {};

template <>
struct PriorityTag<0> {};

class Printer {
 public:
  template <typename T>
  static std::string Stringify(T&& value) {
    Printer printer;
    printer.Print(std::forward<T>(value));
    return printer.GetString();
  }

 private:
  std::string GetString() { return stream_.str(); }

  template <typename T>
  void Print(T&& value) {
    Print(std::forward<T>(value), PriorityTag<11>());
  }

  // TODO(shik): Support bytes with hex output.
  // TODO(shik): Be more careful with char*, which may not have null terminator
  // and we should limit the printing length.
  // TODO(shik): Handle cycle in recursive type.

  // Boolean value is printed as "true" or "false".
  template <typename T>
    requires std::same_as<std::decay_t<T>, bool>
  void Print(T&& value, PriorityTag<11>) {
    stream_ << (value ? "true" : "false");
  }

  // nullptr_t is printed as "nullptr".
  template <typename T>
    requires std::same_as<std::decay_t<T>, std::nullptr_t>
  void Print(T&& value, PriorityTag<10>) {
    stream_ << "nullptr";
  }

  // Char-like value is quoted and escaped as 'x' and '\n'.
  template <typename T>
    requires std::same_as<std::decay_t<T>, char>
  void Print(T&& value, PriorityTag<9>) {
    std::decay_t<T> ch = value;
    std::string_view view(&ch, 1);
    stream_ << "'" << absl::Utf8SafeCEscape(view) << "'";
  }

  // Integral-like value that cannot be handled by standard iostream is printed
  // as normal number.
  template <typename T>
    requires std::same_as<std::decay_t<T>, uint8_t> ||
             std::same_as<std::decay_t<T>, int8_t> ||
             std::is_enum_v<std::decay_t<T>>
  void Print(T&& value, PriorityTag<8>) {
    if constexpr (std::is_signed_v<std::decay_t<T>>) {
      stream_ << static_cast<int64_t>(value);
    } else {
      stream_ << static_cast<uint64_t>(value);
    }
  }

  // String-like value is quoted and escaped as "foo\nbar".
  template <typename T>
    requires std::convertible_to<T, std::string_view>
  void Print(T&& str, PriorityTag<7>) {
    if constexpr (std::is_pointer_v<std::decay_t<T>>) {
      // It's invalid to construct a std::string_view from nullptr until P0903
      // "Define basic_string_view(nullptr)" is accepted. See
      // https://open-std.org/JTC1/SC22/WG21/docs/papers/2018/p0903r2.pdf
      if (str == nullptr) {
        stream_ << "nullptr";
        return;
      }
    }
    auto view = static_cast<std::string_view>(str);
    stream_ << '"' << absl::Utf8SafeCEscape(view) << '"';
  }

  // Container-like range value is printed as [v1, v2, ...].
  template <typename T>
    requires std::ranges::range<T>
  void Print(T&& range, PriorityTag<6>) {
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
  void Print(T&& tuple, PriorityTag<5>) {
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
  void Print(T&& value, PriorityTag<4>) {
    stream_ << "{";
    std::apply(
        [&](const auto&... element) {
          size_t idx = 0;
          ((stream_ << (idx++ > 0 ? ", " : ""), Print(element)), ...);
        },
        AsTuple(std::forward<T>(value)));
    stream_ << "}";
  }

  // Raw pointer value is printed as "nullptr" or a hexadecimal integer.
  // Note that we cannot print the pointee, since it's valid to have a raw
  // pointer that points to one past the last element of the array, which is
  // invalid to dereference.
  template <typename T>
    requires(std::is_pointer_v<std::decay_t<T>>)
  void Print(T&& ptr, PriorityTag<3>) {
    if (ptr == nullptr) {
      stream_ << "nullptr";
    } else {
      stream_ << "0x" << std::hex << reinterpret_cast<uintptr_t>(ptr)
              << std::dec;
    }
  }

  // Smart pointer such as std::unique_ptr or std::shared_ptr is printed as a
  // raw pointer with its pointee.
  template <typename T>
    requires requires(T ptr) {
      typename std::decay_t<T>::element_type;
      { *ptr };  // NOLINT(readability/braces) - `;` is required here.
      { ptr.get() } -> std::convertible_to<const void*>;
    }
  void Print(T&& ptr, PriorityTag<2>) {
    const void* raw = ptr.get();
    Print(raw);
    if (raw != nullptr) {
      stream_ << " (";
      Print(*ptr);
      stream_ << ")";
    }
  }

  // Use operator<< if it's not specialized above. The requires statement is
  // added to make the error message more readable, otherwise compiler will list
  // all possible candidate overloads of operator<<.
  template <typename T>
    requires requires(std::ostream& os, T value) {
      { os << value } -> std::convertible_to<std::ostream&>;
    }
  void Print(T&& value, PriorityTag<1>) {
    stream_ << value;
  }

  // If the value is unprintable at all, simply shows "<unprintable TypeName>".
  template <typename T>
  void Print(T&& value, PriorityTag<0>) {
    stream_ << "<unprintable " << TypeName<std::decay_t<T>>() << ">";
  }

  std::stringstream stream_;
};

inline void DebugPrint(std::string_view file,
                       int line,
                       std::string_view func,
                       std::string_view args) {
  // ANSI color escape codes are applied for highlighting when stderr is a tty.
  constexpr char kBrightBlack[] = "\e[1;30m";
  constexpr char kBrightWhite[] = "\e[1;37m";
  constexpr char kReset[] = "\e[0m";
  static bool use_color = [] { return isatty(fileno(stderr)); }();
  auto may_colorize = [](const char* ansi) { return use_color ? ansi : ""; };

  // Output in the following format:
  // [file:line:func] k1 = v1, k2 = v2, ...
  std::cerr << may_colorize(kBrightBlack) << "[" << file << ":" << line << ":"
            << func << "] " << may_colorize(kBrightWhite) << args
            << may_colorize(kReset) << std::endl;
}

template <typename... T>
std::string Dump(std::initializer_list<std::string_view> names, T&&... values) {
  auto value_strings =
      std::initializer_list<std::string>{Printer::Stringify(values)...};
  std::stringstream stream;
  auto name_it = names.begin();
  auto value_it = value_strings.begin();
  for (size_t i = 0; i < names.size(); ++i) {
    stream << (i > 0 ? ", " : "") << *name_it++ << " = " << *value_it++;
  }
  return stream.str();
}

};  // namespace impl

// Converts the value to a pretty-printed string.
template <typename T>
std::string Stringify(T&& value) {
  return impl::Printer::Stringify(std::forward<T>(value));
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

// DBG_STRINGIFY(foo) => "foo"
#define DBG_STRINGIFY(x) #x

#define DUMP(...) \
  dbg::impl::Dump({DBG_MAP(DBG_STRINGIFY, __VA_ARGS__)}, __VA_ARGS__)

#if defined(NDEBUG) && !defined(DBG_FORCE_ENABLE)
#define DBG(...) static_assert(false, "DBG() MUST NOT be used in production")
#else
#define DBG(...) \
  dbg::impl::DebugPrint(__FILE__, __LINE__, __FUNCTION__, DUMP(__VA_ARGS__))
#endif

#endif  // COMMON_DUMP_H_
