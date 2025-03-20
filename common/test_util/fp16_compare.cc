/*
 * Copyright 2025 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/test_util/fp16_compare.h"

#include <gmock/gmock-matchers.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <ostream>
#include <vector>

namespace {

// Converts an integer from the sign-and-magnitude representation to
// the biased representation.  More precisely, let N be 2 to the
// power of (kBitCount - 1), an integer x is represented by the
// unsigned number x + N.
//
// For instance,
//
//   -N + 1 (the most negative number representable using
//          sign-and-magnitude) is represented by 1;
//   0      is represented by N; and
//   N - 1  (the biggest number representable using
//          sign-and-magnitude) is represented by 2N - 1.
//
// Read https://en.wikipedia.org/wiki/Signed_number_representations
// for more details on signed number representations.
uint32_t SignAndMagnitudeToBiased(uint32_t sam) {
  constexpr uint32_t kSignBitMask = 1u << 31;
  if (kSignBitMask & sam) {
    // sam represents a negative number.
    return ~sam + 1;
  } else {
    // sam represents a positive number.
    return kSignBitMask | sam;
  }
}

// Given two numbers in the sign-and-magnitude representation,
// returns the distance between them as an unsigned number.
uint32_t DistanceBetweenSignAndMagnitudeNumbers(uint32_t sam1, uint32_t sam2) {
  uint32_t biased1 = SignAndMagnitudeToBiased(sam1);
  uint32_t biased2 = SignAndMagnitudeToBiased(sam2);
  return (biased1 >= biased2) ? (biased1 - biased2) : (biased2 - biased1);
}

// Returns true if and only if lhs is at most max_ulps ULP's away from rhs.
// In particular, this function:
//
//   - returns true if both numbers are NAN.
//   - returns false if exact one of numbers is NAN.
//   - treats really large numbers as almost equal to infinity.
//   - thinks +0.0 and -0.0 are 0 ULP's apart.
bool AlmostEquals(float lhs, float rhs, uint32_t max_ulps) {
  if (std::isnan(lhs) || std::isnan(rhs)) {
    return std::isnan(lhs) && std::isnan(rhs);
  }

  return DistanceBetweenSignAndMagnitudeNumbers(std::bit_cast<uint32_t>(lhs),
                                                std::bit_cast<uint32_t>(rhs)) <=
         max_ulps;
}

class Fp16EqMatcher {
 public:
  using is_gtest_matcher = void;

  explicit Fp16EqMatcher(float expected) : expected(expected) {}

  // Compare the actual value and the expected value stored during
  // construction. Omit the explanation opportunity and rely on
  // Describe{Negation}To.
  bool MatchAndExplain(float actual, std::ostream*) const {
    // FP16 only has 10 bits precision while FP32 has 23 bits precision. Thus,
    // to check if results of FP16 are almost equal, we could check the result
    // is within 4 * 2^13 ULPs of FP32, which equals to 4 ULPs of FP16.
    constexpr uint32_t fp16_ulps_in_fp32 = 4 * (1 << 13);
    // The minimum exponent of FP16 is 2^-14, which means the minimum ULP of
    // FP16 is 2^-24. Therefore, when expected is less than 2^-14, i.e. a
    // subnormal FP16 number, the minimum ULP of FP16 should be used instead of
    // ULP of FP32.
    if (std::abs(expected) < 0x1p-14) {
      return std::abs(actual - expected) <= 4 * 0x1p-24;
    }
    return AlmostEquals(actual, expected, fp16_ulps_in_fp32);
  }

  // Describes the property of a value matching this matcher.
  void DescribeTo(std::ostream* os) const { *os << "is " << expected; }

  // Describes the property of a value NOT matching this matcher.
  void DescribeNegationTo(std::ostream* os) const {
    *os << "is not " << expected;
  }

 private:
  const float expected;
};

}  // namespace

testing::Matcher<float> Fp16Eq(float expected) {
  return Fp16EqMatcher(expected);
}

std::vector<testing::Matcher<float>> ArrayFp16Eq(std::span<float> values) {
  std::vector<testing::Matcher<float>> matchers;
  matchers.reserve(values.size());
  std::ranges::transform(values, std::back_inserter(matchers), Fp16Eq);
  return matchers;
}
