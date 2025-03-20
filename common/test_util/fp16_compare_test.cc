/*
 * Copyright 2025 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <gtest/gtest.h>

#include "common/test_util/fp16_compare.h"

// Adapted from
// https://github.com/tensorflow/tensorflow/blob/b0b4ec040b9b8f4b7fefdb5d7c3695349dae1d9d/tensorflow/lite/kernels/test_util_test.cc#L124-L177
TEST(Fp16CompareTest, Eq) {
  // Minimum number that FP16 could represent. When the expected is a subnormal
  // FP16 number, i.e. its exponent is the minimum, -14, this is the ULP used.
  // Given minimum exponent is -14 and fraction has 10 bits, the true minimum
  // of FP16 is 2^(-14-10) = 2^(-24).
  constexpr float fp16_true_min = 0x1p-24;

  // FP16 has 10 bits for tha fraction part, so the ULP error is between
  // 2^-10 / 2 and 2^-10 relative error. Since we emulate a FP16 ULP by 2^13
  // FP32 ULPs, rounding error is negligible. So the tolerated relative error
  // of 4 ULPs is roughly between 4 * 2^-10 / 2 and 4 * 2^-10 ~= 0.195% and
  // 0.39%.
  // 0.15% relative error should be tolerated by 4 ULPs in FP16.
  EXPECT_THAT(0.1f, Fp16Eq(0.10015));
  EXPECT_THAT(100.f, Fp16Eq(100.15));
  EXPECT_THAT(-1.f, Fp16Eq(-1.0015));
  EXPECT_THAT(0.f, Fp16Eq(4 * fp16_true_min));
  EXPECT_THAT(0.f, Fp16Eq(-4 * fp16_true_min));
  // NaN equals to NaN.
  EXPECT_THAT(std::nanf(""), Fp16Eq(std::nanf("")));

  // 0.4% relative error should not be tolerated by 4 ULPs in FP16.
  EXPECT_THAT(0.1f, Not(Fp16Eq(0.1004)));
  EXPECT_THAT(100.f, Not(Fp16Eq(100.4)));
  EXPECT_THAT(-1.f, Not(Fp16Eq(-1.004)));
  EXPECT_THAT(0.f, Not(Fp16Eq(5 * fp16_true_min)));
  EXPECT_THAT(0.f, Not(Fp16Eq(-5 * fp16_true_min)));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
