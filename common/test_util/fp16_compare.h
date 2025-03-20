/*
 * Copyright 2025 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef COMMON_TEST_UTIL_FP16_COMPARE_H_
#define COMMON_TEST_UTIL_FP16_COMPARE_H_

#include <gmock/gmock-matchers.h>
#include <span>
#include <vector>

// Similar to FloatEq and DoubleEq, this matcher use ULP(Units in the Last
// Place)-based comparison to compare the given floating number in the fp16
// sense. In other words, this automatically picks a reasonable error bound
// based on the absolute value of the expected value.
testing::Matcher<float> Fp16Eq(float expected);

// A helper function to help convert the provided values into a series of gmock
// matchers.
std::vector<testing::Matcher<float>> ArrayFp16Eq(std::span<float> values);

#endif  // COMMON_TEST_UTIL_FP16_COMPARE_H_
