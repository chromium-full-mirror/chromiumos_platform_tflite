/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <gtest/gtest.h>

#include <vector>

#include "absl/random/random.h"
#include "common/async_driver.h"
#include "common/simple_model_builder.h"
#include "common/test_util/stable_delegate_env.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"

namespace tflite::cros::tests {

StableDelegateEnvironment* g_env = nullptr;

// Use a relaxed tolerance when comparing the computation results, as the
// primary goal of this test is to verify the asynchronous kernel API flow.
// Some delegates may internally use fp16 for fp32 arithmetic, making the
// default EXPECT_FLOAT_EQ too strict for such cases.
// TODO(shik): If we need checking fp16 results in more places, consider
// implementing a generic fp16 GMock matcher that computes the difference in
// ULPs (Units in the Last Place), similar to the current implementation of
// EXPECT_FLOAT_EQ, to provide a more precise comparison.
constexpr float kEps = 1e-2;

class AsyncDelegateTest : public testing::TestWithParam<int> {};

TEST_P(AsyncDelegateTest, Inference) {
  using TensorArgs = SimpleModelBuilder::TensorArgs;
  const TensorArgs base_args = {
      .type = kTfLiteFloat32,
      .shape = {2, 3},
  };
  auto arg_with_name = [&](const char* name) {
    TensorArgs args = base_args;
    args.name = name;
    return args;
  };

  SimpleModelBuilder mb;

  int a = mb.AddInput(arg_with_name("a"));
  int b = mb.AddInput(arg_with_name("b"));
  int c = mb.AddInput(arg_with_name("c"));
  int d = mb.AddOutput(arg_with_name("d"));

  int a_plus_b = mb.AddInternalTensor(arg_with_name("a_plus_b"));
  mb.AddOperator<TfLiteAddParams>({
      .op = kTfLiteBuiltinAdd,
      .inputs = {a, b},
      .outputs = {a_plus_b},
  });
  mb.AddOperator<TfLiteSubParams>({
      .op = kTfLiteBuiltinSub,
      .inputs = {a_plus_b, c},
      .outputs = {d},
  });

  auto model = mb.Build();
  ASSERT_NE(model, nullptr);

  TfLiteDelegatePtr delegate = g_env->CreateDelegate();
  ASSERT_NE(delegate, nullptr);

  auto driver = AsyncDriver::Create(std::move(delegate), std::move(model));
  ASSERT_NE(driver, nullptr);

  ASSERT_EQ(driver->Prepare(), kTfLiteOk);

  int n = 1;
  for (int dim : base_args.shape) {
    n *= dim;
  }

  const int num_iterations = GetParam();
  for (int iteration = 0; iteration < num_iterations; ++iteration) {
    absl::BitGen gen;
    std::vector<float> a_data(n);
    std::vector<float> b_data(n);
    std::vector<float> c_data(n);
    std::vector<float> d_data(n);
    for (int i = 0; i < n; ++i) {
      a_data[i] = absl::Uniform(gen, 0.0, 1.0);
      b_data[i] = absl::Uniform(gen, 0.0, 1.0);
      c_data[i] = absl::Uniform(gen, 0.0, 1.0);
      d_data[i] = a_data[i] + b_data[i] - c_data[i];
    }
    EXPECT_EQ(driver->SetInputTensor("a", a_data), kTfLiteOk);
    EXPECT_EQ(driver->SetInputTensor("b", b_data), kTfLiteOk);
    EXPECT_EQ(driver->SetInputTensor("c", c_data), kTfLiteOk);

    ASSERT_EQ(driver->Invoke(), kTfLiteOk);

    auto output = driver->GetOutputTensor<float>("d");
    for (int i = 0; i < n; ++i) {
      EXPECT_NEAR(output[i], d_data[i], kEps);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(DifferentIterations,
                         AsyncDelegateTest,
                         testing::Values(1, 10));

}  // namespace tflite::cros::tests

using tflite::cros::tests::g_env;

int main(int argc, char** argv) {
  g_env = new StableDelegateEnvironment();
  if (!g_env->InitFromCommandLine(&argc, argv)) {
    delete g_env;
    return EXIT_FAILURE;
  }

  testing::InitGoogleTest(&argc, argv);
  // GoogleTest takes the ownership of g_env.
  ::testing::AddGlobalTestEnvironment(g_env);
  return RUN_ALL_TESTS();
}
