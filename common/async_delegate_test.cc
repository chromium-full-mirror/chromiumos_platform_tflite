/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "absl/random/random.h"
#include "android/hardware_buffer.h"
#include "common/async_driver.h"
#include "common/scoped_ahwb.h"
#include "common/simple_model_builder.h"
#include "common/test_util/fp16_compare.h"
#include "common/test_util/stable_delegate_env.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"

namespace tflite::cros::tests {

StableDelegateEnvironment* g_env = nullptr;

class AsyncDelegateTest : public testing::TestWithParam<int> {};

TEST_P(AsyncDelegateTest, Inference) {
  auto ahwb_desc_with_size = [](uint32_t size) {
    return AHardwareBuffer_Desc{
        .width = size,
        .height = 1,
        .layers = 1,
        .format = AHARDWAREBUFFER_FORMAT_BLOB,
        .usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                 AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
        .stride = size,
    };
  };

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
    auto a_attrs = driver->GetInputBufferAttributes("a");
    auto b_attrs = driver->GetInputBufferAttributes("b");
    auto c_attrs = driver->GetInputBufferAttributes("c");
    auto d_attrs = driver->GetOutputBufferAttributes("d");
    EXPECT_TRUE(a_attrs.ok());
    EXPECT_TRUE(b_attrs.ok());
    EXPECT_TRUE(c_attrs.ok());
    EXPECT_TRUE(d_attrs.ok());
    EXPECT_EQ(*a_attrs->size, n * sizeof(float));
    EXPECT_EQ(*b_attrs->size, n * sizeof(float));
    EXPECT_EQ(*c_attrs->size, n * sizeof(float));
    EXPECT_EQ(*d_attrs->size, n * sizeof(float));

    ScopedAHardwareBuffer a_buffer(ahwb_desc_with_size(*a_attrs->size));
    ScopedAHardwareBuffer b_buffer(ahwb_desc_with_size(*b_attrs->size));
    ScopedAHardwareBuffer c_buffer(ahwb_desc_with_size(*c_attrs->size));
    ScopedAHardwareBuffer d_buffer(ahwb_desc_with_size(*d_attrs->size));
    driver->SetInputTensorBuffer("a", std::move(a_buffer));
    driver->SetInputTensorBuffer("b", std::move(b_buffer));
    driver->SetInputTensorBuffer("c", std::move(c_buffer));
    driver->SetOutputTensorBuffer("d", std::move(d_buffer));

    absl::BitGen gen;
    std::vector<float> a_data(n);
    std::vector<float> b_data(n);
    std::vector<float> c_data(n);
    std::vector<float> d_data(n);
    for (int i = 0; i < n; ++i) {
      // Limit the input precision to float16 since we are comparing the
      // computation in fp16.
      const _Float16 a = absl::Uniform(gen, 0.0, 1.0);
      const _Float16 b = absl::Uniform(gen, 0.0, 1.0);
      const _Float16 c = absl::Uniform(gen, 0.0, 1.0);
      // If we do not cast back to float16 first, the result will be casted back
      // to float16 only after the full computation.
      const _Float16 d = static_cast<_Float16>(a + b) - c;

      a_data[i] = a;
      b_data[i] = b;
      c_data[i] = c;
      d_data[i] = d;
    }
    EXPECT_EQ(driver->SetInputTensorData("a", a_data), kTfLiteOk);
    EXPECT_EQ(driver->SetInputTensorData("b", b_data), kTfLiteOk);
    EXPECT_EQ(driver->SetInputTensorData("c", c_data), kTfLiteOk);

    ASSERT_EQ(driver->AllocateBuffers(), kTfLiteOk);
    ASSERT_EQ(driver->Invoke(), kTfLiteOk);

    auto output = driver->GetOutputTensorData<float>("d");
    EXPECT_THAT(output, testing::ElementsAreArray(ArrayFp16Eq(d_data)));
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
