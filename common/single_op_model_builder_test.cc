/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/single_op_model_builder.h"

#include <gtest/gtest.h>

#include <memory>

#include "absl/random/random.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/kernels/register.h"

namespace tflite::cros::tests {

using BuiltinOpResolver = ops::builtin::BuiltinOpResolver;

TEST(Builder, Build) {
  const int n = 5;

  // A model with output[i] = input[i] + i.
  std::unique_ptr<FlatBufferModel> model =
      SingleOpModelBuilder()
          .AddInput(kTfLiteFloat32, {n})
          .AddConstInput<float>(kTfLiteFloat32, {n}, {0, 1, 2, 3, 4})
          .AddOutput(kTfLiteFloat32, {n})
          .AddOperator<TfLiteAddParams>(kTfLiteBuiltinAdd)
          .Build();

  // Load the model into an interpreter.
  BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  ASSERT_NE(interpreter, nullptr);

  // Check inputs and ouput tensor sizes.
  const std::vector<int>& inputs = interpreter->inputs();
  ASSERT_EQ(inputs.size(), 2);
  const std::vector<int>& outputs = interpreter->outputs();
  ASSERT_EQ(outputs.size(), 1);

  // Allocate tensors and fill inputs with random data.
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  absl::BitGen gen;
  std::vector<float> expected_output(n);
  for (int i = 0; i < n; ++i) {
    float x = absl::Uniform(gen, 0.0, 1.0);
    interpreter->typed_input_tensor<float>(0)[i] = x;
    expected_output[i] = x + i;
  }

  // Run the inference and check the output to verify the operators in model.
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(interpreter->typed_output_tensor<float>(0)[i],
                    expected_output[i]);
  }
}

}  // namespace tflite::cros::tests

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
