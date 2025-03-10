/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/simple_model_builder.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>

#include "absl/random/random.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/kernels/register.h"

namespace tflite::cros {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Pair;
using ::testing::Pointee;
using BuiltinOpResolver = ops::builtin::BuiltinOpResolver;
using TensorArgs = SimpleModelBuilder::TensorArgs;

// Build a "d = a + b - c" model and verify the result.
TEST(Builder, AddSubModel) {
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

  std::unique_ptr<FlatBufferModel> model = mb.Build();
  ASSERT_NE(model, nullptr);

  // Load the model into an interpreter.
  BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  ASSERT_NE(interpreter, nullptr);

  // Check input/output names are propagated.
  const std::vector<int>& inputs = interpreter->inputs();
  ASSERT_EQ(inputs.size(), 3);
  EXPECT_STREQ(interpreter->GetInputName(0), "a");
  EXPECT_STREQ(interpreter->GetInputName(1), "b");
  EXPECT_STREQ(interpreter->GetInputName(2), "c");

  const std::vector<int>& outputs = interpreter->outputs();
  ASSERT_EQ(outputs.size(), 1);
  EXPECT_STREQ(interpreter->GetOutputName(0), "d");

  // Check signature definition is filled correctly.
  const char* key = SimpleModelBuilder::kSignatureKey;
  ASSERT_THAT(interpreter->signature_keys(), ElementsAre(Pointee(Eq(key))));
  EXPECT_NE(interpreter->GetSignatureRunner(key), nullptr);
  EXPECT_THAT(interpreter->signature_inputs(key),
              ElementsAre(Pair("a", inputs[0]), Pair("b", inputs[1]),
                          Pair("c", inputs[2])));
  EXPECT_THAT(interpreter->signature_outputs(key),
              ElementsAre(Pair("d", outputs[0])));

  // Check all tensors have the expected type and shape.
  ASSERT_EQ(interpreter->tensors_size(), 5);
  for (int i = 0; i < 5; ++i) {
    TfLiteTensor* tensor = interpreter->tensor(i);
    ASSERT_EQ(tensor->type, kTfLiteFloat32);
    ASSERT_TRUE(TfLiteIntArrayEqualsArray(tensor->dims, base_args.shape.size(),
                                          base_args.shape.data()));
  }

  // Allocate tensors and fill inputs with random data.
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  int n = 1;
  for (int dim : base_args.shape) {
    n *= dim;
  }

  absl::BitGen gen;
  std::vector<float> expected_output(n);
  for (int i = 0; i < n; ++i) {
    float a = absl::Uniform(gen, 0.0, 1.0);
    float b = absl::Uniform(gen, 0.0, 1.0);
    float c = absl::Uniform(gen, 0.0, 1.0);
    float d = a + b - c;
    interpreter->typed_input_tensor<float>(0)[i] = a;
    interpreter->typed_input_tensor<float>(1)[i] = b;
    interpreter->typed_input_tensor<float>(2)[i] = c;
    expected_output[i] = d;
  }

  // Run the inference and check the output to verify the operators in model.
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(interpreter->typed_output_tensor<float>(0)[i],
                    expected_output[i]);
  }
}

// Build a "y[i] = x[i] + i" model and verify the result.
TEST(Builder, ModelWithBuffer) {
  int n = 10;

  SimpleModelBuilder mb;
  int x = mb.AddInput({.name = "x", .type = kTfLiteFloat32, .shape = {n}});
  int y = mb.AddOutput({.name = "y", .type = kTfLiteFloat32, .shape = {n}});

  std::vector<float> i_data(n);
  std::iota(i_data.begin(), i_data.end(), 0.0);
  std::vector<uint8_t> i_buffer(n * sizeof(float));
  memcpy(i_buffer.data(), i_data.data(), i_buffer.size());
  int i = mb.AddInternalTensor({
      .name = "i",
      .type = kTfLiteFloat32,
      .shape = {n},
      .buffer = mb.AddBuffer(i_buffer),
  });

  mb.AddOperator<TfLiteAddParams>({
      .op = kTfLiteBuiltinAdd,
      .inputs = {x, i},
      .outputs = {y},
  });
  std::unique_ptr<FlatBufferModel> model = mb.Build();
  ASSERT_NE(model, nullptr);

  // Load the model into an interpreter.
  BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  ASSERT_NE(interpreter, nullptr);

  ASSERT_EQ(interpreter->inputs().size(), 1);
  ASSERT_EQ(interpreter->outputs().size(), 1);

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

TEST(Builder, BuildTwice) {
  SimpleModelBuilder mb;
  int x = mb.AddInput({.name = "x", .type = kTfLiteFloat32, .shape = {1}});
  int y = mb.AddOutput({.name = "y", .type = kTfLiteFloat32, .shape = {1}});
  mb.AddOperator<TfLiteAddParams>({
      .op = kTfLiteBuiltinAdd,
      .inputs = {x, x},
      .outputs = {y},
  });
  EXPECT_NE(mb.Build(), nullptr);
  EXPECT_NE(mb.Build(), nullptr);
}

TEST(Builder, SaveAndInvokeTwice) {
  // Construct the model.
  SimpleModelBuilder mb;
  int x = mb.AddInput({.name = "x", .type = kTfLiteFloat32, .shape = {1}});
  int y = mb.AddOutput({.name = "y", .type = kTfLiteFloat32, .shape = {1}});
  mb.AddOperator<TfLiteAddParams>({
      .op = kTfLiteBuiltinAdd,
      .inputs = {x, x},
      .outputs = {y},
  });

  const std::filesystem::path temp_dir = testing::TempDir();
  for (const std::string_view filename : {"model1.tflite", "model2.tflite"}) {
    // Save the model to the temporary directory with the given name.
    std::filesystem::path temp_file = temp_dir / filename;
    mb.SaveTo(temp_file);

    // Read back the model.
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::VerifyAndBuildFromFile(temp_file.c_str());
    EXPECT_NE(model, nullptr);

    // Load the model into an interpreter.
    BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    ASSERT_NE(interpreter, nullptr);

    // Allocate tensors and fill inputs with random data.
    absl::BitGen gen;
    ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
    float a = absl::Uniform(gen, 0.0, 1.0);
    interpreter->typed_input_tensor<float>(0)[0] = a;

    // Run the inference and check the output to verify the operators in model.
    ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
    EXPECT_FLOAT_EQ(interpreter->typed_output_tensor<float>(0)[0], a + a);
  }
}

}  // namespace tflite::cros

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
