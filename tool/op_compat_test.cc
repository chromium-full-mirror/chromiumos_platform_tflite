/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/single_op_model_builder.h"
#include "common/test_util/stable_delegate_env.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/tools/utils.h"

StableDelegateEnvironment* g_env = nullptr;

namespace tflite::cros::tests {

size_t NumElements(const TfLiteTensor* tensor) {
  size_t n = 1;
  for (int i = 0; i < tensor->dims->size; i++) {
    n *= tensor->dims->data[i];
  }
  return n;
}

template <typename T>
std::vector<double> CopyDataAsDoubleVector(const T* src, size_t n) {
  // Note that we implicitly convert the values from T to double here.
  return std::vector<double>(src, src + n);
}

std::vector<double> CopyTensorAsDoubleVector(const TfLiteTensor* tensor) {
  int n = NumElements(tensor);
  switch (tensor->type) {
    case kTfLiteFloat32:
      return CopyDataAsDoubleVector(tensor->data.f, n);
    case kTfLiteFloat64:
      return CopyDataAsDoubleVector(tensor->data.f64, n);
    case kTfLiteInt8:
      return CopyDataAsDoubleVector(tensor->data.int8, n);
    case kTfLiteUInt8:
      return CopyDataAsDoubleVector(tensor->data.uint8, n);
    case kTfLiteInt16:
      return CopyDataAsDoubleVector(tensor->data.i16, n);
    case kTfLiteUInt16:
      return CopyDataAsDoubleVector(tensor->data.ui16, n);
    case kTfLiteInt32:
      return CopyDataAsDoubleVector(tensor->data.i32, n);
    case kTfLiteUInt32:
      return CopyDataAsDoubleVector(tensor->data.u32, n);
    case kTfLiteBool:
      return CopyDataAsDoubleVector(tensor->data.b, n);
    default:
      ADD_FAILURE() << "Unexpected tensor type "
                    << TfLiteTypeGetName(tensor->type);
      return {};
  }
}

bool IsFullyDelegated(Interpreter* interpreter) {
  for (int index : interpreter->execution_plan()) {
    TfLiteNode node = interpreter->node_and_registration(index)->first;
    if (node.delegate == nullptr) {
      return false;
    }
  }
  return true;
}

using ::testing::DoubleNear;
using ::testing::Pointwise;
// A relaxed tolerance when comparing the computation results since delegates
// may internally use fp16 for fp32 arithmetic, and this test is mainly used for
// checking operator compatibility.
constexpr float kEps = 1e-2;

class OpTest : public testing::Test {
 protected:
  void TestModel(const SingleOpModelBuilder& model_builder) {
    auto delegate = g_env->CreateDelegate();
    ASSERT_NE(delegate, nullptr);

    // Run model with/without the delegate and compare the result.
    std::vector<std::vector<double>> actual, expected;
    random_input_data_.clear();
    RunModel(model_builder.Build(), delegate.get(), actual);
    RunModel(model_builder.Build(), nullptr, expected);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); i++) {
      EXPECT_THAT(actual[i], Pointwise(DoubleNear(kEps), expected[i]));
    }
  }

 private:
  void RunModel(std::unique_ptr<FlatBufferModel> model,
                TfLiteDelegate* delegate,
                std::vector<std::vector<double>>& output_data) {
    ASSERT_NE(model, nullptr);

    // Build interpreter.
    ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
    InterpreterBuilder builder(*model, resolver);
    if (delegate != nullptr) {
      builder.AddDelegate(delegate);
    }
    std::unique_ptr<Interpreter> interpreter;
    ASSERT_EQ(builder(&interpreter), kTfLiteOk);

    // Allocate tensors and fill inputs with random data.
    ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    for (int input : interpreter->inputs()) {
      TfLiteTensor* tensor = interpreter->tensor(input);
      if (tensor->allocation_type == kTfLiteMmapRo) {
        continue;
      }

      // Reuse the same random input from the previous run if exists.
      utils::InputTensorData& data = random_input_data_[input];
      if (data.data == nullptr) {
        float low = 0, high = 0;
        utils::GetDataRangesForType(tensor->type, &low, &high);
        data = utils::CreateRandomTensorData(*tensor, low, high);
      }
      ASSERT_EQ(data.bytes, tensor->bytes);
      memcpy(tensor->data.data, data.data.get(), data.bytes);
    }

    // Run the inference and check the execution is fully delegated.
    ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
    if (delegate != nullptr) {
      EXPECT_TRUE(IsFullyDelegated(interpreter.get()));
    }

    // Copy the output data.
    size_t num_out = interpreter->outputs().size();
    output_data.resize(num_out);
    for (size_t i = 0; i < num_out; i++) {
      output_data[i] = CopyTensorAsDoubleVector(interpreter->output_tensor(i));
    }
  }

  std::map<int, utils::InputTensorData> random_input_data_;
};

// TODO(shik): Extract common logic for testing element-wise unary/binary
// operators as a fixture.
TEST_F(OpTest, Abs) {
  TestModel(SingleOpModelBuilder()
                .AddInput(kTfLiteFloat32, {2, 3})
                .AddOutput(kTfLiteFloat32, {2, 3})
                .AddOperator(kTfLiteBuiltinAbs));
}

TEST_F(OpTest, Add) {
  TestModel(SingleOpModelBuilder()
                .AddInput(kTfLiteFloat32, {2, 3})
                .AddInput(kTfLiteFloat32, {2, 3})
                .AddOutput(kTfLiteFloat32, {2, 3})
                .AddOperator<TfLiteAddParams>(kTfLiteBuiltinAdd));
}

TEST_F(OpTest, Sub) {
  TestModel(SingleOpModelBuilder()
                .AddInput(kTfLiteFloat32, {2, 3})
                .AddInput(kTfLiteFloat32, {2, 3})
                .AddOutput(kTfLiteFloat32, {2, 3})
                .AddOperator<TfLiteSubParams>(kTfLiteBuiltinSub));
}

}  // namespace tflite::cros::tests

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
