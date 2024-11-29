/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fstream>

#include "absl/strings/str_cat.h"
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

struct TestConfig {
  std::vector<Flag> GetFlags() {
    std::vector<Flag> flags = {
        Flag::CreateFlag(
            "dump_model_dir", &dump_model_dir,
            "Where to save the test models. Empty means do not save at all."),
    };
    return flags;
  }

  std::string dump_model_dir = "";
} g_config;

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

class ModelTest : public testing::Test {
 protected:
  void SetRandomInputRange(float low, float high) {
    random_input_range_ = std::make_pair(low, high);
  }

  void UsePositiveInput() {
    // A handle chosen positive random range that should work for most
    // operators. Note that we exclude near zero positive values intentionally
    // to avoid potential numerical issues like log(~0).
    SetRandomInputRange(1, 10);
  }

  void TestModel(const SingleOpModelBuilder& model_builder) {
    if (!g_config.dump_model_dir.empty()) {
      DumpModel(model_builder.Build());
    }

    // Call GetDelegate() instead of CreateDelegate() here to reuse the delegate
    // instance and reduce the overhead of creating/destorying the delegate
    // itself.
    auto delegate = g_env->GetDelegate();
    ASSERT_NE(delegate, nullptr);

    // Run model with/without the delegate and compare the result.
    std::vector<std::vector<double>> actual, expected;
    random_input_data_.clear();
    RunModel(model_builder.Build(), delegate, actual);
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
        if (random_input_range_.has_value()) {
          low = random_input_range_->first;
          high = random_input_range_->second;
        } else {
          utils::GetDataRangesForType(tensor->type, &low, &high);
        }
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

  void DumpModel(std::unique_ptr<FlatBufferModel> model) {
    ASSERT_FALSE(g_config.dump_model_dir.empty());

    const testing::TestInfo* info =
        testing::UnitTest::GetInstance()->current_test_info();
    // Save into {dir}/{suite}.{test}-{counter}.tflite.
    std::string path =
        absl::StrCat(g_config.dump_model_dir, "/", info->test_suite_name(), ".",
                     info->name(), "-", dumped_model_counter_, ".tflite");
    std::ofstream out(path, std::ios::out | std::ios::binary);
    const Allocation* alloc = model->allocation();
    out.write(static_cast<const char*>(alloc->base()), alloc->bytes());
    LOGF(INFO) << "Save to " << path;
    dumped_model_counter_++;
  }

  std::map<int, utils::InputTensorData> random_input_data_;
  std::optional<std::pair<float, float>> random_input_range_;
  int dumped_model_counter_ = 0;
};

class OpTest : public ModelTest {
 protected:
  template <typename T = std::monostate>
  void TestUnaryOp(TfLiteBuiltinOperator op, TfLiteType type, T params = {}) {
    TestOp(op, type, params, /*num_inputs=*/1);
  }

  template <typename T = std::monostate>
  void TestBinaryOp(TfLiteBuiltinOperator op, TfLiteType type, T params = {}) {
    TestOp(op, type, params, /*num_inputs=*/2);
  }

 private:
  template <typename T>
  void TestOp(TfLiteBuiltinOperator op,
              TfLiteType type,
              T params,
              int num_inputs) {
    // TODO(shik): Parameterize tensor ranks and make it controllable as
    // command-line flags. For now, test only 4D which should be supported
    // universally by all delegates we care about.
    const std::vector<int> shape = {1, 2, 3, 4};
    SingleOpModelBuilder builder;
    for (int i = 0; i < num_inputs; i++) {
      builder.AddInput(type, shape);
    }
    TestModel(builder.AddOutput(type, shape).AddOperator<T>(op, params));
  }
};

TEST_F(OpTest, Abs) {
  TestUnaryOp(kTfLiteBuiltinAbs, kTfLiteFloat32);
}

TEST_F(OpTest, Cos) {
  TestUnaryOp(kTfLiteBuiltinCos, kTfLiteFloat32);
}

TEST_F(OpTest, Elu) {
  TestUnaryOp(kTfLiteBuiltinElu, kTfLiteFloat32);
}

TEST_F(OpTest, Exp) {
  TestUnaryOp(kTfLiteBuiltinExp, kTfLiteFloat32);
}

TEST_F(OpTest, Floor) {
  TestUnaryOp(kTfLiteBuiltinFloor, kTfLiteFloat32);
}

TEST_F(OpTest, Gelu) {
  TestUnaryOp<TfLiteGeluParams>(kTfLiteBuiltinGelu, kTfLiteFloat32,
                                {.approximate = false});
  TestUnaryOp<TfLiteGeluParams>(kTfLiteBuiltinGelu, kTfLiteFloat32,
                                {.approximate = true});
}

TEST_F(OpTest, LeakyRelu) {
  TestUnaryOp<TfLiteLeakyReluParams>(kTfLiteBuiltinLeakyRelu, kTfLiteFloat32,
                                     {.alpha = 0.8});
}

TEST_F(OpTest, Log) {
  UsePositiveInput();
  TestUnaryOp(kTfLiteBuiltinLog, kTfLiteFloat32);
}

TEST_F(OpTest, Logistic) {
  TestUnaryOp(kTfLiteBuiltinLogistic, kTfLiteFloat32);
}

TEST_F(OpTest, Neg) {
  TestUnaryOp(kTfLiteBuiltinNeg, kTfLiteFloat32);
}

TEST_F(OpTest, Relu) {
  TestUnaryOp(kTfLiteBuiltinRelu, kTfLiteFloat32);
}

TEST_F(OpTest, Relu6) {
  SetRandomInputRange(-1, 7);
  TestUnaryOp(kTfLiteBuiltinRelu6, kTfLiteFloat32);
}

TEST_F(OpTest, ReluN1To1) {
  SetRandomInputRange(-2, 2);
  TestUnaryOp(kTfLiteBuiltinReluN1To1, kTfLiteFloat32);
}

TEST_F(OpTest, Rsqrt) {
  UsePositiveInput();
  TestUnaryOp(kTfLiteBuiltinRsqrt, kTfLiteFloat32);
}

TEST_F(OpTest, Sign) {
  TestUnaryOp(kTfLiteBuiltinSign, kTfLiteFloat32);
}

TEST_F(OpTest, Sin) {
  TestUnaryOp(kTfLiteBuiltinSin, kTfLiteFloat32);
}

TEST_F(OpTest, Sqrt) {
  UsePositiveInput();
  TestUnaryOp(kTfLiteBuiltinSqrt, kTfLiteFloat32);
}

TEST_F(OpTest, Square) {
  TestUnaryOp(kTfLiteBuiltinSquare, kTfLiteFloat32);
}

TEST_F(OpTest, Tanh) {
  TestUnaryOp(kTfLiteBuiltinTanh, kTfLiteFloat32);
}

TEST_F(OpTest, Add) {
  TestBinaryOp<TfLiteAddParams>(kTfLiteBuiltinAdd, kTfLiteFloat32);
}

TEST_F(OpTest, Div) {
  UsePositiveInput();
  TestBinaryOp<TfLiteDivParams>(kTfLiteBuiltinDiv, kTfLiteFloat32);
}

TEST_F(OpTest, FloorDiv) {
  UsePositiveInput();
  TestBinaryOp(kTfLiteBuiltinFloorDiv, kTfLiteFloat32);
}

TEST_F(OpTest, FloorMod) {
  UsePositiveInput();
  TestBinaryOp(kTfLiteBuiltinFloorMod, kTfLiteFloat32);
}

TEST_F(OpTest, Maximum) {
  TestBinaryOp(kTfLiteBuiltinMaximum, kTfLiteFloat32);
}

TEST_F(OpTest, Minimum) {
  TestBinaryOp(kTfLiteBuiltinMinimum, kTfLiteFloat32);
}

TEST_F(OpTest, Mul) {
  TestBinaryOp(kTfLiteBuiltinMul, kTfLiteFloat32);
}

TEST_F(OpTest, Pow) {
  // The default positive range is too big for power operation.
  SetRandomInputRange(1, 5);
  TestBinaryOp(kTfLiteBuiltinPow, kTfLiteFloat32);
}

TEST_F(OpTest, SquaredDifference) {
  TestBinaryOp(kTfLiteBuiltinSquaredDifference, kTfLiteFloat32);
}

TEST_F(OpTest, Sub) {
  TestBinaryOp<TfLiteSubParams>(kTfLiteBuiltinSub, kTfLiteFloat32);
}

TEST_F(OpTest, Conv2d) {
  SingleOpModelBuilder mb;
  // The input tensors are input, filter and an optional bias.
  mb.AddInput(kTfLiteFloat32, {2, 5, 4, 3});
  mb.AddInput(kTfLiteFloat32, {3, 3, 2, 3});
  mb.AddConstInput<float>(kTfLiteFloat32, {3}, {1, 2, 3});
  mb.AddOutput(kTfLiteFloat32, {2, 5, 4, 3});
  mb.AddOperator<TfLiteConvParams>(kTfLiteBuiltinConv2d,
                                   {
                                       .padding = kTfLitePaddingSame,
                                       .stride_width = 1,
                                       .stride_height = 1,
                                       .activation = kTfLiteActNone,
                                       .dilation_width_factor = 1,
                                       .dilation_height_factor = 1,
                                   });
  TestModel(mb);
}

TEST_F(OpTest, DepthwiseConv2d) {
  SingleOpModelBuilder mb;
  // The input tensors are input, filter and an optional bias.
  mb.AddInput(kTfLiteFloat32, {2, 5, 4, 3});
  mb.AddInput(kTfLiteFloat32, {1, 3, 2, 3});
  mb.AddConstInput<float>(kTfLiteFloat32, {3}, {1, 2, 3});
  mb.AddOutput(kTfLiteFloat32, {2, 5, 4, 3});
  mb.AddOperator<TfLiteDepthwiseConvParams>(kTfLiteBuiltinDepthwiseConv2d,
                                            {
                                                .padding = kTfLitePaddingSame,
                                                .stride_width = 1,
                                                .stride_height = 1,
                                                .depth_multiplier = 1,
                                                .activation = kTfLiteActNone,
                                                .dilation_width_factor = 1,
                                                .dilation_height_factor = 1,
                                            });
  TestModel(mb);
}

TEST_F(OpTest, Concatenation) {
  std::vector<int> shape_in = {1, 2, 3, 4};
  // Concatenation allows negative indexing on the `axis` attribute.
  for (int axis : {1, -1}) {
    std::vector<int> shape_out = shape_in;
    shape_out[(axis + 4) % 4] *= 2;
    SingleOpModelBuilder mb;
    mb.AddInput(kTfLiteFloat32, shape_in);
    mb.AddInput(kTfLiteFloat32, shape_in);
    mb.AddOutput(kTfLiteFloat32, shape_out);
    mb.AddOperator<TfLiteConcatenationParams>(kTfLiteBuiltinConcatenation,
                                              {
                                                  .axis = axis,
                                                  .activation = kTfLiteActNone,
                                              });
    TestModel(mb);
  }
}

}  // namespace tflite::cros::tests

int main(int argc, char** argv) {
  g_env = new StableDelegateEnvironment();
  auto config_flags = ::tflite::cros::tests::g_config.GetFlags();
  if (!g_env->InitFromCommandLine(&argc, argv, config_flags)) {
    delete g_env;
    return EXIT_FAILURE;
  }

  testing::InitGoogleTest(&argc, argv);
  // GoogleTest takes the ownership of g_env.
  ::testing::AddGlobalTestEnvironment(g_env);
  return RUN_ALL_TESTS();
}
