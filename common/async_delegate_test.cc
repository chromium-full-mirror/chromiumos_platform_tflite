/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <gtest/gtest.h>
#include <iostream>

#include "absl/random/random.h"
#include "common/async_driver.h"
#include "common/simple_model_builder.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/tool_params.h"

namespace tflite::cros::tests {

using DelegateProvider = tools::DelegateProvider;
using ProvidedDelegateList = tools::ProvidedDelegateList;
using ProvidedDelegate = tools::ProvidedDelegateList::ProvidedDelegate;
using ToolParams = tools::ToolParams;
using TfLiteDelegatePtr = Interpreter::TfLiteDelegatePtr;

class Environment : public ::testing::Environment {
 public:
  Environment() : delegate_list_(&params_) {}
  ~Environment() override {}

  bool InitFromCommandLine(int* argc, char** argv) {
    constexpr char kSettingsFlagName[] = "stable_delegate_settings_file";

    std::string settings;
    // Use our own flags directly instead of delegate_list_.AppendCmdlineFlags()
    // to make the settings flag required.
    std::vector<tflite::Flag> flags = {
        Flag::CreateFlag(kSettingsFlagName, &settings,
                         "The path to the delegate settings JSON file.",
                         Flag::kRequired),
    };

    if (!tflite::Flags::Parse(argc, const_cast<const char**>(argv), flags)) {
      std::cout << Flags::Usage(argv[0], flags) << std::endl;
      return false;
    }

    delegate_list_.AddAllDelegateParams();
    params_.Set<std::string>(kSettingsFlagName, settings);

    return true;
  }

  TfLiteDelegatePtr CreateDelegate() {
    std::vector<ProvidedDelegate> provided_delegates =
        delegate_list_.CreateAllRankedDelegates();
    EXPECT_EQ(provided_delegates.size(), 1);
    if (provided_delegates.empty()) {
      return tools::CreateNullDelegate();
    }
    return std::move(provided_delegates[0].delegate);
  }

  void SetUp() override {
    // Ensure that the delegate can be created before running any test.
    ASSERT_NE(CreateDelegate(), nullptr);
  }

 private:
  ToolParams params_;
  ProvidedDelegateList delegate_list_;
};

Environment* g_env = nullptr;

// Use a relaxed tolerance when comparing the computation results, as the
// primary goal of this test is to verify the asynchronous kernel API flow.
// Some delegates may internally use fp16 for fp32 arithmetic, making the
// default EXPECT_FLOAT_EQ too strict for such cases.
// TODO(shik): If we need checking fp16 results in more places, consider
// implementing a generic fp16 GMock matcher that computes the difference in
// ULPs (Units in the Last Place), similar to the current implementation of
// EXPECT_FLOAT_EQ, to provide a more precise comparison.
constexpr float kEps = 1e-2;

TEST(AsyncDelegate, Inference) {
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

}  // namespace tflite::cros::tests

using tflite::cros::tests::g_env;

int main(int argc, char** argv) {
  g_env = new tflite::cros::tests::Environment();
  if (!g_env->InitFromCommandLine(&argc, argv)) {
    delete g_env;
    return EXIT_FAILURE;
  }

  testing::InitGoogleTest(&argc, argv);
  // GoogleTest takes the ownership of g_env.
  ::testing::AddGlobalTestEnvironment(g_env);
  return RUN_ALL_TESTS();
}
