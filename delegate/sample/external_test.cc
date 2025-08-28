/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/cleanup/cleanup.h"
#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/delegate_loader.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/tflite_settings_json_parser.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"

namespace tflite {

using delegates::utils::LoadDelegateFromSharedLibrary;
using delegates::utils::TfLiteSettingsJsonParser;

constexpr char kSampleDelegatePath[] =
    "delegate/sample/libtensorflowlite_cros_sample_delegate.so";

constexpr char kSettingsPath[] = "delegate/sample/test_settings.json";

// It's a fp32 "y = x + x + x" test model.
constexpr char kAddModelPath[] =
    "external/org_tensorflow/tensorflow/lite/testdata/add.bin";

TEST(SampleDelegate, Add) {
  // This test both directly and indirectly calls functions which
  // `-fsanitize=function` takes issue with. The indirect calls to it are deep
  // in tensorflow. At a glance, nothing jumps out, but there are conditional
  // type definitions used in the definition of functions in
  // `kSampleDelegatePath` & this binary. Seems most likely that there's
  // a mismatched configuration.
  //
  // Until that's figured out, skip this on ubsan builds, so all other tests
  // can benefit from these checks.
#if defined(CHROMEOS_UBSAN_BUILD)
  GTEST_SKIP()
      << "Skipping due to KI with -fsanitize=function: b/433670501#comment20.";
#endif

  // Load stable delegate.
  const TfLiteStableDelegate* stable_delegate =
      LoadDelegateFromSharedLibrary(kSampleDelegatePath);
  ASSERT_NE(stable_delegate, nullptr);
  ASSERT_NE(stable_delegate->delegate_plugin, nullptr);
  ASSERT_NE(stable_delegate->delegate_plugin->create, nullptr);
  ASSERT_NE(stable_delegate->delegate_plugin->destroy, nullptr);

  // Load settings.
  TfLiteSettingsJsonParser parser;
  const TFLiteSettings* settings = parser.Parse(kSettingsPath);
  ASSERT_NE(settings, nullptr);

  // Create opaque delegate.
  TfLiteOpaqueDelegate* opaque_delegate =
      stable_delegate->delegate_plugin->create(settings);
  ASSERT_NE(opaque_delegate, nullptr);
  absl::Cleanup destory_opaque_delegate = [&] {
    stable_delegate->delegate_plugin->destroy(opaque_delegate);
  };

  // Load model.
  std::unique_ptr<tflite::FlatBufferModel> model =
      FlatBufferModel::BuildFromFile(kAddModelPath);
  ASSERT_NE(model, nullptr);

  // Build interpreter.
  ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);
  builder.AddDelegate(opaque_delegate);
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(builder(&interpreter), kTfLiteOk);

  // Prepare input.
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  float* input = interpreter->typed_input_tensor<float>(0);
  size_t num_elements =
      TfLiteTensorByteSize(interpreter->input_tensor(0)) / sizeof(float);
  for (int i = 0; i < num_elements; ++i) {
    input[i] = 1.23 * i;
  }

  // Run model inference.
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

  // Check output.
  float* output = interpreter->typed_output_tensor<float>(0);
  ASSERT_EQ(TfLiteTensorByteSize(interpreter->output_tensor(0)) / sizeof(float),
            num_elements);
  for (int i = 0; i < num_elements; ++i) {
    // y[i] = x[i] + x[i] + x[i] = 3 * x[i] = 3 * 1.23 * i = 3.69 * i
    EXPECT_FLOAT_EQ(output[i], 3.69 * i);
  }
}

}  // namespace tflite

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
