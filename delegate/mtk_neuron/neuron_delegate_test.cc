/*
 * Copyright 2025 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>
// To keep the codebase simple, we don't want to depend on libchrome or other
// threading libraries. Thus, std::thread is used here to support multithread
// behaviors.
// NOLINTNEXTLINE(build/c++11)
#include <thread>
#include <vector>

#include "absl/random/random.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "common/simple_model_builder.h"
#include "common/sync_driver.h"
#include "common/test_util/aligned_tflite_buffer_allocator.h"
#include "common/test_util/fp16_compare.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/delegate_loader.h"

namespace tflite::cros::tests {

using TfLiteDelegatePtr = Interpreter::TfLiteDelegatePtr;

TFLiteSettingsT GetDefaultSettings() {
  TFLiteSettingsT settings;
  settings.stable_delegate_loader_settings =
      std::make_unique<StableDelegateLoaderSettingsT>();
  settings.stable_delegate_loader_settings->delegate_name =
      "mtk_neuron_delegate";
  settings.stable_delegate_loader_settings->delegate_path =
      "/usr/lib64/libtensorflowlite_mtk_neuron_delegate.so";
  settings.mtk_neuron_settings = std::make_unique<tflite::MtkNeuronSettingsT>();
  settings.mtk_neuron_settings->allow_fp16_precision_for_fp32 = true;
  return settings;
}

TfLiteDelegatePtr GetDelegateFromSettings(TFLiteSettingsT const* settings) {
  auto delegate = tflite::delegates::utils::LoadDelegateFromSharedLibrary(
      settings->stable_delegate_loader_settings->delegate_path);

  flatbuffers::FlatBufferBuilder fbb;
  const flatbuffers::Offset<TFLiteSettings> settings_buffer =
      TFLiteSettings::Pack(fbb, settings);
  fbb.Finish(settings_buffer);

  const TFLiteSettings* settings_ptr =
      flatbuffers::GetRoot<TFLiteSettings>(fbb.GetBufferPointer());
  return std::unique_ptr<TfLiteOpaqueDelegate, void (*)(TfLiteOpaqueDelegate*)>(
      delegate->delegate_plugin->create(settings_ptr),
      delegate->delegate_plugin->destroy);
}

void SumTo(SimpleModelBuilder& mb, std::vector<int> shape, std::string prefix,
           std::vector<int> inputs, int result, int depth = 1) {
  if (inputs.size() == 2) {
    mb.AddOperator<TfLiteAddParams>({
        .op = kTfLiteBuiltinAdd,
        .inputs = inputs,
        .outputs = {result},
    });
  } else {
    std::vector<int> sum_results;
    for (int i = 1; i < std::ssize(inputs); i += 2) {
      sum_results.push_back(mb.AddInternalTensor(SimpleModelBuilder::TensorArgs{
          .name = prefix + absl::StrFormat("sum_results_%d_%d", i, depth),
          .type = kTfLiteFloat32,
          .shape = shape,
      }));
      mb.AddOperator<TfLiteAddParams>({
          .op = kTfLiteBuiltinAdd,
          .inputs = {inputs[i - 1], inputs[i]},
          .outputs = {sum_results.back()},
      });
    }
    if (inputs.size() % 2 == 1) {
      sum_results.push_back(inputs.back());
    }
    SumTo(mb, shape, prefix, sum_results, result, depth + 1);
  }
}

/*
 * The model constructed looks like:
 *                  +-----------+
 *                  |   input   |
 *                  +-----+-----+
 *                        |
 *      +---------+-------+------+-----------+
 *      |         |       |      |           |
 * +----v---+ +---v----+  v  +---v----+ +----v---+
 * | Conv2d | | Conv2d | ... | Conv2d | | Conv2d |
 * +----+---+ +---+----+  |  +---+----+ +--+-----+
 *      |         |       |      |         |
 *    +-v---------v-+     |    +-v---------v-+
 *    |     Add     |     |    |     Add     |
 *    +-------------+     |    +------+------+
 *           |            |           |
 *           +-------+    |    +------+
 *                 +-v----v----v-+
 *                 |     Add     |
 *                 +------+------+
 *                        |
 *      +---------+-------+------+-----------+
 *      |         |       |      |           |
 * +----v---+ +---v----+  v  +---v----+ +----v---+
 * | Conv2d | | Conv2d | ... | Conv2d | | Conv2d |
 * +----+---+ +---+----+  |  +---+----+ +--+-----+
 *      |         |       |      |         |
 *    +-v---------v-+     |    +-v---------v-+
 *    |     Add     |     |    |     Add     |
 *    +------+------+     |    +------+------+
 *           |            |           |
 *           +-------+    |    +------+
 *                   |    |    |
 *                 +-v----v----v-+
 *                 |     Add     |
 *                 +-------------+
 *                        .
 *                        .
 *                        .
 *      +---------+-------+------+-----------+
 *      |         |       |      |           |
 * +----v---+ +---v----+  v  +---v----+ +----v---+
 * | Conv2d | | Conv2d | ... | Conv2d | | Conv2d |
 * +----+---+ +---+----+  |  +---+----+ +--+-----+
 *      |         |       |      |         |
 *    +-v---------v-+     |    +-v---------v-+
 *    |     Add     |     |    |     Add     |
 *    +------+------+     |    +------+------+
 *           |            |           |
 *           +-------+    |    +------+
 *                   |    |    |
 *                 +-v----v----v-+
 *                 |    Output   |
 *                 +-------------+
 */

template <int Batch, int Size, int Channels, int Kernel, int Layers, int Width>
std::unique_ptr<FlatBufferModel> GetModel() {
  absl::BitGen gen;
  SimpleModelBuilder mb;

  // Batch, Height, Width, Channels
  const std::vector<int> content_tensor_shapes = {Batch, Size, Size, Channels};
  auto get_content_tensor_args = [&](std::string name) {
    return SimpleModelBuilder::TensorArgs{
        .name = name,
        .type = kTfLiteFloat32,
        .shape = content_tensor_shapes,
    };
  };

  int input = mb.AddInput(get_content_tensor_args("input"));

  std::vector<std::vector<int>> weights(Width);
  for (int i = 0; i < Width; ++i) {
    std::vector<float> weight_values(Channels * Kernel * Kernel * Channels);
    for (auto& weight_value : weight_values) {
      weight_value = absl::Uniform(gen, 0.0, 1.0);
    }
    std::vector<uint8_t> weight_data(
        reinterpret_cast<const uint8_t*>(weight_values.data()),
        reinterpret_cast<const uint8_t*>(weight_values.data()) +
            weight_values.size() * sizeof(float));
    weights[i].push_back(mb.AddInternalTensor(SimpleModelBuilder::TensorArgs{
        .name = absl::StrFormat("weight_%d", i),
        .type = kTfLiteFloat32,
        // Output Channels, Kernel Height, Kernel Width, Input Channels
        .shape = {Channels, Kernel, Kernel, Channels},
        .buffer = mb.AddBuffer(weight_data),
    }));
  }

  std::vector<uint8_t> bias_data(Channels * sizeof(float));
  int bias = mb.AddInternalTensor(SimpleModelBuilder::TensorArgs{
      .name = "bias",
      .type = kTfLiteFloat32,
      // Output Channels
      .shape = {Channels},
      .buffer = mb.AddBuffer(bias_data),
  });

  int output = mb.AddOutput(get_content_tensor_args("output"));

  for (int i = 1; i <= Layers; ++i) {
    const TfLiteConvParams conv_params = {
        .padding = kTfLitePaddingSame,
        .stride_width = 1,
        .stride_height = 1,
        .activation = kTfLiteActNone,
        .dilation_width_factor = 1,
        .dilation_height_factor = 1,
    };
    int current_output = (i == Layers)
                             ? output
                             : mb.AddInternalTensor(get_content_tensor_args(
                                   absl::StrFormat("intermediate_%d", i)));
    if (Width == 1) {
      mb.AddOperator<TfLiteConvParams>({
          .op = kTfLiteBuiltinConv2d,
          .inputs = {input, weights[0].back(), bias},
          .outputs = {current_output},
          .params = conv_params,
      });
    } else {
      std::vector<int> intermediates;
      for (int j = 0; j < Width; ++j) {
        intermediates.push_back(mb.AddInternalTensor(get_content_tensor_args(
            absl::StrFormat("intermediate_%d_%d", i, j))));
        mb.AddOperator<TfLiteConvParams>({
            .op = kTfLiteBuiltinConv2d,
            .inputs = {input, weights[j].back(), bias},
            .outputs = {intermediates.back()},
            .params = conv_params,
        });
      }
      SumTo(mb, content_tensor_shapes, absl::StrFormat("intermediate_%d_", i),
            intermediates, current_output);
    }
    input = current_output;
  }
  return mb.Build();
}

std::pair<std::unique_ptr<FlatBufferModel>, int>
GetTimeConsumingModelAndBufferSize() {
  // The model with these parameters runs >500ms on Navi.
  constexpr int kBatch = 8;
  constexpr int kSize = 512;
  constexpr int kChannels = 6;
  constexpr int kKernel = 15;
  constexpr int kLayers = 32;
  constexpr int kWidth = 1;
  return {GetModel<kBatch, kSize, kChannels, kKernel, kLayers, kWidth>(),
          kBatch * kSize * kSize * kChannels};
}

std::pair<std::unique_ptr<FlatBufferModel>, int> GetSmallModelAndBufferSize() {
  // The model with these parameters runs <5ms on Navi.
  constexpr int kBatch = 1;
  constexpr int kSize = 2;
  constexpr int kChannels = 1;
  constexpr int kKernel = 1;
  constexpr int kLayers = 1;
  constexpr int kWidth = 1;
  return {GetModel<kBatch, kSize, kChannels, kKernel, kLayers, kWidth>(),
          kBatch * kSize * kSize * kChannels};
}

std::vector<float> GetRandomizedFloatData(int size) {
  absl::BitGen gen;
  std::vector<float> buffer(size);
  for (auto& buffer_value : buffer) {
    buffer_value = absl::Uniform(gen, 0.0, 1.0);
  }
  return buffer;
}

std::unique_ptr<SyncDriver> PrepareSyncDriver(
    const std::unique_ptr<FlatBufferModel>& model, TfLiteDelegatePtr delegate,
    const std::vector<float>& input_buffer) {
  if (model == nullptr || delegate == nullptr) {
    return nullptr;
  }
  auto driver = SyncDriver::Create(
      std::move(delegate), FlatBufferModel::BuildFromModel(model->GetModel()));
  if (driver == nullptr) {
    return nullptr;
  }
  driver->SetInputTensorBuffer(
      "input",
      AllocateTfLiteAlignedTensorBuffer(input_buffer.size() * sizeof(float)));
  if (driver->SetInputTensorData("input", input_buffer) != kTfLiteOk) {
    return nullptr;
  }
  if (driver->AllocateBuffers() != kTfLiteOk) {
    return nullptr;
  }
  return driver;
}

TEST(NeuronDelegateTest, InferenceSelfAborted) {
  TFLiteSettingsT settings = GetDefaultSettings();

  auto [model, buffer_size] = GetTimeConsumingModelAndBufferSize();
  std::vector<float> input = GetRandomizedFloatData(buffer_size);

  // Get the expected output with the given input.
  auto driver =
      PrepareSyncDriver(model, GetDelegateFromSettings(&settings), input);
  ASSERT_NE(driver, nullptr);
  ASSERT_EQ(driver->Invoke(), kTfLiteOk);
  std::vector<float> expected_output =
      driver->GetOutputTensorData<float>("output");

  // Set the inference abort time to 1ms.
  settings.mtk_neuron_settings->inference_abort_time_ms = 1;

  // Try inference the model with an abort time.
  driver = PrepareSyncDriver(model, GetDelegateFromSettings(&settings), input);
  ASSERT_NE(driver, nullptr);

  // Check the model inference is properly aborted.
  absl::Time before_invoke = absl::Now();
  ASSERT_EQ(driver->Invoke(), kTfLiteError);
  absl::Time after_invoke = absl::Now();
  // Use a larger time duration to accommodate inaccuracy and overheads.
  EXPECT_LT(after_invoke - before_invoke, absl::Milliseconds(60));

  // The model inference should be aborted, so the output should be different.
  std::vector<float> aborted_output =
      driver->GetOutputTensorData<float>("output");
  ASSERT_EQ(aborted_output.size(), expected_output.size());
  EXPECT_THAT(
      aborted_output,
      testing::Not(testing::ElementsAreArray(ArrayFp16Eq(expected_output))));
}

TEST(NeuronDelegateTest, ParallelInference) {
  TFLiteSettingsT settings = GetDefaultSettings();

  auto [model, buffer_size] = GetTimeConsumingModelAndBufferSize();
  std::vector<float> input = GetRandomizedFloatData(buffer_size);

  // Create the first driver without inferencing.
  auto driver1 =
      PrepareSyncDriver(model, GetDelegateFromSettings(&settings), input);
  ASSERT_NE(driver1, nullptr);

  // Create the second driver without inferencing.
  auto driver2 =
      PrepareSyncDriver(model, GetDelegateFromSettings(&settings), input);
  ASSERT_NE(driver2, nullptr);

  // Create a child thread to inference the first driver.
  absl::Duration time_used1;
  std::vector<float> output1;
  std::thread child_thread([&] {
    absl::Time before_invoke = absl::Now();
    // Leave the output vector empty if the invocation fail.
    if (driver1->Invoke() == kTfLiteOk) {
      output1 = driver1->GetOutputTensorData<float>("output");
    }
    time_used1 = absl::Now() - before_invoke;
  });

  // Inference the second driver on main thread.
  absl::Time before_invoke = absl::Now();
  ASSERT_EQ(driver2->Invoke(), kTfLiteOk);
  std::vector<float> output2 = driver2->GetOutputTensorData<float>("output");
  absl::Duration time_used2 = absl::Now() - before_invoke;

  child_thread.join();

  // The result should be the same since they are the same model and the same
  // input.
  EXPECT_EQ(output1, output2);
  // If the time spent is similar, we consider the model to be ran in parallel.
  // Here we allow a 15% relative error range.
  EXPECT_LT(FDivDuration(AbsDuration(time_used1 - time_used2),
                         std::max(time_used1, time_used2)),
            0.15);
}

TEST(NeuronDelegateTest, BlockedByPreviousInference) {
  // Set optimization hints to OPTIMIZATION_LOW_LATENCY to ensure the model
  // utilizing every NPU cores.
  using MtkNeuronSettings_::OptimizationHint;
  TFLiteSettingsT settings = GetDefaultSettings();
  settings.mtk_neuron_settings->optimization_hints =
      std::vector<OptimizationHint>{
          OptimizationHint::OptimizationHint_OPTIMIZATION_LOW_LATENCY};

  auto [model, buffer_size] = GetTimeConsumingModelAndBufferSize();
  std::vector<float> input = GetRandomizedFloatData(buffer_size);

  // Create the first driver without inferencing.
  auto driver1 =
      PrepareSyncDriver(model, GetDelegateFromSettings(&settings), input);
  ASSERT_NE(driver1, nullptr);

  // Create the second driver without inferencing.
  auto driver2 =
      PrepareSyncDriver(model, GetDelegateFromSettings(&settings), input);
  ASSERT_NE(driver2, nullptr);

  // Create a child thread to inference the first driver.
  absl::Duration time_used1;
  std::vector<float> output1;
  std::thread child_thread([&] {
    absl::Time before_invoke = absl::Now();
    // Leave the output vector empty if the invocation fail.
    if (driver1->Invoke() == kTfLiteOk) {
      output1 = driver1->GetOutputTensorData<float>("output");
    }
    time_used1 = absl::Now() - before_invoke;
  });

  // Inference the second driver on main thread.
  absl::Time before_invoke = absl::Now();
  ASSERT_EQ(driver2->Invoke(), kTfLiteOk);
  std::vector<float> output2 = driver2->GetOutputTensorData<float>("output");
  absl::Duration time_used2 = absl::Now() - before_invoke;

  child_thread.join();

  // The result should be the same since they are the same model and the same
  // input.
  EXPECT_EQ(output1, output2);
  // If the time spent is around 2x, we consider the model to be blocked by the
  // previous model. Here we allow a 15% absolute error range.
  if (time_used1 < time_used2) {
    std::swap(time_used1, time_used2);
  }
  EXPECT_NEAR(FDivDuration(time_used1, time_used2), 2, 0.15);
}

TEST(NeuronDelegateTest, AbortedWhenWaiting) {
  // Set optimization hints to OPTIMIZATION_LOW_LATENCY to ensure the model
  // utilizing every NPU cores.
  using MtkNeuronSettings_::OptimizationHint;
  TFLiteSettingsT settings = GetDefaultSettings();
  settings.mtk_neuron_settings->optimization_hints =
      std::vector<OptimizationHint>{
          OptimizationHint::OptimizationHint_OPTIMIZATION_LOW_LATENCY};

  // Create the driver for a time consuming model without inferencing.
  auto [big_model, big_model_buffer_size] =
      GetTimeConsumingModelAndBufferSize();
  auto big_model_driver =
      PrepareSyncDriver(big_model, GetDelegateFromSettings(&settings),
                        GetRandomizedFloatData(big_model_buffer_size));
  ASSERT_NE(big_model_driver, nullptr);

  // Since the small model usually finish in <5ms and the large model finish in
  // >500ms, pick 50ms should be safe.
  settings.mtk_neuron_settings->inference_abort_time_ms = 50;

  // Create the driver for a small model without inferencing.
  auto [small_model, small_model_buffer_size] = GetSmallModelAndBufferSize();
  auto small_model_driver =
      PrepareSyncDriver(small_model, GetDelegateFromSettings(&settings),
                        GetRandomizedFloatData(small_model_buffer_size));
  ASSERT_NE(small_model_driver, nullptr);

  // Create a child thread to inference the time consuming model.
  TfLiteStatus big_model_inference_status;
  std::thread child_thread(
      [&] { big_model_inference_status = big_model_driver->Invoke(); });

  // Sleep 100ms to ensure the time consuming model is running first.
  absl::SleepFor(absl::Milliseconds(100));
  absl::Time before_invoke = absl::Now();
  EXPECT_NE(small_model_driver->Invoke(), kTfLiteOk);
  absl::Duration small_model_latency = absl::Now() - before_invoke;

  child_thread.join();

  // Time consuming model should be normally inferenced.
  EXPECT_EQ(big_model_inference_status, kTfLiteOk);
  // The time spent for small model should similar to abort_time_ms.
  // Setting 100ms here to allow some overheads. This should not affect
  // the effectiveness of the check since the time consuming model
  // normally spent more than 500ms.
  EXPECT_LT(small_model_latency, absl::Milliseconds(100));
}

}  // namespace tflite::cros::tests

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
