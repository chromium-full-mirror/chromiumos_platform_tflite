/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/openvino_delegate_core.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <unordered_set>

#include "delegate/intel_openvino/openvino_graph_builder.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/core/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"

/* This file creates unit tests for openvino_delegate_core.cc to be run on host
 * machine without NPU */

namespace tflite {
namespace openvinodelegate {

class OpenVINODelegateCoreTest : public testing::Test {
 protected:
  TfLiteInterpreter* interpreter_ = nullptr;
  TfLiteOpaqueDelegate* opaque_delegate_ = nullptr;
  TfLiteModel* model_ = nullptr;
};

TEST_F(OpenVINODelegateCoreTest, InitializeBuilder) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* opaque_context,
                                       TfLiteOpaqueDelegate* opaque_delegate_,
                                       void* data) -> TfLiteStatus {
    auto reg_ex = TfLiteRegistrationExternalCreate(
        kTfLiteBuiltinDelegate, "Test driver Openvino delegate", /*version=*/1,
        /*data=*/nullptr);
    TfLiteRegistrationExternalSetInit(
        reg_ex,
        [](TfLiteOpaqueContext* opaque_context, const char* buffer,
           size_t length) -> void* {
          const TfLiteOpaqueDelegateParams* params =
              reinterpret_cast<const TfLiteOpaqueDelegateParams*>(buffer);
          const std::unordered_set<int> inputs(
              &params->input_tensors->data[0],
              &params->input_tensors->data[params->input_tensors->size]);

          auto ov_delegate_core_test =
              std::make_unique<tflite::openvinodelegate::OpenVINODelegateCore>(
                  "");
          TfLiteOpenVINODelegateOptions delegate_options;
          EXPECT_EQ(kTfLiteOk, ov_delegate_core_test->CreateModel(
                                   opaque_context, params, &delegate_options));
          void* void_fake_ptr = nullptr;
          return void_fake_ptr;
        });
    TfLiteRegistrationExternalSetInvoke(
        reg_ex,
        [](TfLiteOpaqueContext* context, TfLiteOpaqueNode* opaque_node)
            -> TfLiteStatus { return kTfLiteOk; });

    TfLiteRegistrationExternalSetFree(
        reg_ex, [](TfLiteOpaqueContext* context, void* data) {});

    TfLiteIntArray* execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, reg_ex, execution_plan, opaque_delegate_);
    return kTfLiteOk;
  };

  model_ = TfLiteModelCreateFromFile(
      "external/org_tensorflow/tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model_, nullptr);
  opaque_delegate_ = TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate_);
  interpreter_ = TfLiteInterpreterCreate(model_, options);
  ASSERT_NE(interpreter_, nullptr);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter_);
  TfLiteModelDelete(model_);
  TfLiteOpaqueDelegateDelete(opaque_delegate_);
}

TEST_F(OpenVINODelegateCoreTest, CreateModel_InvalidContext) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* opaque_context,
                                       TfLiteOpaqueDelegate* opaque_delegate_,
                                       void* data) -> TfLiteStatus {
    auto reg_ex = TfLiteRegistrationExternalCreate(
        kTfLiteBuiltinDelegate, "Test driver Openvino delegate", /*version=*/1,
        /*data=*/nullptr);
    TfLiteRegistrationExternalSetInit(
        reg_ex,
        [](TfLiteOpaqueContext* opaque_context, const char* buffer,
           size_t length) -> void* {
          const TfLiteOpaqueDelegateParams* params =
              reinterpret_cast<const TfLiteOpaqueDelegateParams*>(buffer);
          const std::unordered_set<int> inputs(
              &params->input_tensors->data[0],
              &params->input_tensors->data[params->input_tensors->size]);

          // test for context with null ptr
          opaque_context = nullptr;
          auto ov_delegate_core_test =
              std::make_unique<tflite::openvinodelegate::OpenVINODelegateCore>(
                  "");
          TfLiteOpenVINODelegateOptions delegate_options;
          EXPECT_EQ(kTfLiteError,
                    ov_delegate_core_test->CreateModel(opaque_context, params,
                                                       &delegate_options));
          void* void_fake_ptr = nullptr;
          return void_fake_ptr;
        });

    TfLiteRegistrationExternalSetInvoke(
        reg_ex,
        [](TfLiteOpaqueContext* context, TfLiteOpaqueNode* opaque_node)
            -> TfLiteStatus { return kTfLiteOk; });

    TfLiteRegistrationExternalSetFree(
        reg_ex, [](TfLiteOpaqueContext* context, void* data) {});

    TfLiteIntArray* execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, reg_ex, execution_plan, opaque_delegate_);
    return kTfLiteOk;
  };

  model_ = TfLiteModelCreateFromFile(
      "external/org_tensorflow/tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model_, nullptr);
  opaque_delegate_ = TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate_);
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate_);
  interpreter_ = TfLiteInterpreterCreate(model_, options);
  ASSERT_NE(interpreter_, nullptr);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter_);
  TfLiteModelDelete(model_);
  TfLiteOpaqueDelegateDelete(opaque_delegate_);
}

TEST_F(OpenVINODelegateCoreTest, InitializeBuilder_InvalidParams) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* opaque_context,
                                       TfLiteOpaqueDelegate* opaque_delegate_,
                                       void* data) -> TfLiteStatus {
    auto reg_ex = TfLiteRegistrationExternalCreate(
        kTfLiteBuiltinDelegate, "Test driver Openvino delegate", /*version=*/1,
        /*data=*/nullptr);
    TfLiteRegistrationExternalSetInit(
        reg_ex,
        [](TfLiteOpaqueContext* opaque_context, const char* buffer,
           size_t length) -> void* {
          // test for params with null ptr
          const TfLiteOpaqueDelegateParams* params = nullptr;
          auto ov_delegate_core_test =
              std::make_unique<tflite::openvinodelegate::OpenVINODelegateCore>(
                  "");
          TfLiteOpenVINODelegateOptions delegate_options;
          EXPECT_EQ(kTfLiteError,
                    ov_delegate_core_test->CreateModel(opaque_context, params,
                                                       &delegate_options));
          void* void_fake_ptr = nullptr;
          return void_fake_ptr;
        });

    TfLiteRegistrationExternalSetInvoke(
        reg_ex,
        [](TfLiteOpaqueContext* context, TfLiteOpaqueNode* opaque_node)
            -> TfLiteStatus { return kTfLiteOk; });

    TfLiteRegistrationExternalSetFree(
        reg_ex, [](TfLiteOpaqueContext* context, void* data) {});

    TfLiteIntArray* execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, reg_ex, execution_plan, opaque_delegate_);
    return kTfLiteOk;
  };

  model_ = TfLiteModelCreateFromFile(
      "external/org_tensorflow/tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model_, nullptr);
  opaque_delegate_ = TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate_);
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate_);
  interpreter_ = TfLiteInterpreterCreate(model_, options);
  ASSERT_NE(interpreter_, nullptr);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter_);
  TfLiteModelDelete(model_);
  TfLiteOpaqueDelegateDelete(opaque_delegate_);
}

TEST_F(OpenVINODelegateCoreTest, CreateModelAndCache) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* opaque_context,
                                       TfLiteOpaqueDelegate* opaque_delegate_,
                                       void* data) -> TfLiteStatus {
    auto reg_ex = TfLiteRegistrationExternalCreate(
        kTfLiteBuiltinDelegate, "Test driver Openvino delegate", /*version=*/1,
        /*data=*/nullptr);
    TfLiteRegistrationExternalSetInit(
        reg_ex,
        [](TfLiteOpaqueContext* opaque_context, const char* buffer,
           size_t length) -> void* {
          const TfLiteOpaqueDelegateParams* params =
              reinterpret_cast<const TfLiteOpaqueDelegateParams*>(buffer);
          const std::unordered_set<int> inputs(
              &params->input_tensors->data[0],
              &params->input_tensors->data[params->input_tensors->size]);

          auto ov_delegate_core_test =
              std::make_unique<tflite::openvinodelegate::OpenVINODelegateCore>(
                  "");
          TfLiteOpenVINODelegateOptions delegate_options;
          delegate_options.cache_dir = "/tmp/cache_test";
          delegate_options.model_token = "abcdefgh";
          std::filesystem::create_directory("/tmp/cache_test");
          EXPECT_EQ(kTfLiteOk, ov_delegate_core_test->CreateModel(
                                   opaque_context, params, &delegate_options));
          EXPECT_EQ(true,
                    std::filesystem::exists("/tmp/cache_test/abcdefgh.xml"));
          EXPECT_EQ(true,
                    std::filesystem::exists("/tmp/cache_test/abcdefgh.bin"));
          void* void_fake_ptr = nullptr;
          return void_fake_ptr;
        });

    TfLiteRegistrationExternalSetInvoke(
        reg_ex,
        [](TfLiteOpaqueContext* context, TfLiteOpaqueNode* opaque_node)
            -> TfLiteStatus { return kTfLiteOk; });

    TfLiteRegistrationExternalSetFree(
        reg_ex, [](TfLiteOpaqueContext* context, void* data) {});

    TfLiteIntArray* execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, reg_ex, execution_plan, opaque_delegate_);
    return kTfLiteOk;
  };

  model_ = TfLiteModelCreateFromFile(
      "external/org_tensorflow/tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model_, nullptr);
  opaque_delegate_ = TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate_);
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate_);
  interpreter_ = TfLiteInterpreterCreate(model_, options);
  ASSERT_NE(interpreter_, nullptr);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter_);
  TfLiteModelDelete(model_);
  TfLiteOpaqueDelegateDelete(opaque_delegate_);
}

}  // namespace openvinodelegate
}  // namespace tflite

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
