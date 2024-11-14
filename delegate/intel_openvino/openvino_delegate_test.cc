/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/openvino_delegate.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/core/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"

/* This file creates unit tests for openvino_delegate.cc to be run on host
 * machine without NPU */

namespace tflite {
namespace openvinodelegate {

std::function<void(TfLiteOpaqueContext *opaque_context, TfLiteOpaqueNode *node)>
    test_function_;

class OpenVINODelegateTestPeer {
 public:
  OpenVINODelegateTestPeer() {}

  static bool CheckInputsType(const OpenVINODelegate &test_delegate,
                              TfLiteType tensor_type,
                              TfLiteType expected_type) {
    return test_delegate.CheckInputType(tensor_type, expected_type);
  }
  static bool CheckDataTypeSupported(
      const OpenVINODelegate &test_delegate, const TfLiteOpaqueContext *context,
      const TfLiteOpaqueNode *node,
      const std::vector<std::vector<TfLiteType>> supported_types) {
    return test_delegate.CheckDataTypeSupported(context, node, supported_types);
  }
  static bool CheckDims(const OpenVINODelegate &test_delegate,
                        const TfLiteOpaqueContext *context,
                        const TfLiteOpaqueNode *node,
                        const std::vector<std::vector<int>> dims_size) {
    return test_delegate.CheckDims(context, node, dims_size);
  }
};

class OpenVINODelegateTest : public testing::Test {
 protected:
  void SetUpDelegate(std::function<void(TfLiteOpaqueContext *opaque_context,
                                        TfLiteOpaqueNode *node)>
                         test_function) {
    TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
    test_function_ = test_function;
    opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext *opaque_context,
                                         TfLiteOpaqueDelegate *opaque_delegate_,
                                         void *data) -> TfLiteStatus {
      // Test that an unnamed delegate kernel can be passed to the TF Lite
      // runtime.

      TfLiteIntArray *execution_plan;
      TF_LITE_ENSURE_STATUS(
          TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
      for (int i = 0; i < execution_plan->size; ++i) {
        TfLiteOpaqueNode *node = nullptr;
        TfLiteOperator *registration = nullptr;
        TfLiteOpaqueContextGetNodeAndRegistration(opaque_context, i, &node,
                                                  &registration);
        test_function_(opaque_context, node);
      }

      TfLiteOperator *registration_external =
          TfLiteOperatorCreate(kTfLiteBuiltinDelegate,
                               /*name=*/nullptr,
                               /*version=*/1,
                               /*data=*/nullptr);
      return TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
          opaque_context, registration_external, execution_plan,
          opaque_delegate_);
    };
    opaque_delegate_ = TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);

    ::tflite::ops::builtin::BuiltinOpResolver resolver;
    ::tflite::InterpreterBuilder builder(*model_, resolver);
    builder.AddDelegate(opaque_delegate_);
    EXPECT_EQ(kTfLiteOk, builder(&interpreter_));
    ASSERT_NE(interpreter_, nullptr);
  }

  void SetUp() override {
    model_ = ::tflite::FlatBufferModel::BuildFromFile(
        "external/org_tensorflow/tensorflow/lite/testdata/add.bin");
    ASSERT_NE(model_, nullptr);
  }

  void TearDown() override { TfLiteOpaqueDelegateDelete(opaque_delegate_); }

  std::unique_ptr<::tflite::Interpreter> interpreter_;
  TfLiteOpaqueDelegate *opaque_delegate_ = nullptr;
  std::unique_ptr<::tflite::FlatBufferModel> model_;
};

TEST_F(OpenVINODelegateTest, CheckSupportedInputsTypeTest) {
  auto test_func = [](TfLiteOpaqueContext *opaque_context,
                      TfLiteOpaqueNode *node) -> void {
    TfLiteOpenVINODelegateOptions options_del;

    OpenVINODelegate ov_del_test = OpenVINODelegate(&options_del);
    ::tflite::openvinodelegate::OpenVINODelegateTestPeer test_peer;
    EXPECT_EQ(true, test_peer.CheckDataTypeSupported(
                        ov_del_test, opaque_context, node,
                        {{kTfLiteFloat32}, {kTfLiteFloat32}}));
  };
  SetUpDelegate(test_func);
}

TEST_F(OpenVINODelegateTest, CheckUnsupportedInputsTypeTest) {
  auto test_func = [](TfLiteOpaqueContext *opaque_context,
                      TfLiteOpaqueNode *node) -> void {
    TfLiteOpenVINODelegateOptions options_del;
    OpenVINODelegate ov_del_test = OpenVINODelegate(&options_del);
    ::tflite::openvinodelegate::OpenVINODelegateTestPeer test_peer;
    EXPECT_EQ(false, test_peer.CheckDataTypeSupported(
                         ov_del_test, opaque_context, node,
                         {{kTfLiteInt16}, {kTfLiteInt16}}));
  };
  SetUpDelegate(test_func);
}

TEST_F(OpenVINODelegateTest, CheckUnsupportedInputsTypeTest2) {
  auto test_func = [](TfLiteOpaqueContext *opaque_context,
                      TfLiteOpaqueNode *node) -> void {
    TfLiteOpenVINODelegateOptions options_del;
    OpenVINODelegate ov_del_test = OpenVINODelegate(&options_del);
    ::tflite::openvinodelegate::OpenVINODelegateTestPeer test_peer;
    EXPECT_EQ(false, test_peer.CheckDataTypeSupported(
                         ov_del_test, opaque_context, node,
                         {{kTfLiteFloat32}, {kTfLiteInt16}}));
  };
  SetUpDelegate(test_func);
}

TEST_F(OpenVINODelegateTest, CheckSupportedInputsTypeTest2) {
  auto test_func = [](TfLiteOpaqueContext *opaque_context,
                      TfLiteOpaqueNode *node) -> void {
    TfLiteOpenVINODelegateOptions options_del;
    OpenVINODelegate ov_del_test = OpenVINODelegate(&options_del);
    ::tflite::openvinodelegate::OpenVINODelegateTestPeer test_peer;
    EXPECT_EQ(true, test_peer.CheckDataTypeSupported(
                        ov_del_test, opaque_context, node,
                        {{kTfLiteFloat32}, {kTfLiteInt16, kTfLiteFloat32}}));
  };
  SetUpDelegate(test_func);
}

TEST_F(OpenVINODelegateTest, CheckInputsIsNodeSupportedByDelegate1) {
  auto test_func = [](TfLiteOpaqueContext *opaque_context,
                      TfLiteOpaqueNode *node) -> void {
    TfLiteOpenVINODelegateOptions options_del;
    OpenVINODelegate ov_del_test = OpenVINODelegate(&options_del);
    EXPECT_EQ(false, ov_del_test.IsNodeSupportedByDelegate(nullptr, node,
                                                           opaque_context));
  };
  SetUpDelegate(test_func);
}

TEST_F(OpenVINODelegateTest, CheckInputsSupportedByDelegate2) {
  auto test_func = [](TfLiteOpaqueContext *opaque_context,
                      TfLiteOpaqueNode *node) -> void {
    TfLiteOpenVINODelegateOptions options_del;
    OpenVINODelegate ov_del_test = OpenVINODelegate(&options_del);
    EXPECT_EQ(false, ov_del_test.IsNodeSupportedByDelegate(nullptr, nullptr,
                                                           opaque_context));
  };
  SetUpDelegate(test_func);
}

TEST_F(OpenVINODelegateTest, CheckInputsSupportedByDelegate3) {
  auto test_func = [](TfLiteOpaqueContext *opaque_context,
                      TfLiteOpaqueNode *node) -> void {
    TfLiteOpenVINODelegateOptions options_del;
    OpenVINODelegate ov_del_test = OpenVINODelegate(&options_del);
    EXPECT_EQ(false,
              ov_del_test.IsNodeSupportedByDelegate(nullptr, node, nullptr));
  };
  SetUpDelegate(test_func);
}

TEST_F(OpenVINODelegateTest, CheckNoSupportedTypes) {
  auto test_func = [](TfLiteOpaqueContext *opaque_context,
                      TfLiteOpaqueNode *node) -> void {
    TfLiteOpenVINODelegateOptions options_del;
    OpenVINODelegate ov_del_test = OpenVINODelegate(&options_del);
    ::tflite::openvinodelegate::OpenVINODelegateTestPeer test_peer;
    EXPECT_EQ(false, test_peer.CheckDataTypeSupported(
                         ov_del_test, opaque_context, node, {{}, {}}));
  };
  SetUpDelegate(test_func);
}

TEST_F(OpenVINODelegateTest, CheckDimsTest) {
  auto test_func = [](TfLiteOpaqueContext *opaque_context,
                      TfLiteOpaqueNode *node) -> void {
    TfLiteOpenVINODelegateOptions options_del;
    OpenVINODelegate ov_del_test = OpenVINODelegate(&options_del);
    ::tflite::openvinodelegate::OpenVINODelegateTestPeer test_peer;
    EXPECT_EQ(false, test_peer.CheckDims(ov_del_test, opaque_context, node,
                                         {{4}, {4}, {4}}));
  };
  SetUpDelegate(test_func);
}

TEST_F(OpenVINODelegateTest, CheckDimsDynamicTest) {
  auto test_func = [](TfLiteOpaqueContext *opaque_context,
                      TfLiteOpaqueNode *node) -> void {
    TfLiteOpenVINODelegateOptions options_del;
    OpenVINODelegate ov_del_test = OpenVINODelegate(&options_del);
    ::tflite::openvinodelegate::OpenVINODelegateTestPeer test_peer;
    EXPECT_EQ(false, test_peer.CheckDims(ov_del_test, opaque_context, node,
                                         {{0}, {4}}));
  };
  SetUpDelegate(test_func);
}

}  // namespace openvinodelegate
}  // namespace tflite

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
