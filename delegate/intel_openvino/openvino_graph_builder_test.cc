/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delegate/intel_openvino/openvino_graph_builder.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>

#include <memory>
#include <string>
#include <unordered_set>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/core/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"

/* This file creates unit tests for openvino_graph_builder_test.cc to be run on
 * host machine without NPU */

namespace tflite {
namespace openvinodelegate {

constexpr char kPluginsXmlPath[] = "/etc/openvino/plugins.xml";

std::function<void(TfLiteOpaqueTensor *opaque_tensor, const int index)>
    test_function_;

class OpenVINOGraphBuilderTest : public testing::Test {
 protected:
  void SetUp() override {
    model = ::tflite::FlatBufferModel::BuildFromFile(
        "external/org_tensorflow/tensorflow/lite/testdata/add.bin");
    ASSERT_NE(model, nullptr);
  }
  void TearDown() override { TfLiteOpaqueDelegateDelete(opaque_delegate_); }

 protected:
  std::unique_ptr<::tflite::Interpreter> interpreter_;
  TfLiteOpaqueDelegate *opaque_delegate_ = nullptr;
  std::unique_ptr<::tflite::FlatBufferModel> model;
};

TfLiteOpaqueTensor *CreateOpaqueTensor(TfLiteTensor *t) {
  return reinterpret_cast<TfLiteOpaqueTensor *>(t);
}

TEST_F(OpenVINOGraphBuilderTest, AddInputParamsTest_VALID) {
  TfLiteTensor t;
  const int kNumElements = 32;
  const int kBytes = sizeof(float) * kNumElements;
  memset(&t, 0, sizeof(TfLiteTensor));
  t.bytes = kBytes;
  t.data_is_stale = true;
  t.allocation_type = kTfLiteDynamic;
  t.type = kTfLiteFloat32;
  t.dims = TfLiteIntArrayCreate(2);
  t.dims->data[0] = 4;
  t.dims->data[1] = 8;
  t.dims_signature = TfLiteIntArrayCopy(t.dims);
  t.buffer_handle = 5;

  TfLiteOpaqueTensor *opaque_t = CreateOpaqueTensor(&t);

  auto openvino_graph_builder_test =
      std::make_unique<OpenVINOGraphBuilder>(std::make_unique<NodeManager>());

  EXPECT_EQ(kTfLiteOk,
            openvino_graph_builder_test->AddInputParams(opaque_t, 0));
  EXPECT_EQ(true, openvino_graph_builder_test->getNodeManagerSize() == 1);
}

TEST_F(OpenVINOGraphBuilderTest, AddInputParamsTest_InvalidTensor) {
  auto openvino_graph_builder_test =
      std::make_unique<OpenVINOGraphBuilder>(std::make_unique<NodeManager>());

  EXPECT_EQ(kTfLiteError,
            openvino_graph_builder_test->AddInputParams(nullptr, 0));
  EXPECT_EQ(false, openvino_graph_builder_test->getNodeManagerSize() == 1);
}

TEST_F(OpenVINOGraphBuilderTest, AddInputParamsTest_InvalidIndex) {
  TfLiteTensor t;
  const int kNumElements = 32;
  const int kBytes = sizeof(float) * kNumElements;
  memset(&t, 0, sizeof(TfLiteTensor));
  t.bytes = kBytes;
  // t.delegate = &delegate;
  t.data_is_stale = true;
  t.allocation_type = kTfLiteDynamic;
  t.type = kTfLiteFloat32;
  t.dims = TfLiteIntArrayCreate(2);
  t.dims->data[0] = 4;
  t.dims->data[1] = 8;
  t.dims_signature = TfLiteIntArrayCopy(t.dims);
  t.buffer_handle = 5;

  TfLiteOpaqueTensor *opaque_t = CreateOpaqueTensor(&t);

  auto openvino_graph_builder_test =
      std::make_unique<OpenVINOGraphBuilder>(std::make_unique<NodeManager>());

  EXPECT_EQ(kTfLiteError,
            openvino_graph_builder_test->AddInputParams(opaque_t, -1));
  EXPECT_EQ(false, openvino_graph_builder_test->getNodeManagerSize() == 1);
}

TEST_F(OpenVINOGraphBuilderTest, CreateConstNode_Invalid) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  bool delegate_prepared = false;

  opaque_delegate_builder.data = &delegate_prepared;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext *opaque_context,
                                       TfLiteOpaqueDelegate *opaque_delegate_,
                                       void *data) -> TfLiteStatus {
    auto delegate_prepared = static_cast<bool *>(data);
    *delegate_prepared = true;
    TfLiteIntArray *execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    for (int i = 0; i < execution_plan->size; ++i) {
      TfLiteOpaqueNode *node = nullptr;
      TfLiteOperator *registration = nullptr;
      TfLiteOpaqueContextGetNodeAndRegistration(opaque_context, i, &node,
                                                &registration);
      auto openvino_graph_builder_test = std::make_unique<OpenVINOGraphBuilder>(
          std::make_unique<NodeManager>());

      EXPECT_EQ(kTfLiteError, openvino_graph_builder_test->CreateConstNode(
                                  opaque_context, 0));
    }

    TfLiteOperator *registration_external =
        TfLiteOperatorCreate(kTfLiteBuiltinDelegate,
                             /*name*/ nullptr,
                             /*version=*/1,
                             /*user_data=*/nullptr);
    return TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, registration_external, execution_plan,
        opaque_delegate_);
  };

  opaque_delegate_ = TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  builder.AddDelegate(opaque_delegate_);

  EXPECT_EQ(kTfLiteOk, builder(&interpreter_));
  EXPECT_TRUE(delegate_prepared);
  ASSERT_NE(interpreter_, nullptr);
}

TEST_F(OpenVINOGraphBuilderTest, CreateConstNode_InvalidContext) {
  auto openvino_graph_builder_test =
      std::make_unique<OpenVINOGraphBuilder>(std::make_unique<NodeManager>());
  EXPECT_EQ(kTfLiteError,
            openvino_graph_builder_test->CreateConstNode(nullptr, 0));
}

TEST_F(OpenVINOGraphBuilderTest, CreateConstNode_InvalidIndex) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  bool delegate_prepared = false;

  opaque_delegate_builder.data = &delegate_prepared;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext *opaque_context,
                                       TfLiteOpaqueDelegate *opaque_delegate_,
                                       void *data) -> TfLiteStatus {
    auto delegate_prepared = static_cast<bool *>(data);
    *delegate_prepared = true;
    TfLiteIntArray *execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    for (int i = 0; i < execution_plan->size; ++i) {
      TfLiteOpaqueNode *node = nullptr;
      TfLiteOperator *registration = nullptr;
      TfLiteOpaqueContextGetNodeAndRegistration(opaque_context, i, &node,
                                                &registration);
      auto openvino_graph_builder_test = std::make_unique<OpenVINOGraphBuilder>(
          std::make_unique<NodeManager>());

      EXPECT_EQ(kTfLiteError, openvino_graph_builder_test->CreateConstNode(
                                  opaque_context, 1));
    }

    TfLiteOperator *registration_external =
        TfLiteOperatorCreate(kTfLiteBuiltinDelegate,
                             /*name*/ nullptr,
                             /*version=*/1,
                             /*user_data=*/nullptr);
    return TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, registration_external, execution_plan,
        opaque_delegate_);
  };
  opaque_delegate_ = TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  builder.AddDelegate(opaque_delegate_);

  EXPECT_EQ(kTfLiteOk, builder(&interpreter_));
  EXPECT_TRUE(delegate_prepared);
  ASSERT_NE(interpreter_, nullptr);
}

TEST_F(OpenVINOGraphBuilderTest, UpdateResultNodes_Valid) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  struct UserData {
    bool delegate_prepared_ = false;
    ov::CompiledModel compiled_model_;
  } user_data;

  opaque_delegate_builder.data = &user_data;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext *opaque_context,
                                       TfLiteOpaqueDelegate *opaque_delegate_,
                                       void *data) -> TfLiteStatus {
    static_cast<UserData *>(data)->delegate_prepared_ = true;
    auto reg_ex = TfLiteOperatorCreate(kTfLiteBuiltinDelegate,
                                       "Test driver Openvino delegate",
                                       /*version=*/1, data);
    TfLiteOperatorSetInitWithData(
        reg_ex,
        [](void *user_data, TfLiteOpaqueContext *opaque_context,
           const char *buffer, size_t length) -> void * {
          auto openvino_graph_builder_test =
              std::make_unique<OpenVINOGraphBuilder>(
                  std::make_unique<NodeManager>());
          std::vector<int> compute_inputs_ = {};
          std::vector<int> outputs_ = {};
          const TfLiteOpaqueDelegateParams *params =
              reinterpret_cast<const TfLiteOpaqueDelegateParams *>(buffer);
          const std::unordered_set<int> inputs(
              &params->input_tensors->data[0],
              &params->input_tensors->data[params->input_tensors->size]);

          for (int o = 0; o < params->output_tensors->size; o++) {
            const int output_tensor_idx = params->output_tensors->data[o];
            outputs_.push_back(output_tensor_idx);
          }

          for (int i = 0; i < params->nodes_to_replace->size; ++i) {
            const int delegate_node_id = params->nodes_to_replace->data[i];
            TfLiteOpaqueNode *node = nullptr;
            TfLiteOperator *registration = nullptr;
            TfLiteOpaqueContextGetNodeAndRegistration(
                opaque_context, delegate_node_id, &node, &registration);

            int inputs_size = TfLiteOpaqueNodeNumberOfInputs(node);
            for (int k = 0; k < inputs_size; k++) {
              const int *inputs_data;
              int num_inputs;
              EXPECT_EQ(kTfLiteOk, TfLiteOpaqueNodeInputs(node, &inputs_data,
                                                          &num_inputs));
              const int t = inputs_data[k];
              const void *data = nullptr;
              auto opaque_tensor =
                  TfLiteOpaqueContextGetOpaqueTensor(opaque_context, t);
              EXPECT_NE(opaque_tensor, nullptr);

              auto allocation_type =
                  TfLiteOpaqueTensorGetAllocationType(opaque_tensor);
              if (allocation_type == kTfLiteMmapRo) {
                data = TfLiteOpaqueTensorData(opaque_tensor);
                EXPECT_NE(data, nullptr);
                if (openvino_graph_builder_test->CreateConstNode(
                        opaque_context, t) != kTfLiteOk)
                  exit(0);
              }

              if (inputs.count(t) != 0) {
                if (data == nullptr) {
                  if (openvino_graph_builder_test->AddInputParams(
                          opaque_tensor, t) != kTfLiteOk)
                    exit(0);
                  compute_inputs_.push_back(t);
                }
              }
            }
            if (openvino_graph_builder_test->CreateNodeFromTfLiteOp(
                    registration, node, opaque_context) != kTfLiteOk)
              exit(0);
          }

          EXPECT_EQ(kTfLiteOk, openvino_graph_builder_test->UpdateResultNodes(
                                   opaque_context, outputs_));
          std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(
              openvino_graph_builder_test->getResultNodes(),
              openvino_graph_builder_test->getInputParams());

          ov::Core ov_core_(kPluginsXmlPath);
          std::string deviceStr = "CPU";
          ov::InferRequest infer_request_;
          auto &compiled_model_ =
              static_cast<UserData *>(user_data)->compiled_model_;
          compiled_model_ = ov_core_.compile_model(model, deviceStr);

          infer_request_ = compiled_model_.create_infer_request();
          return &compiled_model_;
        });

    TfLiteOperatorSetInvoke(reg_ex,
                            [](TfLiteOpaqueContext *context,
                               TfLiteOpaqueNode *opaque_node) -> TfLiteStatus {
                              return kTfLiteOk;
                            });

    TfLiteOperatorSetFree(reg_ex,
                          [](TfLiteOpaqueContext *context, void *data) {});

    TfLiteIntArray *execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, reg_ex, execution_plan, opaque_delegate_);
    return kTfLiteOk;
  };

  TfLiteModel *model = TfLiteModelCreateFromFile(
      "external/org_tensorflow/tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteOpaqueDelegate *opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  EXPECT_TRUE(user_data.delegate_prepared_);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

TEST_F(OpenVINOGraphBuilderTest, UpdateResultNodes_InvalidContext) {
  auto openvino_graph_builder_test =
      std::make_unique<OpenVINOGraphBuilder>(std::make_unique<NodeManager>());
  EXPECT_EQ(kTfLiteError,
            openvino_graph_builder_test->UpdateResultNodes(nullptr, {0}));
}

TEST_F(OpenVINOGraphBuilderTest, UpdateResultNodes_InvalidIndex) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  struct UserData {
    bool delegate_prepared_ = false;
    ov::CompiledModel compiled_model_;
  } user_data;

  opaque_delegate_builder.data = &user_data;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext *opaque_context,
                                       TfLiteOpaqueDelegate *opaque_delegate_,
                                       void *data) -> TfLiteStatus {
    static_cast<UserData *>(data)->delegate_prepared_ = true;
    auto reg_ex = TfLiteOperatorCreate(kTfLiteBuiltinDelegate,
                                       "Test driver Openvino delegate",
                                       /*version=*/1, data);
    TfLiteOperatorSetInitWithData(
        reg_ex,
        [](void *user_data, TfLiteOpaqueContext *opaque_context,
           const char *buffer, size_t length) -> void * {
          auto openvino_graph_builder_test =
              std::make_unique<OpenVINOGraphBuilder>(
                  std::make_unique<NodeManager>());
          std::vector<int> compute_inputs_ = {};
          std::vector<int> outputs_ = {};
          const TfLiteOpaqueDelegateParams *params =
              reinterpret_cast<const TfLiteOpaqueDelegateParams *>(buffer);
          const std::unordered_set<int> inputs(
              &params->input_tensors->data[0],
              &params->input_tensors->data[params->input_tensors->size]);

          for (int o = 0; o < params->output_tensors->size; o++) {
            const int output_tensor_idx = params->output_tensors->data[o];
            outputs_.push_back(output_tensor_idx);
          }

          for (int i = 0; i < params->nodes_to_replace->size; ++i) {
            const int delegate_node_id = params->nodes_to_replace->data[i];
            TfLiteOpaqueNode *node = nullptr;
            TfLiteOperator *registration = nullptr;
            TfLiteOpaqueContextGetNodeAndRegistration(
                opaque_context, delegate_node_id, &node, &registration);

            int inputs_size = TfLiteOpaqueNodeNumberOfInputs(node);
            for (int k = 0; k < inputs_size; k++) {
              const int *inputs_data;
              int num_inputs;
              EXPECT_EQ(kTfLiteOk, TfLiteOpaqueNodeInputs(node, &inputs_data,
                                                          &num_inputs));
              const int t = inputs_data[k];
              const void *data = nullptr;
              auto opaque_tensor =
                  TfLiteOpaqueContextGetOpaqueTensor(opaque_context, t);
              EXPECT_NE(opaque_tensor, nullptr);
              auto allocation_type =
                  TfLiteOpaqueTensorGetAllocationType(opaque_tensor);
              if (allocation_type == kTfLiteMmapRo) {
                data = TfLiteOpaqueTensorData(opaque_tensor);
                EXPECT_NE(data, nullptr);

                if (openvino_graph_builder_test->CreateConstNode(
                        opaque_context, t) != kTfLiteOk)
                  exit(0);
              }
              if (inputs.count(t) != 0) {
                if (data == nullptr) {
                  static int count = 0;
                  if (count == 0) {
                    if (openvino_graph_builder_test->AddInputParams(
                            opaque_tensor, t) != kTfLiteOk)
                      exit(0);
                    compute_inputs_.push_back(t);
                    count++;
                  }
                }
              }
            }
            if (openvino_graph_builder_test->CreateNodeFromTfLiteOp(
                    registration, node, opaque_context) != kTfLiteOk)
              exit(0);
          }

          EXPECT_EQ(kTfLiteError,
                    openvino_graph_builder_test->UpdateResultNodes(
                        opaque_context, {}));
          std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(
              openvino_graph_builder_test->getResultNodes(),
              openvino_graph_builder_test->getInputParams());

          ov::Core ov_core_(kPluginsXmlPath);
          std::string deviceStr = "CPU";
          ov::InferRequest infer_request_;
          auto &compiled_model_ =
              static_cast<UserData *>(user_data)->compiled_model_;
          compiled_model_ = ov_core_.compile_model(model, deviceStr);

          infer_request_ = compiled_model_.create_infer_request();
          return &compiled_model_;
        });

    TfLiteOperatorSetInvoke(reg_ex,
                            [](TfLiteOpaqueContext *context,
                               TfLiteOpaqueNode *opaque_node) -> TfLiteStatus {
                              return kTfLiteOk;
                            });

    TfLiteOperatorSetFree(reg_ex,
                          [](TfLiteOpaqueContext *context, void *data) {});

    TfLiteIntArray *execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, reg_ex, execution_plan, opaque_delegate_);
    return kTfLiteOk;
  };

  TfLiteModel *model = TfLiteModelCreateFromFile(
      "external/org_tensorflow/tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteOpaqueDelegate *opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  EXPECT_TRUE(user_data.delegate_prepared_);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

TEST_F(OpenVINOGraphBuilderTest, CreateOpClass_Valid) {
  TfLiteOperator registration;
  registration.version = 1;
  registration.builtin_code = kTfLiteBuiltinAdd;
  registration.node_index = 1;

  std::shared_ptr<OperationsBase> op_base;
  auto openvino_graph_builder_test =
      std::make_unique<OpenVINOGraphBuilder>(std::make_unique<NodeManager>());

  EXPECT_EQ(kTfLiteOk,
            openvino_graph_builder_test->CreateOpClass(&registration, op_base));
}

TEST_F(OpenVINOGraphBuilderTest, CreateOpClass_InvalidRegistration) {
  std::shared_ptr<OperationsBase> op_base;
  auto openvino_graph_builder_test =
      std::make_unique<OpenVINOGraphBuilder>(std::make_unique<NodeManager>());

  EXPECT_EQ(kTfLiteError,
            openvino_graph_builder_test->CreateOpClass(nullptr, op_base));
}

TEST_F(OpenVINOGraphBuilderTest, CreateNodeFromTfLiteOp_Valid) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  struct UserData {
    bool delegate_prepared_ = false;
    ov::CompiledModel compiled_model_;
  } user_data;

  opaque_delegate_builder.data = &user_data;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext *opaque_context,
                                       TfLiteOpaqueDelegate *opaque_delegate_,
                                       void *data) -> TfLiteStatus {
    static_cast<UserData *>(data)->delegate_prepared_ = true;
    auto reg_ex = TfLiteOperatorCreate(kTfLiteBuiltinDelegate,
                                       "Test driver Openvino delegate",
                                       /*version=*/1, data);
    TfLiteOperatorSetInitWithData(
        reg_ex,
        [](void *user_data, TfLiteOpaqueContext *opaque_context,
           const char *buffer, size_t length) -> void * {
          auto openvino_graph_builder_test =
              std::make_unique<OpenVINOGraphBuilder>(
                  std::make_unique<NodeManager>());
          std::vector<int> compute_inputs_ = {};
          std::vector<int> outputs_ = {};
          const TfLiteOpaqueDelegateParams *params =
              reinterpret_cast<const TfLiteOpaqueDelegateParams *>(buffer);
          const std::unordered_set<int> inputs(
              &params->input_tensors->data[0],
              &params->input_tensors->data[params->input_tensors->size]);

          for (int o = 0; o < params->output_tensors->size; o++) {
            const int output_tensor_idx = params->output_tensors->data[o];
            outputs_.push_back(output_tensor_idx);
          }

          for (int i = 0; i < params->nodes_to_replace->size; ++i) {
            const int delegate_node_id = params->nodes_to_replace->data[i];
            TfLiteOpaqueNode *node = nullptr;
            TfLiteOperator *registration = nullptr;
            TfLiteOpaqueContextGetNodeAndRegistration(
                opaque_context, delegate_node_id, &node, &registration);

            int inputs_size = TfLiteOpaqueNodeNumberOfInputs(node);
            for (int k = 0; k < inputs_size; k++) {
              const int *inputs_data;
              int num_inputs;
              EXPECT_EQ(kTfLiteOk, TfLiteOpaqueNodeInputs(node, &inputs_data,
                                                          &num_inputs));
              const int t = inputs_data[k];
              const void *data = nullptr;
              auto opaque_tensor =
                  TfLiteOpaqueContextGetOpaqueTensor(opaque_context, t);
              EXPECT_NE(opaque_tensor, nullptr);
              auto allocation_type =
                  TfLiteOpaqueTensorGetAllocationType(opaque_tensor);
              if (allocation_type == kTfLiteMmapRo) {
                data = TfLiteOpaqueTensorData(opaque_tensor);
                EXPECT_NE(data, nullptr);

                if (openvino_graph_builder_test->CreateConstNode(
                        opaque_context, t) != kTfLiteOk)
                  exit(0);
              }
              if (inputs.count(t) != 0) {
                if (data == nullptr) {
                  static int count = 0;
                  if (count == 0) {
                    if (openvino_graph_builder_test->AddInputParams(
                            opaque_tensor, t) != kTfLiteOk)
                      exit(0);
                    compute_inputs_.push_back(t);
                    count++;
                  }
                }
              }
            }
            EXPECT_EQ(kTfLiteOk,
                      openvino_graph_builder_test->CreateNodeFromTfLiteOp(
                          registration, node, opaque_context));
          }

          openvino_graph_builder_test->UpdateResultNodes(opaque_context, {});
          std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(
              openvino_graph_builder_test->getResultNodes(),
              openvino_graph_builder_test->getInputParams());

          ov::Core ov_core_(kPluginsXmlPath);
          std::string deviceStr = "CPU";
          ov::InferRequest infer_request_;
          auto &compiled_model_ =
              static_cast<UserData *>(user_data)->compiled_model_;
          compiled_model_ = ov_core_.compile_model(model, deviceStr);

          infer_request_ = compiled_model_.create_infer_request();
          return &compiled_model_;
        });

    TfLiteOperatorSetInvoke(reg_ex,
                            [](TfLiteOpaqueContext *context,
                               TfLiteOpaqueNode *opaque_node) -> TfLiteStatus {
                              return kTfLiteOk;
                            });

    TfLiteOperatorSetFree(reg_ex,
                          [](TfLiteOpaqueContext *context, void *data) {});

    TfLiteIntArray *execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, reg_ex, execution_plan, opaque_delegate_);
    return kTfLiteOk;
  };

  TfLiteModel *model = TfLiteModelCreateFromFile(
      "external/org_tensorflow/tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteOpaqueDelegate *opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  EXPECT_TRUE(user_data.delegate_prepared_);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

TEST_F(OpenVINOGraphBuilderTest, CreateNodeFromTfLiteOp_InvalidRegistration) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  struct UserData {
    bool delegate_prepared_ = false;
    ov::CompiledModel compiled_model_;
  } user_data;

  opaque_delegate_builder.data = &user_data;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext *opaque_context,
                                       TfLiteOpaqueDelegate *opaque_delegate_,
                                       void *data) -> TfLiteStatus {
    static_cast<UserData *>(data)->delegate_prepared_ = true;
    auto reg_ex = TfLiteOperatorCreate(kTfLiteBuiltinDelegate,
                                       "Test driver Openvino delegate",
                                       /*version=*/1, data);
    TfLiteOperatorSetInitWithData(
        reg_ex,
        [](void *user_data, TfLiteOpaqueContext *opaque_context,
           const char *buffer, size_t length) -> void * {
          auto openvino_graph_builder_test =
              std::make_unique<OpenVINOGraphBuilder>(
                  std::make_unique<NodeManager>());
          std::vector<int> compute_inputs_ = {};
          std::vector<int> outputs_ = {};
          const TfLiteOpaqueDelegateParams *params =
              reinterpret_cast<const TfLiteOpaqueDelegateParams *>(buffer);
          const std::unordered_set<int> inputs(
              &params->input_tensors->data[0],
              &params->input_tensors->data[params->input_tensors->size]);

          for (int o = 0; o < params->output_tensors->size; o++) {
            const int output_tensor_idx = params->output_tensors->data[o];
            outputs_.push_back(output_tensor_idx);
          }

          for (int i = 0; i < params->nodes_to_replace->size; ++i) {
            const int delegate_node_id = params->nodes_to_replace->data[i];
            TfLiteOpaqueNode *node = nullptr;
            TfLiteOperator *registration = nullptr;
            TfLiteOpaqueContextGetNodeAndRegistration(
                opaque_context, delegate_node_id, &node, &registration);

            int inputs_size = TfLiteOpaqueNodeNumberOfInputs(node);
            for (int k = 0; k < inputs_size; k++) {
              const int *inputs_data;
              int num_inputs;
              EXPECT_EQ(kTfLiteOk, TfLiteOpaqueNodeInputs(node, &inputs_data,
                                                          &num_inputs));
              const int t = inputs_data[k];
              const void *data = nullptr;
              auto opaque_tensor =
                  TfLiteOpaqueContextGetOpaqueTensor(opaque_context, t);
              EXPECT_NE(opaque_tensor, nullptr);

              auto allocation_type =
                  TfLiteOpaqueTensorGetAllocationType(opaque_tensor);
              if (allocation_type == kTfLiteMmapRo) {
                data = TfLiteOpaqueTensorData(opaque_tensor);
                EXPECT_NE(data, nullptr);

                if (openvino_graph_builder_test->CreateConstNode(
                        opaque_context, t) != kTfLiteOk)
                  exit(0);
              }
              if (inputs.count(t) != 0) {
                if (data == nullptr) {
                  static int count = 0;
                  if (count == 0) {
                    if (openvino_graph_builder_test->AddInputParams(
                            opaque_tensor, t) != kTfLiteOk)
                      exit(0);
                    compute_inputs_.push_back(t);
                    count++;
                  }
                }
              }
            }
            EXPECT_EQ(kTfLiteError,
                      openvino_graph_builder_test->CreateNodeFromTfLiteOp(
                          nullptr, node, opaque_context));
          }

          openvino_graph_builder_test->UpdateResultNodes(opaque_context, {});
          std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(
              openvino_graph_builder_test->getResultNodes(),
              openvino_graph_builder_test->getInputParams());

          ov::Core ov_core_(kPluginsXmlPath);
          std::string deviceStr = "CPU";
          ov::InferRequest infer_request_;
          auto &compiled_model_ =
              static_cast<UserData *>(user_data)->compiled_model_;
          compiled_model_ = ov_core_.compile_model(model, deviceStr);

          infer_request_ = compiled_model_.create_infer_request();
          return &compiled_model_;
        });

    TfLiteOperatorSetInvoke(reg_ex,
                            [](TfLiteOpaqueContext *context,
                               TfLiteOpaqueNode *opaque_node) -> TfLiteStatus {
                              return kTfLiteOk;
                            });

    TfLiteOperatorSetFree(reg_ex,
                          [](TfLiteOpaqueContext *context, void *data) {});

    TfLiteIntArray *execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, reg_ex, execution_plan, opaque_delegate_);
    return kTfLiteOk;
  };

  TfLiteModel *model = TfLiteModelCreateFromFile(
      "external/org_tensorflow/tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteOpaqueDelegate *opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  EXPECT_TRUE(user_data.delegate_prepared_);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

TEST_F(OpenVINOGraphBuilderTest, CreateNodeFromTfLiteOp_InvalidNode) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  struct UserData {
    bool delegate_prepared_ = false;
    ov::CompiledModel compiled_model_;
  } user_data;

  opaque_delegate_builder.data = &user_data;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext *opaque_context,
                                       TfLiteOpaqueDelegate *opaque_delegate_,
                                       void *data) -> TfLiteStatus {
    static_cast<UserData *>(data)->delegate_prepared_ = true;
    auto reg_ex = TfLiteOperatorCreate(kTfLiteBuiltinDelegate,
                                       "Test driver Openvino delegate",
                                       /*version=*/1, data);
    TfLiteOperatorSetInitWithData(
        reg_ex,
        [](void *user_data, TfLiteOpaqueContext *opaque_context,
           const char *buffer, size_t length) -> void * {
          auto openvino_graph_builder_test =
              std::make_unique<OpenVINOGraphBuilder>(
                  std::make_unique<NodeManager>());
          std::vector<int> compute_inputs_ = {};
          std::vector<int> outputs_ = {};
          const TfLiteOpaqueDelegateParams *params =
              reinterpret_cast<const TfLiteOpaqueDelegateParams *>(buffer);
          const std::unordered_set<int> inputs(
              &params->input_tensors->data[0],
              &params->input_tensors->data[params->input_tensors->size]);

          for (int o = 0; o < params->output_tensors->size; o++) {
            const int output_tensor_idx = params->output_tensors->data[o];
            outputs_.push_back(output_tensor_idx);
          }

          for (int i = 0; i < params->nodes_to_replace->size; ++i) {
            const int delegate_node_id = params->nodes_to_replace->data[i];
            TfLiteOpaqueNode *node = nullptr;
            TfLiteOperator *registration = nullptr;
            TfLiteOpaqueContextGetNodeAndRegistration(
                opaque_context, delegate_node_id, &node, &registration);

            int inputs_size = TfLiteOpaqueNodeNumberOfInputs(node);
            for (int k = 0; k < inputs_size; k++) {
              const int *inputs_data;
              int num_inputs;
              EXPECT_EQ(kTfLiteOk, TfLiteOpaqueNodeInputs(node, &inputs_data,
                                                          &num_inputs));
              const int t = inputs_data[k];
              const void *data = nullptr;
              auto opaque_tensor =
                  TfLiteOpaqueContextGetOpaqueTensor(opaque_context, t);
              EXPECT_NE(opaque_tensor, nullptr);

              auto allocation_type =
                  TfLiteOpaqueTensorGetAllocationType(opaque_tensor);
              if (allocation_type == kTfLiteMmapRo) {
                data = TfLiteOpaqueTensorData(opaque_tensor);
                EXPECT_NE(data, nullptr);

                if (openvino_graph_builder_test->CreateConstNode(
                        opaque_context, t) != kTfLiteOk)
                  exit(0);
              }
              if (inputs.count(t) != 0) {
                if (data == nullptr) {
                  static int count = 0;
                  if (count == 0) {
                    if (openvino_graph_builder_test->AddInputParams(
                            opaque_tensor, t) != kTfLiteOk)
                      exit(0);
                    compute_inputs_.push_back(t);
                    count++;
                  }
                }
              }
            }
            EXPECT_EQ(kTfLiteError,
                      openvino_graph_builder_test->CreateNodeFromTfLiteOp(
                          registration, nullptr, opaque_context));
          }

          openvino_graph_builder_test->UpdateResultNodes(opaque_context, {});
          std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(
              openvino_graph_builder_test->getResultNodes(),
              openvino_graph_builder_test->getInputParams());

          ov::Core ov_core_(kPluginsXmlPath);
          std::string deviceStr = "CPU";
          ov::InferRequest infer_request_;
          auto &compiled_model_ =
              static_cast<UserData *>(user_data)->compiled_model_;
          compiled_model_ = ov_core_.compile_model(model, deviceStr);

          infer_request_ = compiled_model_.create_infer_request();
          return &compiled_model_;
        });

    TfLiteOperatorSetInvoke(reg_ex,
                            [](TfLiteOpaqueContext *context,
                               TfLiteOpaqueNode *opaque_node) -> TfLiteStatus {
                              return kTfLiteOk;
                            });

    TfLiteOperatorSetFree(reg_ex,
                          [](TfLiteOpaqueContext *context, void *data) {});

    TfLiteIntArray *execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, reg_ex, execution_plan, opaque_delegate_);
    return kTfLiteOk;
  };

  TfLiteModel *model = TfLiteModelCreateFromFile(
      "external/org_tensorflow/tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteOpaqueDelegate *opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  EXPECT_TRUE(user_data.delegate_prepared_);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

TEST_F(OpenVINOGraphBuilderTest, CreateNodeFromTfLiteOp_InvalidContext) {
  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  struct UserData {
    bool delegate_prepared_ = false;
    ov::CompiledModel compiled_model_;
  } user_data;

  opaque_delegate_builder.data = &user_data;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext *opaque_context,
                                       TfLiteOpaqueDelegate *opaque_delegate_,
                                       void *data) -> TfLiteStatus {
    static_cast<UserData *>(data)->delegate_prepared_ = true;
    auto reg_ex = TfLiteOperatorCreate(kTfLiteBuiltinDelegate,
                                       "Test driver Openvino delegate",
                                       /*version=*/1, data);
    TfLiteOperatorSetInitWithData(
        reg_ex,
        [](void *user_data, TfLiteOpaqueContext *opaque_context,
           const char *buffer, size_t length) -> void * {
          auto openvino_graph_builder_test =
              std::make_unique<OpenVINOGraphBuilder>(
                  std::make_unique<NodeManager>());
          std::vector<int> compute_inputs_ = {};
          std::vector<int> outputs_ = {};
          const TfLiteOpaqueDelegateParams *params =
              reinterpret_cast<const TfLiteOpaqueDelegateParams *>(buffer);
          const std::unordered_set<int> inputs(
              &params->input_tensors->data[0],
              &params->input_tensors->data[params->input_tensors->size]);

          for (int o = 0; o < params->output_tensors->size; o++) {
            const int output_tensor_idx = params->output_tensors->data[o];
            outputs_.push_back(output_tensor_idx);
          }

          for (int i = 0; i < params->nodes_to_replace->size; ++i) {
            const int delegate_node_id = params->nodes_to_replace->data[i];
            TfLiteOpaqueNode *node = nullptr;
            TfLiteOperator *registration = nullptr;
            TfLiteOpaqueContextGetNodeAndRegistration(
                opaque_context, delegate_node_id, &node, &registration);

            int inputs_size = TfLiteOpaqueNodeNumberOfInputs(node);
            for (int k = 0; k < inputs_size; k++) {
              const int *inputs_data;
              int num_inputs;
              EXPECT_EQ(kTfLiteOk, TfLiteOpaqueNodeInputs(node, &inputs_data,
                                                          &num_inputs));
              const int t = inputs_data[k];
              const void *data = nullptr;
              auto opaque_tensor =
                  TfLiteOpaqueContextGetOpaqueTensor(opaque_context, t);
              auto allocation_type =
                  TfLiteOpaqueTensorGetAllocationType(opaque_tensor);
              EXPECT_NE(opaque_tensor, nullptr);

              if (allocation_type == kTfLiteMmapRo) {
                data = TfLiteOpaqueTensorData(opaque_tensor);
                EXPECT_NE(data, nullptr);

                if (openvino_graph_builder_test->CreateConstNode(
                        opaque_context, t) != kTfLiteOk)
                  exit(0);
              }
              if (inputs.count(t) != 0) {
                if (data == nullptr) {
                  static int count = 0;
                  if (count == 0) {
                    if (openvino_graph_builder_test->AddInputParams(
                            opaque_tensor, t) != kTfLiteOk)
                      exit(0);
                    compute_inputs_.push_back(t);
                    count++;
                  }
                }
              }
            }
            EXPECT_EQ(kTfLiteError,
                      openvino_graph_builder_test->CreateNodeFromTfLiteOp(
                          registration, node, nullptr));
          }

          openvino_graph_builder_test->UpdateResultNodes(opaque_context, {});
          std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(
              openvino_graph_builder_test->getResultNodes(),
              openvino_graph_builder_test->getInputParams());

          ov::Core ov_core_(kPluginsXmlPath);
          std::string deviceStr = "CPU";
          ov::InferRequest infer_request_;
          auto &compiled_model_ =
              static_cast<UserData *>(user_data)->compiled_model_;
          compiled_model_ = ov_core_.compile_model(model, deviceStr);

          infer_request_ = compiled_model_.create_infer_request();
          return &compiled_model_;
        });

    TfLiteOperatorSetInvoke(reg_ex,
                            [](TfLiteOpaqueContext *context,
                               TfLiteOpaqueNode *opaque_node) -> TfLiteStatus {
                              return kTfLiteOk;
                            });

    TfLiteOperatorSetFree(reg_ex,
                          [](TfLiteOpaqueContext *context, void *data) {});

    TfLiteIntArray *execution_plan;
    TF_LITE_ENSURE_STATUS(
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
    TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, reg_ex, execution_plan, opaque_delegate_);
    return kTfLiteOk;
  };

  TfLiteModel *model = TfLiteModelCreateFromFile(
      "external/org_tensorflow/tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteOpaqueDelegate *opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  EXPECT_TRUE(user_data.delegate_prepared_);

  TfLiteInterpreterOptionsDelete(options);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

}  // namespace openvinodelegate
}  // namespace tflite

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
