/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#include "common/simple_async_delegate.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "tensorflow/lite/array.h"
#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/core/c/c_api_opaque.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite::cros {
namespace {

// The custom op for the pre-compiled MTK graph.
constexpr char kCrosMtkPreCompile[] = "cros-mtk-pre-compile";

TfLiteRegistrationExternal* CreateDelegateKernelRegistration(
    SimpleAsyncDelegateInterface* delegate) {
  TfLiteRegistrationExternal* kernel_registration =
      TfLiteOperatorCreate(kTfLiteBuiltinDelegate, delegate->Name(),
                           /*version=*/1, /*user_data=*/nullptr);

  TfLiteOperatorSetFreeWithData(
      kernel_registration,
      [](void* user_data, TfLiteOpaqueContext* context, void* buffer) -> void {
        delete reinterpret_cast<SimpleAsyncDelegateKernelInterface*>(buffer);
      });

  TfLiteOperatorSetInitWithData(
      kernel_registration,
      [](void* user_data, TfLiteOpaqueContext* context, const char* buffer,
         size_t length) -> void* {
        const TfLiteOpaqueDelegateParams* params =
            reinterpret_cast<const TfLiteOpaqueDelegateParams*>(buffer);
        if (params == nullptr) {
          return nullptr;
        }
        auto* delegate_data = reinterpret_cast<SimpleAsyncDelegateInterface*>(
            params->delegate_data);
        std::unique_ptr<SimpleAsyncDelegateKernelInterface> delegate_kernel(
            delegate_data->CreateDelegateKernelInterface());
        if (delegate_kernel->Init(context, params) != kTfLiteOk) {
          return nullptr;
        }
        return delegate_kernel.release();
      });
  TfLiteOperatorSetPrepareWithData(
      kernel_registration,
      [](void* user_data, TfLiteOpaqueContext* context,
         TfLiteOpaqueNode* opaque_node) -> TfLiteStatus {
        SimpleAsyncDelegateKernelInterface* delegate_kernel =
            reinterpret_cast<SimpleAsyncDelegateKernelInterface*>(
                TfLiteOpaqueNodeGetUserData(opaque_node));
        return delegate_kernel->Prepare(context, opaque_node);
      });
  TfLiteOperatorSetInvokeWithData(
      kernel_registration,
      [](void* user_data, TfLiteOpaqueContext* context,
         TfLiteOpaqueNode* opaque_node) -> TfLiteStatus {
        SimpleAsyncDelegateKernelInterface* delegate_kernel =
            reinterpret_cast<SimpleAsyncDelegateKernelInterface*>(
                TfLiteOpaqueNodeGetUserData(opaque_node));
        TFLITE_DCHECK(delegate_kernel != nullptr);
        return delegate_kernel->Eval(context, opaque_node);
      });
  TfLiteOperatorSetAsyncKernelWithData(
      kernel_registration,
      [](void* user_data, TfLiteOpaqueContext* context,
         TfLiteOpaqueNode* opaque_node) -> TfLiteAsyncKernel* {
        SimpleAsyncDelegateKernelInterface* delegate_kernel =
            reinterpret_cast<SimpleAsyncDelegateKernelInterface*>(
                TfLiteOpaqueNodeGetUserData(opaque_node));
        TFLITE_DCHECK(delegate_kernel != nullptr);
        return delegate_kernel->AsyncKernel(context, opaque_node);
      });

  return kernel_registration;
}

std::vector<int> GetSupportedNodes(
    TfLiteOpaqueContext* opaque_context,
    SimpleAsyncDelegateInterface* simple_async_delegate,
    IntArrayUniquePtr plan) {
  std::vector<int> supported_nodes;

  // TODO(b/374028782): Remove this after we have fine-grained delegation
  // control in TFLite.
  for (int i = 0; i < plan->size; ++i) {
    const int node_id = plan->data[i];

    TfLiteOpaqueNode* opaque_node;
    TfLiteRegistrationExternal* registration_external;
    TfLiteOpaqueContextGetNodeAndRegistration(
        opaque_context, node_id, &opaque_node, &registration_external);

    const char* custom = TfLiteOperatorGetCustomName(registration_external);
    if (custom && strcmp(custom, kCrosMtkPreCompile) == 0) {
      supported_nodes.push_back(node_id);
    }
  }
  // If the graph contains pre-compiled graph, we don't need to compile the
  // remaining graph again. And the remaining graph should be running on CPU.
  if (!supported_nodes.empty()) {
    return supported_nodes;
  }

  for (int i = 0; i < plan->size; ++i) {
    const int node_id = plan->data[i];

    TfLiteOpaqueNode* opaque_node;
    TfLiteRegistrationExternal* registration_external;
    TfLiteOpaqueContextGetNodeAndRegistration(
        opaque_context, node_id, &opaque_node, &registration_external);

    if (simple_async_delegate->IsNodeSupportedByDelegate(
            registration_external, opaque_node, opaque_context)) {
      supported_nodes.push_back(node_id);
    }
  }
  return supported_nodes;
}

TfLiteStatus DelegatePrepare(TfLiteOpaqueContext* opaque_context,
                             TfLiteOpaqueDelegate* opaque_delegate,
                             void* data) {
  auto* simple_async_delegate =
      reinterpret_cast<SimpleAsyncDelegateInterface*>(data);
  TF_LITE_ENSURE_STATUS(simple_async_delegate->Initialize(opaque_context));

  TfLiteIntArray* execution_plan;
  TF_LITE_ENSURE_STATUS(
      TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan));
  IntArrayUniquePtr plan(TfLiteIntArrayCopy(execution_plan));

  std::vector<int> supported_nodes =
      GetSupportedNodes(opaque_context, simple_async_delegate, std::move(plan));

  TfLiteRegistrationExternal* delegate_kernel_registration =
      CreateDelegateKernelRegistration(simple_async_delegate);

  // Transfers ownership of delegate_kernel_registration to the opaque_context.
  return TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
      opaque_context, delegate_kernel_registration,
      BuildTfLiteArray(supported_nodes).get(), opaque_delegate);
}
}  // namespace

TfLiteOpaqueDelegate* SimpleAsyncDelegateFactory::CreateAsyncDelegate(
    std::unique_ptr<SimpleAsyncDelegateInterface> delegate,
    int64_t flags) {
  if (delegate == nullptr) {
    return {};
  }

  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  opaque_delegate_builder.Prepare = &DelegatePrepare;
  opaque_delegate_builder.flags = flags;
  opaque_delegate_builder.data = delegate.release();

  return TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
}

void SimpleAsyncDelegateFactory::DeleteAsyncDelegate(
    TfLiteOpaqueDelegate* opaque_delegate) {
  if (!opaque_delegate)
    return;
  auto* delegate = reinterpret_cast<SimpleAsyncDelegateInterface*>(
      TfLiteOpaqueDelegateGetData(opaque_delegate));
  delete delegate;
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

}  // namespace tflite::cros
