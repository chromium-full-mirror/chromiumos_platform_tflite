/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef COMMON_SIMPLE_ASYNC_DELEGATE_H_
#define COMMON_SIMPLE_ASYNC_DELEGATE_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"

namespace tflite::cros {

using TfLiteOpaqueDelegateUniquePtr =
    std::unique_ptr<TfLiteOpaqueDelegate, void (*)(TfLiteOpaqueDelegate*)>;

// Users should inherit from this class and implement the interface below.
// Each instance represents a single part of the graph (subgraph).
class SimpleAsyncDelegateKernelInterface {
 public:
  virtual ~SimpleAsyncDelegateKernelInterface() = default;

  // Initializes a delegated subgraph.
  // The nodes in the subgraph are inside
  // TfLiteOpaqueDelegateParams->nodes_to_replace
  virtual TfLiteStatus Init(TfLiteOpaqueContext* context,
                            const TfLiteOpaqueDelegateParams* params) = 0;

  // Will be called by the framework. Should handle any needed preparation
  // for the subgraph e.g. allocating buffers, compiling model.
  // Returns status, and signalling any errors.
  virtual TfLiteStatus Prepare(TfLiteOpaqueContext* context,
                               TfLiteOpaqueNode* node) = 0;

  // Actual subgraph inference should happen on this call.
  // Returns status, and signalling any errors.
  // NOTE: Tensor data pointers (tensor->data) can change every inference, so
  // the implementation of this method needs to take that into account.
  virtual TfLiteStatus Eval(TfLiteOpaqueContext* context,
                            TfLiteOpaqueNode* node) = 0;

  // Retrieves the async kernel.
  // Returns nullptr if the delegate does not support asynchronous execution.
  virtual TfLiteAsyncKernel* AsyncKernel(TfLiteOpaqueContext* context,
                                         TfLiteOpaqueNode* node) = 0;
};

// Pure Interface that clients should implement.
// The Interface represents a delegate's capabilities and provides a factory
// for SimpleAsyncDelegateKernelInterface.
//
// Clients should implement the following methods:
// - IsNodeSupportedByDelegate - Initialize
// - Name
// - CreateDelegateKernelInterface
class SimpleAsyncDelegateInterface {
 public:
  virtual ~SimpleAsyncDelegateInterface() = default;

  // Returns true if 'node' is supported by the delegate. False otherwise.
  virtual bool IsNodeSupportedByDelegate(
      const TfLiteRegistrationExternal* registration_external,
      const TfLiteOpaqueNode* node,
      TfLiteOpaqueContext* context) const = 0;

  // Initialize the delegate before finding and replacing TfLite nodes with
  // delegate kernels, for example, retrieving some TFLite settings from
  // 'context'.
  virtual TfLiteStatus Initialize(TfLiteOpaqueContext* context) = 0;

  // Returns a name that identifies the delegate.
  // This name is used for debugging/logging/profiling.
  virtual const char* Name() const = 0;

  // Returns instance of an object that implements the interface
  // SimpleAsyncDelegateKernelInterface.
  // An instance of SimpleAsyncDelegateKernelInterface represents one subgraph
  // to be delegated.
  // Caller takes ownership of the returned object.
  virtual std::unique_ptr<SimpleAsyncDelegateKernelInterface>
  CreateDelegateKernelInterface() = 0;
};

// Factory class that provides static methods to deal with SimpleAsyncDelegate
// creation and deletion.
class SimpleAsyncDelegateFactory {
 public:
  // Creates TfLiteDelegate from the provided SimpleAsyncDelegateInterface.
  // The returned TfLiteDelegate should be deleted using DeleteAsyncDelegate.
  // A simple usage of the flags bit mask:
  // CreateAsyncDelegate(..., kTfLiteDelegateFlagsAllowDynamicTensors |
  // kTfLiteDelegateFlagsRequirePropagatedShapes)
  static TfLiteOpaqueDelegate* CreateAsyncDelegate(
      std::unique_ptr<SimpleAsyncDelegateInterface> delegate,
      int64_t flags = kTfLiteDelegateFlagsNone);

  // Deletes 'delegate' the passed pointer must be the one returned from
  // CreateAsyncDelegate. This function will destruct the SimpleAsyncDelegate
  // object too.
  static void DeleteAsyncDelegate(TfLiteOpaqueDelegate* opaque_delegate);

  // A convenient function wrapping the above two functions and returning a
  // std::unique_ptr type for auto memory management.
  inline static TfLiteOpaqueDelegateUniquePtr Create(
      std::unique_ptr<SimpleAsyncDelegateInterface> delegate) {
    return TfLiteOpaqueDelegateUniquePtr(
        CreateAsyncDelegate(std::move(delegate)), DeleteAsyncDelegate);
  }
};

}  // namespace tflite::cros

#endif  // COMMON_SIMPLE_ASYNC_DELEGATE_H_
