/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef COMMON_TEST_UTIL_STABLE_DELEGATE_ENV_H_
#define COMMON_TEST_UTIL_STABLE_DELEGATE_ENV_H_

#include <gtest/gtest.h>

#include <vector>

#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/tool_params.h"

// An environment that loads the settings file from command line and creates a
// stable delegate accordingly.
class StableDelegateEnvironment : public ::testing::Environment {
  using ProvidedDelegateList = tflite::tools::ProvidedDelegateList;
  using ToolParams = tflite::tools::ToolParams;
  using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;

 public:
  StableDelegateEnvironment();
  ~StableDelegateEnvironment() override;

  // Initializes the environment with optional extra command line flags that
  // should be parsed together with the stable delegate flag.
  bool InitFromCommandLine(int* argc,
                           char** argv,
                           std::vector<tflite::Flag> flags = {});

  // Creates a new delegate instance. The caller owns the delegate.
  TfLiteDelegatePtr CreateDelegate();

  // Retrieves a delegate instance, possibly reusing a cached instance. The
  // testing environment owns the delegate.
  TfLiteDelegate* GetDelegate();

 protected:
  void SetUp() override;
  void TearDown() override;

 private:
  ToolParams params_;
  ProvidedDelegateList delegate_list_;
  TfLiteDelegatePtr cached_delegate_ = tflite::tools::CreateNullDelegate();
};

#endif  // COMMON_TEST_UTIL_STABLE_DELEGATE_ENV_H_
