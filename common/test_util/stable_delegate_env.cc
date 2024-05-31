/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/test_util/stable_delegate_env.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

StableDelegateEnvironment::StableDelegateEnvironment()
    : delegate_list_(&params_) {}

StableDelegateEnvironment::~StableDelegateEnvironment() {}

bool StableDelegateEnvironment::InitFromCommandLine(
    int* argc,
    char** argv,
    std::vector<tflite::Flag> flags) {
  using tflite::Flag;
  using tflite::Flags;

  constexpr char kSettingsFlagName[] = "stable_delegate_settings_file";

  std::string settings;
  // Use our own flags directly instead of delegate_list_.AppendCmdlineFlags()
  // to make the settings flag required.
  flags.push_back(Flag::CreateFlag(
      kSettingsFlagName, &settings,
      "The path to the delegate settings JSON file.", Flag::kRequired));

  if (!Flags::Parse(argc, const_cast<const char**>(argv), flags)) {
    std::cout << Flags::Usage(argv[0], flags) << std::endl;
    return false;
  }

  delegate_list_.AddAllDelegateParams();
  params_.Set<std::string>(kSettingsFlagName, settings);

  return true;
}

tflite::Interpreter::TfLiteDelegatePtr
StableDelegateEnvironment::CreateDelegate() {
  std::vector<ProvidedDelegateList::ProvidedDelegate> provided_delegates =
      delegate_list_.CreateAllRankedDelegates();
  EXPECT_EQ(provided_delegates.size(), 1);
  if (provided_delegates.empty()) {
    return tflite::tools::CreateNullDelegate();
  }
  return std::move(provided_delegates[0].delegate);
}

TfLiteDelegate* StableDelegateEnvironment::GetDelegate() {
  if (cached_delegate_ == nullptr) {
    cached_delegate_ = CreateDelegate();
  }
  return cached_delegate_.get();
}

void StableDelegateEnvironment::SetUp() {
  // Ensure that the delegate can be created before running any test.
  ASSERT_NE(CreateDelegate(), nullptr);
}

void StableDelegateEnvironment::TearDown() {
  cached_delegate_.reset();
}
