/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/single_op_model_builder.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"

namespace tflite::cros {

SingleOpModelBuilder& SingleOpModelBuilder::AddInput(
    TfLiteType type,
    const std::vector<int>& shape) {
  std::string name = absl::StrCat("input", inputs_.size());
  int input = simple_model_builder_.AddInput({
      .name = name,
      .type = type,
      .shape = shape,
  });
  inputs_.push_back(input);
  return *this;
}

SingleOpModelBuilder& SingleOpModelBuilder::AddOutput(
    TfLiteType type,
    const std::vector<int>& shape) {
  std::string name = absl::StrCat("output", outputs_.size());
  int output = simple_model_builder_.AddOutput({
      .name = name,
      .type = type,
      .shape = shape,
  });
  outputs_.push_back(output);
  return *this;
}

std::unique_ptr<FlatBufferModel> SingleOpModelBuilder::Build() const {
  return simple_model_builder_.Build();
}

}  // namespace tflite::cros
