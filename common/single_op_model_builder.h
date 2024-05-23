/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef COMMON_SINGLE_OP_MODEL_BUILDER_H_
#define COMMON_SINGLE_OP_MODEL_BUILDER_H_

#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "common/log.h"
#include "common/simple_model_builder.h"

namespace tflite::cros {

class SingleOpModelBuilder {
 public:
  using Self = SingleOpModelBuilder;

  // Copyable and movable.
  SingleOpModelBuilder() = default;
  SingleOpModelBuilder(const SingleOpModelBuilder& other) = default;
  SingleOpModelBuilder& operator=(const SingleOpModelBuilder& other) = default;
  SingleOpModelBuilder(SingleOpModelBuilder&& other) = default;
  SingleOpModelBuilder& operator=(SingleOpModelBuilder&& other) = default;

  Self& AddInput(TfLiteType type, const std::vector<int>& shape);

  template <typename T>
  Self& AddConstInput(TfLiteType type,
                      const std::vector<int>& shape,
                      const std::vector<T>& data) {
    std::vector<uint8_t> raw_data(data.size() * sizeof(T));
    memcpy(raw_data.data(), data.data(), raw_data.size());
    int buffer = simple_model_builder_.AddBuffer(std::move(raw_data));
    int input = simple_model_builder_.AddInput({
        .type = type,
        .shape = shape,
        .buffer = buffer,
    });
    inputs_.push_back(input);
    return *this;
  }

  Self& AddOutput(TfLiteType type, const std::vector<int>& shape);

  template <typename T = std::monostate>
  Self& AddOperator(TfLiteBuiltinOperator op, T params = {}) {
    CHECK(!operator_added_);
    simple_model_builder_.AddOperator<T>({
        .op = op,
        .inputs = inputs_,
        .outputs = outputs_,
        .params = params,
    });
    return *this;
  }

  std::unique_ptr<FlatBufferModel> Build() const;

 private:
  SimpleModelBuilder simple_model_builder_;
  std::vector<int> inputs_;
  std::vector<int> outputs_;
  bool operator_added_ = false;
};

}  // namespace tflite::cros
#endif  // COMMON_SINGLE_OP_MODEL_BUILDER_H_
