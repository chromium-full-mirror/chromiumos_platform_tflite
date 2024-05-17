/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "common/log.h"
#include "flatbuffers/idl.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tool/schema_fbs_embed-inl.h"

namespace tflite::cros {

void Main(int argc, const char** argv) {
  std::string input_path;
  std::string output_path = "/dev/stdout";
  std::vector<Flag> flags = {
      Flag::CreateFlag("input", &input_path, "input model path to .tflite",
                       Flag::kRequired),
      Flag::CreateFlag("output", &output_path, "output path for JSON",
                       Flag::kOptional),
  };
  if (!Flags::Parse(&argc, argv, flags) || argc != 1) {
    std::cerr << Flags::Usage("tflite_to_json", flags) << std::endl;
    exit(1);
  }

  std::unique_ptr<tflite::FlatBufferModel> model =
      FlatBufferModel::VerifyAndBuildFromFile(input_path.c_str());
  CHECK(model != nullptr) << "Failed to load model";

  flatbuffers::Parser parser;
  parser.opts.strict_json = true;
  // schema_fbs is defined in schema_fbs_embed-inl.h without the null
  // terminator, so we construct a std::string here first.
  std::string schema(reinterpret_cast<const char*>(schema_fbs), schema_fbs_len);
  CHECK(parser.Parse(schema.c_str())) << "Failed to parse schema";

  std::string json;
  const char* err = flatbuffers::GenTextFromTable(parser, model->GetModel(),
                                                  "tflite.Model", &json);
  CHECK(err == nullptr) << "Failed to generate json: " << err;

  std::ofstream out(output_path);
  out << json;
}

};  // namespace tflite::cros

int main(int argc, const char** argv) {
  tflite::cros::Main(argc, argv);
  return 0;
}
