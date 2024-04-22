/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <fstream>
#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/schema/schema_utils.h"

ABSL_FLAG(std::string, input, "", "input tflite model");
ABSL_FLAG(std::string, output, "", "output tflite model");
ABSL_FLAG(int, op_index, -1, "operator index");

namespace {

constexpr absl::string_view kUsage = R"(
decompose_model <list|decompose> [options]

Available subcommands:
list --input=<input tflite file>
  Lists all operators in the |input| and shows the their information.

decompose --input=<input tflite file> --output=<output tflite file>
          --op_index=<operator index>
  Decomposes the |input| model by extracting the operator with index |op_index|
  and rebuild a single-op model with the extracted operator. The new model
  is saved to |output|.
)";

void PrintUsage() {
  std::cerr << kUsage << std::endl;
}

std::string TableStringify(const std::vector<std::vector<std::string>>& rows) {
  if (rows.empty()) {
    return "";
  }
  std::vector<int> widths(rows[0].size());
  for (auto& row : rows) {
    for (int i = 0; i < row.size(); ++i) {
      widths[i] = std::max(widths[i], static_cast<int>(row[i].size()));
    }
  }
  std::string message;
  for (auto& row : rows) {
    std::vector<std::string> items;
    for (int i = 0; i < row.size(); ++i) {
      items.push_back(absl::StrFormat("%-*s", widths[i], row[i]));
    }
    absl::StrAppendFormat(&message, "%s\n", absl::StrJoin(items, "    "));
  }
  return message;
}

std::unique_ptr<tflite::ModelT> GetUnpackedModel(const std::string& path) {
  std::unique_ptr<tflite::FlatBufferModel> fb_model =
      tflite::FlatBufferModel::BuildFromFile(path.c_str());
  if (!fb_model) {
    std::cerr << "Failed to build model from file." << std::endl;
    return nullptr;
  }
  std::unique_ptr<tflite::ModelT> model(fb_model->GetModel()->UnPack());
  if (!model) {
    std::cerr << "Failed to unpack model." << std::endl;
    return nullptr;
  }
  return model;
}

// Decomposes |model|. Extracts the operator of index |op_index| as a single-op
// model and re-pack it into |model|.
bool DecomposeModel(tflite::ModelT& model, int op_index) {
  if (model.subgraphs.size() != 1) {
    return false;
  }
  auto& subgraph = model.subgraphs[0];

  // Prunes the other operators and keeps only operator with index |op_index|.
  // TODO: b/331900692 - Also prunes the unused tensor.
  auto& operators = subgraph->operators;
  if (operators.size() <= op_index) {
    return false;
  }
  operators[0] = std::move(operators[op_index]);
  operators.resize(1);

  // Reassigns input/output of the remaining operator with empty buffers to
  // input/output of the subgraph.
  auto& op = operators[0];
  subgraph->inputs.clear();
  for (int input : op->inputs) {
    int buffer_idx = subgraph->tensors[input]->buffer;
    // Input of the subgraph should contain only empty buffers.
    if (model.buffers[buffer_idx]->data.size() == 0) {
      subgraph->inputs.push_back(input);
    }
    // TODO: b/331900692 - Randomize the buffer data of the other inputs.
  }
  subgraph->outputs = op->outputs;
  return true;
}

bool Decompose(const std::string& input_path,
               const std::string& output_path,
               int op_index) {
  std::unique_ptr<tflite::ModelT> model = GetUnpackedModel(input_path);
  if (!model) {
    return false;
  }

  if (!DecomposeModel(*model, op_index)) {
    std::cerr << "Failed to decompose model." << std::endl;
    return false;
  }
  // Packs the model back.
  flatbuffers::FlatBufferBuilder fbb;
  const flatbuffers::Offset<tflite::Model> model_buffer =
      tflite::Model::Pack(fbb, model.get());
  FinishModelBuffer(fbb, model_buffer);

  std::ofstream out(output_path);
  out.write(reinterpret_cast<char*>(fbb.GetBufferPointer()), fbb.GetSize());
  return true;
}

std::string DumpTensorsInfo(
    const std::vector<int>& tensor_indices,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tensors) {
  return absl::StrJoin(
      tensor_indices, "  ", [&tensors](std::string* out, int index) {
        auto& tensor = tensors[index];
        std::string message =
            absl::StrFormat("%s[%s]#%d", EnumNameTensorType(tensor->type),
                            absl::StrJoin(tensor->shape, ","), index);
        absl::StrAppend(out, message);
      });
}

bool List(const std::string& input_path) {
  std::unique_ptr<tflite::ModelT> model = GetUnpackedModel(input_path);
  if (!model) {
    return false;
  }
  if (model->subgraphs.size() != 1) {
    return false;
  }
  auto& subgraph = model->subgraphs[0];
  auto& operators = subgraph->operators;
  std::vector<std::vector<std::string>> table;
  table.push_back({"index", "operator", "input: type[shape]#index",
                   "output: type[shape]#index"});
  for (int i = 0; i < operators.size(); ++i) {
    auto& op = operators[i];
    auto& op_code = model->operator_codes[op->opcode_index];
    tflite::BuiltinOperator builtin_code =
        tflite::GetBuiltinCode(op_code.get());
    std::string op_name = tflite::EnumNameBuiltinOperator(builtin_code);
    std::string input_dump = DumpTensorsInfo(op->inputs, subgraph->tensors);
    std::string output_dump = DumpTensorsInfo(op->outputs, subgraph->tensors);
    table.push_back({std::to_string(i), op_name, input_dump, output_dump});
  }
  std::cout << TableStringify(table) << std::endl;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage(kUsage);
  std::vector<char*> args = absl::ParseCommandLine(argc, argv);
  if (args.size() < 2) {
    PrintUsage();
    return 1;
  }
  std::string subcmd(args[1]);
  if (subcmd == "decompose") {
    std::string input_path = absl::GetFlag(FLAGS_input);
    std::string output_path = absl::GetFlag(FLAGS_output);
    int op_index = absl::GetFlag(FLAGS_op_index);
    if (input_path.empty() || output_path.empty() || op_index == -1) {
      PrintUsage();
      return 1;
    }
    if (!Decompose(input_path, output_path, op_index)) {
      return 1;
    }
  } else if (subcmd == "list") {
    std::string input_path = absl::GetFlag(FLAGS_input);
    if (input_path.empty()) {
      PrintUsage();
      return 1;
    }
    if (!List(input_path)) {
      return 1;
    }
  } else {
    PrintUsage();
    return 1;
  }
  return 0;
}
