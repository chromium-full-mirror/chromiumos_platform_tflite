/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "common/log.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/schema/schema_utils.h"

ABSL_FLAG(std::string, input, "", "input tflite model");
ABSL_FLAG(std::string, output, "", "output tflite model");
ABSL_FLAG(std::vector<std::string>,
          op_index,
          std::vector<std::string>{},
          "operator indices");

namespace {

constexpr absl::string_view kUsage = R"(
decompose_model <list|decompose> [options]

Available subcommands:
list --input=<input tflite file>
  Lists all operators in the |input| and shows the their information.

decompose --input=<input tflite file> --output=<output tflite file>
          --op_index=<operator indices>
  Decomposes the |input| model by extracting the operators with indices
  |op_index| and rebuild a model with the extracted operators, where |op_index|
  is the list of indices seperated by comma, e.g. `--op_index=0,2,3`.
  The new model is saved to |output|.
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
    LOG(ERROR) << "Failed to build model from file";
    return nullptr;
  }
  std::unique_ptr<tflite::ModelT> model(fb_model->GetModel()->UnPack());
  if (!model) {
    LOG(ERROR) << "Failed to unpack model";
    return nullptr;
  }
  return model;
}

// Decomposes |model|. Extracts the operators of indices |op_indices| as a new
// model and re-pack it into |model|.
bool DecomposeModel(tflite::ModelT& model, const std::vector<int>& op_indices) {
  if (model.subgraphs.size() != 1) {
    LOG(ERROR) << "There is more than one subgraph";
    return false;
  }
  auto& subgraph = model.subgraphs[0];

  // Prunes the other operators and keeps only operators with indices
  // |op_indices|.
  auto& operators = subgraph->operators;
  for (int i = 0; i < op_indices.size(); ++i) {
    if (operators.size() <= op_indices[i]) {
      LOG(ERROR) << "operator index is out of range: " << op_indices[i]
                 << ", size = " << operators.size();
      return false;
    }
    operators[i] = std::move(operators[op_indices[i]]);
  }
  operators.resize(op_indices.size());

  // Prunes the unused tensors.
  std::map<int, int> tensor_map;
  int tensor_map_index = 0;
  for (auto& op : operators) {
    for (auto& tensors : {std::ref(op->inputs), std::ref(op->outputs)}) {
      for (int& index : tensors.get()) {
        // Optional input -1 should be skipped.
        if (index < 0) {
          continue;
        }
        auto it = tensor_map.find(index);
        if (it == tensor_map.end()) {
          tensor_map.emplace(index, tensor_map_index++);
        }
        index = tensor_map[index];
      }
    }
  }
  std::vector<std::unique_ptr<tflite::TensorT>> new_tensors(tensor_map_index);
  for (auto [k, v] : tensor_map) {
    new_tensors[v] = std::move(subgraph->tensors[k]);
  }
  subgraph->tensors = std::move(new_tensors);

  // Prunes the unused buffers.
  // TODO: b/331900692 - Randomize the buffer data.
  // 0th entry of the buffer array should be an empty buffer for tensors
  // without associated buffers.
  std::map<int, int> buffer_map = {{0, 0}};
  int buffer_map_index = 1;
  for (const auto& tensor : subgraph->tensors) {
    int buffer_idx = tensor->buffer;
    auto it = buffer_map.find(buffer_idx);
    if (it == buffer_map.end()) {
      buffer_map.emplace(buffer_idx, buffer_map_index++);
    }
    tensor->buffer = buffer_map[buffer_idx];
  }
  std::vector<std::unique_ptr<tflite::BufferT>> new_buffers(buffer_map_index);
  for (auto [k, v] : buffer_map) {
    new_buffers[v] = std::move(model.buffers[k]);
  }
  model.buffers = std::move(new_buffers);

  // TODO: b/331900692 - Consider also pruning unused operator code.

  // Reassigns input/output of the remaining operators with empty buffers to
  // input/output of the subgraph.
  subgraph->inputs.clear();
  subgraph->outputs.clear();
  std::set<int> op_inputs, op_outputs;
  for (auto& op : operators) {
    for (int index : op->inputs) {
      if (index < 0) {
        continue;
      }
      int buffer_idx = subgraph->tensors[index]->buffer;
      // Input of the subgraph should contain only empty buffers.
      if (model.buffers[buffer_idx]->data.size() == 0) {
        op_inputs.insert(index);
      }
    }
    for (int index : op->outputs) {
      op_outputs.insert(index);
    }
  }
  // Any tensor as both input and output of operators is not considered as the
  // input/output of the subgraph.
  std::vector<int> inputs, outputs;
  std::ranges::set_difference(op_inputs, op_outputs,
                              std::back_inserter(subgraph->inputs));
  std::ranges::set_difference(op_outputs, op_inputs,
                              std::back_inserter(subgraph->outputs));
  return true;
}

bool Decompose(const std::string& input_path,
               const std::string& output_path,
               const std::vector<int>& op_indices) {
  std::unique_ptr<tflite::ModelT> model = GetUnpackedModel(input_path);
  if (!model) {
    return false;
  }

  if (!DecomposeModel(*model, op_indices)) {
    LOG(ERROR) << "Failed to decompose model";
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
        if (index < 0) {
          absl::StrAppend(out, "#" + std::to_string(index));
          return;
        }
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
    LOG(ERROR) << "There is more than one subgraph";
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
  std::cout << "Subgraph inputs: "
            << DumpTensorsInfo(subgraph->inputs, subgraph->tensors)
            << std::endl;
  std::cout << "Subgraph outputs: "
            << DumpTensorsInfo(subgraph->outputs, subgraph->tensors)
            << std::endl;
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
    std::vector<std::string> raw_op_indices = absl::GetFlag(FLAGS_op_index);
    if (input_path.empty() || output_path.empty() || raw_op_indices.empty()) {
      PrintUsage();
      return 1;
    }
    std::vector<int> op_indices;
    for (std::string& raw_index : raw_op_indices) {
      int index = -1;
      if (!absl::SimpleAtoi(raw_index, &index) || index < 0) {
        LOG(ERROR) << "Failed to convert index: " << raw_index;
        return 1;
      }
      op_indices.push_back(index);
    }
    std::sort(op_indices.begin(), op_indices.end());
    auto it = std::unique(op_indices.begin(), op_indices.end());
    op_indices.erase(it, op_indices.end());
    if (!Decompose(input_path, output_path, op_indices)) {
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
