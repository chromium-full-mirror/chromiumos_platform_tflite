/*
 * Copyright (C) 2021 MediaTek Inc., this file is modified on 02/26/2021
 * by MediaTek Inc. based on MIT License .
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the ""Software""), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED ""AS IS"", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef DELEGATE_MTK_NEURON_NEURON_DELEGATE_KERNEL_H_
#define DELEGATE_MTK_NEURON_NEURON_DELEGATE_KERNEL_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common/simple_async_delegate.h"
#include "delegate/mtk_neuron/neuron_async_kernel.h"
#include "delegate/mtk_neuron/neuron_delegate.h"
#include "delegate/mtk_neuron/neuron_delegate_builder.h"
#include "delegate/mtk_neuron/neuron_implementation.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/core/async/c/task.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"

#define NEURON_REUSABLE_EXECUTION

namespace tflite {
namespace neuron {

class DelegateAsyncKernel;
// The kernel that represents the node sub set of TFLite being run on Neuron
// API.
struct NeuronOpMappingArgs {
  TfLiteOpaqueContext* context;
  NeuronOpBuilder* builder;
  TfLiteOpaqueNode* node;
  std::vector<int>* model_state_outputs;
  std::vector<int>* model_state_tfl_inputs;
  std::vector<std::tuple<int, int>>* feedback_loops;
  /// M: NeuroPilot {@
  TfLiteRegistrationExternal* registration;
  /// M: NeuroPilot @}
  int* neuron_errno;
};

// Manage Neuron shared memory handle
class NNMemory {
 public:
  NNMemory(const NeuronApi* neuronapi, const char* name, size_t size,
           bool use_ahwb, bool use_cacheable_buffer);

  ~NNMemory();

  uint8_t* get_data_ptr() { return data_ptr_; }
  NeuronMemory* get_handle() { return nn_memory_handle_; }
  size_t get_byte_size() { return byte_size_; }

  int ion_memory_lock() {
#if defined(__ANDROID__)
    if (buffer_) {
      return AHardwareBuffer_lock(buffer_,
                                  AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                                      AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
                                  -1, NULL,
                                  reinterpret_cast<void**>(&data_ptr_));
    }
#endif
    return 0;
  }
  int ion_memory_unlock() {
#if defined(__ANDROID__)
    if (buffer_) {
      return AHardwareBuffer_unlock(buffer_, nullptr);
    }
#endif
    return 0;
  }

 private:
  // NeuronApi instance to use. Not owned by this object.
  const NeuronApi* neuronapi_;
  int fd_ = -1;
  size_t byte_size_ = 0;
  uint8_t* data_ptr_ = nullptr;
  bool use_ahwb_ = false;
  NeuronMemory* nn_memory_handle_ = nullptr;
#if defined(__ANDROID__)
  AHardwareBuffer* buffer_ = nullptr;
#else
  int dma_heap_fd_ = 0;
#endif
};

// RAII Neuron Model Destructor for use with std::unique_ptr
class NNFreeModel {
 public:
  explicit NNFreeModel(const NeuronApi* neuronapi) : neuronapi_(neuronapi) {}
  void operator()(NeuronModel* model) { neuronapi_->NeuronModel_free(model); }

 private:
  // NeuronApi instance to use. Not owned by this object.
  const NeuronApi* neuronapi_;
};

// RAII Neuron Compilation Destructor for use with std::unique_ptr
class NNFreeCompilation {
 public:
  explicit NNFreeCompilation(const NeuronApi* neuronapi)
      : neuronapi_(neuronapi) {}
  void operator()(NeuronCompilation* compilation) {
    neuronapi_->NeuronCompilation_free(compilation);
  }

 private:
  // NeuronApi instance to use. Not owned by this object.
  const NeuronApi* neuronapi_;
};

// RAII Neuron Execution Destructor for use with std::unique_ptr
class NNFreeExecution {
 public:
  explicit NNFreeExecution(const NeuronApi* neuronapi)
      : neuronapi_(neuronapi) {}
  void operator()(NeuronExecution* execution) {
    neuronapi_->NeuronExecution_free(execution);
  }

 private:
  // NeuronApi instance to use. Not owned by this object.
  const NeuronApi* neuronapi_;
};

// TODO(Code): Mode NNCompilationArgs to another Util file.
class NNCompilationArgs {
 public:
  static const std::string& GetArgs(ExecutionPreference preference) {
#ifdef MTK_GITHUB_RELEASE
    static const std::string kFastSingleAnswerArgs = "--opt 3";
#else
    static const std::string kFastSingleAnswerArgs = "--opt-accuracy --opt 3";
#endif
    return kFastSingleAnswerArgs;
  }
};

enum DelegateSpecificMode {
  // Normal Mode, get information from context
  kNormalMode = 0,
  // Get model information from one_node_and_reg_
  // Currently only for getSupportedForOneNode
  kOneNodeMode = 1,
  kSubgraphMode = 2,
  kContextMode = 3,
};

// Neuron delegate kernel.
class NeuronStableDelegateKernel
    : public cros::SimpleAsyncDelegateKernelInterface {
 public:
  static TfLiteStatus Map(TfLiteOpaqueContext* context, int builtin_code,
                          int version, int neuron_platform_version,
                          const NeuronOpMappingArgs& mapping_args,
                          NeuronOperationType* nn_op_type);
  NeuronStableDelegateKernel();
  explicit NeuronStableDelegateKernel(const NeuronApi* neuronapi,
                                      const NeuronDelegateOptions& options)
      : async_kernel_(std::make_unique<DelegateAsyncKernel>(this)),
        neuronapi_(neuronapi),
        options_(options),
        nn_model_(nullptr, NNFreeModel(neuronapi_)),
        nn_compilation_(nullptr, NNFreeCompilation(neuronapi_)),
#ifdef NEURON_REUSABLE_EXECUTION
        nn_execution_(nullptr, NNFreeExecution(neuronapi_))
#endif
  {
  }

  virtual ~NeuronStableDelegateKernel() = default;

  TfLiteStatus Init(TfLiteOpaqueContext* context,
                    const TfLiteOpaqueDelegateParams* params) override;

  TfLiteStatus Prepare(TfLiteOpaqueContext* context,
                       TfLiteOpaqueNode* node) override;
  TfLiteStatus Eval(TfLiteOpaqueContext* context,
                    TfLiteOpaqueNode* node) override;
  TfLiteStatus Eval(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node,
                    TfLiteExecutionTask* task);
  TfLiteAsyncKernel* AsyncKernel(TfLiteOpaqueContext* context,
                                 TfLiteOpaqueNode* node) override;
  TfLiteStatus GetSupportedOperationsForOneNode(
      TfLiteOpaqueContext* context, TfLiteOpaqueNode* node,
      TfLiteRegistrationExternal* registration, bool* is_supported);
  TfLiteStatus GetSupportedOperationsForWholeGraph(TfLiteOpaqueContext* context,
                                                   bool* is_supported);
  TfLiteStatus GetSupportedOperations(TfLiteOpaqueContext* context,
                                      const TfLiteIntArray* input_tensors,
                                      const TfLiteIntArray* output_tensors,
                                      bool* is_supported);
  TfLiteStatus RegisterBuffer(TfLiteBufferHandle buf_handle, int buf_fd,
                              size_t buf_size);
  TfLiteStatus UnregisterBuffer(TfLiteBufferHandle buf_handle);

 private:
  std::unique_ptr<DelegateAsyncKernel> async_kernel_;
  const NeuronApi* neuronapi_;
  NeuronDelegateOptions options_;
  // True if initialization has been completed successfully
  bool initialised_;
  std::unique_ptr<NeuronModel, NNFreeModel> nn_model_;
  std::unique_ptr<NeuronCompilation, NNFreeCompilation> nn_compilation_;
#ifdef NEURON_REUSABLE_EXECUTION
  std::unique_ptr<NeuronExecution, NNFreeExecution> nn_execution_;
#endif
  // Node indices that this delegate is responsible for. Indices here
  // indexes into the nodes array in the TfLiteContext.
  std::vector<int> nodes_;
  // Track indices we use
  OperandMapping operand_mapping_;

  std::map<TfLiteBufferHandle, NeuronMemory*> tensor_memory_map_kernel_;

  std::vector<int> model_state_outputs_;
  std::vector<int> model_state_tfl_inputs_;
  // This is the equivalent of the pair model_state_outputs_,
  // model_state_tfl_inputs_ for all tensors where we have to keep the output
  // data available for TFLite model users
  std::vector<std::tuple<int, int>> feedback_loops_;

  std::unique_ptr<NNMemory> nn_input_memory_;
  std::unique_ptr<NNMemory> nn_output_memory_;

  std::vector<uint8_t> nn_compilation_cache_token_;

  // Prevent set memory multi-times when we use ahwb
  bool need_set_memory_ = true;

  // Number of the ops in NeuronModel
  int total_op_num_ = 0;

  // Neuron delegate specific run mode
  DelegateSpecificMode delegate_specific_mode_ = kNormalMode;
  // For One Node Mode
  std::pair<TfLiteOpaqueNode*, TfLiteRegistrationExternal*> one_node_and_reg_ =
      {nullptr, nullptr};

  void AddDequantizeOperatorsWhereNeeded(const TfLiteOpaqueContext* context,
                                         int builtin_code,
                                         const TfLiteOpaqueNode* node,
                                         NeuronOpBuilder* builder);

  TfLiteStatus AddOpsAndTensors(TfLiteOpaqueContext* context);

  TfLiteStatus BuildGraph(TfLiteOpaqueContext* context,
                          const TfLiteOpaqueDelegateParams* params,
                          const TfLiteIntArray* input_tensors,
                          const TfLiteIntArray* output_tensors);
};

}  // namespace neuron
}  // namespace tflite

#endif  // DELEGATE_MTK_NEURON_NEURON_DELEGATE_KERNEL_H_
