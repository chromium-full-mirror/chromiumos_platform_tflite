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

#ifndef DELEGATE_MTK_NEURON_NEURON_DELEGATE_OPTIONS_H_
#define DELEGATE_MTK_NEURON_NEURON_DELEGATE_OPTIONS_H_

#include <string>

namespace tflite {
namespace neuron {

enum ExecutionPreference {
  kUndefined = -1,
  kLowPower = 0,
  kFastSingleAnswer = 1,
  kSustainedSpeed = 2,
  kTurboBoost = 3,
};

enum ExecutionPriority {
  kPriorityUndefined = -1,
  kPriorityLow = 90,
  kPriorityMedium = 100,
  kPriorityHigh = 110,
};

enum OptimizationHint {
  kOptimizationNone = 0,
  kOptimizationLowLatency = 1 << 0,
  kOptimizationDeepFusion = 1 << 1,
  kOptimizationBatchProcessor = 1 << 2,
  kOptimizationDefault = kOptimizationNone,
};

enum NeuronOperationCheckMode {
  kNoOperationCheck = 0,
  kPerNodeOperationCheck = 1,
  kPreOperationCheck = 2,
};

typedef struct {
  // Default execution_preference = kFastSingleAnswer
  ExecutionPreference execution_preference;
  // Default execution_priority = kPriorityHigh
  ExecutionPriority execution_priority;
  // Default optimization_hint = kOptimizationDefault
  int optimization_hint;
  // Default allow_fp16 = false
  bool allow_fp16;
  // The nul-terminated cache dir.
  // Default to an empty string, which implies the Neuron will not try caching
  // the compilation.
  std::string cache_dir;
  // The unique nul-terminated token string.
  // Default to an empty string, which implies the Neuron will not try caching
  // the compilation. It is the caller's responsibility to ensure there is no
  // clash of the tokens.
  // NOTE: when using compilation caching, it is not recommended to use the
  // same delegate instance for multiple models.
  std::string model_token;
  // Whether to use ahwb
  bool use_ahwb;
  // Whether to use cacheable ahwb
  bool use_cacheable_buffer;
  // Set compile options
  std::string compile_options;
  // Set target device
  std::string accelerator_name;
  // Whether to use supported operation check or not.
  NeuronOperationCheckMode operation_check_mode;
  // Execution deadline for the inference (in msec)
  uint16_t inference_deadline;
  // Execution abort time is the hint for the the maximum inference time for
  // the execution (in msec).
  uint16_t inference_abort_time;
  // Optional path to the platform-dependent Neuron configuration file.
  // Unused on ChromeOS.
  std::string neuron_config_path;
} NeuronDelegateOptions;

}  // namespace neuron
}  // namespace tflite

#endif  // DELEGATE_MTK_NEURON_NEURON_DELEGATE_OPTIONS_H_
