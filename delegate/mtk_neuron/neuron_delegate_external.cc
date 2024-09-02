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

#include <memory>
#include <sstream>
#include <utility>

#include "common/minimal_logging.h"
#include "delegate/mtk_neuron/neuron_delegate.h"
#include "tensorflow/lite/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/acceleration/configuration/c/stable_delegate.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/stable_delegate_interface.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"

namespace tflite {
namespace neuron {
namespace {

// Joins the string vector from FlatBuffers with the provided separator as a
// single string.
std::string JoinFlatBuffersStringVector(
    const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>&
        fb_string_vector,
    std::string_view sep) {
  std::stringstream ss;
  for (int i = 0; i < fb_string_vector.size(); ++i) {
    if (i > 0) {
      ss << sep;
    }
    ss << fb_string_vector[i]->c_str();
  }
  return ss.str();
}

tflite::neuron::ExecutionPreference ConvertExecutionPrefence(
    tflite::MtkNeuronSettings_::ExecutionPreference
        neuron_execution_preference) {
  using enum tflite::MtkNeuronSettings_::ExecutionPreference;
  using TflitePreference = tflite::neuron::ExecutionPreference;
  switch (neuron_execution_preference) {
    case ExecutionPreference_PREFERENCE_UNDEFINED:
      return TflitePreference::kUndefined;
    case ExecutionPreference_PREFERENCE_LOW_POWER:
      return TflitePreference::kLowPower;
    case ExecutionPreference_PREFERENCE_FAST_SINGLE_ANSWER:
      return TflitePreference::kFastSingleAnswer;
    case ExecutionPreference_PREFERENCE_SUSTAINED_SPEED:
      return TflitePreference::kSustainedSpeed;
    case ExecutionPreference_PREFERENCE_TURBO_BOOST:
      return TflitePreference::kTurboBoost;
    default:
      return TflitePreference::kFastSingleAnswer;
  }
}

tflite::neuron::ExecutionPriority ConvertExecutionPriority(
    tflite::MtkNeuronSettings_::ExecutionPriority neuron_execution_priority) {
  using enum tflite::MtkNeuronSettings_::ExecutionPriority;
  using TflitePriority = tflite::neuron::ExecutionPriority;
  switch (neuron_execution_priority) {
    case ExecutionPriority_PRIORITY_UNDEFINED:
      return TflitePriority::kPriorityUndefined;
    case ExecutionPriority_PRIORITY_LOW:
      return TflitePriority::kPriorityLow;
    case ExecutionPriority_PRIORITY_MEDIUM:
      return TflitePriority::kPriorityMedium;
    case ExecutionPriority_PRIORITY_HIGH:
      return TflitePriority::kPriorityHigh;
    default:
      return TflitePriority::kPriorityHigh;
  }
}

tflite::neuron::OptimizationHint ConvertOptimizationHint(
    tflite::MtkNeuronSettings_::OptimizationHint neuron_optimization_hint) {
  using enum tflite::MtkNeuronSettings_::OptimizationHint;
  using TfLiteHint = tflite::neuron::OptimizationHint;
  switch (neuron_optimization_hint) {
    case OptimizationHint_OPTIMIZATION_NONE:
      return TfLiteHint::kOptimizationNone;
    case OptimizationHint_OPTIMIZATION_LOW_LATENCY:
      return TfLiteHint::kOptimizationLowLatency;
    case OptimizationHint_OPTIMIZATION_DEEP_FUSION:
      return TfLiteHint::kOptimizationDeepFusion;
    case OptimizationHint_OPTIMIZATION_BATCH_PROCESSING:
      return TfLiteHint::kOptimizationBatchProcessor;
    default:
      return TfLiteHint::kOptimizationNone;
  }
}

tflite::neuron::NeuronOperationCheckMode ConvertOperationCheckMode(
    tflite::MtkNeuronSettings_::OperationCheckMode
        neuron_operation_check_mode) {
  using enum tflite::MtkNeuronSettings_::OperationCheckMode;
  using NeuronOperationCheckMode = tflite::neuron::NeuronOperationCheckMode;
  switch (neuron_operation_check_mode) {
    case OperationCheckMode_NO_OPERATION_CHECK:
      return NeuronOperationCheckMode::kNoOperationCheck;
    case OperationCheckMode_PER_NODE_OPERATION_CHECK:
      return NeuronOperationCheckMode::kPerNodeOperationCheck;
    case OperationCheckMode_PRE_OPERATION_CHECK:
      return NeuronOperationCheckMode::kPreOperationCheck;
    default:
      return NeuronOperationCheckMode::kNoOperationCheck;
  }
}

TfLiteOpaqueDelegate* NeuronStableDelegateCreateFunc(const void* settings) {
  const ::tflite::TFLiteSettings* tflite_settings =
      static_cast<const ::tflite::TFLiteSettings*>(settings);
  NeuronDelegateOptions options = TfLiteNeuronDelegateOptionsDefault();

  // Parse MtkNeuronSettings to NeuronDelegateOptions
  if (const auto* neuron_settings = tflite_settings->mtk_neuron_settings();
      neuron_settings != nullptr) {
    if (neuron_settings->execution_preference()) {
      options.execution_preference =
          ConvertExecutionPrefence(neuron_settings->execution_preference());
    }
    if (neuron_settings->execution_priority()) {
      options.execution_priority =
          ConvertExecutionPriority(neuron_settings->execution_priority());
    }
    if (const auto* hints = neuron_settings->optimization_hints();
        hints != nullptr && hints->size() != 0) {
      // Convert the repeated hints field into corresponding bitwise flags.
      options.optimization_hint = 0;
      for (int hint : *neuron_settings->optimization_hints()) {
        options.optimization_hint |= ConvertOptimizationHint(
            static_cast<MtkNeuronSettings_::OptimizationHint>(hint));
      }
    }
    if (neuron_settings->operation_check_mode()) {
      options.operation_check_mode =
          ConvertOperationCheckMode(neuron_settings->operation_check_mode());
    }

    options.inference_deadline =
        static_cast<uint16_t>(neuron_settings->inference_deadline_ms());
    options.inference_abort_time =
        static_cast<uint16_t>(neuron_settings->inference_abort_time_ms());
    options.allow_fp16 = neuron_settings->allow_fp16_precision_for_fp32();
    options.use_ahwb = neuron_settings->use_ahwb();
    options.use_cacheable_buffer = neuron_settings->use_cacheable_buffer();
    if (const auto* opts = neuron_settings->compile_options();
        opts != nullptr && opts->size() != 0) {
      options.compile_options = JoinFlatBuffersStringVector(*opts, " ");
    }
    if (const auto* names = neuron_settings->accelerator_names();
        names != nullptr && names->size() != 0) {
      options.accelerator_name = JoinFlatBuffersStringVector(*names, ",");
    }
    if (const auto* path = neuron_settings->neuron_config_path();
        path != nullptr && path->size() != 0) {
      options.neuron_config_path = path->str();
    }
  } else {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                    "Use Default Neuron Delegate Options.");
  }

  if (const auto* caching_settings =
          tflite_settings->compilation_caching_settings();
      caching_settings != nullptr) {
    if (const auto* dir = caching_settings->cache_dir();
        dir != nullptr && dir->size() != 0) {
      options.cache_dir = dir->str();
    }
    if (const auto* token = caching_settings->model_token();
        token != nullptr && token->size() != 0) {
      options.model_token = token->str();
    }
  }

  auto delegate =
      std::make_unique<tflite::neuron::NeuronStableDelegate>(options);
  return tflite::cros::SimpleAsyncDelegateFactory::CreateAsyncDelegate(
      std::move(delegate));
}

void NeuronStableDelegateDestroyFunc(
    TfLiteOpaqueDelegate* neuron_stable_delegate) {
  tflite::cros::SimpleAsyncDelegateFactory::DeleteAsyncDelegate(
      neuron_stable_delegate);
}

int NeuronStableDelegateErrnoFunc(
    TfLiteOpaqueDelegate* neuron_stable_delegate) {
  // no-op
  return 0;
}

const TfLiteOpaqueDelegatePlugin neuron_stable_delegate_plugin = {
    NeuronStableDelegateCreateFunc, NeuronStableDelegateDestroyFunc,
    NeuronStableDelegateErrnoFunc};

const TfLiteStableDelegate neuron_stable_delegate = {
    /*delegate_abi_version=*/TFL_STABLE_DELEGATE_ABI_VERSION,
    /*delegate_name=*/kNeuronStableDelegateName,
    /*delegate_version=*/kNeuronStableDelegateVersion,
    /*delegate_plugin=*/&neuron_stable_delegate_plugin};

}  // namespace
}  // namespace neuron
}  // namespace tflite

extern "C" const TfLiteStableDelegate TFL_TheStableDelegate =
    tflite::neuron::neuron_stable_delegate;
