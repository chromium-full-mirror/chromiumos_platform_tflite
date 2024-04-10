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

#include <cstdlib>
#include <memory>
#include <utility>

#include "delegate/mtk_neuron/neuron_delegate.h"
#include "tensorflow/lite/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/acceleration/configuration/c/stable_delegate.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/stable_delegate_interface.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace neuron {
namespace {

tflite::neuron::ExecutionPreference ConvertExecutionPrefence(
    tflite::NeuronExecutionPreference neuron_execution_preference) {
  using TflitePreference = tflite::neuron::ExecutionPreference;
  switch (neuron_execution_preference) {
    case NeuronExecutionPreference_NEURON_UNDEFINED:
      return TflitePreference::kUndefined;
    case NeuronExecutionPreference_NEURON_LOW_POWER:
      return TflitePreference::kLowPower;
    case NeuronExecutionPreference_NEURON_FAST_SINGLE_ANSWER:
      return TflitePreference::kFastSingleAnswer;
    case NeuronExecutionPreference_NEURON_SUSTAINED_SPEED:
      return TflitePreference::kSustainedSpeed;
    case NeuronExecutionPreference_NEURON_TURBO_BOOST:
      return TflitePreference::kTurboBoost;
    default:
      return TflitePreference::kFastSingleAnswer;
  }
}

tflite::neuron::ExecutionPriority ConvertExecutionPriority(
    tflite::NeuronExecutionPriority neuron_execution_priority) {
  using TflitePriority = tflite::neuron::ExecutionPriority;
  switch (neuron_execution_priority) {
    case NeuronExecutionPriority_NEURON_PRIORITY_UNDEFINED:
      return TflitePriority::kPriorityUndefined;
    case NeuronExecutionPriority_NEURON_PRIORITY_LOW:
      return TflitePriority::kPriorityLow;
    case NeuronExecutionPriority_NEURON_PRIORITY_MEDIUM:
      return TflitePriority::kPriorityMedium;
    case NeuronExecutionPriority_NEURON_PRIORITY_HIGH:
      return TflitePriority::kPriorityHigh;
    default:
      return TflitePriority::kPriorityHigh;
  }
}

tflite::neuron::NeuronOperationCheckMode ConvertOperationCheckMode(
    tflite::NeuronOperationCheckMode neuron_operation_check_mode) {
  using NeuronOperationCheckMode = tflite::neuron::NeuronOperationCheckMode;
  switch (neuron_operation_check_mode) {
    case NeuronOperationCheckMode_NO_OPERATION_CHECK:
      return NeuronOperationCheckMode::kNoOperationCheck;
    case NeuronOperationCheckMode_PER_NODE_OPERATION_CHECK:
      return NeuronOperationCheckMode::kPerNodeOperationCheck;
    case NeuronOperationCheckMode_PRE_OPERATION_CHECK:
      return NeuronOperationCheckMode::kPreOperationCheck;
    default:
      return NeuronOperationCheckMode::kNoOperationCheck;
  }
}

TfLiteOpaqueDelegate* NeuronStableDelegateCreateFunc(const void* settings) {
  const ::tflite::TFLiteSettings* tflite_settings =
      static_cast<const ::tflite::TFLiteSettings*>(settings);
  const ::tflite::NeuronDelegateSettings* neuron_delegate_settings =
      tflite_settings->neuron_delegate_settings();

  NeuronDelegateOptions options = TfLiteNeuronDelegateOptionsDefault();

  // Parse NeuronDelegateSettings to NeuronDelegateOptions
  if (neuron_delegate_settings) {
    // TODO(b:333500607): NEURON_LOW_POWER is impossible to set.
    options.execution_preference =
        neuron_delegate_settings->execution_preference()
            ? ConvertExecutionPrefence(
                  neuron_delegate_settings->execution_preference())
            : options.execution_preference;
    options.execution_priority =
        neuron_delegate_settings->execution_priority()
            ? ConvertExecutionPriority(
                  neuron_delegate_settings->execution_priority())
            : options.execution_priority;
    options.optimization_hint =
        neuron_delegate_settings->optimization_hint()
            ? (int32_t)(neuron_delegate_settings->optimization_hint())
            : options.optimization_hint;
    if ((int32_t)(neuron_delegate_settings->optimization_hint() >> 3)) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "Unsupported hints: %d",
                      options.optimization_hint);
    }
    options.allow_fp16 = neuron_delegate_settings->allow_fp16()
                             ? neuron_delegate_settings->allow_fp16()
                             : options.allow_fp16;
    if (neuron_delegate_settings->cache_dir() &&
        neuron_delegate_settings->cache_dir()->size() != 0) {
      std::string cache_dir = neuron_delegate_settings->cache_dir()->str();
      options.cache_dir = cache_dir;
    }
    if (neuron_delegate_settings->model_token() &&
        neuron_delegate_settings->model_token()->size() != 0) {
      std::string model_token = neuron_delegate_settings->model_token()->str();
      options.model_token = model_token;
    }
    options.use_ahwb = neuron_delegate_settings->use_ahwb()
                           ? neuron_delegate_settings->use_ahwb()
                           : options.use_ahwb;
    options.use_cacheable_buffer =
        neuron_delegate_settings->use_cacheable_buffer()
            ? neuron_delegate_settings->use_cacheable_buffer()
            : options.use_cacheable_buffer;
    if (neuron_delegate_settings->compile_options() &&
        neuron_delegate_settings->compile_options()->size() != 0) {
      std::string compile_options =
          neuron_delegate_settings->compile_options()->str();
      options.compile_options = compile_options;
    }
    if (neuron_delegate_settings->accelerator_name() &&
        neuron_delegate_settings->accelerator_name()->size() != 0) {
      std::string accelerator_name =
          neuron_delegate_settings->accelerator_name()->str();
      options.accelerator_name = accelerator_name;
    }
    options.operation_check_mode =
        neuron_delegate_settings->operation_check_mode()
            ? ConvertOperationCheckMode(
                  neuron_delegate_settings->operation_check_mode())
            : options.operation_check_mode;
  } else {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                    "Use Default Neuron Delegate Options.");
  }
  auto delegate =
      std::make_unique<tflite::neuron::NeuronStableDelegate>(options);
  return tflite::TfLiteOpaqueDelegateFactory::CreateSimpleDelegate(
      std::move(delegate));
}

void NeuronStableDelegateDestroyFunc(
    TfLiteOpaqueDelegate* neuron_stable_delegate) {
  tflite::TfLiteOpaqueDelegateFactory::DeleteSimpleDelegate(
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
