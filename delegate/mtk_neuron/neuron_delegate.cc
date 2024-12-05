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

#include "delegate/mtk_neuron/neuron_delegate.h"

#include <farmhash.h>

#include <string>
#include <vector>

#include "common/minimal_logging.h"
#include "common/simple_async_delegate.h"
#include "delegate/mtk_neuron/mtk/mtk_minimal_logging.h"
#include "delegate/mtk_neuron/neuron_delegate_kernel.h"
#include "delegate/mtk_neuron/neuron_delegate_utils.h"
#include "delegate/mtk_neuron/neuron_delegate_validation.h"
#include "delegate/mtk_neuron/neuron_version.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define LOG_TAG "NeuronDelegate"

namespace tflite {
namespace neuron {

NeuronDelegateOptions TfLiteNeuronDelegateOptionsDefault() {
  NeuronDelegateOptions options = {
      .execution_preference = kFastSingleAnswer,
      .execution_priority = kPriorityHigh,
      .optimization_hint = kOptimizationDefault,
      .allow_fp16 = false,
      .cache_dir = "",
      .model_token = "",
      .use_ahwb = false,
      .use_cacheable_buffer = true,
      .compile_options = "",
      .accelerator_name = "",
      .operation_check_mode = kNoOperationCheck,
      .inference_deadline = 0,
      .inference_abort_time = 0,
  };

  // Return default options
  return options;
}

bool NeuronStableDelegate::IsNodeSupportedByDelegate(
    const TfLiteRegistrationExternal* registration,
    const TfLiteOpaqueNode* node, TfLiteOpaqueContext* context) const {
  std::vector<NeuronValidationFailure> failure;
  // TODO(b:333498833): Fix fragile dependency of node ordering.
  if (options_.operation_check_mode == kPreOperationCheck) {
    node_id_++;
    if (node_is_supported_.size() > 0 && node_id_ < node_is_supported_.size()) {
      return node_is_supported_[node_id_];
    }
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "PreOpCheck: error size of vector.");
  }
  bool supported =
      Validate(registration, node, context, neuron_, &options_, &failure);
  if (!supported) {
    TFLITE_LOG_PROD(
        tflite::TFLITE_LOG_ERROR, "OP %s (v%d) is not supported (%s)",
        tflite::EnumNameBuiltinOperator(static_cast<BuiltinOperator>(
            TfLiteOperatorGetBuiltInCode(registration))),
        TfLiteRegistrationExternalGetVersion(registration),
        failure.size() > 0 ? failure[0].message.c_str() : "");
  }
  return supported;
}

TfLiteStatus NeuronStableDelegate::Initialize(TfLiteOpaqueContext* context) {
#ifdef MTK_INTERNAL_USE
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                  "Neuron stable delegate for internal use. (version: %s)",
                  NEURON_DELEGATE_VERSION_STRING);
#else
  TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                       "Neuron stable delegate version: %s",
                       NEURON_DELEGATE_VERSION_STRING);
#endif

  if (options_.operation_check_mode == kPreOperationCheck) {
    GetPreOpCheckResult(context);
  }

  return kTfLiteOk;
}

const char* NeuronStableDelegate::Name() const {
  return kNeuronStableDelegateName;
}

std::unique_ptr<cros::SimpleAsyncDelegateKernelInterface>
NeuronStableDelegate::CreateDelegateKernelInterface() {
  return std::make_unique<NeuronStableDelegateKernel>(neuron_, options_);
}

void NeuronStableDelegate::GetPreOpCheckResult(TfLiteOpaqueContext* context) {
  TfLiteStatus result = PreGetSupportedForWholeGraph(
      context, neuron_, &options_, &node_is_supported_);
  if (result != kTfLiteOk) {
    options_.operation_check_mode = kPerNodeOperationCheck;
  }
}

}  // namespace neuron
}  // namespace tflite
