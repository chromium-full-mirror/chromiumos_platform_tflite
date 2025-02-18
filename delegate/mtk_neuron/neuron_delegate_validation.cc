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

#include "delegate/mtk_neuron/neuron_delegate_validation.h"

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "common/minimal_logging.h"
#include "delegate/mtk_neuron/neuron_delegate.h"
#include "delegate/mtk_neuron/neuron_delegate_kernel.h"
#include "delegate/mtk_neuron/neuron_delegate_utils.h"
#include "delegate/mtk_neuron/neuron_implementation.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace neuron {
namespace {

constexpr int32_t kMinMagicNumberForNeuron33 = 28;

struct OpValidationContext {
  bool is_valid;
  std::vector<NeuronValidationFailure>* validation_failures;
};

#define EXPECT_INPUT_TYPE_IN(actual_type, ...)                     \
  ExpectTypeIn(actual_type, {__VA_ARGS__},                         \
               NeuronValidationFailureType::kUnsupportedInputType, \
               "Input type not in expected list " #__VA_ARGS__, &val_ctx)

void AddValidationFailure(NeuronValidationFailureType failure_type,
                          const char* message, OpValidationContext* val_ctx) {
  val_ctx->is_valid = false;

  if (val_ctx->validation_failures) {
    val_ctx->validation_failures->push_back({failure_type, message});
  }
}

template <typename... Args>
void AddValidationFailureFmt(OpValidationContext* val_ctx,
                             NeuronValidationFailureType failure_type,
                             const char* message_fmt, Args... args) {
  val_ctx->is_valid = false;
#ifdef NEURON_VERBOSE_VALIDATION
  if (val_ctx->validation_failures) {
    size_t req_buf_size = snprintf(nullptr, 0, message_fmt, args...) + 1;
    std::unique_ptr<char[]> tmp_buf(new char[req_buf_size]);
    snprintf(tmp_buf.get(), req_buf_size, message_fmt, args...);

    val_ctx->validation_failures->push_back({failure_type, tmp_buf.get()});
  }
#endif
}

bool Expect(bool condition, NeuronValidationFailureType failure_type,
            const char* message, OpValidationContext* val_ctx) {
  if (!condition) {
    AddValidationFailure(failure_type, message, val_ctx);
    return false;
  }
  return true;
}

template <typename... Args>
bool ExpectFmt(bool condition, OpValidationContext* val_ctx,
               NeuronValidationFailureType failure_type,
               const char* message_fmt, Args... args) {
  if (!condition) {
    AddValidationFailureFmt(val_ctx, failure_type, message_fmt, args...);
    return false;
  }
  return true;
}

bool ExpectTypeIn(TfLiteType actual_type,
                  std::initializer_list<TfLiteType> allowed_types,
                  NeuronValidationFailureType failure_type, const char* msg,
                  OpValidationContext* val_ctx) {
  return Expect(std::find(allowed_types.begin(), allowed_types.end(),
                          actual_type) != allowed_types.end(),
                failure_type, msg, val_ctx);
}

[[maybe_unused]] bool ExpectMinAndroidSdkVersion(int curr_version,
                                                 int min_version,
                                                 OpValidationContext* val_ctx) {
  return ExpectFmt(curr_version >= min_version, val_ctx,
                   NeuronValidationFailureType::kUnsupportedAndroidVersion,
                   "Android sdk version less than %d", min_version);
}

bool ExpectMaxOpVersion(int curr_version, int max_version,
                        OpValidationContext* val_ctx) {
  return ExpectFmt(curr_version <= max_version, val_ctx,
                   NeuronValidationFailureType::kUnsupportedOperatorVersion,
                   "OP Version higher than %d", max_version);
}

bool ExpectOpVersion(int curr_version, int max_version,
                     OpValidationContext* val_ctx) {
  return ExpectFmt(curr_version <= max_version, val_ctx,
                   NeuronValidationFailureType::kUnsupportedOperatorVersion,
                   "OP Version different from %d", max_version);
}

[[maybe_unused]] bool ExpectIsFloatOperator(const TfLiteOpaqueContext* context,
                                            const TfLiteOpaqueNode* node,
                                            OpValidationContext* val_ctx) {
  const auto input_type =
      TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
  return Expect(IsFloat(input_type),
                NeuronValidationFailureType::kUnsupportedInputType,
                "Input should be Float", val_ctx);
}

[[maybe_unused]] bool ExpectIsFloatOrUint8Operator(
    const TfLiteOpaqueContext* context, const TfLiteOpaqueNode* node,
    OpValidationContext* val_ctx) {
  const auto input_type =
      TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
  return Expect(IsFloatOrUInt8(input_type),
                NeuronValidationFailureType::kUnsupportedInputType,
                "Input should be Float or UINT8", val_ctx);
}

bool ExpectIsFloatOrQuant8Operator(const TfLiteOpaqueContext* context,
                                   const TfLiteOpaqueNode* node,
                                   OpValidationContext* val_ctx) {
  const auto input_type =
      TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
  return Expect(IsFloat(input_type) || IsQuantized(input_type),
                NeuronValidationFailureType::kUnsupportedInputType,
                "Input should be Float or Quant8", val_ctx);
}

bool IsFloatQuantizedOrInt32(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
    case kTfLiteUInt8:
    case kTfLiteInt8:
    case kTfLiteInt32:
      return true;
    default:
      return false;
  }
}
bool ExpectIsFloatQuant8OrInt32Operator(const TfLiteOpaqueContext* context,
                                        const TfLiteOpaqueNode* node,
                                        OpValidationContext* val_ctx) {
  const auto input_type =
      TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
  return Expect(IsFloatQuantizedOrInt32(input_type),
                NeuronValidationFailureType::kUnsupportedInputType,
                "Input should be Float, Quant8, or Int32", val_ctx);
}

// When using Neuron, the condition below must be true
// for quantized versions of the following ops:
// * CONV_2D
// * DEPTHWISE_CONV_2D
// * FULLY_CONNECTED (where filter actually stands for weights)
bool ExpectIsRestrictedScalesCompliant(const TfLiteOpaqueContext* context,
                                       const TfLiteOpaqueNode* node,
                                       OpValidationContext* val_ctx) {
  const float input_scale = TfLiteOpaqueTensorGetQuantizationParams(
                                TfLiteOpaqueNodeGetInput(context, node, 0))
                                .scale;
  const float filter_scale = TfLiteOpaqueTensorGetQuantizationParams(
                                 TfLiteOpaqueNodeGetInput(context, node, 1))
                                 .scale;
  float output_scale =
      TfLiteOpaqueTensorGetQuantizationParams(
          TfLiteOpaqueNodeGetOutput(const_cast<TfLiteOpaqueContext*>(context),
                                    node, 0))
          .scale;
  return Expect(
      input_scale * filter_scale < output_scale,
      NeuronValidationFailureType::kNotRestrictedScaleCompliant,
      "When using Neuron, input_scale * filter_scale < output_scale:", val_ctx);
}

// Pack one node to NeuronModel_getSupportedOperations
bool ExpectIsSupportedOperationForOneNode(
    const TfLiteRegistrationExternal* registration,
    const TfLiteOpaqueNode* node, TfLiteOpaqueContext* context,
    const NeuronApi* neuronapi, const NeuronDelegateOptions* options,
    OpValidationContext* val_ctx) {
  bool is_supported = false;
  auto kernel_state =
      std::make_unique<NeuronStableDelegateKernel>(neuronapi, *options);
  if (kernel_state->GetSupportedOperationsForOneNode(
          context, const_cast<TfLiteOpaqueNode*>(node),
          const_cast<TfLiteRegistrationExternal*>(registration),
          &is_supported) != kTfLiteOk) {
    AddValidationFailure(NeuronValidationFailureType::kUnsupportedOperator,
                         "GetSupportedOperationsForOneNode returns error",
                         val_ctx);
    return false;
  }
  return Expect(is_supported, NeuronValidationFailureType::kUnsupportedOperator,
                "result of getSupportedOperations is false", val_ctx);
}

}  // namespace

// In SPLIT_V, it is legal to specify -1 in size_splits representing an unknown
// split size taking as many values as possible. This function computes and
// returns the actual value of this unknown size, or returns -1 if all split
// sizes are known. The caller is responsible for making sure the size_splits
// and axis tensor are constants.
int ComputeSplitVUnknownSplitSize(const TfLiteOpaqueContext* context,
                                  const TfLiteOpaqueNode* node) {
  const auto& input = TfLiteOpaqueNodeGetInput(context, node, 0);
  const auto& size_splits_tensor = TfLiteOpaqueNodeGetInput(context, node, 1);
  const auto& axis_tensor = TfLiteOpaqueNodeGetInput(context, node, 2);

  const auto* size_splits =
      reinterpret_cast<int32_t*>(TfLiteOpaqueTensorData(size_splits_tensor));
  int num_splits = TfLiteOpaqueTensorDim(size_splits_tensor, 0);
  bool has_unknown_split_size = false;
  int sum_of_known_split_sizes = 0;
  for (int i = 0; i < num_splits; i++) {
    if (size_splits[i] == -1) {
      has_unknown_split_size = true;
    } else {
      sum_of_known_split_sizes += size_splits[i];
    }
  }

  int axis = reinterpret_cast<int32_t*>(TfLiteOpaqueTensorData(axis_tensor))[0];
  axis = axis < 0 ? axis + TfLiteOpaqueTensorNumDims(input) : axis;
  int total_size = TfLiteOpaqueTensorDim(input, axis);
  return has_unknown_split_size ? total_size - sum_of_known_split_sizes : -1;
}

TfLiteStatus PreGetSupportedForWholeGraph(TfLiteOpaqueContext* context,
                                          const NeuronApi* neuronapi,
                                          const NeuronDelegateOptions* options,
                                          std::vector<int>* is_supported) {
  TfLiteIntArray* execution_plan = nullptr;
  if (TfLiteOpaqueContextGetExecutionPlan(context, &execution_plan) !=
      kTfLiteOk) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "PreOpCheck: Fail to get exexution plan");
    return kTfLiteError;
  }
  int num_node = execution_plan->size;
  TfLiteOpaqueNode* ref_node;
  TfLiteRegistrationExternal* ref_reg;
  std::vector<NeuronValidationFailure> failure;
  // Check if all nodes pass the validate function
  for (int i = 0; i < num_node; i++) {
    if (TfLiteOpaqueContextGetNodeAndRegistration(context, i, &ref_node,
                                                  &ref_reg) != kTfLiteOk) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                      "PreOpCheck: Fail to get Node and Registration.");
      return kTfLiteError;
    }
    bool node_result =
        Validate(ref_reg, ref_node, context, neuronapi, options, &failure);
    if (!node_result) {
      TFLITE_LOG_PROD(
          tflite::TFLITE_LOG_ERROR,
          "PreOpCheck: OP %s (v%d) is not supported (%s)",
          tflite::EnumNameBuiltinOperator(static_cast<BuiltinOperator>(
              TfLiteOperatorGetBuiltInCode(ref_reg))),
          TfLiteRegistrationExternalGetVersion(ref_reg),
          failure.size() > 0 ? failure[0].message.c_str() : "");
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                      "PreOpCheck: Fail validation because of above op.");
      return kTfLiteError;
    }
  }
  // GetSupportedOperationsForWholeGraph
  auto support_flags = std::make_unique<bool[]>(num_node);
  auto kernel_state =
      std::make_unique<NeuronStableDelegateKernel>(neuronapi, *options);
  if (kernel_state->GetSupportedOperationsForWholeGraph(
          context, support_flags.get()) != kTfLiteOk) {
    TFLITE_LOG_PROD(
        tflite::TFLITE_LOG_ERROR,
        "PreOpCheck: GetSupportedOperationsForWholeGraph returns error");
    return kTfLiteError;
  }
  is_supported->clear();
  for (int i = 0; i < num_node; i++) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                    "PreOpCheck: support_flags[%d] = %d", i, support_flags[i]);
    is_supported->push_back(support_flags[i] ? 1 : 0);
  }
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE, "PreOpCheck: succesfully");
  return kTfLiteOk;
}

bool Validate(const TfLiteRegistrationExternal* registration,
              const TfLiteOpaqueNode* node, TfLiteOpaqueContext* context,
              const NeuronApi* neuronapi, const NeuronDelegateOptions* options,
              std::vector<NeuronValidationFailure>* map_failures) {
  static const NeuronRuntimeVersion neuron_version = [&] {
    NeuronRuntimeVersion version = {};
    if (neuronapi->Neuron_getVersion(&version) != NEURON_NO_ERROR) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                      "Failed to get neuron runtime version.");
    }
    return version;
  }();
  OpValidationContext val_ctx{true, map_failures};

  const auto builtin_code = TfLiteOperatorGetBuiltInCode(registration);
  const auto version = TfLiteRegistrationExternalGetVersion(registration);
  switch (builtin_code) {
    case kTfLiteBuiltinAdd: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinArgMax:
    case kTfLiteBuiltinArgMin: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      const TfLiteType input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      const TfLiteType axis_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 1));
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat16, kTfLiteFloat32,
                           kTfLiteUInt8, kTfLiteInt8);
      Expect(axis_type == kTfLiteInt32,
             NeuronValidationFailureType::kUnsupportedOutputType,
             "Neuron only supports int32 axis.", &val_ctx);
      if (builtin_code == kTfLiteBuiltinArgMax) {
        auto builtin = reinterpret_cast<TfLiteArgMaxParams*>(
            TfLiteOpaqueNodeGetBuiltinData(node));
        Expect(builtin->output_type == kTfLiteInt32,
               NeuronValidationFailureType::kUnsupportedOutputType,
               "Neuron only supports int32 output.", &val_ctx);
      } else {
        auto builtin = reinterpret_cast<TfLiteArgMinParams*>(
            TfLiteOpaqueNodeGetBuiltinData(node));
        Expect(builtin->output_type == kTfLiteInt32,
               NeuronValidationFailureType::kUnsupportedOutputType,
               "Neuron only supports int32 output.", &val_ctx);
      }
    } break;
    case kTfLiteBuiltinMul: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinAveragePool2d: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinMaxPool2d: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinL2Pool2d: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinConv2d: {
      ExpectMaxOpVersion(version, 3, &val_ctx);
      const TfLiteType input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      const TfLiteType filter_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 1));
      if (input_type == kTfLiteUInt8) {
        ExpectIsRestrictedScalesCompliant(context, node, &val_ctx);
      }
      Expect(filter_type != kTfLiteInt4,
             NeuronValidationFailureType::kUnsupportedOutputType,
             "Unsupported filter of type kTfLiteInt4", &val_ctx);
      // TODO(Code): Add support for Conv2D with omitted bias.
      Expect(TfLiteOpaqueNodeNumberOfInputs(node) == 3,
             NeuronValidationFailureType::kMissingRequiredOperand,
             "Conv2D with omitted bias not supported", &val_ctx);
    } break;
    case kTfLiteBuiltinDepthwiseConv2d: {
      ExpectMaxOpVersion(version, 3, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
      const TfLiteType input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      const TfLiteType filter_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 1));
      if (input_type == kTfLiteUInt8) {
        ExpectIsRestrictedScalesCompliant(context, node, &val_ctx);
      }
      Expect(filter_type != kTfLiteInt4,
             NeuronValidationFailureType::kUnsupportedOutputType,
             "Unsupported filter of type kTfLiteInt4", &val_ctx);
    } break;
    case kTfLiteBuiltinFullyConnected: {
      ExpectMaxOpVersion(version, 5, &val_ctx);
      const TfLiteType input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      const TfLiteType filter_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 1));
      if (input_type == kTfLiteUInt8) {
        ExpectIsRestrictedScalesCompliant(context, node, &val_ctx);
      }
      Expect(filter_type != kTfLiteInt4,
             NeuronValidationFailureType::kUnsupportedOutputType,
             "Unsupported filter of type kTfLiteInt4", &val_ctx);
      const int* inputs;
      int num_inputs;
      TF_LITE_OPAQUE_ENSURE_STATUS(
          TfLiteOpaqueNodeInputs(node, &inputs, &num_inputs));
      const bool is_bias_present = TfLiteOpaqueNodeNumberOfInputs(node) == 3 &&
                                   inputs[2] != kTfLiteOptionalTensor;
      if (is_bias_present) {
        const TfLiteType bias_type =
            TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 2));
        Expect(bias_type != kTfLiteInt64,
               NeuronValidationFailureType::kUnsupportedOutputType,
               "Unsupported bias of type kTfLiteInt64", &val_ctx);
      }
    } break;
    case kTfLiteBuiltinHardSwish: {
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinSoftmax: {
      ExpectOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
      const auto& output = TfLiteOpaqueNodeGetOutput(context, node, 0);
      ExpectTypeIn(TfLiteOpaqueTensorType(output), {kTfLiteFloat32},
                   NeuronValidationFailureType::kUnsupportedOutputType,
                   "Output type should be one of kTfLiteFloat32.", &val_ctx);
      const auto& input = TfLiteOpaqueNodeGetInput(context, node, 0);
      const int input_rank = TfLiteOpaqueTensorNumDims(input);
      Expect(input_rank <= 4,
             NeuronValidationFailureType::kUnsupportedOperandRank,
             "Input rank should be <= 4", &val_ctx);
    } break;
    case kTfLiteBuiltinReshape: {
      ExpectOpVersion(version, 1, &val_ctx);
      ExpectIsFloatQuant8OrInt32Operator(context, node, &val_ctx);
      if (TfLiteOpaqueNodeNumberOfInputs(node) >= 2) {
        Expect(TfLiteOpaqueTensorGetAllocationType(
                   TfLiteOpaqueNodeGetInput(context, node, 1)) == kTfLiteMmapRo,
               NeuronValidationFailureType::kInputTensorShouldHaveConstantShape,
               "The shape input tensor must be constant.", &val_ctx);
      }
      if (TfLiteOpaqueNodeNumberOfInputs(node) == 1) {
        // reject scalar reshaping
        auto* params = reinterpret_cast<TfLiteReshapeParams*>(
            TfLiteOpaqueNodeGetBuiltinData(node));
        int num_dimensions = params->num_dimensions;
        if (num_dimensions == 1 && params->shape[0] == 0) {
          // Legacy tflite models use a shape parameter of [0] to indicate
          // scalars.
          num_dimensions = 0;
        }
        Expect(num_dimensions > 0,
               NeuronValidationFailureType::kUnsupportedOperandRank,
               "New shape rank should be > 0", &val_ctx);
      }
    } break;
    case kTfLiteBuiltinResizeBilinear: {
      ExpectMaxOpVersion(version, 3, &val_ctx);
      const auto& input = TfLiteOpaqueNodeGetInput(context, node, 0);
      Expect(TfLiteOpaqueTensorNumDims(input) == 4,
             NeuronValidationFailureType::kUnsupportedOperandRank,
             "Input should have rank 4", &val_ctx);
      Expect(TfLiteOpaqueNodeNumberOfInputs(node) >= 2,
             NeuronValidationFailureType::kUnsupportedOperatorVariant,
             "Expected at least 2 inputs", &val_ctx);
      if (TfLiteOpaqueNodeNumberOfInputs(node) >= 2) {
        Expect(TfLiteOpaqueTensorGetAllocationType(
                   TfLiteOpaqueNodeGetInput(context, node, 1)) == kTfLiteMmapRo,
               NeuronValidationFailureType::kInputTensorShouldHaveConstantShape,
               "The size input tensor must be constant.", &val_ctx);
      }
    } break;
    case kTfLiteBuiltinResizeNearestNeighbor: {
      ExpectMaxOpVersion(version, 3, &val_ctx);
    } break;
    case kTfLiteBuiltinSqueeze: {
      ExpectOpVersion(version, 1, &val_ctx);
    } break;
    case kTfLiteBuiltinL2Normalization: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      auto builtin = reinterpret_cast<TfLiteL2NormParams*>(
          TfLiteOpaqueNodeGetBuiltinData(node));
      Expect(builtin->activation == kTfLiteActNone,
             NeuronValidationFailureType::kNoActivationExpected,
             "Expected no activation", &val_ctx);
    } break;
    case kTfLiteBuiltinConcatenation: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      Expect(reinterpret_cast<TfLiteConcatenationParams*>(
                 TfLiteOpaqueNodeGetBuiltinData(node))
                     ->activation == kTfLiteActNone,
             NeuronValidationFailureType::kNoActivationExpected,
             "No activation function supported", &val_ctx);
      Expect(TfLiteOpaqueTensorNumDims(
                 TfLiteOpaqueNodeGetInput(context, node, 0)) <= 4,
             NeuronValidationFailureType::kUnsupportedOperandRank,
             "Input rank should be less than 4", &val_ctx);

      const auto& input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat16, kTfLiteFloat32,
                           kTfLiteUInt8, kTfLiteInt8);
    } break;
    case kTfLiteBuiltinDequantize: {
      // Allow dequantizing fp16->fp32.
      const auto& input = TfLiteOpaqueNodeGetInput(context, node, 0);
      if (TfLiteOpaqueTensorType(input) == kTfLiteFloat16 &&
          TfLiteOpaqueTensorGetAllocationType(input) == kTfLiteMmapRo) {
        return true;
      }
      ExpectOpVersion(version, 2, &val_ctx);
      EXPECT_INPUT_TYPE_IN(TfLiteOpaqueTensorType(input), kTfLiteUInt8,
                           kTfLiteInt8);
    } break;
    case kTfLiteBuiltinFloor: {
      ExpectOpVersion(version, 1, &val_ctx);
    } break;
    case kTfLiteBuiltinRelu:
    case kTfLiteBuiltinReluN1To1:
    case kTfLiteBuiltinRelu6:
    case kTfLiteBuiltinLogistic:
    case kTfLiteBuiltinTanh: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinSub: {
      ExpectMaxOpVersion(version, 3, &val_ctx);
      const int input0_rank =
          TfLiteOpaqueTensorNumDims(TfLiteOpaqueNodeGetInput(context, node, 0));
      const int input1_rank =
          TfLiteOpaqueTensorNumDims(TfLiteOpaqueNodeGetInput(context, node, 1));
      Expect(input0_rank <= 4 && input1_rank <= 4,
             NeuronValidationFailureType::kUnsupportedOperandRank,
             "Input rank must be <= 4", &val_ctx);
    } break;
    case kTfLiteBuiltinDiv: {
      ExpectOpVersion(version, 1, &val_ctx);
    } break;
    case kTfLiteBuiltinPad:
    case kTfLiteBuiltinPadv2: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
      const TfLiteOpaqueTensor* input_tensor =
          TfLiteOpaqueNodeGetInput(context, node, 0);
      for (int i = 0; i < TfLiteOpaqueTensorNumDims(input_tensor); i++) {
        Expect(TfLiteOpaqueTensorDim(input_tensor, i) != 0,
               NeuronValidationFailureType::kUnsupportedOperandValue,
               "Neuron pad ops do not support input tensors with no elements",
               &val_ctx);
      }
      Expect(TfLiteOpaqueNodeNumberOfInputs(node) >= 2,
             NeuronValidationFailureType::kUnsupportedOperatorVariant,
             "Expecting at least 2 inputs", &val_ctx);
    } break;
    case kTfLiteBuiltinSpaceToBatchNd: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
    } break;
    case kTfLiteBuiltinBatchToSpaceNd: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      auto crops = TfLiteOpaqueNodeGetInput(context, node, 2);
      auto crops_data =
          reinterpret_cast<int32_t*>(TfLiteOpaqueTensorData(crops));
      Expect(crops_data && TfLiteOpaqueTensorByteSize(crops) == 16 &&
                 crops_data[0] == 0 && crops_data[1] == 0 &&
                 crops_data[2] == 0 && crops_data[3] == 0,
             NeuronValidationFailureType::kUnsupportedOperandValue,
             "All crops should be 0.", &val_ctx);
    } break;
    case kTfLiteBuiltinStridedSlice: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
    } break;
    case kTfLiteBuiltinTranspose: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      // Note that the permutation input tensor value dictates the output
      // dimensions.
      // TODO(Code): Support dynamically-sized tensors in delegates.
      Expect((TfLiteOpaqueNodeNumberOfInputs(node) > 1) &&
                 (TfLiteOpaqueTensorGetAllocationType(TfLiteOpaqueNodeGetInput(
                      context, node, 1)) == kTfLiteMmapRo),
             NeuronValidationFailureType::kInputTensorShouldHaveConstantShape,
             "Dynamically-sized tensors not supported.", &val_ctx);
    } break;
    case kTfLiteBuiltinAbs:
    case kTfLiteBuiltinExp:
    case kTfLiteBuiltinLog:
    case kTfLiteBuiltinPow: {
      ExpectOpVersion(version, 1, &val_ctx);
      const auto input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat16, kTfLiteFloat32);
    } break;
    case kTfLiteBuiltinSlice: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      const auto input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      const auto begin_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 1));
      const auto size_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 2));
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteInt32,
                           kTfLiteUInt8, kTfLiteInt8);
      Expect(begin_type == kTfLiteInt32,
             NeuronValidationFailureType::kUnsupportedInputType,
             "Begin type should be Int32", &val_ctx);
      Expect(size_type == kTfLiteInt32,
             NeuronValidationFailureType::kUnsupportedInputType,
             "Size type should be Int32", &val_ctx);
    } break;
    case kTfLiteBuiltinTransposeConv: {
      ExpectMaxOpVersion(version, 3, &val_ctx);
      // Remove dynamically-sized tensor check
    } break;
    case kTfLiteBuiltinSqrt: {
      ExpectOpVersion(version, 1, &val_ctx);
    } break;
    case kTfLiteBuiltinSpaceToDepth: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      ExpectIsFloatOrQuant8Operator(context, node, &val_ctx);
    } break;
    case kTfLiteBuiltinMean: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      Expect(TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(
                 context, node, 0)) == kTfLiteFloat32 ||
                 IsQuantized(TfLiteOpaqueTensorType(
                     TfLiteOpaqueNodeGetInput(context, node, 0))),
             NeuronValidationFailureType::kUnsupportedInputType,
             "Expected Float32 or Quantized input", &val_ctx);
    } break;
    case kTfLiteBuiltinEmbeddingLookup: {
      ExpectOpVersion(version, 1, &val_ctx);
    } break;
    case kTfLiteBuiltinHashtableLookup: {
      ExpectOpVersion(version, 1, &val_ctx);
    } break;
    case kTfLiteBuiltinMaximum:
    case kTfLiteBuiltinMinimum: {
      ExpectMaxOpVersion(version, 3, &val_ctx);
      const auto input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteUInt8,
                           kTfLiteInt8, kTfLiteInt32);
      const TfLiteOpaqueTensor* operand0 =
          TfLiteOpaqueNodeGetInput(context, node, 0);
      if (TfLiteOpaqueTensorNumDims(operand0) == 0) {
        Expect(TfLiteOpaqueTensorGetAllocationType(operand0) == kTfLiteMmapRo,
               NeuronValidationFailureType::kUnsupportedInputType,
               "Scalar operand should be constant", &val_ctx);
      }
      const TfLiteOpaqueTensor* operand1 =
          TfLiteOpaqueNodeGetInput(context, node, 1);
      if (TfLiteOpaqueTensorNumDims(operand1) == 0) {
        Expect(TfLiteOpaqueTensorGetAllocationType(operand1) == kTfLiteMmapRo,
               NeuronValidationFailureType::kUnsupportedInputType,
               "Scalar operand should be constant", &val_ctx);
      }
    } break;
    case kTfLiteBuiltinCast: {
      ExpectOpVersion(version, 1, &val_ctx);
      const TfLiteType input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      const TfLiteType output_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetOutput(context, node, 0));
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteInt32, kTfLiteUInt8, kTfLiteInt8);
      if (neuron_version.major == 8) {
        // Neuron Pilot 8 is coupled with MT8196, which has MVPU support.
        // MVPU can handle Int32 -> Float32 conversion.
        ExpectTypeIn(output_type,
                     {kTfLiteInt32, kTfLiteUInt8, kTfLiteInt8, kTfLiteFloat32},
                     NeuronValidationFailureType::kUnsupportedOutputType,
                     "Output type should be one of kTfLiteInt32, "
                     "kTfLiteUInt8, kTfLiteInt8, kTfLiteFloat32.",
                     &val_ctx);
      } else {
        ExpectTypeIn(output_type, {kTfLiteInt32, kTfLiteUInt8, kTfLiteInt8},
                     NeuronValidationFailureType::kUnsupportedOutputType,
                     "Output type should be one of kTfLiteInt32, "
                     "kTfLiteUInt8, kTfLiteInt8.",
                     &val_ctx);
      }
    } break;
    case kTfLiteBuiltinLeakyRelu:
    case kTfLiteBuiltinPrelu: {
      ExpectOpVersion(version, 1, &val_ctx);
      const auto input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteUInt8,
                           kTfLiteInt8);
    } break;
    case kTfLiteBuiltinLogicalOr:
    case kTfLiteBuiltinLogicalAnd:
    case kTfLiteBuiltinLogicalNot: {
      ExpectOpVersion(version, 1, &val_ctx);
      const auto input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      Expect(input_type == kTfLiteBool,
             NeuronValidationFailureType::kUnsupportedInputType,
             "Input should be bool", &val_ctx);
    } break;
    case kTfLiteBuiltinLess:
    case kTfLiteBuiltinLessEqual:
    case kTfLiteBuiltinGreater:
    case kTfLiteBuiltinGreaterEqual:
    case kTfLiteBuiltinEqual:
    case kTfLiteBuiltinNotEqual: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      const auto input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteUInt8,
                           kTfLiteInt8, kTfLiteBool, kTfLiteInt32);
    } break;
    case kTfLiteBuiltinNeg: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      const auto input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteInt32);
    } break;
    case kTfLiteBuiltinTopkV2: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      const auto& input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteInt32,
                           kTfLiteUInt8, kTfLiteInt8);
      const auto& k_param = TfLiteOpaqueNodeGetInput(context, node, 1);
      Expect(TfLiteOpaqueTensorType(k_param) == kTfLiteInt32 &&
                 TfLiteOpaqueTensorGetAllocationType(k_param) == kTfLiteMmapRo,
             NeuronValidationFailureType::kUnsupportedInputType,
             "K param should be a constant of type Int32", &val_ctx);
    } break;
    case kTfLiteBuiltinSelect: {
      ExpectMaxOpVersion(version, 2, &val_ctx);
      const auto value_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 1));
      EXPECT_INPUT_TYPE_IN(value_type, kTfLiteFloat32, kTfLiteInt32,
                           kTfLiteUInt8, kTfLiteInt8);
      const TfLiteOpaqueTensor* condition_tensor =
          TfLiteOpaqueNodeGetInput(context, node, 0);
      const TfLiteOpaqueTensor* input_tensor =
          TfLiteOpaqueNodeGetInput(context, node, 1);
      bool same_num_of_dim = TfLiteOpaqueTensorNumDims(condition_tensor) ==
                             TfLiteOpaqueTensorNumDims(input_tensor);
      Expect(same_num_of_dim,
             NeuronValidationFailureType::kUnsupportedOperandValue,
             "Condition and inputs tensors shuld have the number of dims",
             &val_ctx);
      if (same_num_of_dim) {
        for (int i = 0; i < TfLiteOpaqueTensorNumDims(condition_tensor); i++) {
          Expect(TfLiteOpaqueTensorDim(condition_tensor, i) ==
                     TfLiteOpaqueTensorDim(input_tensor, i),
                 NeuronValidationFailureType::kUnsupportedOperandValue,
                 "Condition and inputs tensors shuld have the same shape",
                 &val_ctx);
        }
      }
    } break;
    case kTfLiteBuiltinGather: {
      ExpectOpVersion(version, 2, &val_ctx);
      const auto input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      const auto& positions = TfLiteOpaqueNodeGetInput(context, node, 1);

      // TODO: neuropilot workaround for gather(remove float32 support)
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat16, kTfLiteInt32,
                           kTfLiteUInt8, kTfLiteInt8);

      Expect(TfLiteOpaqueTensorType(positions) == kTfLiteInt32,
             NeuronValidationFailureType::kUnsupportedInputType,
             "Positions type should be one of kTfLiteInt32", &val_ctx);
      Expect(TfLiteOpaqueTensorNumDims(positions) != 0,
             NeuronValidationFailureType::kUnsupportedOperandRank,
             "0-dimension args are not supported by Neuron.", &val_ctx);
    } break;
    case kTfLiteBuiltinSplit: {
      ExpectOpVersion(version, 3, &val_ctx);
      const TfLiteOpaqueTensor* input =
          TfLiteOpaqueNodeGetInput(context, node, 1);
      EXPECT_INPUT_TYPE_IN(TfLiteOpaqueTensorType(input), kTfLiteFloat32,
                           kTfLiteUInt8, kTfLiteInt8, kTfLiteInt32);
      const TfLiteOpaqueTensor* axis =
          TfLiteOpaqueNodeGetInput(context, node, 0);
      Expect(TfLiteOpaqueTensorType(axis) == kTfLiteInt32 &&
                 TfLiteOpaqueTensorGetAllocationType(axis) == kTfLiteMmapRo,
             NeuronValidationFailureType::kUnsupportedInputType,
             "Neuron only supports constant int32 axis tensor.", &val_ctx);
    } break;
    case kTfLiteBuiltinSplitV: {
      ExpectOpVersion(version, 2, &val_ctx);
      // Tensor indices: value: 0, size_splits: 1, axis: 2
      const TfLiteOpaqueTensor* input =
          TfLiteOpaqueNodeGetInput(context, node, 0);
      const TfLiteOpaqueTensor* size_splits =
          TfLiteOpaqueNodeGetInput(context, node, 1);
      const TfLiteOpaqueTensor* axis =
          TfLiteOpaqueNodeGetInput(context, node, 2);
      EXPECT_INPUT_TYPE_IN(TfLiteOpaqueTensorType(input), kTfLiteFloat32,
                           kTfLiteUInt8, kTfLiteInt8, kTfLiteInt32);
      bool size_splits_is_int32_const_vector =
          TfLiteOpaqueTensorType(size_splits) == kTfLiteInt32 &&
          TfLiteOpaqueTensorNumDims(size_splits) == 1 &&
          TfLiteOpaqueTensorGetAllocationType(size_splits) == kTfLiteMmapRo;
      bool axis_is_int32_const =
          TfLiteOpaqueTensorType(axis) == kTfLiteInt32 &&
          TfLiteOpaqueTensorGetAllocationType(axis) == kTfLiteMmapRo;
      Expect(size_splits_is_int32_const_vector,
             NeuronValidationFailureType::kUnsupportedInputType,
             "Neuron only supports constant int32 size_splits vector.",
             &val_ctx);
      Expect(axis_is_int32_const,
             NeuronValidationFailureType::kUnsupportedInputType,
             "Neuron only supports constant int32 axis tensor.", &val_ctx);
      if (size_splits_is_int32_const_vector && axis_is_int32_const) {
        int32_t* size_splits_data =
            reinterpret_cast<int32_t*>(TfLiteOpaqueTensorData(size_splits));
        Expect(std::all_of(
                   size_splits_data,
                   size_splits_data + TfLiteOpaqueTensorDim(size_splits, 0),
                   [](auto size) { return size != 0; }),
               NeuronValidationFailureType::kUnsupportedInputType,
               "Neuron only supports non-zero split sizes.", &val_ctx);
#if 1
        Expect(ComputeSplitVUnknownSplitSize(context, node) != 0,
               NeuronValidationFailureType::kUnsupportedInputType,
               "Neuron only supports non-zero split sizes.", &val_ctx);
#endif
      }
    } break;
    case kTfLiteBuiltinQuantize: {
      ExpectOpVersion(version, 2, &val_ctx);
      const auto value_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      Expect(value_type == kTfLiteFloat32 || value_type == kTfLiteInt16 ||
                 IsQuantized(value_type),
             NeuronValidationFailureType::kUnsupportedInputType,
             "Value should be quantized or Float32.", &val_ctx);
      if (value_type == kTfLiteInt16 || IsQuantized(value_type)) {
        const auto quantization_params =
            TfLiteOpaqueTensorGetQuantizationParams(
                TfLiteOpaqueNodeGetInput(context, node, 0));
        Expect(quantization_params.scale > 0.f,
               NeuronValidationFailureType::kUnsupportedQuantizationParameters,
               "Input quantization scale should be > 0.", &val_ctx);
        // MT8188 MDLA does not support asymi16
        if (value_type == kTfLiteInt16 && neuron_version.major == 6) {
          Expect(
              quantization_params.zero_point == 0,
              NeuronValidationFailureType::kUnsupportedQuantizationParameters,
              "Input quantization zero point should be 0.", &val_ctx);
        }
      }
      const auto output_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetOutput(context, node, 0));
      ExpectTypeIn(
          output_type, {kTfLiteInt16, kTfLiteUInt8, kTfLiteInt8},
          NeuronValidationFailureType::kUnsupportedOutputType,
          "Output should be kTfLiteInt16 or kTfLiteUInt8 or kTfLiteInt8.",
          &val_ctx);
      const auto quantization_params = TfLiteOpaqueTensorGetQuantizationParams(
          TfLiteOpaqueNodeGetOutput(context, node, 0));
      Expect(quantization_params.scale > 0.f,
             NeuronValidationFailureType::kUnsupportedQuantizationParameters,
             "Output quantization scale should be > 0.", &val_ctx);
    } break;
    case kTfLiteBuiltinReduceAny:
    case kTfLiteBuiltinReduceMin:
    case kTfLiteBuiltinReduceMax: {
      ExpectOpVersion(version, 2, &val_ctx);
      const TfLiteOpaqueTensor* axis_tensor =
          TfLiteOpaqueNodeGetInput(context, node, 1);
      if (TfLiteOpaqueTensorNumDims(axis_tensor) == 0) {
        Expect(TfLiteOpaqueTensorNumDims(axis_tensor) > 0,
               NeuronValidationFailureType::kUnsupportedOperator,
               "axis dimension size should be > 0.", &val_ctx);
      }
    } break;
    case kTfLiteBuiltinDepthToSpace: {
      const TfLiteType input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteFloat32, kTfLiteUInt8,
                           kTfLiteInt8);
    } break;
    case kTfLiteBuiltinSum: {
      ExpectOpVersion(version, 1, &val_ctx);
      const TfLiteOpaqueTensor* axis_tensor =
          TfLiteOpaqueNodeGetInput(context, node, 1);
      if (TfLiteOpaqueTensorNumDims(axis_tensor) == 0) {
        Expect(TfLiteOpaqueTensorNumDims(axis_tensor) > 0,
               NeuronValidationFailureType::kUnsupportedOperator,
               "axis dimension size should be > 0.", &val_ctx);
      }
    } break;
    case kTfLiteBuiltinPack: {
      ExpectOpVersion(version, 2, &val_ctx);
      const auto input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      EXPECT_INPUT_TYPE_IN(input_type, kTfLiteInt32, kTfLiteFloat32,
                           kTfLiteInt8);
      auto builtin = reinterpret_cast<TfLitePackParams*>(
          TfLiteOpaqueNodeGetBuiltinData(node));
      Expect(
          builtin->axis != -1 &&
              builtin->axis != TfLiteOpaqueTensorNumDims(
                                   TfLiteOpaqueNodeGetInput(context, node, 0)),
          NeuronValidationFailureType::kUnsupportedOperandValue,
          "Neuron does not support axis being the last dimension", &val_ctx);
    } break;
    case kTfLiteBuiltinSquaredDifference:
    case kTfLiteBuiltinRsqrt:
    case kTfLiteBuiltinMirrorPad:
    case kTfLiteBuiltinUnpack: {
      // Use BuiltinOP as MTK EXT OP
    } break;
    case kTfLiteBuiltinReverseV2: {
      // Use BuiltinOP as MTK EXT OP
      const TfLiteType input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      Expect(input_type != kTfLiteInt64,
             NeuronValidationFailureType::kUnsupportedOutputType,
             "Unsupported filter of type kTfLiteInt64", &val_ctx);
    } break;
#ifndef MTK_INTERNAL_USE
    case kTfLiteBuiltinCustom: {
      if (strcmp(TfLiteRegistrationExternalGetCustomName(registration),
                 "cros-mtk-pre-compile") == 0) {
        break;
      }
      int32_t magic_number = -1;
      // get NP magic number
      if (neuronapi->Neuron_getNeuroPilotMagicNumber != nullptr) {
        RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
            context, neuronapi->Neuron_getNeuroPilotMagicNumber(&magic_number),
            "get magic number");
      }
      if (strcmp(TfLiteRegistrationExternalGetCustomName(registration),
                 "TFLite_Detection_PostProcess") == 0 &&
          magic_number >= kMinMagicNumberForNeuron33) {
        ExpectIsSupportedOperationForOneNode(registration, node, context,
                                             neuronapi, options, &val_ctx);
        break;
      }
      [[fallthrough]];
    }  // no break, pass to default
#endif
    default:
      // All other operators are not mapped.
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                      "Unsupported operation type %d", builtin_code);
      AddValidationFailure(NeuronValidationFailureType::kUnsupportedOperator,
                           "Unsupported operation type.", &val_ctx);
  }
  if (val_ctx.is_valid) {
    if (options->operation_check_mode == kPerNodeOperationCheck) {
      ExpectIsSupportedOperationForOneNode(registration, node, context,
                                           neuronapi, options, &val_ctx);
    }
  }
  return val_ctx.is_valid;
}

}  // namespace neuron
}  // namespace tflite
