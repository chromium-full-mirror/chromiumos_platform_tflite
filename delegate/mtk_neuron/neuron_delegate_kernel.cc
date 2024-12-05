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

#include "delegate/mtk_neuron/neuron_delegate_kernel.h"

#include <farmhash.h>
#include <inttypes.h>
#include <linux/dma-heap.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <algorithm>
#include <iostream>

#include "common/minimal_logging.h"
#include "delegate/mtk_neuron/mtk/mtk_minimal_logging.h"
#include "delegate/mtk_neuron/mtk/mtk_utils.h"
#include "delegate/mtk_neuron/mtk/mtkext_ops.h"
#include "delegate/mtk_neuron/neuron_delegate_builder.h"
#include "delegate/mtk_neuron/neuron_delegate_utils.h"
#include "delegate/mtk_neuron/neuron_implementation.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/util.h"

#define LOG_TAG "NeuronDelegate"

namespace tflite {
namespace neuron {

namespace {

bool IsScalarInputSupported(int builtin_code) {
  switch (builtin_code) {
    case kTfLiteBuiltinAdd:
    case kTfLiteBuiltinMul:
    case kTfLiteBuiltinSub:
    case kTfLiteBuiltinDiv:
    case kTfLiteBuiltinEqual:
    case kTfLiteBuiltinNotEqual:
    case kTfLiteBuiltinGreater:
    case kTfLiteBuiltinGreaterEqual:
    case kTfLiteBuiltinLess:
    case kTfLiteBuiltinLessEqual:
    case kTfLiteBuiltinPow:
    case kTfLiteBuiltinMaximum:
    case kTfLiteBuiltinMinimum:
    case kTfLiteBuiltinPrelu:
    case kTfLiteBuiltinLeakyRelu:
    case kTfLiteBuiltinReduceMax:
    case kTfLiteBuiltinSum:
      return true;
    default:
      return false;
  }
}

// Check if the operation requires explict conversion from int8 to uint8 values.
[[maybe_unused]] bool NeedInt8Conversion(const TfLiteOpaqueContext* context,
                                         int builtin_code,
                                         const TfLiteOpaqueNode* node) {
  const TfLiteType input_type =
      TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
  switch (builtin_code) {
    case kTfLiteBuiltinConv2d:
    case kTfLiteBuiltinDepthwiseConv2d:
    case kTfLiteBuiltinFullyConnected: {
      if (input_type == kTfLiteInt8) {
        const TfLiteOpaqueTensor* weights_tensor =
            TfLiteOpaqueNodeGetInput(context, node, 1);
        if ((TfLiteOpaqueTensorType(weights_tensor) == kTfLiteInt8 ||
             TfLiteOpaqueTensorType(weights_tensor) == kTfLiteUInt8) &&
            TfLiteOpaqueTensorGetQuantization(weights_tensor).type ==
                kTfLiteAffineQuantization) {
          return true;
        }
      }
      return false;
    }
    case kTfLiteBuiltinTransposeConv: {
      // Transpose convolution has a different order of inputs:
      // 0: output_shape, 1: filter, 2: input, 3: bias.
      const TfLiteType input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 2));
      if (input_type == kTfLiteInt8) {
        return true;
      }
      return false;
    }
    case kTfLiteBuiltinSelect: {
      const auto value_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 1));
      return value_type == kTfLiteInt8;
    }
    case kTfLiteBuiltinAdd:
    case kTfLiteBuiltinArgMax:
    case kTfLiteBuiltinArgMin:
    case kTfLiteBuiltinAveragePool2d:
    case kTfLiteBuiltinBatchToSpaceNd:
    case kTfLiteBuiltinConcatenation:
    case kTfLiteBuiltinEqual:
    case kTfLiteBuiltinExpandDims:
    case kTfLiteBuiltinGather:
    case kTfLiteBuiltinGreater:
    case kTfLiteBuiltinGreaterEqual:
    case kTfLiteBuiltinHardSwish:
    case kTfLiteBuiltinL2Normalization:
    case kTfLiteBuiltinLeakyRelu:
    case kTfLiteBuiltinLess:
    case kTfLiteBuiltinLessEqual:
    case kTfLiteBuiltinLogistic:
    case kTfLiteBuiltinMaximum:
    case kTfLiteBuiltinMaxPool2d:
    case kTfLiteBuiltinMean:
    case kTfLiteBuiltinMinimum:
    case kTfLiteBuiltinMul:
    case kTfLiteBuiltinNotEqual:
    case kTfLiteBuiltinPad:
    case kTfLiteBuiltinPadv2:
    case kTfLiteBuiltinPrelu:
    case kTfLiteBuiltinReduceMax:
    case kTfLiteBuiltinReduceMin:
    case kTfLiteBuiltinRelu:
    case kTfLiteBuiltinReluN1To1:
    case kTfLiteBuiltinRelu6:
    case kTfLiteBuiltinResizeBilinear:
    case kTfLiteBuiltinResizeNearestNeighbor:
    case kTfLiteBuiltinReshape:
    case kTfLiteBuiltinSlice:
    case kTfLiteBuiltinSoftmax:
    case kTfLiteBuiltinSpaceToBatchNd:
    case kTfLiteBuiltinSpaceToDepth:
    case kTfLiteBuiltinDepthToSpace:
    case kTfLiteBuiltinStridedSlice:
    case kTfLiteBuiltinSub:
    case kTfLiteBuiltinTanh:
    case kTfLiteBuiltinTile:
    case kTfLiteBuiltinTopkV2:
    case kTfLiteBuiltinTranspose: {
      return input_type == kTfLiteInt8;
    }
    default:
      return false;
  }
}  // namespace

constexpr int kLstmFullKernelInputSize = 24;
// The 20 input version is deprecated and kept only to
// support old model. The latest version of the LSTM Full Kernel
// is the one with 24 inputs
constexpr int kLstmFullKernelNoOptionalParamsInputSize = 20;
constexpr int kLstmBasicKernelInputSize = 5;

inline bool isLstmBasicKernel(const TfLiteOpaqueNode* node) {
  return TfLiteOpaqueNodeNumberOfInputs(node) == kLstmBasicKernelInputSize;
}

inline bool isLstmFullKernel(const TfLiteOpaqueNode* node) {
  return TfLiteOpaqueNodeNumberOfInputs(node) == kLstmFullKernelInputSize ||
         TfLiteOpaqueNodeNumberOfInputs(node) ==
             kLstmFullKernelNoOptionalParamsInputSize;
}

bool IsHybridOperator(const TfLiteOpaqueContext* context, int builtin_code,
                      const TfLiteOpaqueNode* node) {
  switch (builtin_code) {
    case kTfLiteBuiltinConv2d:
    case kTfLiteBuiltinFullyConnected: {
      const TfLiteType input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      const TfLiteType filter_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 1));
      return IsFloat(input_type) && IsQuantized(filter_type);
    }
    case kTfLiteBuiltinLstm: {
      // Input #1 is optional so use #2 to determine if hybrid.
      const TfLiteType input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      const TfLiteType weights_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 2));
      return isLstmFullKernel(node) && IsFloat(input_type) &&
             IsQuantized(weights_type);
    }
    case kTfLiteBuiltinUnidirectionalSequenceLstm: {
      // Input #1 is optional so use #2 to determine if hybrid.
      const TfLiteType input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      const TfLiteType weights_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 2));
      return IsFloat(input_type) && IsQuantized(weights_type);
    }
    case kTfLiteBuiltinBidirectionalSequenceLstm: {
      // Input #1 is optional so use #2 to determine if hybrid.
      const TfLiteType input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      const TfLiteType weights_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 2));
      return IsFloat(input_type) && IsQuantized(weights_type);
    }
    case kTfLiteBuiltinUnidirectionalSequenceRnn: {
      const TfLiteType input_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 0));
      const TfLiteType weights_type =
          TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetInput(context, node, 1));
      return IsFloat(input_type) && IsQuantized(weights_type);
    }
    default:
      return false;
  }
}

constexpr size_t kDefaultByteAlignmentForNeuron = 128;

static size_t getNumPaddingBytes(size_t byte_size) {
  size_t num_padding_bytes = 0;
  if (byte_size % kDefaultByteAlignmentForNeuron) {
    num_padding_bytes = kDefaultByteAlignmentForNeuron -
                        (byte_size % kDefaultByteAlignmentForNeuron);
  }
  return num_padding_bytes;
}

// Compute the hash of a TfLiteIntArray.
uint64_t GetHash(const TfLiteIntArray* int_array, uint64_t combine_with = 0) {
  constexpr auto kHashConst = 0x9e3779b97f4a7800ULL;
  uint64_t result = combine_with;
  for (auto i : TfLiteIntArrayView(int_array)) {
    result = result ^ (i + kHashConst + (result << 10) + (result >> 4));
  }
  return result;
}

uint64_t GetHash(const int* int_array, int num_of_array,
                 uint64_t combine_with = 0) {
  constexpr auto kHashConst = 0x9e3779b97f4a7800ULL;
  uint64_t result = combine_with;
  for (int i = 0; i < num_of_array; i++) {
    result =
        result ^ (int_array[i] + kHashConst + (result << 10) + (result >> 4));
  }
  return result;
}

[[maybe_unused]] uint64_t GetInOutPutHash(TfLiteOpaqueContext* context,
                                          const TfLiteIntArray* int_array) {
  constexpr auto kHashConst = 0x9e3779b97f4a7800ULL;
  uint64_t result = 0;
  for (auto i : TfLiteIntArrayView(int_array)) {
    result = result ^ (TfLiteOpaqueTensorByteSize(
                           TfLiteOpaqueContextGetOpaqueTensor(context, i)) +
                       kHashConst + (result << 10) + (result >> 4));
  }
  return result;
}

TfLiteStatus GetAcceleratorDevice(const NeuronApi* neuron_api,
                                  TfLiteOpaqueContext* context,
                                  const char* device_name,
                                  NeuronDevice** result) {
  if (!device_name) {
    return kTfLiteError;
  }
  uint32_t num_devices = 0;
  *result = nullptr;

  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
      context, neuron_api->Neuron_getDeviceCount(&num_devices),
      "getting device count");

  for (uint32_t i = 0; i < num_devices; ++i) {
    const char* name = nullptr;
    NeuronDevice* device = nullptr;
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context, neuron_api->Neuron_getDevice(i, &device), "getting device");
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context, neuron_api->NeuronDevice_getName(device, &name),
        "getting device name");
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE, "Got device name: %s", name);
    if (std::string(device_name) == std::string(name)) {
      *result = device;
      return kTfLiteOk;
    }
  }

  TfLiteOpaqueContextReportError(
      context, "Could not find the specified Neuron accelerator: %s.",
      device_name);
  return kTfLiteError;
}

NeuronAdapterFuseCode GetEquivalentToNeuronFuseCode(
    TfLiteFusedActivation activation) {
  switch (activation) {
    case kTfLiteActNone:
      return NEURON_FUSED_NONE;
    case kTfLiteActRelu:
      return NEURON_FUSED_RELU;
    case kTfLiteActReluN1To1:
      return NEURON_FUSED_RELU1;
    case kTfLiteActRelu6:
      return NEURON_FUSED_RELU6;
    default:
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                      "Unsupported FusedActivation: %d", activation);
      return NEURON_FUSED_NONE;
  }
}
}  // namespace

NNMemory::NNMemory(const NeuronApi* neuronapi, const char* name, size_t size,
                   bool use_ahwb, bool use_cacheable_buffer) {
  neuronapi_ = neuronapi;
  byte_size_ = size;
  use_ahwb_ = use_ahwb;
  if (use_ahwb && name && size > 0) {
#if defined(__ANDROID__)
    uint64_t usage;
    if (use_cacheable_buffer) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                      "use cacheable ahardwarebuffer");
      usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
              AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;
    } else {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                      "use non cacheable ahardwarebuffer");
      usage = AHARDWAREBUFFER_USAGE_CPU_READ_RARELY |
              AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY;
    }
    AHardwareBuffer_Desc desc{
        .width = static_cast<uint32_t>(size),
        .height = 1,
        .layers = 1,
        .format = AHARDWAREBUFFER_FORMAT_BLOB,
        .usage = usage,
        .stride = static_cast<uint32_t>(size),
    };
    if (AHardwareBuffer_allocate(&desc, &buffer_) == 0) {
      neuronapi_->NeuronMemory_createFromAHardwareBuffer(buffer_,
                                                         &nn_memory_handle_);
      ion_memory_lock();
    }
#else
    int ret = false;

    struct dma_heap_allocation_data heap_data = {
        .len = size,
        .fd_flags = O_RDWR | O_CLOEXEC,
    };

    dma_heap_fd_ = open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
    if (dma_heap_fd_ < 0) {
      printf("open dma-heap fail\n");
      return;
    }

    ret = ioctl(dma_heap_fd_, DMA_HEAP_IOCTL_ALLOC, &heap_data);
    if (ret < 0) {
      printf("alloc dmabuf from dmabufheap fail\n");
      return;
    }

    fd_ = heap_data.fd;
    if (fd_ >= 0) {
      /* map user va */
      data_ptr_ = reinterpret_cast<uint8_t*>(
          mmap(NULL, byte_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
      if (data_ptr_ != MAP_FAILED) {
        neuronapi_->NeuronMemory_createFromFd(size, PROT_READ | PROT_WRITE, fd_,
                                              0, &nn_memory_handle_);
      }
    }
#endif
  } else if (name && size > 0) {
    byte_size_ = size;
    data_ptr_ = reinterpret_cast<uint8_t*>(malloc(size));
  }
}

NNMemory::~NNMemory() {
  ion_memory_unlock();
  if (!use_ahwb_ && data_ptr_) {
    free(data_ptr_);
  }
  if (nn_memory_handle_) {
    neuronapi_->NeuronMemory_free(nn_memory_handle_);
  }
#if defined(__ANDROID__)
  if (buffer_) {
    AHardwareBuffer_release(buffer_);
  }
#else
  if (use_ahwb_) {
    if (data_ptr_) munmap(data_ptr_, byte_size_);
    if (fd_ >= 0) close(fd_);
    if (dma_heap_fd_ >= 0) close(dma_heap_fd_);
  }
#endif
}

TfLiteStatus NeuronStableDelegateKernel::Map(
    TfLiteOpaqueContext* context, int builtin_code, int version,
    int android_sdk_version, const NeuronOpMappingArgs& mapping_args,
    NeuronOperationType* nn_op_type) {
  auto add_zero_bias = [mapping_args](int input_id, int filter_id,
                                      int num_elements) -> void {
    // Neuron requires a bias tensor, so we allocate a new tensor to fill
    // it with zeroes. It is deleted with other tensors in the context
    // during subgraph destructor call.
    TfLiteOpaqueTensorBuilder* tensor_builder =
        TfLiteOpaqueTensorBuilderCreate();
    const auto input_type = TfLiteOpaqueTensorType(
        TfLiteOpaqueContextGetOpaqueTensor(mapping_args.context, input_id));
    void* tensor_data = nullptr;
    if (input_type == kTfLiteFloat32) {
      TfLiteOpaqueTensorBuilderSetType(tensor_builder, kTfLiteFloat32);
      tensor_data = malloc(num_elements * sizeof(float));
    } else {
      TfLiteOpaqueTensorBuilderSetType(tensor_builder, kTfLiteInt32);
      tensor_data = malloc(num_elements * sizeof(int));
      const TfLiteOpaqueTensor* input_tensor =
          TfLiteOpaqueContextGetOpaqueTensor(mapping_args.context, input_id);
      const TfLiteOpaqueTensor* filter_tensor =
          TfLiteOpaqueContextGetOpaqueTensor(mapping_args.context, filter_id);
      // Neuron requires bias scale to be a product of an input scale and
      // a filter scale.
      TfLiteQuantizationParams quant_params{};
      quant_params.scale =
          TfLiteOpaqueTensorGetQuantizationParams(input_tensor).scale *
          TfLiteOpaqueTensorGetQuantizationParams(filter_tensor).scale;
      TfLiteOpaqueTensorBuilderSetQuantizationParams(tensor_builder,
                                                     quant_params);
    }
    TfLiteOpaqueTensorBuilderSetAllocationType(tensor_builder, kTfLiteDynamic);

    // The dynamic tensor need to set the tensor data. Set an empty pointer and
    // set the data value later after resizeing tensor dimension.
    TfLiteOpaqueTensorBuilderSetData(tensor_builder, tensor_data);

    int bias_index = -1;
    TfLiteOpaqueContextAddTensor(mapping_args.context, tensor_builder,
                                 &bias_index);
    TfLiteOpaqueTensorBuilderDelete(tensor_builder);
    TfLiteOpaqueTensor* bias_tensor =
        TfLiteOpaqueContextGetOpaqueTensor(mapping_args.context, bias_index);

    // Create an array with a required bias shape and resize the bias
    // tensor.
    TfLiteIntArray* bias_shape = TfLiteIntArrayCreate(1);
    bias_shape->data[0] = num_elements;
    TfLiteOpaqueContextResizeTensor(mapping_args.context, bias_tensor,
                                    bias_shape);
    // Set tensor's values to zeroes and add it using AddVector*, so
    // that the values are copied to Neuron. We don't use the AddTensor
    // function because it doesn't copy values and the tensor we just
    // created is not in the node->inputs.
    if (input_type == kTfLiteFloat32) {
      memset(TfLiteOpaqueTensorData(bias_tensor), 0,
             num_elements * sizeof(float));
      mapping_args.builder->AddVectorFloat32Operand(
          reinterpret_cast<const float*>(TfLiteOpaqueTensorData(bias_tensor)),
          num_elements);
    } else {
      memset(TfLiteOpaqueTensorData(bias_tensor), 0,
             num_elements * sizeof(int));
      mapping_args.builder->AddVectorInt32Operand(
          reinterpret_cast<const int32_t*>(TfLiteOpaqueTensorData(bias_tensor)),
          num_elements,
          TfLiteOpaqueTensorGetQuantizationParams(bias_tensor).scale,
          /*zero_point=*/0);
    }
  };
  switch (builtin_code) {
    case kTfLiteBuiltinAdd: {
      auto builtin = reinterpret_cast<TfLiteAddParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      *nn_op_type = NEURON_ADD;
    } break;
    case kTfLiteBuiltinArgMax: {
      *nn_op_type = NEURON_ARGMAX;
    } break;
    case kTfLiteBuiltinArgMin: {
      *nn_op_type = NEURON_ARGMIN;
    } break;
    case kTfLiteBuiltinMul: {
      auto builtin = reinterpret_cast<TfLiteMulParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      *nn_op_type = NEURON_MUL;
    } break;
    case kTfLiteBuiltinAveragePool2d: {
      mapping_args.builder->AddPoolingParams(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      *nn_op_type = NEURON_AVERAGE_POOL_2D;
    } break;
    case kTfLiteBuiltinMaxPool2d: {
      mapping_args.builder->AddPoolingParams(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      *nn_op_type = NEURON_MAX_POOL_2D;
    } break;
    case kTfLiteBuiltinL2Pool2d: {
      mapping_args.builder->AddPoolingParams(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      *nn_op_type = NEURON_L2_POOL_2D;
    } break;
    case kTfLiteBuiltinConv2d: {
      auto builtin = reinterpret_cast<TfLiteConvParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarInt32Operand(builtin->padding);
      mapping_args.builder->AddScalarInt32Operand(builtin->stride_width);
      mapping_args.builder->AddScalarInt32Operand(builtin->stride_height);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      mapping_args.builder->AddScalarBoolOperand(false);  // Use NHWC format
      mapping_args.builder->AddScalarInt32Operand(
          builtin->dilation_width_factor);
      mapping_args.builder->AddScalarInt32Operand(
          builtin->dilation_height_factor);
      *nn_op_type = NEURON_CONV_2D;
    } break;
    case kTfLiteBuiltinDepthwiseConv2d: {
      auto builtin = reinterpret_cast<TfLiteDepthwiseConvParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarInt32Operand(builtin->padding);
      mapping_args.builder->AddScalarInt32Operand(builtin->stride_width);
      mapping_args.builder->AddScalarInt32Operand(builtin->stride_height);
      mapping_args.builder->AddScalarInt32Operand(builtin->depth_multiplier);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      mapping_args.builder->AddScalarBoolOperand(false);  // Use NHWC format
      if (builtin->dilation_width_factor != 1 ||
          builtin->dilation_height_factor != 1) {
        mapping_args.builder->AddScalarInt32Operand(
            builtin->dilation_width_factor);
        mapping_args.builder->AddScalarInt32Operand(
            builtin->dilation_height_factor);
      }
      *nn_op_type = NEURON_DEPTHWISE_CONV_2D;
    } break;
    case kTfLiteBuiltinFullyConnected: {
      const int* inputs;
      int num_inputs;
      TF_LITE_OPAQUE_ENSURE_STATUS(
          TfLiteOpaqueNodeInputs(mapping_args.node, &inputs, &num_inputs));
      const bool is_bias_present =
          TfLiteOpaqueNodeNumberOfInputs(mapping_args.node) == 3 &&
          inputs[2] != kTfLiteOptionalTensor;
      if (!is_bias_present) {
        const int input_tensor_id = inputs[/*kInputTensor*/ 0];
        const int filter_tensor_id = inputs[/*kWeightsTensor*/ 1];
        const int num_units =
            TfLiteOpaqueTensorDim(TfLiteOpaqueContextGetOpaqueTensor(
                                      mapping_args.context, filter_tensor_id),
                                  0);
        add_zero_bias(input_tensor_id, filter_tensor_id, num_units);
      }
      auto builtin = reinterpret_cast<TfLiteFullyConnectedParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      *nn_op_type = NEURON_FULLY_CONNECTED;
    } break;
    case kTfLiteBuiltinHardSwish: {
      *nn_op_type = NEURON_HARD_SWISH;
    } break;
    case kTfLiteBuiltinSoftmax: {
      auto builtin = reinterpret_cast<TfLiteSoftmaxParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarFloat32Operand(builtin->beta);
      // Optional scalar specifying the dimension the activation would be
      // performed on is not added. Default to -1.
      *nn_op_type = NEURON_SOFTMAX;
    } break;
    case kTfLiteBuiltinReshape: {
      if (TfLiteOpaqueNodeNumberOfInputs(mapping_args.node) == 1) {
        // if no new_shape tensor, construct the new shape from params.
        auto* params = reinterpret_cast<TfLiteReshapeParams*>(
            TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
        int num_dimensions = params->num_dimensions;
        std::vector<int32_t> output_shape(num_dimensions);
        for (int i = 0; i < num_dimensions; ++i) {
          output_shape[i] = params->shape[i];
        }
        mapping_args.builder->AddVectorInt32Operand(
            output_shape.data(), static_cast<uint32_t>(num_dimensions));
      }
      *nn_op_type = NEURON_RESHAPE;
    } break;
    case kTfLiteBuiltinResizeBilinear: {
      auto* output =
          TfLiteOpaqueNodeGetOutput(mapping_args.context, mapping_args.node, 0);
      const int output_height = TfLiteOpaqueTensorDim(output, 1);
      const int output_width = TfLiteOpaqueTensorDim(output, 2);
      mapping_args.builder->AddScalarInt32Operand(output_width);
      mapping_args.builder->AddScalarInt32Operand(output_height);
      mapping_args.builder->AddScalarBoolOperand(false);  // Use NHWC format
      auto builtin = reinterpret_cast<TfLiteResizeBilinearParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarBoolOperand(builtin->align_corners);
      mapping_args.builder->AddScalarBoolOperand(builtin->half_pixel_centers);
      *nn_op_type = NEURON_RESIZE_BILINEAR;
    } break;
    case kTfLiteBuiltinResizeNearestNeighbor: {
      const TfLiteOpaqueTensor* new_shape =
          TfLiteOpaqueNodeGetInput(mapping_args.context, mapping_args.node, 1);
      // Neuron uses scalar inputs for height and width.
      mapping_args.builder->AddScalarInt32Operand(
          reinterpret_cast<int32_t*>(TfLiteOpaqueTensorData(new_shape))[1]);
      mapping_args.builder->AddScalarInt32Operand(
          reinterpret_cast<int32_t*>(TfLiteOpaqueTensorData(new_shape))[0]);
      mapping_args.builder->AddScalarBoolOperand(false);  // Use NHWC format
      auto builtin = reinterpret_cast<TfLiteResizeNearestNeighborParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      if (builtin->align_corners == true ||
          builtin->half_pixel_centers == true) {
        mapping_args.builder->AddScalarBoolOperand(builtin->align_corners);
        mapping_args.builder->AddScalarBoolOperand(builtin->half_pixel_centers);
      }
      *nn_op_type = NEURON_RESIZE_NEAREST_NEIGHBOR;
    } break;
    case kTfLiteBuiltinSqueeze: {
      auto builtin = reinterpret_cast<TfLiteSqueezeParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      // Note that we add the squeeze dimensions even if the dimensions
      // were unspecified (empty), as Neuron requires the operand.
      mapping_args.builder->AddVectorInt32Operand(
          builtin->num_squeeze_dims ? builtin->squeeze_dims : nullptr,
          static_cast<uint32_t>(builtin->num_squeeze_dims));
      *nn_op_type = NEURON_SQUEEZE;
    } break;
    case kTfLiteBuiltinL2Normalization: {
      *nn_op_type = NEURON_L2_NORMALIZATION;
    } break;
    case kTfLiteBuiltinLocalResponseNormalization: {
      auto builtin = reinterpret_cast<TfLiteLocalResponseNormParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarInt32Operand(builtin->radius);
      mapping_args.builder->AddScalarFloat32Operand(builtin->bias);
      mapping_args.builder->AddScalarFloat32Operand(builtin->alpha);
      mapping_args.builder->AddScalarFloat32Operand(builtin->beta);
      *nn_op_type = NEURON_LOCAL_RESPONSE_NORMALIZATION;
    } break;
    case kTfLiteBuiltinConcatenation: {
      auto builtin = reinterpret_cast<TfLiteConcatenationParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      int axis = builtin->axis < 0
                     ? TfLiteOpaqueTensorNumDims(TfLiteOpaqueNodeGetInput(
                           mapping_args.context, mapping_args.node, 0)) +
                           builtin->axis
                     : builtin->axis;
      mapping_args.builder->AddScalarInt32Operand(axis);
      *nn_op_type = NEURON_CONCATENATION;
    } break;
    case kTfLiteBuiltinDequantize: {
      *nn_op_type = NEURON_DEQUANTIZE;
    } break;
    case kTfLiteBuiltinFloor: {
      *nn_op_type = NEURON_FLOOR;
    } break;
    case kTfLiteBuiltinRelu: {
      *nn_op_type = NEURON_RELU;
    } break;
    case kTfLiteBuiltinReluN1To1: {
      *nn_op_type = NEURON_RELU1;
    } break;
    case kTfLiteBuiltinRelu6: {
      *nn_op_type = NEURON_RELU6;
    } break;
    case kTfLiteBuiltinLogistic: {
      *nn_op_type = NEURON_LOGISTIC;
    } break;
    case kTfLiteBuiltinTanh: {
      *nn_op_type = NEURON_TANH;
    } break;
    case kTfLiteBuiltinSub: {
      auto builtin = reinterpret_cast<TfLiteSubParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      *nn_op_type = NEURON_SUB;
    } break;
    case kTfLiteBuiltinDiv: {
      auto builtin = reinterpret_cast<TfLiteDivParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      *nn_op_type = NEURON_DIV;
    } break;
    case kTfLiteBuiltinPad:
    case kTfLiteBuiltinPadv2: {
      // We want to map to PAD as much as possible since it is more widely
      // supported. We map to PadV2 only when there is the need to specify
      // the padding value
      if (TfLiteOpaqueNodeNumberOfInputs(mapping_args.node) == 2) {
        *nn_op_type = NEURON_PAD;
      } else {
        const int* inputs;
        int num_inputs;
        TF_LITE_OPAQUE_ENSURE_STATUS(
            TfLiteOpaqueNodeInputs(mapping_args.node, &inputs, &num_inputs));
        const int constant_value_id = inputs[2];
        if (constant_value_id == kTfLiteOptionalTensor) {
          *nn_op_type = NEURON_PAD;
        } else {
          *nn_op_type = NEURON_PAD_V2;
        }
      }
    } break;
    case kTfLiteBuiltinSpaceToBatchNd: {
      *nn_op_type = NEURON_SPACE_TO_BATCH_ND;
    } break;
    case kTfLiteBuiltinBatchToSpaceNd: {
      *nn_op_type = NEURON_BATCH_TO_SPACE_ND;
    } break;
    case kTfLiteBuiltinStridedSlice: {
      auto builtin = reinterpret_cast<TfLiteStridedSliceParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarInt32Operand(builtin->begin_mask);
      mapping_args.builder->AddScalarInt32Operand(builtin->end_mask);
      mapping_args.builder->AddScalarInt32Operand(builtin->shrink_axis_mask);
      *nn_op_type = NEURON_STRIDED_SLICE;
    } break;
    case kTfLiteBuiltinTranspose: {
      *nn_op_type = NEURON_TRANSPOSE;
    } break;
    case kTfLiteBuiltinAbs: {
      *nn_op_type = NEURON_ABS;
    } break;
    case kTfLiteBuiltinExp: {
      *nn_op_type = NEURON_EXP;
    } break;
    case kTfLiteBuiltinLog: {
      *nn_op_type = NEURON_LOG;
    } break;
    case kTfLiteBuiltinPow: {
      *nn_op_type = NEURON_POW;
    } break;
    case kTfLiteBuiltinSlice: {
      *nn_op_type = NEURON_SLICE;
    } break;
    case kTfLiteBuiltinTransposeConv: {
      const int* inputs;
      int num_inputs;
      TF_LITE_OPAQUE_ENSURE_STATUS(
          TfLiteOpaqueNodeInputs(mapping_args.node, &inputs, &num_inputs));
      int input_tensor_flags = 0;
      const int input_tensor_id = inputs[/*kDataInputTensor*/ 2];
      const int weight_tensor_id = inputs[/*kWeightsTensor*/ 1];

      // Transpose convolution doesn't have hybrid variation.
      const bool hybrid_op = false;

      mapping_args.builder->AddTensorInput(
          input_tensor_id, hybrid_op,
          input_tensor_flags | NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED);

      // Transpose convlution uses per-channel quantization with int8 inputs
      // even if the number of channels in quantization parameters is equal to 1
      // (as opposed to conv2d, which uses per-tensor quantization in this
      // case).
      // TODO(MTK): Remove force per-channel flag here to avoid 1D quantize
      // param using per-channel quant. Check the reason for using force
      // per-channel quant here.
      mapping_args.builder->AddTensorInput(weight_tensor_id, hybrid_op,
                                           input_tensor_flags);

      const bool is_bias_present =
          TfLiteOpaqueNodeNumberOfInputs(mapping_args.node) == 4 &&
          inputs[/*kBiasTensor*/ 3] != kTfLiteOptionalTensor;

      if (is_bias_present) {
        mapping_args.builder->AddTensorInput(inputs[/*kBiasTensor*/ 3],
                                             hybrid_op);
      } else {
        const TfLiteOpaqueTensor* output_shape = TfLiteOpaqueNodeGetInput(
            mapping_args.context, mapping_args.node, 0);
        const int output_depth =
            reinterpret_cast<int32_t*>(TfLiteOpaqueTensorData(output_shape))[3];
        add_zero_bias(input_tensor_id, weight_tensor_id, output_depth);
      }
      mapping_args.builder->AddTensorInput(inputs[/*kOutputShapeTensor*/ 0],
                                           hybrid_op);

      auto builtin = reinterpret_cast<TfLiteTransposeConvParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarInt32Operand(builtin->padding);
      mapping_args.builder->AddScalarInt32Operand(builtin->stride_width);
      mapping_args.builder->AddScalarInt32Operand(builtin->stride_height);
      mapping_args.builder->AddScalarInt32Operand(
          GetEquivalentToNeuronFuseCode(builtin->activation));
      // Use NHWC layout for input and output.
      mapping_args.builder->AddScalarBoolOperand(false);
      *nn_op_type = NEURON_TRANSPOSE_CONV_2D;
    } break;
    case kTfLiteBuiltinSqrt: {
      *nn_op_type = NEURON_SQRT;
    } break;
    case kTfLiteBuiltinSpaceToDepth: {
      auto builtin = reinterpret_cast<TfLiteSpaceToDepthParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarInt32Operand(builtin->block_size);
      *nn_op_type = NEURON_SPACE_TO_DEPTH;
    } break;
    case kTfLiteBuiltinMean: {
      auto builtin = reinterpret_cast<TfLiteReducerParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      int32_t keep_dims = 0;
      if (builtin->keep_dims) keep_dims = 1;
      mapping_args.builder->AddScalarInt32Operand(keep_dims);
      *nn_op_type = NEURON_MEAN;
    } break;
    case kTfLiteBuiltinEmbeddingLookup: {
      *nn_op_type = NEURON_EMBEDDING_LOOKUP;
    } break;
    case kTfLiteBuiltinHashtableLookup: {
      *nn_op_type = NEURON_HASHTABLE_LOOKUP;
    } break;
    case kTfLiteBuiltinMaximum: {
      *nn_op_type = NEURON_MAXIMUM;
    } break;
    case kTfLiteBuiltinMinimum: {
      *nn_op_type = NEURON_MINIMUM;
    } break;
    case kTfLiteBuiltinCast: {
      *nn_op_type = NEURON_CAST;
    } break;
    case kTfLiteBuiltinLeakyRelu: {
      const auto input_type = TfLiteOpaqueTensorType(
          TfLiteOpaqueNodeGetInput(mapping_args.context, mapping_args.node, 0));
      auto builtin = reinterpret_cast<TfLiteLeakyReluParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));

      TfLiteTensor alpha_tensor;
      alpha_tensor.type = input_type;
      alpha_tensor.allocation_type = kTfLiteDynamic;
      alpha_tensor.dims = TfLiteIntArrayCreate(1);
      alpha_tensor.dims->data[0] = 1;
      alpha_tensor.params.zero_point = 0;

      int new_tensor_index = -1;
      if (input_type == kTfLiteFloat32) {
        alpha_tensor.params.scale = 0;
        std::vector<float> alpha_value = {builtin->alpha};
        mapping_args.builder->AddNewInputConstantTensor(
            NEURON_TENSOR_FLOAT32, kTfLiteFloat32, alpha_tensor.dims,
            alpha_value, alpha_tensor.params, &new_tensor_index);
      } else if (input_type == kTfLiteInt8) {
        alpha_tensor.params.scale = builtin->alpha;
        std::vector<int8_t> alpha_value = {1};
        mapping_args.builder->AddNewInputConstantTensor(
            NEURON_TENSOR_QUANT8_ASYMM_SIGNED, kTfLiteInt8, alpha_tensor.dims,
            alpha_value, alpha_tensor.params, &new_tensor_index);
      } else {
        alpha_tensor.params.scale = builtin->alpha;
        std::vector<uint8_t> alpha_value = {1};
        mapping_args.builder->AddNewInputConstantTensor(
            NEURON_TENSOR_QUANT8_ASYMM, kTfLiteUInt8, alpha_tensor.dims,
            alpha_value, alpha_tensor.params, &new_tensor_index);
      }

      TfLiteIntArrayFree(alpha_tensor.dims);
      *nn_op_type = NEURON_PRELU;
    } break;
    case kTfLiteBuiltinPrelu: {
      *nn_op_type = NEURON_PRELU;
    } break;
    case kTfLiteBuiltinLogicalOr: {
      *nn_op_type = NEURON_LOGICAL_OR;
    } break;
    case kTfLiteBuiltinLogicalAnd: {
      *nn_op_type = NEURON_LOGICAL_AND;
    } break;
    case kTfLiteBuiltinLogicalNot: {
      *nn_op_type = NEURON_LOGICAL_NOT;
    } break;
    case kTfLiteBuiltinLess: {
      *nn_op_type = NEURON_LESS;
    } break;
    case kTfLiteBuiltinLessEqual: {
      *nn_op_type = NEURON_LESS_EQUAL;
    } break;
    case kTfLiteBuiltinGreater: {
      *nn_op_type = NEURON_GREATER;
    } break;
    case kTfLiteBuiltinGreaterEqual: {
      *nn_op_type = NEURON_GREATER_EQUAL;
    } break;
    case kTfLiteBuiltinEqual: {
      *nn_op_type = NEURON_EQUAL;
    } break;
    case kTfLiteBuiltinNotEqual: {
      *nn_op_type = NEURON_NOT_EQUAL;
    } break;
    case kTfLiteBuiltinNeg: {
      *nn_op_type = NEURON_NEG;
    } break;
    case kTfLiteBuiltinTopkV2: {
      const TfLiteOpaqueTensor* k_param =
          TfLiteOpaqueNodeGetInput(mapping_args.context, mapping_args.node, 1);
      mapping_args.builder->AddScalarInt32Operand(
          *reinterpret_cast<int32_t*>(TfLiteOpaqueTensorData(k_param)));
      *nn_op_type = NEURON_TOPK_V2;
    } break;
    case kTfLiteBuiltinSelect: {
      *nn_op_type = NEURON_SELECT;
    } break;
    case kTfLiteBuiltinGather: {
      auto builtin = reinterpret_cast<TfLiteGatherParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      const int* inputs;
      int num_inputs;
      TF_LITE_OPAQUE_ENSURE_STATUS(
          TfLiteOpaqueNodeInputs(mapping_args.node, &inputs, &num_inputs));
      mapping_args.builder->AddTensorInput(inputs[0],
                                           /* hybrid_op */ false,
                                           /* scalar_as_tensor */ false);

      mapping_args.builder->AddScalarInt32Operand(builtin->axis);

      mapping_args.builder->AddTensorInput(inputs[1],
                                           /* hybrid_op */ false,
                                           /* scalar_as_tensor */ false);

      *nn_op_type = NEURON_GATHER;
    } break;
    case kTfLiteBuiltinSplit: {
      const TfLiteOpaqueTensor* axis =
          TfLiteOpaqueNodeGetInput(mapping_args.context, mapping_args.node, 0);
      auto builtin = reinterpret_cast<TfLiteSplitParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarInt32Operand(
          *reinterpret_cast<int32_t*>(TfLiteOpaqueTensorData(axis)));
      mapping_args.builder->AddScalarInt32Operand(builtin->num_splits);
      *nn_op_type = NEURON_SPLIT;
    } break;

    case kTfLiteBuiltinQuantize: {
      /*
            // TODO(Code): Add AddDequantize here to avoid Neuron adapter error
            // temporally Check if we can remove the AddDequantize latter, since
            // Neuron adapter should support requantize directly
      */
      const int* inputs;
      int num_inputs;
      TF_LITE_OPAQUE_ENSURE_STATUS(
          TfLiteOpaqueNodeInputs(mapping_args.node, &inputs, &num_inputs));
      auto input_type = TfLiteOpaqueTensorType(
          TfLiteOpaqueNodeGetInput(mapping_args.context, mapping_args.node, 0));
      auto output_type = TfLiteOpaqueTensorType(TfLiteOpaqueNodeGetOutput(
          mapping_args.context, mapping_args.node, 0));
      if (IsQuantized(input_type) && IsQuantized(output_type) &&
          input_type != output_type) {
        // Use requantize instead of quantize
        const char* custom_name = "requantizemtk";
        size_t oem_scalar_size = 0;
        uint8_t* oem_scalar = nullptr;
        oem_scalar_size =
            ::tflite::mtk::PackOemScalarString(custom_name, &oem_scalar);
        if (oem_scalar != nullptr) {
          mapping_args.builder->AddScalarExtensionOperand(oem_scalar,
                                                          oem_scalar_size);
          free(oem_scalar);
        }
        mapping_args.builder->GetExtensionOperationType(nn_op_type);
      } else {
        *nn_op_type = NEURON_QUANTIZE;
      }
    } break;
    case kTfLiteBuiltinReduceAny: {
      auto builtin = reinterpret_cast<TfLiteReducerParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarBoolOperand(builtin->keep_dims);
      *nn_op_type = NEURON_REDUCE_ANY;
    } break;
    case kTfLiteBuiltinReduceMin: {
      auto builtin = reinterpret_cast<TfLiteReducerParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarBoolOperand(builtin->keep_dims);
      *nn_op_type = NEURON_REDUCE_MIN;
    } break;
    case kTfLiteBuiltinReduceMax: {
      auto builtin = reinterpret_cast<TfLiteReducerParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarBoolOperand(builtin->keep_dims);
      *nn_op_type = NEURON_REDUCE_MAX;
    } break;
    case kTfLiteBuiltinDepthToSpace: {
      auto builtin = reinterpret_cast<TfLiteDepthToSpaceParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarInt32Operand(builtin->block_size);
      *nn_op_type = NEURON_DEPTH_TO_SPACE;
    } break;
    case kTfLiteBuiltinSum: {
      auto builtin = reinterpret_cast<TfLiteReducerParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarBoolOperand(builtin->keep_dims);
      *nn_op_type = NEURON_REDUCE_SUM;
    } break;
    case kTfLiteBuiltinSquaredDifference: {
      const char* custom_name = "MTKEXT_SQUARED_DIFFERENCE";
      size_t oem_scalar_size = 0;
      uint8_t* oem_scalar = nullptr;
      oem_scalar_size =
          ::tflite::mtk::PackOemScalarString(custom_name, &oem_scalar);
      if (oem_scalar != nullptr) {
        mapping_args.builder->AddScalarExtensionOperand(oem_scalar,
                                                        oem_scalar_size);
        free(oem_scalar);
      }
      mapping_args.builder->GetExtensionOperationType(nn_op_type);
    } break;
    case kTfLiteBuiltinRsqrt: {
      const char* custom_name = "MTKEXT_RSQRT";
      size_t oem_scalar_size = 0;
      uint8_t* oem_scalar = nullptr;
      oem_scalar_size =
          ::tflite::mtk::PackOemScalarString(custom_name, &oem_scalar);
      if (oem_scalar != nullptr) {
        mapping_args.builder->AddScalarExtensionOperand(oem_scalar,
                                                        oem_scalar_size);
        free(oem_scalar);
      }
      mapping_args.builder->GetExtensionOperationType(nn_op_type);
    } break;
    case kTfLiteBuiltinUnpack: {
      const char* custom_name = "unpackmtk";
      size_t oem_scalar_size = 0;
      uint8_t* oem_scalar = nullptr;
      auto builtin = reinterpret_cast<TfLiteUnpackParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarInt32Operand(builtin->axis);
      oem_scalar_size =
          ::tflite::mtk::PackOemScalarString(custom_name, &oem_scalar);
      if (oem_scalar != nullptr) {
        mapping_args.builder->AddScalarExtensionOperand(oem_scalar,
                                                        oem_scalar_size);
        free(oem_scalar);
      }
      mapping_args.builder->GetExtensionOperationType(nn_op_type);
    } break;
    case kTfLiteBuiltinReverseV2: {
      const char* custom_name = "reversemtk";
      size_t oem_scalar_size = 0;
      uint8_t* oem_scalar = nullptr;
      oem_scalar_size =
          ::tflite::mtk::PackOemScalarString(custom_name, &oem_scalar);
      if (oem_scalar != nullptr) {
        mapping_args.builder->AddScalarExtensionOperand(oem_scalar,
                                                        oem_scalar_size);
        free(oem_scalar);
      }
      mapping_args.builder->GetExtensionOperationType(nn_op_type);
    } break;
    case kTfLiteBuiltinMirrorPad: {
      const char* custom_name = "mirrorpadmtk";
      size_t oem_scalar_size = 0;
      uint8_t* oem_scalar = nullptr;
      auto builtin = reinterpret_cast<TfLiteMirrorPaddingParams*>(
          TfLiteOpaqueNodeGetBuiltinData(mapping_args.node));
      mapping_args.builder->AddScalarInt32Operand(builtin->mode);
      oem_scalar_size =
          ::tflite::mtk::PackOemScalarString(custom_name, &oem_scalar);
      if (oem_scalar != nullptr) {
        mapping_args.builder->AddScalarExtensionOperand(oem_scalar,
                                                        oem_scalar_size);
        free(oem_scalar);
      }
      mapping_args.builder->GetExtensionOperationType(nn_op_type);
    } break;
    case kTfLiteBuiltinPack: {
      *nn_op_type = NEURON_PACK;
    } break;
    case kTfLiteBuiltinCustom: {
      if (strcmp(TfLiteRegistrationExternalGetCustomName(
                     mapping_args.registration),
                 "TFLite_Detection_PostProcess") == 0) {
        const char* custom_name = "detectionpostprocessingmtk";
        size_t oem_scalar_size = 0;
        uint8_t* oem_scalar = nullptr;
        const int* inputs;
        int num_inputs;
        TF_LITE_OPAQUE_ENSURE_STATUS(
            TfLiteOpaqueNodeInputs(mapping_args.node, &inputs, &num_inputs));
        const int* outputs;
        int num_outputs;
        TF_LITE_OPAQUE_ENSURE_STATUS(
            TfLiteOpaqueNodeOutputs(mapping_args.node, &outputs, &num_outputs));
        mapping_args.builder->AddTensorInput(inputs[1],
                                             /* hybrid_op */ false,
                                             /* scalar_as_tensor */ false);
        mapping_args.builder->AddTensorInput(inputs[0],
                                             /* hybrid_op */ false,
                                             /* scalar_as_tensor */ false);
        mapping_args.builder->AddTensorInput(inputs[2],
                                             /* hybrid_op */ false,
                                             /* scalar_as_tensor */ false);
        auto builtin = reinterpret_cast<
            ::tflite::ops::mtkext::detection_postprocessing::OpData*>(
            TfLiteOpaqueNodeGetUserData(mapping_args.node));
        mapping_args.builder->AddScalarFloat32Operand(builtin->scale_values.y);
        mapping_args.builder->AddScalarFloat32Operand(builtin->scale_values.x);
        mapping_args.builder->AddScalarFloat32Operand(builtin->scale_values.h);
        mapping_args.builder->AddScalarFloat32Operand(builtin->scale_values.w);
        mapping_args.builder->AddScalarBoolOperand(
            builtin->use_regular_non_max_suppression);
        mapping_args.builder->AddScalarInt32Operand(builtin->max_detections);
        mapping_args.builder->AddScalarInt32Operand(
            builtin->max_classes_per_detection);
        mapping_args.builder->AddScalarInt32Operand(
            builtin->detections_per_class);
        mapping_args.builder->AddScalarFloat32Operand(
            builtin->non_max_suppression_score_threshold);
        mapping_args.builder->AddScalarFloat32Operand(
            builtin->intersection_over_union_threshold);
        mapping_args.builder->AddScalarBoolOperand(false);
        oem_scalar_size =
            ::tflite::mtk::PackOemScalarString(custom_name, &oem_scalar);
        if (oem_scalar != nullptr) {
          mapping_args.builder->AddScalarExtensionOperand(oem_scalar,
                                                          oem_scalar_size);
          free(oem_scalar);
        }
        mapping_args.builder->GetExtensionOperationType(nn_op_type);
        mapping_args.builder->AddTensorOutput(outputs[2]);
        mapping_args.builder->AddTensorOutput(outputs[0]);
        mapping_args.builder->AddTensorOutput(outputs[1]);
        mapping_args.builder->AddTensorOutput(outputs[3]);
        break;  // needed
      }
      if (strcmp(TfLiteRegistrationExternalGetCustomName(
                     mapping_args.registration),
                 "cros-mtk-pre-compile") == 0) {
        const void* init_data = nullptr;
        int size = 0;
        TF_LITE_ENSURE_STATUS(TfLiteOpaqueNodeGetCustomInitialData(
            mapping_args.node, &init_data, &size));
        mapping_args.builder->AddPreCompileExtensionOperand(
            static_cast<const uint8_t*>(init_data), size);
        mapping_args.builder->AddPreCompileExtensionOperand(nn_op_type);
        break;  // needed
      }
      [[fallthrough]];
    }  // no break, pass to default
    default:
      // All other operators are not mapped.
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "Unmapped OP: %d", builtin_code);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus NeuronStableDelegateKernel::Init(
    TfLiteOpaqueContext* context, const TfLiteOpaqueDelegateParams* params) {
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE, "Init");
  if (params->delegate == nullptr) return kTfLiteDelegateError;

  for (auto node_index : TfLiteIntArrayView(params->nodes_to_replace)) {
    nodes_.push_back(node_index);
  }

  if (!nn_model_) {
    NeuronModel* model = nullptr;
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(context,
                                        neuronapi_->NeuronModel_create(&model),
                                        "creating Neuron model");
    nn_model_.reset(model);
    TF_LITE_ENSURE_STATUS(BuildGraph(context, params, params->input_tensors,
                                     params->output_tensors));
  }

  need_set_memory_ = true;
  initialised_ = true;
  return kTfLiteOk;
}

TfLiteStatus NeuronStableDelegateKernel::Prepare(TfLiteOpaqueContext* context,
                                                 TfLiteOpaqueNode* node) {
  if (!initialised_) {
    return kTfLiteError;
  }

  if (nn_compilation_) {
    return kTfLiteOk;
  }

  NeuronCompilation* compilation = nullptr;
  uint32_t num_devices = 0;

  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
      context, neuronapi_->Neuron_getDeviceCount(&num_devices),
      "getting device count");

  std::vector<NeuronDevice*> device(num_devices);

  int device_count = 0;
  if (strlen(options_.accelerator_name.c_str()) != 0) {
    char* device_split_name =
        std::strtok(const_cast<char*>(options_.accelerator_name.c_str()), ",");
    while (device_split_name) {
      NeuronDevice* mtk_device = nullptr;
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                      "NeuronCompilation_createForDevices: %s",
                      device_split_name);
      if (GetAcceleratorDevice(neuronapi_, context, device_split_name,
                               &mtk_device) == kTfLiteOk) {
        device[device_count] = mtk_device;
        device_count++;
      }
      device_split_name = std::strtok(NULL, ",");
    }
  }
  if (device_count) {
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context,
        neuronapi_->NeuronCompilation_createForDevices(
            nn_model_.get(), device.data(), device_count, &compilation),
        "creating Neuron compilation user specified device");
  } else {
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context,
        neuronapi_->NeuronCompilation_create(nn_model_.get(), &compilation),
        "creating Neuron compilation");
  }

  int result = neuronapi_->NeuronCompilation_setPreference(
      compilation, options_.execution_preference);
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                  "NeuronCompilation_setPreference: %d",
                  options_.execution_preference);
  if (result != NEURON_NO_ERROR) {
    neuronapi_->NeuronCompilation_free(compilation);
    compilation = nullptr;
  }
  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(context, result,
                                      "setting compilation preferences");
  result = neuronapi_->NeuronCompilation_setPriority(
      compilation, options_.execution_priority);
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                  "NeuronCompilation_setPriority: %d",
                  options_.execution_priority);
  if (result != NEURON_NO_ERROR) {
    neuronapi_->NeuronCompilation_free(compilation);
    compilation = nullptr;
  }
  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(context, result,
                                      "setting compilation priority");
  if (strlen(options_.cache_dir.c_str()) != 0 &&
      !nn_compilation_cache_token_.empty()) {
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context,
        neuronapi_->NeuronCompilation_setCaching(
            compilation, options_.cache_dir.c_str(),
            nn_compilation_cache_token_.data()),
        "set caching");
  }

  // Use optimization string first
  if (neuronapi_->NeuronCompilation_setOptimizationString != nullptr) {
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context,
        neuronapi_->NeuronCompilation_setOptimizationString(
            compilation,
            NNCompilationArgs::GetArgs(options_.execution_preference).c_str()),
        "set optimization string");
  }
  // sample for print model token
  // std::string test = "{";
  // for (auto i:nn_compilation_cache_token_) {
  //        test += std::to_string(i);
  //        test += ",";
  //}
  // test += "};";

  if (neuronapi_->NeuronCompilation_setOptimizationHint != nullptr) {
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context,
        neuronapi_->NeuronCompilation_setOptimizationHint(
            compilation, options_.optimization_hint),
        "set optimization hint");
  }
  if (strlen(options_.compile_options.c_str()) != 0) {
    TFLITE_MTK_LOG_INFO("NeuronCompilation_setCompileOptions: %s",
                        options_.compile_options.c_str());
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context,
        neuronapi_->NeuronCompilation_setOptimizationString(
            compilation, options_.compile_options.c_str()),
        "set compile options");
  }
  uint32_t apu_mem_size = 0;
  if (neuronapi_->Neuron_getL1MemorySizeKb(&apu_mem_size) == NEURON_NO_ERROR &&
      apu_mem_size > 0) {
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context,
        neuronapi_->NeuronCompilation_setL1MemorySizeKb(compilation,
                                                        apu_mem_size),
        "set apu l1 memory size");
  }

  // NeuronCompilation_getSupportedOperations must be called before
  // calling NeuronCompilation_finish
  if (delegate_specific_mode_ == kOneNodeMode ||
      delegate_specific_mode_ == kContextMode) {
    nn_compilation_.reset(compilation);
    return kTfLiteOk;
  }

  const int finish_result = neuronapi_->NeuronCompilation_finish(compilation);
  if (finish_result != NEURON_NO_ERROR) {
    neuronapi_->NeuronCompilation_free(compilation);
    compilation = nullptr;
  }

  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(context, finish_result,
                                      "completing Neuron compilation");
  nn_compilation_.reset(compilation);
#ifdef NEURON_REUSABLE_EXECUTION
  NeuronExecution* execution = nullptr;
  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
      context,
      neuronapi_->NeuronExecution_create(nn_compilation_.get(), &execution),
      "creating Neuron execution");
  nn_execution_.reset(execution);
#endif
  return kTfLiteOk;
}

TfLiteStatus NeuronStableDelegateKernel::Eval(TfLiteOpaqueContext* context,
                                              TfLiteOpaqueNode* node) {
  return this->Eval(context, node, nullptr);
}

TfLiteStatus NeuronStableDelegateKernel::Eval(TfLiteOpaqueContext* context,
                                              TfLiteOpaqueNode* node,
                                              TfLiteExecutionTask* task) {
  // MTK_ATRACE_CALL();
#ifdef NEURON_REUSABLE_EXECUTION
  NeuronExecution* execution = nn_execution_.get();
#else
  NeuronExecution* execution = nullptr;
  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
      context,
      neuronapi_->NeuronExecution_create(nn_compilation_.get(), &execution),
      "creating Neuron execution");
  std::unique_ptr<NeuronExecution, NNFreeExecution> execution_unique_ptr(
      execution, NNFreeExecution(neuronapi_));
#endif
  if (options_.use_ahwb && !options_.use_cacheable_buffer &&
      neuronapi_->NeuronExecution_setCacheFlushHint != nullptr) {
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context,
        neuronapi_->NeuronExecution_setCacheFlushHint(
            execution, NEURON_CACHE_FLUSH_DISABLE_SYNC_INPUT |
                           NEURON_CACHE_FLUSH_DISABLE_INVALIDATE_OUTPUT),
        "set cache flush hint");
  }
  // Set the input tensor buffers. Note: we access tflite tensors using
  // absolute indices but Neuron indices inputs by relative indices.
  int relative_input_index = 0;
  size_t input_offset = 0;
  const int* inputs;
  int num_inputs;
  TF_LITE_OPAQUE_ENSURE_STATUS(
      TfLiteOpaqueNodeInputs(node, &inputs, &num_inputs));
  for (auto i = 0; i < num_inputs; i++) {
    int absolute_input_index = inputs[i];
    if (absolute_input_index == kTfLiteOptionalTensor) {
      continue;
    }

    NeuronOperandType* input_nn_operand_type_ptr = nullptr;
    TfLiteOpaqueTensor* tensor =
        TfLiteOpaqueContextGetOpaqueTensor(context, absolute_input_index);
    TfLiteBufferHandle buffer_handle =
        TfLiteExecutionTaskGetBufferByIndex(task, absolute_input_index);
    if (TfLiteOpaqueTensorGetAllocationType(tensor) != kTfLiteMmapRo) {
      TfLiteType nn_type_equivalent =
          operand_mapping_.lite_index_to_neuron_type_conversion(
              absolute_input_index);
      if (buffer_handle != kTfLiteNullBufferHandle &&
          tensor_memory_map_kernel_.find(buffer_handle) !=
              tensor_memory_map_kernel_.end()) {
        RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
            context,
            neuronapi_->NeuronExecution_setInputFromMemory(
                execution, relative_input_index, input_nn_operand_type_ptr,
                tensor_memory_map_kernel_.at(buffer_handle), 0,
                TfLiteOpaqueTensorByteSize(tensor)),
            "associating Neuron execution input with a memory object");
        relative_input_index++;
        continue;
      }
      int tensor_size = 0;
      if (nn_type_equivalent != kTfLiteNoType) {
        const auto num_elements = NumElements(tensor);
        uint8_t* input_ptr = nn_input_memory_->get_data_ptr() + input_offset;
        if (TfLiteOpaqueTensorType(tensor) == kTfLiteUInt8 &&
            nn_type_equivalent == kTfLiteInt32) {
          uint8_t* uint8_tensor_data =
              static_cast<uint8_t*>(TfLiteOpaqueTensorData(tensor));
          for (int i = 0; i < num_elements; ++i) {
            reinterpret_cast<int32_t*>(input_ptr)[i] =
                static_cast<const int32_t>(uint8_tensor_data[i]);
          }
        } else if (TfLiteOpaqueTensorType(tensor) == kTfLiteInt8 &&
                   nn_type_equivalent == kTfLiteUInt8) {
          // Explicitly convert int8 values to uint8 values.
          for (int i = 0; i < num_elements; ++i) {
            input_ptr[i] = static_cast<const uint8_t>(
                static_cast<int32_t*>(TfLiteOpaqueTensorData(tensor))[i] + 128);
          }
        } else if (TfLiteOpaqueTensorType(tensor) == kTfLiteInt8 &&
                   nn_type_equivalent == kTfLiteInt32) {
          for (int i = 0; i < num_elements; ++i) {
            reinterpret_cast<int32_t*>(input_ptr)[i] =
                static_cast<const int32_t*>(TfLiteOpaqueTensorData(tensor))[i] +
                128;
          }
        } else {
          TfLiteOpaqueContextReportError(
              context,
              "Neuron Delegate: unsupported tensor types conversion: "
              "from type code %d to type code %d.\n",
              TfLiteOpaqueTensorType(tensor), nn_type_equivalent);
          return kTfLiteError;
        }
        size_t type_size;
        TF_LITE_ENSURE_OK(context,
                          TfLiteOpaqueContextGetSizeOfType(
                              context, nn_type_equivalent, &type_size));
        tensor_size = NumElements(tensor) * type_size;
        if (options_.use_ahwb) {
          if (need_set_memory_) {
            RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
                context,
                neuronapi_->NeuronExecution_setInputFromMemory(
                    execution, relative_input_index, input_nn_operand_type_ptr,
                    nn_input_memory_->get_handle(), input_offset, tensor_size),
                "Associating the execution input with a memory object");
          }
        } else {
          RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
              context,
              neuronapi_->NeuronExecution_setInput(
                  execution, relative_input_index, input_nn_operand_type_ptr,
                  input_ptr, tensor_size),
              "Associating the execution input with a memory object");
        }
      } else {
        // copy data to pre-allocated shared memory.
        memcpy(nn_input_memory_->get_data_ptr() + input_offset,
               TfLiteOpaqueTensorData(tensor),
               TfLiteOpaqueTensorByteSize(tensor));
        if (options_.use_ahwb) {
          if (need_set_memory_) {
            RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
                context,
                neuronapi_->NeuronExecution_setInputFromMemory(
                    execution, relative_input_index, input_nn_operand_type_ptr,
                    nn_input_memory_->get_handle(), input_offset,
                    TfLiteOpaqueTensorByteSize(tensor)),
                "Associating the execution input with a memory object");
          }
        } else {
          RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
              context,
              neuronapi_->NeuronExecution_setInput(
                  execution, relative_input_index, input_nn_operand_type_ptr,
                  nn_input_memory_->get_data_ptr() + input_offset,
                  TfLiteOpaqueTensorByteSize(tensor)),
              "Associating the execution input with a memory object");
        }
        tensor_size = TfLiteOpaqueTensorByteSize(tensor);
      }

      input_offset += tensor_size;
      input_offset += getNumPaddingBytes(tensor_size);
      relative_input_index++;
    }
  }

  // Set the output tensor buffers.
  int relative_output_index = 0;
  size_t output_offset = 0;
  const int* outputs;
  int num_outputs;
  TF_LITE_OPAQUE_ENSURE_STATUS(
      TfLiteOpaqueNodeOutputs(node, &outputs, &num_outputs));
  for (auto i = 0; i < num_outputs; i++) {
    int output_index = outputs[i];
    // If the Neuron implementation doesn't have some of the outputs
    // they are left unmapped and we should not try to read their value here
    if (operand_mapping_.lite_index_to_neuron(output_index) == -1) {
      continue;
    }

    NeuronOperandType* output_nn_operand_type_ptr = nullptr;
    TfLiteOpaqueTensor* tensor =
        TfLiteOpaqueContextGetOpaqueTensor(context, output_index);
    TfLiteBufferHandle buffer_handle =
        TfLiteExecutionTaskGetBufferByIndex(task, output_index);
    if (buffer_handle != kTfLiteNullBufferHandle &&
        tensor_memory_map_kernel_.find(buffer_handle) !=
            tensor_memory_map_kernel_.end()) {
      RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
          context,
          neuronapi_->NeuronExecution_setOutputFromMemory(
              execution, relative_output_index, output_nn_operand_type_ptr,
              tensor_memory_map_kernel_.at(buffer_handle), 0,
              TfLiteOpaqueTensorByteSize(tensor)),
          "associating NNAPI execution output to a memory object");
    } else {
      if (options_.use_ahwb) {
        if (need_set_memory_) {
          RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
              context,
              neuronapi_->NeuronExecution_setOutputFromMemory(
                  execution, relative_output_index, output_nn_operand_type_ptr,
                  nn_output_memory_->get_handle(), output_offset,
                  TfLiteOpaqueTensorByteSize(tensor)),
              "associating NNAPI execution output to a memory object");
        }
      } else {
        RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
            context,
            neuronapi_->NeuronExecution_setOutput(
                execution, relative_output_index, output_nn_operand_type_ptr,
                nn_output_memory_->get_data_ptr() + output_offset,
                TfLiteOpaqueTensorByteSize(tensor)),
            "associating NNAPI execution output to a memory object");
      }

      output_offset += TfLiteOpaqueTensorByteSize(tensor);
      output_offset += getNumPaddingBytes(TfLiteOpaqueTensorByteSize(tensor));
    }
    relative_output_index++;
  }

  // The state_out of previous invocation need to be mapped to state_in of
  // current invocation.
  for (size_t i = 0; i < model_state_tfl_inputs_.size(); i++) {
    int state_tensor_idx = model_state_tfl_inputs_[i];
    TfLiteOpaqueTensor* tensor =
        TfLiteOpaqueContextGetOpaqueTensor(context, state_tensor_idx);
    if (options_.use_ahwb) {
      if (need_set_memory_) {
        RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
            context,
            neuronapi_->NeuronExecution_setOutputFromMemory(
                execution, relative_output_index, nullptr,
                nn_output_memory_->get_handle(), output_offset,
                TfLiteOpaqueTensorByteSize(tensor)),
            "associating Neuron execution output to a buffer");
      }
    } else {
      RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
          context,
          neuronapi_->NeuronExecution_setOutput(
              execution, relative_output_index, nullptr,
              nn_output_memory_->get_data_ptr() + output_offset,
              TfLiteOpaqueTensorByteSize(tensor)),
          "associating Neuron execution output to a buffer");
    }
    output_offset += TfLiteOpaqueTensorByteSize(tensor);
    output_offset += getNumPaddingBytes(TfLiteOpaqueTensorByteSize(tensor));
    relative_output_index++;
  }

  [[maybe_unused]] uint32_t boost_duration_default = 1000;
#ifndef MTK_INTERNAL_USE
  if (options_.execution_preference == ExecutionPreference::kFastSingleAnswer ||
      options_.execution_preference == ExecutionPreference::kTurboBoost) {
    boost_duration_default = 2000;
  } else {
    boost_duration_default = 10;
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context, neuronapi_->NeuronExecution_setBoostHint(execution, 80),
        "set boost hint");
  }
#endif

  if (options_.inference_deadline > 0 &&
      neuronapi_->NeuronExecution_setDeadline != nullptr) {
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context,
        neuronapi_->NeuronExecution_setDeadline(execution,
                                                options_.inference_deadline),
        "set deadline");
  }

  if (options_.inference_abort_time > 0 &&
      neuronapi_->NeuronExecution_setAbortTime != nullptr) {
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context,
        neuronapi_->NeuronExecution_setAbortTime(execution,
                                                 options_.inference_abort_time),
        "set abort time");
  }

  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
      context, neuronapi_->NeuronExecution_compute(execution),
      "running computation");
  // copy results from shared memory to the destination.
  output_offset = 0;
  for (auto i = 0; i < num_outputs; i++) {
    int output_index = outputs[i];
    TfLiteOpaqueTensor* tensor =
        TfLiteOpaqueContextGetOpaqueTensor(context, output_index);
    TfLiteBufferHandle buffer_handle =
        TfLiteExecutionTaskGetBufferByIndex(task, output_index);
    if (buffer_handle != kTfLiteNullBufferHandle) {
      continue;
    }
    TfLiteType nn_type_equivalent =
        operand_mapping_.lite_index_to_neuron_type_conversion(output_index);
    if (TfLiteOpaqueTensorType(tensor) == kTfLiteInt8 &&
        nn_type_equivalent == kTfLiteUInt8) {
      // Explicitly convert uint8 values to int8 values.
      uint8_t* output_ptr = reinterpret_cast<uint8_t*>(
          nn_output_memory_->get_data_ptr() + output_offset);
      const auto num_elements = NumElements(tensor);
      for (int i = 0; i < num_elements; ++i) {
        output_ptr[i] =
            static_cast<uint8_t>(static_cast<int32_t>(output_ptr[i]) - 128);
      }
    }
    memcpy(TfLiteOpaqueTensorData(tensor),
           nn_output_memory_->get_data_ptr() + output_offset,
           TfLiteOpaqueTensorByteSize(tensor));
    output_offset += TfLiteOpaqueTensorByteSize(tensor);
    output_offset += getNumPaddingBytes(TfLiteOpaqueTensorByteSize(tensor));
  }
  need_set_memory_ = false;

  // copy output of all output tensors in feedback_loops_ into the
  // associated input
  for (auto feedback_loop : feedback_loops_) {
    int output_tensor_idx;
    int input_tensor_idx;
    std::tie(output_tensor_idx, input_tensor_idx) = feedback_loop;
    TfLiteOpaqueTensor* src =
        TfLiteOpaqueContextGetOpaqueTensor(context, outputs[output_tensor_idx]);
    TfLiteOpaqueTensor* dest =
        TfLiteOpaqueContextGetOpaqueTensor(context, inputs[input_tensor_idx]);

    memcpy(TfLiteOpaqueTensorData(dest), TfLiteOpaqueTensorData(src),
           TfLiteOpaqueTensorByteSize(src));
  }

  return kTfLiteOk;
}

TfLiteAsyncKernel* NeuronStableDelegateKernel::AsyncKernel(
    TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
  return async_kernel_->kernel();
}

void NeuronStableDelegateKernel::AddDequantizeOperatorsWhereNeeded(
    const TfLiteOpaqueContext* context, int builtin_code,
    const TfLiteOpaqueNode* node, NeuronOpBuilder* builder) {
  int input_tensor_index = -1;
  std::vector<int> inputs_to_potentially_dequantize;

  switch (builtin_code) {
    case kTfLiteBuiltinConv2d:
    case kTfLiteBuiltinFullyConnected: {
      input_tensor_index = 0;
      // Weights and bias are inputs #1 and #2 respectively and may require
      // dequantization.
      inputs_to_potentially_dequantize = {1, 2};
      break;
    }
    case kTfLiteBuiltinLstm: {
      input_tensor_index = 0;
      inputs_to_potentially_dequantize = {1,  2,  3,  4,  5,  6,  7,
                                          8,  9,  10, 11, 12, 13, 14,
                                          15, 16, 17, 20, 21, 22, 23};
      break;
    }
    default:
      return;
  }
  const int* inputs;
  int num_inputs;
  TfLiteOpaqueNodeInputs(node, &inputs, &num_inputs);
  int tensor_id = inputs[input_tensor_index];
  if (tensor_id < 0) return;

  // Nothing to do if the input is not floating-point.
  if (!IsFloat(TfLiteOpaqueTensorType(
          TfLiteOpaqueContextGetOpaqueTensor(context, tensor_id))))
    return;

  for (int i : inputs_to_potentially_dequantize) {
    if (i < 0 || i >= TfLiteOpaqueNodeNumberOfInputs(node))
      continue;  // Ignore invalid index.
    tensor_id = inputs[i];
    if (tensor_id < 0) continue;  // Ignore optional input.

    const TfLiteType type = TfLiteOpaqueTensorType(
        TfLiteOpaqueContextGetOpaqueTensor(context, tensor_id));
    // Nothing to do for this tensor if it's not quantized.
    if (!IsQuantized(type)) continue;

    // Insert Dequantize operator if it hasn't been done already and change
    // the node's input accordingly.
    builder->AddDequantize(i, inputs[i], type);
  }
}

TfLiteStatus NeuronStableDelegateKernel::AddOpsAndTensors(
    TfLiteOpaqueContext* context) {
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE, "AddOpsAndTensors");
  DequantizeMapping dequantize_mapping;
  // The operand builder allows creating a single op. It is created outside
  // the for loop to avoid reallocating the vectors.
  NeuronOpBuilder builder(neuronapi_, context, &operand_mapping_,
                          &dequantize_mapping, nn_model_.get());
  // Add Tensors.
  for (auto node_index : nodes_) {
    // Obtain the op and registration.
    TfLiteOpaqueNode* node;
    TfLiteRegistrationExternal* reg;
    if (delegate_specific_mode_ == kOneNodeMode) {
      node = one_node_and_reg_.first;
      reg = one_node_and_reg_.second;
    } else {
      TF_LITE_ENSURE_STATUS(TfLiteOpaqueContextGetNodeAndRegistration(
          context, node_index, &node, &reg));
    }

#ifdef LOWERING_PACK
    // Delegate PACK by lowering it into CONCAT + RESHAPE.
    if (TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinPack) {
      TF_LITE_ENSURE_STATUS(builder.TransformPackIntoSupportedOps(node, reg));
      continue;
    }
#endif
    // Delegate SPLIT_V by lowering it into SLICEs.
    if (TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinSplitV) {
      TF_LITE_ENSURE_STATUS(
          builder.TransformSplitVIntoSupportedOps(node_index, node, reg));
      continue;
    }
    // Remove MirrorPad set same quantization params of input & output tensors
    // patch due to the limitation of the MirrorPad spec.
    const bool hybrid_op =
        IsHybridOperator(context, TfLiteOperatorGetBuiltInCode(reg), node);
    const bool scalar_as_tensor =
        IsScalarInputSupported(TfLiteOperatorGetBuiltInCode(reg));
    const bool need_int8_conversion = false;
    const bool use_int8_asymm_signed = true;
    const int* node_inputs;
    int node_num_inputs;
    TF_LITE_OPAQUE_ENSURE_STATUS(
        TfLiteOpaqueNodeInputs(node, &node_inputs, &node_num_inputs));

    int input_tensor_flags = 0;
    if (scalar_as_tensor) {
      input_tensor_flags |= NN_TENSOR_FLAG_SCALAR_AS_TENSOR;
    }
    if (use_int8_asymm_signed) {
      input_tensor_flags |= NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED;
    }
#ifdef LOWERING_HARD_SWISH
    // h_swish will be lowered into supported Neuron operations.
    if (TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinHardSwish) {
      builder.AddHardSwish(node_inputs[0], node->outputs->data[0],
                           need_int8_conversion);
      continue;
    }
#endif
#ifndef LOWERING_PACK
    // For PACK, NNAPI expects the axis scalar before all input tensors.
    if (TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinPack) {
      const auto* builtin = reinterpret_cast<TfLitePackParams*>(
          TfLiteOpaqueNodeGetBuiltinData(node));
      // Neuron only accepts non-negative axis.
      auto* input_tensor = TfLiteOpaqueNodeGetInput(context, node, 0);
      int axis = builtin->axis < 0 ? TfLiteOpaqueTensorNumDims(input_tensor) +
                                         builtin->axis + 1
                                   : builtin->axis;
      TF_LITE_ENSURE_STATUS(builder.AddScalarInt32Operand(axis));
    }
#endif
    // Map inputs to Neuron tensor indices.
    for (int input_pos = 0; input_pos < TfLiteOpaqueNodeNumberOfInputs(node);
         ++input_pos) {
      if (TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinTransposeConv) {
        // Everything is added during Map since input tensors
        // have different order.
        continue;
      }
      if (TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinFullyConnected &&
          node_inputs[input_pos] == kTfLiteOptionalTensor) {
        // skip optional bias and handle it during mapping
        continue;
      }
      const auto input_index = node_inputs[input_pos];
      if (need_int8_conversion &&
          (input_pos == 0 ||
           TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinFullyConnected ||
           TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinConv2d ||
           TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinDepthwiseConv2d ||
           TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinAdd ||
           TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinMul ||
           TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinSub ||
           TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinConcatenation ||
           TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinMaximum ||
           TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinMinimum ||
           TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinLeakyRelu ||
           TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinLess ||
           TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinLessEqual ||
           TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinPrelu ||
           TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinGreater ||
           TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinGreaterEqual ||
           TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinEqual ||
           TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinNotEqual ||
           TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinSelect)) {
        // Only selected inputs require int8 conversion.
        TF_LITE_ENSURE_STATUS(builder.AddTensorInput(
            input_index, hybrid_op,
            input_tensor_flags | NN_TENSOR_FLAG_INT8_CONVERSION));
        continue;
      }
      if (TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinLstm &&
          isLstmFullKernel(node) && input_pos >= 20) {
        // Skip layer normalization weights. They are added in the Map
        // function (after all the other inputs added there) since layer
        // normalization weights are the last four inputs of the LSTM op in
        // Neuron.
        continue;
      }
      if (TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinLstm &&
          isLstmBasicKernel(node)) {
        // Configuring all inputs in the Map function
        continue;
      }
      if (TfLiteOperatorGetBuiltInCode(reg) ==
          kTfLiteBuiltinUnidirectionalSequenceLstm) {
        if (input_pos >= 20) {
          // Skip layer normalization weights. They are added in the Map
          // function (after all the other inputs added there) since layer
          // normalization weights are the last four inputs of the
          // unidirectional sequence LSTM op in Neuron.
          continue;
        }
        if (input_index == kTfLiteOptionalTensor) {
          TF_LITE_ENSURE_STATUS(builder.AddVectorFloat32Operand(nullptr, 0));
          continue;
        }
      }
      if ((TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinSplit) &&
          (input_index == node_inputs[0])) {
        // Skip the axis input tensor; it will be added as a scalar operand
        // by the Map() mapping.
        continue;
      }

      if ((TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinPadv2 ||
           TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinPad) &&
          input_pos == 1) {
        const TfLiteOpaqueTensor* padding =
            TfLiteOpaqueContextGetOpaqueTensor(context, input_index);

        if (TfLiteOpaqueTensorType(padding) == kTfLiteInt64) {
          input_tensor_flags |= NN_TENSOR_FLAG_INT64_CONVERSION;
        }
        // Pad and Padv2 have an optional parameter for a pad value which has
        // to be converted to a scalar type in Neuron.
      } else if ((TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinPadv2 ||
                  TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinPad) &&
                 TfLiteOpaqueNodeNumberOfInputs(node) == 3 && input_pos == 2) {
        const int constant_value_id = node_inputs[2];
        if (constant_value_id == kTfLiteOptionalTensor) {
          continue;
        }
        const TfLiteOpaqueTensor* constant_value =
            TfLiteOpaqueContextGetOpaqueTensor(context, constant_value_id);

        switch (TfLiteOpaqueTensorType(constant_value)) {
          case kTfLiteFloat32:
            if (TfLiteOpaqueTensorGetAllocationType(constant_value) ==
                kTfLiteMmapRo) {
              builder.AddScalarFloat32Operand(*reinterpret_cast<float*>(
                  TfLiteOpaqueTensorData(constant_value)));
            } else {
              builder.AddSingleValueTensorAsScalarOperand(constant_value_id,
                                                          NEURON_FLOAT32);
            }
            break;
          case kTfLiteUInt8:
            if (TfLiteOpaqueTensorGetAllocationType(constant_value) ==
                kTfLiteMmapRo) {
              builder.AddScalarInt32Operand(*static_cast<int32_t*>(
                  TfLiteOpaqueTensorData(constant_value)));
            } else {
              builder.AddSingleValueTensorAsScalarOperand(constant_value_id,
                                                          NEURON_INT32);
            }
            break;
          case kTfLiteInt8:
            if (TfLiteOpaqueTensorGetAllocationType(constant_value) ==
                kTfLiteMmapRo) {
              builder.AddScalarInt32Operand(
                  *static_cast<int32_t*>(
                      TfLiteOpaqueTensorData(constant_value)) +
                  128);
            } else {
              builder.AddSingleValueTensorAsScalarOperand(constant_value_id,
                                                          NEURON_INT32);
            }
            break;
          default:
            TfLiteOpaqueContextReportError(
                context, "Unsupported type of pad value for pad_v2\n");
            return kTfLiteError;
        }
        continue;
      }

      if (input_index == kTfLiteOptionalTensor &&
          (TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinLstm ||
           TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinSvdf ||
           TfLiteOperatorGetBuiltInCode(reg) ==
               kTfLiteBuiltinBidirectionalSequenceLstm)) {
        // properly handle the optional tensor for LSTM and SVDF.
        // currently only support float32.
        TF_LITE_ENSURE_STATUS(builder.AddVectorFloat32Operand(nullptr, 0));
      } else if (TfLiteOperatorGetBuiltInCode(reg) ==
                     kTfLiteBuiltinResizeBilinear ||
                 TfLiteOperatorGetBuiltInCode(reg) ==
                     kTfLiteBuiltinResizeNearestNeighbor) {
        if (input_pos == 0) {
          // Only the first input tensor is added. The second one,
          // specifying the output height and width, is not added and
          // instead the height and width will be added individually as
          // scalars by the mapping function returned by Map().
          TF_LITE_ENSURE_STATUS(builder.AddTensorInput(input_index, hybrid_op));
        }
      } else if (TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinTopkV2 &&
                 input_pos > 0) {
        // The K parameter tensor is not handled here but by the functor
        // returned by Map, the input tensor is instead added in
        // the else clause below
        continue;
      } else if (TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinGather) {
        // Everything is added during Map since input tensors
        // have different order.
        continue;
      } else if (TfLiteOperatorGetBuiltInCode(reg) ==
                     kTfLiteBuiltinExpandDims &&
                 input_pos == 1) {
        // The axis param is added during Map
        continue;
      } else if (TfLiteOperatorGetBuiltInCode(reg) ==
                     kTfLiteBuiltinBatchToSpaceNd &&
                 input_pos == 2) {
        // Neuron does not support crops.
        // The Map function will check if all crops are zero.
        continue;
      } else if (TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinArgMin ||
                 TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinArgMax) {
        // The first input tensor is added as is. The second one, specifying
        // the axis, needs to be converted to a scalar since TFLite uses a
        // tensor but Neuron uses a scalar as the axis.
        if (input_pos == 0) {
          TF_LITE_ENSURE_STATUS(builder.AddTensorInput(input_index, hybrid_op));
        } else {
          const int axis_id = node_inputs[1];
          const TfLiteOpaqueTensor* axis_tensor =
              TfLiteOpaqueContextGetOpaqueTensor(context, axis_id);
          switch (TfLiteOpaqueTensorType(axis_tensor)) {
            case kTfLiteInt32:
              if (TfLiteOpaqueTensorGetAllocationType(axis_tensor) ==
                  kTfLiteMmapRo) {
                TF_LITE_ENSURE_STATUS(
                    builder.AddScalarInt32Operand(*static_cast<int32_t*>(
                        TfLiteOpaqueTensorData(axis_tensor))));
              } else {
                TF_LITE_ENSURE_STATUS(
                    builder.AddSingleValueTensorAsScalarOperand(axis_id,
                                                                NEURON_INT32));
              }
              break;
            case kTfLiteInt64:
              // Map() function already makes sure int64 input is constant.
              TF_LITE_ENSURE_STATUS(builder.AddScalarInt32Operand(
                  *static_cast<int32_t*>(TfLiteOpaqueTensorData(axis_tensor))));
              break;
            default:
              return kTfLiteError;
          }
        }
      } else if (TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinCustom &&
                 strcmp(TfLiteRegistrationExternalGetCustomName(reg),
                        "TFLite_Detection_PostProcess") == 0) {
        // Everything is added during Map since input tensors
        // have different order.
        continue;
      } else {
        TF_LITE_ENSURE_STATUS(
            builder.AddTensorInput(input_index, hybrid_op, input_tensor_flags));
      }
    }
    // If we have target accelerators the target SDK version might be
    // different than the current android version.
    int android_sdk_version = neuronapi_->android_sdk_version;

    // Get op type and operands
    // Fails if the Validate function failed
    NeuronOperationType nn_op_type;
    TF_LITE_ENSURE_STATUS(Map(context, TfLiteOperatorGetBuiltInCode(reg),
                              TfLiteRegistrationExternalGetVersion(reg),
                              android_sdk_version,
                              {context, &builder, node, &model_state_outputs_,
                               &model_state_tfl_inputs_, &feedback_loops_, reg},
                              &nn_op_type));
    // Map outputs to Neuron tensor indices.
    int output_tensor_flags = 0;
    if (need_int8_conversion) {
      output_tensor_flags |= NN_TENSOR_FLAG_INT8_CONVERSION;
    }
    if (use_int8_asymm_signed) {
      output_tensor_flags |= NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED;
    }
    // fc_nn_intermediate_output_index is used to indicate whether additional
    // RESHAPE op is needed.
    int fc_nn_intermediate_output_index = -1;
    const int* outputs;
    int num_outputs;
    TF_LITE_OPAQUE_ENSURE_STATUS(
        TfLiteOpaqueNodeOutputs(node, &outputs, &num_outputs));
    for (int output_pos = 0; output_pos < TfLiteOpaqueNodeNumberOfOutputs(node);
         ++output_pos) {
      auto output_index = outputs[output_pos];

      // Outputs for  basic LSTM cell are set in the Map function since
      if (TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinLstm &&
          isLstmBasicKernel(node)) {
        continue;
      }
      // Handle FC with keep_num_dims==true.
      if (TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinFullyConnected &&
          reinterpret_cast<TfLiteFullyConnectedParams*>(
              TfLiteOpaqueNodeGetBuiltinData(node))
              ->keep_num_dims) {
        auto* output_tensor =
            TfLiteOpaqueContextGetOpaqueTensor(context, output_index);

        int num_units = TfLiteOpaqueTensorDim(
            output_tensor, TfLiteOpaqueTensorNumDims(output_tensor) - 1);
        std::vector<uint32_t> output_dims(2);
        output_dims[0] = NumElements(output_tensor) / num_units;
        output_dims[1] = num_units;
        TF_LITE_ENSURE_STATUS(builder.AddIntermediateOutputTensor(
            TfLiteOpaqueTensorType(output_tensor), output_dims.size(),
            output_dims.data(),
            TfLiteOpaqueTensorGetQuantizationParams(output_tensor).scale,
            TfLiteOpaqueTensorGetQuantizationParams(output_tensor).zero_point,
            &fc_nn_intermediate_output_index));
      } else if (TfLiteOperatorGetBuiltInCode(reg) == kTfLiteBuiltinCustom &&
                 strcmp(TfLiteRegistrationExternalGetCustomName(reg),
                        "TFLite_Detection_PostProcess") == 0) {
        // Everything is added during Map since output tensors
        // have different order.
        continue;
      } else {
        TF_LITE_ENSURE_STATUS(
            builder.AddTensorOutput(output_index, output_tensor_flags));
      }
    }

    // Dequantize operators may have to be added in case inputs are to be
    // floating-point.
    AddDequantizeOperatorsWhereNeeded(
        context, TfLiteOperatorGetBuiltInCode(reg), node, &builder);
    TF_LITE_ENSURE_STATUS(builder.FinalizeAddOperation(nn_op_type));
    if (fc_nn_intermediate_output_index > -1) {
      TF_LITE_ENSURE_STATUS(
          builder.AppendReshape(fc_nn_intermediate_output_index, outputs[0]));
    }
  }
  total_op_num_ = builder.GetTotalOpNum();
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE, "AddOpsAndTensors done");
  return kTfLiteOk;
}

TfLiteStatus NeuronStableDelegateKernel::BuildGraph(
    TfLiteOpaqueContext* context, const TfLiteOpaqueDelegateParams* params,
    const TfLiteIntArray* input_tensors, const TfLiteIntArray* output_tensors) {
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE, "BuildGraph");
  // Build the ops and tensors.
  TF_LITE_ENSURE_STATUS(AddOpsAndTensors(context));

  // Map input and output tensor indices to Neuron
  std::vector<uint32_t> inputs;
  inputs.reserve(input_tensors->size);
  std::vector<uint32_t> outputs;
  outputs.reserve(output_tensors->size);

  size_t total_input_byte_size = 0;
  // Make the TensorFlow Lite inputs and outputs to neuron indices.
  for (int i : TfLiteIntArrayView(input_tensors)) {
    // Constant tensors are not Neuron inputs.
    if (i != kTfLiteOptionalTensor &&
        TfLiteOpaqueTensorGetAllocationType(
            TfLiteOpaqueContextGetOpaqueTensor(context, i)) != kTfLiteMmapRo &&
        // The delegate might not have mapped this input (this can
        // happen if one tensor is split in several ones)
        operand_mapping_.lite_index_to_neuron(i) != -1) {
      inputs.push_back(operand_mapping_.lite_index_to_neuron(i));
      const TfLiteType nn_type_conversion =
          operand_mapping_.lite_index_to_neuron_type_conversion(i);
      int tensor_size = 0;
      if (nn_type_conversion == kTfLiteNoType) {
        tensor_size = TfLiteOpaqueTensorByteSize(
            TfLiteOpaqueContextGetOpaqueTensor(context, i));
      } else {
        size_t type_size;
        TF_LITE_ENSURE_OK(context,
                          TfLiteOpaqueContextGetSizeOfType(
                              context, nn_type_conversion, &type_size));
        tensor_size =
            NumElements(TfLiteOpaqueContextGetOpaqueTensor(context, i)) *
            type_size;
      }
      total_input_byte_size += tensor_size;
      total_input_byte_size += getNumPaddingBytes(tensor_size);
    }
  }
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE, "total_input_byte_size: %d",
                  total_input_byte_size);

  size_t total_output_byte_size = 0;
  for (int i : TfLiteIntArrayView(output_tensors)) {
    const int output_tensor_neuron_index =
        operand_mapping_.lite_index_to_neuron(i);
    // Unmapped outputs are not added
    if (output_tensor_neuron_index != -1) {
      outputs.push_back(output_tensor_neuron_index);
    }
    total_output_byte_size += TfLiteOpaqueTensorByteSize(
        TfLiteOpaqueContextGetOpaqueTensor(context, i));
    total_output_byte_size += getNumPaddingBytes(TfLiteOpaqueTensorByteSize(
        TfLiteOpaqueContextGetOpaqueTensor(context, i)));
  }
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE, "total_output_byte_size: %d",
                  total_output_byte_size);

  // Add state output tensors as model outputs.
  for (int i : model_state_outputs_) {
    outputs.push_back(i);
  }
  // gen model token
  nn_compilation_cache_token_.clear();
  std::string model_token_str =
      options_.model_token + "/" +
      std::to_string(TfLiteOpaqueContextGetNumTensors(context));
  uint64_t token_parts[4];
  token_parts[0] =
      ::util::Fingerprint64(model_token_str.c_str(), model_token_str.size());
  if (params != nullptr) {
    token_parts[1] = GetHash(params->nodes_to_replace);
    token_parts[2] = GetHash(params->input_tensors);
    token_parts[3] = GetHash(params->output_tensors);
  } else {
    token_parts[1] = GetHash(nodes_.data(), nodes_.size());
    token_parts[2] = GetHash(input_tensors);
    token_parts[3] = GetHash(output_tensors);
  }
  std::vector<uint8_t> nnapi_cache_token(32, 0);
  uint8_t* p = reinterpret_cast<uint8_t*>(token_parts);
  for (int i = 0; i < 4 * sizeof(uint64_t); i++) {
    nnapi_cache_token[i] = p[i];
  }
  nn_compilation_cache_token_ = nnapi_cache_token;

  std::string test = "{";
  for (auto i : nnapi_cache_token) {
    test += std::to_string(i);
    test += ",";
  }
  test += "};";

  TFLITE_MTK_LOG_VERBOSE("Kernel cache_token: %s", test.c_str());

  // Tell Neuron to declare inputs/outputs
  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
      context,
      neuronapi_->NeuronModel_identifyInputsAndOutputs(
          nn_model_.get(), inputs.size(), inputs.data(), outputs.size(),
          outputs.data()),
      "identifying model inputs and outputs");
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                  "NeuronModel_identifyInputsAndOutputs");

  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
      context,
      neuronapi_->NeuronModel_relaxComputationFloat32toFloat16(
          nn_model_.get(), options_.allow_fp16),
      "set relaxed computation mode for fp32 if possible");
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                  "NeuronModel_relaxComputationFloat32toFloat16: %d",
                  options_.allow_fp16);

  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
      context, neuronapi_->NeuronModel_finish(nn_model_.get()),
      "finalizing the model");
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE, "NeuronModel_finish");
  // Create shared memory pool for inputs and outputs.
  nn_input_memory_.reset(new NNMemory(neuronapi_, "input_pool",
                                      total_input_byte_size, options_.use_ahwb,
                                      options_.use_cacheable_buffer));
  nn_output_memory_.reset(
      new NNMemory(neuronapi_, "output_pool", total_output_byte_size,
                   options_.use_ahwb, options_.use_cacheable_buffer));

  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE, "BuildGraph done");
  return kTfLiteOk;
}

TfLiteStatus NeuronStableDelegateKernel::GetSupportedOperations(
    TfLiteOpaqueContext* context, const TfLiteIntArray* input_tensors,
    const TfLiteIntArray* output_tensors, bool* is_supported) {
  TF_LITE_ENSURE_STATUS(
      BuildGraph(context, nullptr, input_tensors, output_tensors));
  initialised_ = true;
  auto support_flags = std::make_unique<bool[]>(total_op_num_);
  // Determine the list of operations the device actually supports
  if (neuronapi_->NeuronCompilation_getSupportedOperations != nullptr) {
    TF_LITE_ENSURE_STATUS(Prepare(context, nullptr));
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                    "NeuronCompilation_getSupportedOperations");
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context,
        neuronapi_->NeuronCompilation_getSupportedOperations(
            nn_compilation_.get(), total_op_num_, support_flags.get()),
        "get supported operations for devices");
  } else {
    uint32_t num_devices = 0;

    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context, neuronapi_->Neuron_getDeviceCount(&num_devices),
        "getting device count");

    std::vector<NeuronDevice*> device(num_devices);
    int device_count = 0;
    if (strlen(options_.accelerator_name.c_str()) != 0) {
      char* device_split_name = std::strtok(
          const_cast<char*>(options_.accelerator_name.c_str()), ",");
      while (device_split_name) {
        NeuronDevice* mtk_device = nullptr;
        if (GetAcceleratorDevice(neuronapi_, context, device_split_name,
                                 &mtk_device) == kTfLiteOk) {
          device[device_count] = mtk_device;
          device_count++;
          TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE, "Get device: %s",
                          device_split_name);
        }
        device_split_name = std::strtok(NULL, ",");
      }
    }
    if (device_count) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                      "NeuronModel_getSupportedOperationsForDevices");
      RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
          context,
          neuronapi_->NeuronModel_getSupportedOperationsForDevices(
              nn_model_.get(), device.data(), device_count,
              support_flags.get()),
          "get supported operations for devices");

    } else {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                      "NeuronModel_getSupportedOperations");
      RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
          context,
          neuronapi_->NeuronModel_getSupportedOperations(
              nn_model_.get(), support_flags.get(), total_op_num_),
          "get supported operations");
    }
  }

  // If the number of nodes is different from the nodes that we actually
  // added to neuron, we cannot identify which nodes the support status
  // results correspond to.
  if (total_op_num_ != nodes_.size()) {
    for (int i = 1; i < total_op_num_; ++i) {
      if (support_flags[i] != support_flags[0]) {
        std::fill(is_supported, is_supported + nodes_.size(), false);
        TFLITE_LOG_PROD(
            tflite::TFLITE_LOG_WARNING,
            "Currently cannot support if the number of nodes "
            "is different from the nodes we actually added to neuron.");
        return kTfLiteUnresolvedOps;
      }
    }
  }
  // Only copy the size of the original nodes to avoid exceeding the bounds.
  std::copy(support_flags.get(), support_flags.get() + nodes_.size(),
            is_supported);
  return kTfLiteOk;
}

TfLiteStatus NeuronStableDelegateKernel::GetSupportedOperationsForOneNode(
    TfLiteOpaqueContext* context, TfLiteOpaqueNode* node,
    TfLiteRegistrationExternal* registration, bool* is_supported) {
  delegate_specific_mode_ = kOneNodeMode;
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                  "GetSupportedOperationsForOneNode: %s",
                  EnumNameBuiltinOperator(static_cast<BuiltinOperator>(
                      TfLiteOperatorGetBuiltInCode(registration))));
  // One Node Mode with unknown node index, nodes_ is always set to [0].
  nodes_.push_back(0);
  one_node_and_reg_.first = node;
  one_node_and_reg_.second = registration;
  NeuronModel* model = nullptr;
  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
      context, neuronapi_->NeuronModel_create(&model), "creating Neuron model");
  nn_model_.reset(model);
  const int *inputs, *outputs;
  int num_inputs, num_outputs;
  TF_LITE_OPAQUE_ENSURE_STATUS(
      TfLiteOpaqueNodeInputs(node, &inputs, &num_inputs));
  TF_LITE_OPAQUE_ENSURE_STATUS(
      TfLiteOpaqueNodeOutputs(node, &outputs, &num_outputs));
  TfLiteIntArray* input_tensors = TfLiteIntArrayCreate(num_inputs);
  TfLiteIntArray* output_tensors = TfLiteIntArrayCreate(num_outputs);
  if (input_tensors) {
    memcpy(input_tensors->data, inputs, sizeof(int) * num_inputs);
  }
  if (output_tensors) {
    memcpy(output_tensors->data, outputs, sizeof(int) * num_outputs);
  }

  auto re = GetSupportedOperations(context, input_tensors, output_tensors,
                                   is_supported);

  if (input_tensors) {
    TfLiteIntArrayFree(input_tensors);
  }
  if (output_tensors) {
    TfLiteIntArrayFree(output_tensors);
  }
  // If there's any nodes that added to neuron is unsupported,
  // it return kTfLiteUnresolvedOps and mark this TFLite node as unsupported.
  if (re != kTfLiteOk && re != kTfLiteUnresolvedOps) {
    return re;
  }
  return kTfLiteOk;
}

TfLiteStatus NeuronStableDelegateKernel::GetSupportedOperationsForWholeGraph(
    TfLiteOpaqueContext* context, bool* is_supported) {
  delegate_specific_mode_ = kContextMode;
  TfLiteIntArray* execution_plan = nullptr;
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                  "GetSupportedOperationsForWholeGraph");
  if (TfLiteOpaqueContextGetExecutionPlan(context, &execution_plan) !=
      kTfLiteOk) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR, "Fail to get exexution plan");
    return kTfLiteError;
  }
  nodes_.resize(execution_plan->size);
  memcpy(nodes_.data(), execution_plan->data,
         sizeof(int) * execution_plan->size);

  NeuronModel* model = nullptr;
  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
      context, neuronapi_->NeuronModel_create(&model), "creating Neuron model");
  nn_model_.reset(model);
  const int *inputs, *outputs;
  int num_inputs, num_outputs;
  TF_LITE_OPAQUE_ENSURE_STATUS(
      TfLiteOpaqueContextGetInputs(context, &inputs, &num_inputs));
  TF_LITE_OPAQUE_ENSURE_STATUS(
      TfLiteOpaqueContextGetOutputs(context, &outputs, &num_outputs));
  TfLiteIntArray* input_tensors = TfLiteIntArrayCreate(num_inputs);
  TfLiteIntArray* output_tensors = TfLiteIntArrayCreate(num_outputs);
  if (input_tensors) {
    memcpy(input_tensors->data, inputs, sizeof(int) * num_inputs);
  }
  if (output_tensors) {
    memcpy(output_tensors->data, outputs, sizeof(int) * num_outputs);
  }

  TF_LITE_OPAQUE_ENSURE_STATUS(GetSupportedOperations(
      context, input_tensors, output_tensors, is_supported));

  if (input_tensors) {
    TfLiteIntArrayFree(input_tensors);
  }
  if (output_tensors) {
    TfLiteIntArrayFree(output_tensors);
  }
  return kTfLiteOk;
}

TfLiteStatus NeuronStableDelegateKernel::RegisterBuffer(
    TfLiteBufferHandle buf_handle, int buf_fd, size_t buf_size) {
  NeuronMemory* memory = nullptr;
  neuronapi_->NeuronMemory_createFromFd(buf_size, PROT_READ | PROT_WRITE,
                                        buf_fd, 0, &memory);
  if (memory == nullptr) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "RegisterBuffer called with unknown memory");
    return kTfLiteError;
  }
  tensor_memory_map_kernel_.emplace(buf_handle, memory);
  return kTfLiteOk;
}

TfLiteStatus NeuronStableDelegateKernel::UnregisterBuffer(
    TfLiteBufferHandle buf_handle) {
  auto it = tensor_memory_map_kernel_.find(buf_handle);
  if (it == tensor_memory_map_kernel_.end()) {
    return kTfLiteError;
  }
  neuronapi_->NeuronMemory_free(it->second);
  tensor_memory_map_kernel_.erase(it);
  return kTfLiteOk;
}

}  // namespace neuron
}  // namespace tflite
