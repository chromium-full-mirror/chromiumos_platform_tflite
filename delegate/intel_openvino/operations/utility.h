/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DELEGATE_INTEL_OPENVINO_OPERATIONS_UTILITY_H_
#define DELEGATE_INTEL_OPENVINO_OPERATIONS_UTILITY_H_

#include <openvino/openvino.hpp>

#include "tensorflow/lite/c/c_api_types.h"

namespace tflite {
namespace openvinodelegate {

inline ov::element::Type GetOVElementType(TfLiteType tensor_type) {
  ov::element::Type ov_element_type;
  switch (tensor_type) {
    case kTfLiteFloat32:
      ov_element_type = ov::element::f32;
      break;
    case kTfLiteInt32:
      ov_element_type = ov::element::i32;
      break;
    case kTfLiteUInt8:
      ov_element_type = ov::element::u8;
      break;
    case kTfLiteInt64:
      ov_element_type = ov::element::i64;
      break;
    case kTfLiteBool:
      ov_element_type = ov::element::boolean;
      break;
    case kTfLiteInt16:
      ov_element_type = ov::element::i16;
      break;
    case kTfLiteInt8:
      ov_element_type = ov::element::i8;
      break;
    case kTfLiteFloat16:
      ov_element_type = ov::element::f16;
      break;
    case kTfLiteFloat64:
      ov_element_type = ov::element::f64;
      break;
    case kTfLiteUInt64:
      ov_element_type = ov::element::u64;
      break;
    case kTfLiteUInt32:
      ov_element_type = ov::element::u32;
      break;
    case kTfLiteUInt16:
      ov_element_type = ov::element::u16;
      break;
    case kTfLiteInt4:
      ov_element_type = ov::element::i4;
      break;
    default:
      ov_element_type = ov::element::undefined;
  }
  return ov_element_type;
}

}  // namespace openvinodelegate
}  // namespace tflite

#endif  // DELEGATE_INTEL_OPENVINO_OPERATIONS_UTILITY_H_
