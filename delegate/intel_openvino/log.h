/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef DELEGATE_INTEL_OPENVINO_LOG_H_
#define DELEGATE_INTEL_OPENVINO_LOG_H_

// Wraps the logging macros in a header so it's easier to swap the implementaion
// on difference platforms if needed.

#include "common/log.h"

#define TFLITE_LOG(severity) LOG(severity)

#endif  // DELEGATE_INTEL_OPENVINO_LOG_H_
