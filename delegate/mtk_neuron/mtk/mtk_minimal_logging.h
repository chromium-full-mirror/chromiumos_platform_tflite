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

#ifndef DELEGATE_MTK_NEURON_MTK_MTK_MINIMAL_LOGGING_H_
#define DELEGATE_MTK_NEURON_MTK_MTK_MINIMAL_LOGGING_H_

#include "common/minimal_logging.h"

#define TFLITE_MTK_LOG_VERBOSE(format, ...)                        \
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE, LOG_TAG ": " format, \
                  ##__VA_ARGS__)

#define TFLITE_MTK_LOG_INFO(format, ...) \
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, LOG_TAG ": " format, ##__VA_ARGS__)

#define TFLITE_MTK_LOG_WARN(format, ...)                           \
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING, LOG_TAG ": " format, \
                  ##__VA_ARGS__)

#define TFLITE_MTK_LOG_ERROR(format, ...) \
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR, LOG_TAG ": " format, ##__VA_ARGS__)

#endif  // DELEGATE_MTK_NEURON_MTK_MTK_MINIMAL_LOGGING_H_
