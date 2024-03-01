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

#ifndef MTK_TENSORFLOW_LITE_MINIMAL_LOGGING_H_
#define MTK_TENSORFLOW_LITE_MINIMAL_LOGGING_H_

#include <cstdarg>
#include <string>
#include "tensorflow/lite/minimal_logging.h"
#include "delegate/mtk_neuron/mtk/mtk_utils.h"
#include "delegate/mtk_neuron/neuron_delegate_utils.h"

namespace tflite {
namespace logging_internal {
namespace mtk {

inline bool IsPrintLogSupported() {
  static bool enabled = tflite::mtk::PropertyGetBool("debug.mtk_tflite.vlog",
                                                    false);
  static bool isUserBuildType = tflite::neuron::GetPropertyValue("ro.build.type") == "user";
  return enabled && !isUserBuildType;
}

inline void Log(LogSeverity severity, const char* tag,
    const char* format, ...) {
  static bool support = IsPrintLogSupported();
  if (!support && severity != TFLITE_LOG_ERROR) {
    return;
  }

  va_list args;
  va_start(args, format);
  tflite::logging_internal::MinimalLogger::LogFormatted(severity, format, args);
  va_end(args);
}

}  // namespace mtk
}  // namespace logging_internal
}  // namespace tflite

#define TFLITE_MTK_LOG_INFO(format, ...) \
  tflite::logging_internal::mtk::Log(tflite::TFLITE_LOG_INFO, \
                                    LOG_TAG, format, ##__VA_ARGS__);

#define TFLITE_MTK_LOG_WARN(format, ...) \
  tflite::logging_internal::mtk::Log(tflite::TFLITE_LOG_WARNING, \
                                    LOG_TAG, format, ##__VA_ARGS__);

#define TFLITE_MTK_LOG_ERROR(format, ...) \
  tflite::logging_internal::mtk::Log(tflite::TFLITE_LOG_ERROR, \
                                    LOG_TAG, format, ##__VA_ARGS__);

#define TFLITE_MTK_LOG_OP_DUMP(format, ...) \
  tflite::logging_internal::mtk::Log(tflite::TFLITE_LOG_INFO, \
                                    "TFLite OP Dump", format, ##__VA_ARGS__);

#endif  // MTK_TENSORFLOW_LITE_MINIMAL_LOGGING_H_
