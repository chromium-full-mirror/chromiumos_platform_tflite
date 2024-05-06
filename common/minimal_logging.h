/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef COMMON_MINIMAL_LOGGING_H_
#define COMMON_MINIMAL_LOGGING_H_

// This header provides the same TFLITE_LOG*() macros as
// tensorflow/lite/minimal_logging.h with the ChromeOS logging implementation.
//
// The third-party delegates can integrate ChromeOS logging easily by just
// switching the include statement, instead of changing every TFLITE_LOG*
// statement.
//
// TODO(shik): Consider upstreaming this into TFLite.

#include "absl/strings/str_format.h"  // IWYU pragma: keep
#include "common/log.h"
#include "tensorflow/lite/logger.h"  // IWYU pragma: keep

#define TFLITE_LOG_PROD(severity, format, ...)                      \
  do {                                                              \
    switch (severity) {                                             \
      case tflite::TFLITE_LOG_VERBOSE:                              \
        VLOGF(1) << absl::StreamFormat(format, ##__VA_ARGS__);      \
        break;                                                      \
      case tflite::TFLITE_LOG_INFO:                                 \
        LOGF(INFO) << absl::StreamFormat(format, ##__VA_ARGS__);    \
        break;                                                      \
      case tflite::TFLITE_LOG_WARNING:                              \
        LOGF(WARNING) << absl::StreamFormat(format, ##__VA_ARGS__); \
        break;                                                      \
      case tflite::TFLITE_LOG_ERROR:                                \
        LOGF(ERROR) << absl::StreamFormat(format, ##__VA_ARGS__);   \
        break;                                                      \
    }                                                               \
  } while (false)

#define TFLITE_LOG_PROD_ONCE(severity, format, ...)     \
  do {                                                  \
    [[maybe_unused]] static const bool log_once = [&] { \
      TFLITE_LOG(severity, format, ##__VA_ARGS__);      \
      return true;                                      \
    }();                                                \
  } while (false)

#ifdef NDEBUG
// In prod builds, never log, but ensure the code is well-formed and compiles.
#define TFLITE_LOG(severity, format, ...)               \
  do {                                                  \
    if (false) {                                        \
      TFLITE_LOG_PROD(severity, format, ##__VA_ARGS__); \
    }                                                   \
  } while (false)
#define TFLITE_LOG_ONCE TFLITE_LOG
#else  // NDEBUG
// In debug builds, always log.
#define TFLITE_LOG TFLITE_LOG_PROD
#define TFLITE_LOG_ONCE TFLITE_LOG_PROD_ONCE
#endif  // NDEBUG

#endif  // COMMON_MINIMAL_LOGGING_H_
