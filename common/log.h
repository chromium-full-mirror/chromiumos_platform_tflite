/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef COMMON_LOG_H_
#define COMMON_LOG_H_

#include "absl/log/absl_log.h"

// TODO(shik): Wrap other logging macros as well.
#define LOG(severity)                       \
  ABSL_LOG(severity).NoPrefix().ToSinkOnly( \
      ::tflite::cros::log_internal::SyslogStderrSink::GetInstance())

namespace tflite::cros {

namespace log_internal {

// A LogSink class that mimics the usual libbrillo logging behavior on ChromeOS,
// which will sends the message to both syslog and stderr.
//
// This class is thread-safe.
class SyslogStderrSink : public absl::LogSink {
 public:
  static SyslogStderrSink* GetInstance();

  void Send(const absl::LogEntry& entry) override;
};

}  // namespace log_internal

}  // namespace tflite::cros

#endif  // COMMON_LOG_H_
