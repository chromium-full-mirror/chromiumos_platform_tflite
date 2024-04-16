/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef COMMON_LOG_H_
#define COMMON_LOG_H_

#include "absl/log/absl_log.h"  // IWYU pragma: keep

// Exports DUMP() here to allow `LOG(INFO) << DUMP(...)` usage.
#include "common/dump.h"  // IWYU pragma: export

#define LOG_IMPL_ABSL(macro, ...) \
  ABSL_##macro(__VA_ARGS__)       \
      .NoPrefix()                 \
      .ToSinkOnly(tflite::cros::log_impl::SyslogStderrSink::GetInstance())

#define LOG_IMPL_ABSL_F(...) LOG_IMPL_ABSL(__VA_ARGS__) << __FUNCTION__ << ": "

// Normal logging.
#define LOG(severity) LOG_IMPL_ABSL(LOG, severity)

// Logging with function name.
#define LOGF(severity) LOG_IMPL_ABSL_F(LOG, severity)

// Logging with errno description.
#define PLOG(severity) LOG_IMPL_ABSL(PLOG, severity)

// Logging with function name and errno description.
#define PLOGF(severity) LOG_IMPL_ABSL_F(PLOG, severity)

// TODO(shik): VLOG() is not yet available in the Abseil used by Tensorflow yet.
// Consider adding it once available.

namespace tflite::cros {

namespace log_impl {

// A LogSink class that mimics the usual libbrillo logging behavior on ChromeOS,
// which will sends the message to both syslog and stderr.
//
// This class is thread-safe.
class SyslogStderrSink : public absl::LogSink {
 public:
  static SyslogStderrSink* GetInstance();

  void Send(const absl::LogEntry& entry) override;
};

}  // namespace log_impl

}  // namespace tflite::cros

#endif  // COMMON_LOG_H_
