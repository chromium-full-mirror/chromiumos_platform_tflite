/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef COMMON_LOG_H_
#define COMMON_LOG_H_

#include "absl/log/absl_check.h"  // IWYU pragma: keep
#include "absl/log/absl_log.h"    // IWYU pragma: keep

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

// TODO(shik): ABSL_VLOG() is not yet available in the Abseil used by Tensorflow
// yet. Replace WithVerbosity with ABSL_VLOG() once available.

// Logging with verbosity level that can configured at runtime.
// `VLOG` statements are logged at `INFO` severity if they are logged at all;
// the numeric levels are on a different scale than the proper severity levels.
#define VLOG(verbosity) LOG_IMPL_ABSL(LOG, INFO).WithVerbosity(verbosity)

// Logging with verbosity level and function name.
#define VLOGF(verbosity) \
  LOG_IMPL_ABSL(LOG, INFO).WithVerbosity(verbosity) << __FUNCTION__ << ": "

// Checks the condition and terminates the program if it's false.
// Note that DCHECK() is intentionally not added here as per the style guide.
// https://chromium.googlesource.com/chromium/src/+/HEAD/styleguide/c++/checks.md
#define CHECK(condition) LOG_IMPL_ABSL(CHECK, condition)
#define CHECK_EQ(val1, val2) LOG_IMPL_ABSL(CHECK_EQ, val1, val2)
#define CHECK_NE(val1, val2) LOG_IMPL_ABSL(CHECK_NE, val1, val2)
#define CHECK_LT(val1, val2) LOG_IMPL_ABSL(CHECK_LT, val1, val2)
#define CHECK_LE(val1, val2) LOG_IMPL_ABSL(CHECK_LE, val1, val2)
#define CHECK_GT(val1, val2) LOG_IMPL_ABSL(CHECK_GT, val1, val2)
#define CHECK_GE(val1, val2) LOG_IMPL_ABSL(CHECK_GE, val1, val2)

// `PCHECK` behaves like `CHECK` but appends a description of the current state
// of `errno` to the failure message.
#define PCHECK(condition) LOG_IMPL_ABSL(PCHECK, condition)

namespace tflite::cros {

namespace log_impl {

// A LogSink class that mimics the usual libbrillo logging behavior on ChromeOS,
// which will sends the message to both syslog and stderr.
//
// This class is thread-safe.
class SyslogStderrSink : public absl::LogSink {
 public:
  static SyslogStderrSink* GetInstance();

  explicit SyslogStderrSink(int global_vlog_level)
      : global_vlog_level_(global_vlog_level) {}

  void Send(const absl::LogEntry& entry) override;

 private:
  int global_vlog_level_ = 0;
};

}  // namespace log_impl

}  // namespace tflite::cros

#endif  // COMMON_LOG_H_
