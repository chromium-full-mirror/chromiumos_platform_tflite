/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/log.h"

#include <errno.h>
#include <syslog.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_format.h"
#include "tensorflow/lite/tools/command_line_flags.h"

namespace tflite::cros::log_impl {

namespace {

const char* GetProgramName() {
  // This is defined in <errno.h>.
  const char* name = program_invocation_short_name;
  return name != nullptr ? name : "";
}

int GetSyslogLevel(const absl::LogEntry& entry) {
  if (entry.verbosity() != absl::LogEntry::kNoVerbosityLevel) {
    return LOG_DEBUG;
  }

  switch (entry.log_severity()) {
    case absl::LogSeverity::kInfo:
      return LOG_INFO;
    case absl::LogSeverity::kWarning:
      return LOG_WARNING;
    case absl::LogSeverity::kError:
      return LOG_ERR;
    case absl::LogSeverity::kFatal:
      return LOG_CRIT;
  }

  // Safe fallback. Should not happen normally.
  return LOG_DEBUG;
}

// Gets the `--v=` flag from command line.
int GetVLogLevelFromCmdline() {
  // We can not call absl::ParseCommandLine() in main because the logging
  // library might be used in stable delegate, which will be loaded dynamically
  // with dlopen(). Reconstruct argv from proc fs instead.
  FILE* fp = fopen("/proc/self/cmdline", "r");
  if (fp == nullptr) {
    return 0;
  }
  absl::Cleanup close_file = [&] { fclose(fp); };

  std::vector<std::string> args;
  char* arg = nullptr;
  size_t size = 0;
  while (getdelim(&arg, &size, '\0', fp) != -1) {
    args.push_back(arg);
  }
  free(arg);

  int vlog_level = 0;
  std::vector<Flag> flags = {
      Flag::CreateFlag("v", &vlog_level,
                       "Show all VLOG(m) messages for m <= this."),
  };
  // We need to copy the arguments to a non-owning temporary vector since
  // Flags::Parse() will try to remove the parsed arguments from argv.
  int argc = args.size();
  std::vector<const char*> argv;
  for (const auto& arg : args) {
    argv.push_back(arg.data());
  }
  Flags::Parse(&argc, argv.data(), flags);

  return vlog_level;
}

std::string FormatPidTid(int pid, int tid) {
  // Only show thread id when it's different from the process id.
  return pid == tid ? absl::StrFormat("%d", pid)
                    : absl::StrFormat("%d:%d", pid, tid);
}

}  // namespace

SyslogStderrSink* SyslogStderrSink::GetInstance() {
  // Leaky singleton.
  using Self = SyslogStderrSink;
  alignas(Self) static uint8_t storage[sizeof(Self)];
  static Self* instance = [&] {
    int vlog_level = GetVLogLevelFromCmdline();
    return new (storage) Self(vlog_level);
  }();
  return instance;
}

void SyslogStderrSink::Send(const absl::LogEntry& entry) {
  if (entry.verbosity() > global_vlog_level_) {
    return;
  }

  syslog(GetSyslogLevel(entry), "%s",
         entry.text_message_with_prefix_and_newline_c_str());

  std::string timestamp = absl::FormatTime(
      "%Y-%m-%d%ET%H:%M:%E6SZ", entry.timestamp(), absl::UTCTimeZone());
  const char* severity = absl::LogSeverityName(entry.log_severity());

  // A syslog compatible format used on ChromeOS. Reference:
  // https://source.chromium.org/chromium/chromium/src/+/main:base/logging_chromeos.cc
  //
  // We don't provide control knobs for simplicity. The format looks like:
  // <timestamp> <severity> <program>[<pid>:<tid>]: [<file>(<line>)] <message>
  //
  // TODO(shik): Only log to stderr if stdin is a tty.
  std::string body = absl::StrFormat(
      "%v %s %s[%s]: [%v(%d)] %v", timestamp, severity, GetProgramName(),
      FormatPidTid(getpid(), entry.tid()), entry.source_basename(),
      entry.source_line(), entry.text_message_with_newline());
  fwrite(body.data(), body.size(), 1, stderr);

  // TODO(shik): Print stacktrace if available.
}

}  // namespace tflite::cros::log_impl
