/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <fcntl.h>
#include <unistd.h>

#include <string>

#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "common/log.h"
#include "common/scoped_fd.h"

namespace tflite::cros {

int64_t ReadInt64FromFile(const char* path) {
  ScopedFd fd(open(path, O_RDONLY));
  PCHECK(fd.is_valid());

  char buf[32] = {};
  ssize_t count = read(fd, buf, sizeof(buf));
  CHECK_LT(count, sizeof(buf));

  int64_t value = 0;
  CHECK(absl::SimpleAtoi(buf, &value));
  return value;
}

class IntelNPULoad {
 public:
  bool Init() {
    if (access(kBusyTimePath, R_OK) != 0) {
      LOG(ERROR) << "Failed to read " << kBusyTimePath;
      return false;
    }
    Update();
    return true;
  }

  std::string GetValue() {
    Update();
    CHECK_NE(curr_timestamp_, prev_timestamp_);
    double dt = absl::ToDoubleNanoseconds(curr_timestamp_ - prev_timestamp_);
    double db = absl::ToDoubleNanoseconds(curr_busy_time_ - prev_busy_time_);
    double util = std::min(db / dt, 1.0);
    return absl::StrFormat("%.1f%%", util * 100);
  }

 private:
  // The file content is a single integer that represent the busy time since
  // boot in microseconds.
  // TODO: b:380808338 - Resolve this dynamically from driver or udev instead of
  // hard-coding it statically.
  static constexpr char kBusyTimePath[] =
      "/sys/devices/pci0000:00/0000:00:0b.0/npu_busy_time_us";

  void Update() {
    auto now = absl::Now();
    if (now == curr_timestamp_) {
      return;
    }

    prev_timestamp_ = curr_timestamp_;
    prev_busy_time_ = curr_busy_time_;

    curr_timestamp_ = now;
    curr_busy_time_ = absl::Microseconds(ReadInt64FromFile(kBusyTimePath));
  }

  absl::Time prev_timestamp_;
  absl::Duration prev_busy_time_;

  absl::Time curr_timestamp_;
  absl::Duration curr_busy_time_;
};

// TODO: b:380808338 - Make this configurable with command line flags.
constexpr absl::Duration kSampleInterval = absl::Seconds(1);

// TODO: b:380808338 - Support output in JSONL.
// TODO: b:380808338 - Support output in CSV.
void RunSyslogLoop() {
  // TODO: b:380808338 - Support CPU load.
  // TODO: b:380808338 - Support GPU load.
  // TODO: b:380808338 - Support MTK NPU load.
  IntelNPULoad npu;
  CHECK(npu.Init());
  while (true) {
    absl::SleepFor(kSampleInterval);
    LOG(INFO) << "npu: " << npu.GetValue();
  }
}

}  // namespace tflite::cros

int main() {
  tflite::cros::RunSyslogLoop();
  return 0;
}
