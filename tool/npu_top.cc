/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <fcntl.h>
#include <unistd.h>

#include <cinttypes>
#include <fstream>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
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

std::string ReadFirstLineFromFile(const char* path) {
  std::ifstream file(path);
  CHECK(file.is_open());

  std::string line;
  CHECK(std::getline(file, line));

  return line;
}

// Returns the duration elapsed since the first call with a monotonically
// non-decreasing clock. Ideal for measuring elapsed time.
absl::Duration SteadyClockTicks() {
  static const auto origin = std::chrono::steady_clock::now();
  return absl::FromChrono(std::chrono::steady_clock::now() - origin);
}

template <typename T>
std::string FormatUtilization(T busy, T total) {
  if (busy < 0) {
    LOG(WARNING) << "busy < 0, the counter might be wrapped";
    busy = 0;
  }
  CHECK_GT(total, 0);
  double util = static_cast<double>(busy) / static_cast<double>(total);
  return absl::StrFormat("%.1f%%", util * 100);
}

absl::Status CheckFileReadable(const char* path) {
  if (access(path, R_OK) != 0) {
    return absl::NotFoundError(absl::StrCat("Failed to read ", path));
  }
  return absl::OkStatus();
}

// Abstract base class representing a data source.
class Source {
 public:
  Source() {}
  virtual ~Source() = default;

  virtual absl::Status CheckAvailability() = 0;
  virtual void Init() = 0;
  virtual std::string GetValue() = 0;
};

class CpuLoad : public Source {
 public:
  absl::Status CheckAvailability() override {
    return CheckFileReadable(kProcStatPath);
  }

  void Init() override { Update(); }

  std::string GetValue() override {
    Update();
    return FormatUtilization(curr_data_.busy - prev_data_.busy,
                             curr_data_.total - prev_data_.total);
  }

 private:
  static constexpr char kProcStatPath[] = "/proc/stat";

  struct Data {
    int64_t busy = 0;
    int64_t total = 0;
  };

  enum class ProcStatIndex {
    kUser,
    kNice,
    kSystem,
    kIdle,
    kIowait,
    kIrq,
    KSoftirq,
    kSteal,
    kGuest,
    kGuestNice,
  };

  void Update() {
    Data data = ParseProcStat();
    if (data.total == curr_data_.total) {
      return;
    }
    prev_data_ = curr_data_;
    curr_data_ = data;
  }

  static Data ParseProcStat() {
    // The content of the /proc/stat file begins like this:
    // ```
    // # head -n 1 /proc/stat
    // cpu  247132 21 94411 31505941 13513 0 895 0 0 0
    // ```
    //
    // The numbers represent the time, measured in USER_HZ units, that the
    // system has spent in various states. For more details, refer to `man 5
    // proc`.
    std::string line = ReadFirstLineFromFile(kProcStatPath);
    std::vector<std::string_view> cols =
        absl::StrSplit(line, absl::ByChar(' '), absl::SkipWhitespace());

    // The header ("cpu") and the first four numbers (user, nice, system, idle)
    // are guaranteed to be present in the file.
    CHECK_GE(cols.size(), 5);
    CHECK_EQ(cols[0], "cpu");

    std::vector<int64_t> times;
    for (size_t i = 1; i < cols.size(); ++i) {
      int64_t time = 0;
      CHECK(absl::SimpleAtoi(cols[i], &time));
      times.push_back(time);
    }

    auto get = [&](ProcStatIndex idx) -> int64_t {
      size_t i = static_cast<size_t>(idx);
      return i < times.size() ? times[i] : 0;
    };

    // Follow the same calculation logic used by the `procps` package,
    // which provides tools like `top` and `ps`. For reference:
    // https://gitlab.com/procps-ng/procps/-/blob/5ce26ac301d6133e3e86aa9da89035bd8cd89886/library/stat.c#L556-585
    using enum ProcStatIndex;
    int64_t idle = get(kIdle) + get(kIowait);
    int64_t busy = get(kUser) + get(kNice) + get(kSystem) + get(kIrq) +
                   get(KSoftirq) + get(kSteal);
    return Data{.busy = busy, .total = idle + busy};
  }

  Data prev_data_;
  Data curr_data_;
};

class IntelNpuLoad : public Source {
 public:
  absl::Status CheckAvailability() override {
    return CheckFileReadable(kBusyTimePath);
  }

  void Init() override { Update(); }

  std::string GetValue() override {
    Update();
    double db = absl::ToDoubleNanoseconds(curr_busy_time_ - prev_busy_time_);
    double dt = absl::ToDoubleNanoseconds(curr_ticks_ - prev_ticks_);
    return FormatUtilization(db, dt);
  }

 private:
  // The file content is a single integer that represent the busy time since
  // boot in microseconds.
  // TODO: b:380808338 - Resolve this dynamically from driver or udev instead of
  // hard-coding it statically.
  static constexpr char kBusyTimePath[] =
      "/sys/devices/pci0000:00/0000:00:0b.0/npu_busy_time_us";

  void Update() {
    auto ticks = SteadyClockTicks();
    if (ticks == curr_ticks_) {
      return;
    }

    prev_ticks_ = curr_ticks_;
    prev_busy_time_ = curr_busy_time_;

    curr_ticks_ = ticks;
    curr_busy_time_ = absl::Microseconds(ReadInt64FromFile(kBusyTimePath));
  }

  absl::Duration prev_ticks_;
  absl::Duration prev_busy_time_;

  absl::Duration curr_ticks_;
  absl::Duration curr_busy_time_;
};

class IntelGpuLoad : public Source {
 public:
  absl::Status CheckAvailability() override {
    auto status = CheckFileReadable(kI915EngineInfoPath);
    if (!status.ok()) {
      return status;
    }

    return ParseGpuBusyTime().status();
  }

  void Init() override { Update(); }

  std::string GetValue() override {
    Update();
    double db = absl::ToDoubleNanoseconds(curr_busy_time_ - prev_busy_time_);
    double dt = absl::ToDoubleNanoseconds(curr_ticks_ - prev_ticks_);
    return FormatUtilization(db, dt);
  }

 private:
  static constexpr char kI915EngineInfoPath[] =
      "/run/debugfs_gpu/i915_engine_info";

  void Update() {
    auto ticks = SteadyClockTicks();
    if (ticks == curr_ticks_) {
      return;
    }

    absl::StatusOr<absl::Duration> busy_time = ParseGpuBusyTime();
    CHECK(busy_time.ok());

    prev_ticks_ = curr_ticks_;
    prev_busy_time_ = curr_busy_time_;

    curr_ticks_ = ticks;
    curr_busy_time_ = *busy_time;
  }

  static absl::StatusOr<absl::Duration> ParseGpuBusyTime() {
    // Cumulative runtime information has been available on ChromeOS since
    // Kernel v5.10. In earlier versions, the engine info file may exist, but it
    // will not include the "Runtime:" stat.
    //
    // The implementation can be found in intel_engine_dump() at
    // drivers/gpu/drm/i915/gt/intel_engine_cs.c.
    std::ifstream file(kI915EngineInfoPath);
    if (!file.is_open()) {
      return absl::NotFoundError(
          absl::StrCat("Failed to open ", kI915EngineInfoPath));
    }

    std::string line;
    absl::Duration busy_time;
    bool found_runtime = false;
    while (std::getline(file, line)) {
      if (!absl::StartsWith(line, "\tRuntime: ")) {
        continue;
      }

      // Multiple engines can run concurrently, causing the busy time to
      // increase faster than wall-clock time. For simplicity, we aggregate them
      // and clamp the usage percentage to 100% in the frontend. If needed, we
      // can consider exposing the per-engine breakdown.
      int64_t engine_runtime_ms = 0;
      if (sscanf(line.c_str(), "\tRuntime: %" PRId64 "ms",
                 &engine_runtime_ms) != 1) {
        continue;
      }
      busy_time += absl::Milliseconds(engine_runtime_ms);
      found_runtime = true;
    }

    if (!found_runtime) {
      return absl::NotFoundError(
          absl::StrCat("No runtime information in ", kI915EngineInfoPath));
    }

    return busy_time;
  }

  absl::Duration prev_ticks_;
  absl::Duration prev_busy_time_;

  absl::Duration curr_ticks_;
  absl::Duration curr_busy_time_;
};

class ArmMaliGpuLoad : public Source {
 public:
  absl::Status CheckAvailability() override {
    // TODO: b:380275060 - The dvfs utilization is available but wrong on rauru
    // board now.
    return CheckFileReadable(kDvfsUtilizationPath);
  }

  void Init() override { Update(); }

  std::string GetValue() override {
    Update();
    return FormatUtilization(curr_data_.busy - prev_data_.busy,
                             curr_data_.total - prev_data_.total);
  }

 private:
  static constexpr char kDvfsUtilizationPath[] =
      "/sys/kernel/debug/mali0/dvfs_utilization";

  struct Data {
    uint32_t busy = 0;
    uint32_t total = 0;
  };

  void Update() {
    Data data = ParseDvfsUtilization();
    if (data.total == curr_data_.total) {
      return;
    }

    prev_data_ = curr_data_;
    curr_data_ = data;
  }

  static Data ParseDvfsUtilization() {
    std::string content = ReadFirstLineFromFile(kDvfsUtilizationPath);

    // Example output:
    // ```
    // # cat /sys/kernel/debug/mali0/dvfs_utilization
    // busy_time: 114016010 idle_time: 590833816
    // ```
    //
    // The time value is in units of 256ns and is of uint32_t, so the wraparound
    // may happend and will be handled implicitly when taking difference.
    //
    // Reference: Mali DDK base_get_gpu_utilization_info() in
    // https://source.corp.google.com/h/chrome-internal/chromeos/superproject/+/main:src/partner_private/mali-ddk-avalon/base/tests/common/mali_base_helpers_power.h
    uint32_t busy = 0;
    uint32_t idle = 0;
    int scanned =
        sscanf(content.c_str(), "busy_time: %" PRIu32 " idle_time: %" PRIu32,
               &busy, &idle);
    CHECK_EQ(scanned, 2) << "Failed to parse DVFS utilization data: "
                         << content;

    return Data{.busy = busy, .total = busy + idle};
  }

  Data prev_data_;
  Data curr_data_;
};

class Top {
 public:
  Top() {
    TryAddSource<CpuLoad>("cpu");

    // TODO: b:380808338 - Create a generic GpuLoad that wraps platform specific
    // sources.
    TryAddSource<IntelGpuLoad>("gpu");
    TryAddSource<ArmMaliGpuLoad>("gpu");

    TryAddSource<IntelNpuLoad>("npu");
    // TODO: b:380808338 - Support MTK NPU load.
  }

  // TODO: b:380808338 - Support output in JSONL.
  // TODO: b:380808338 - Support output in CSV.
  void RunSyslogLoop() {
    for (auto& entry : source_entries) {
      entry.source->Init();
    }
    while (true) {
      absl::SleepFor(kSampleInterval);
      std::vector<std::string> pieces;
      for (auto& entry : source_entries) {
        pieces.push_back(
            absl::StrFormat("%s: %s", entry.name, entry.source->GetValue()));
      }
      LOG(INFO) << absl::StrJoin(pieces, " ");
    }
  }

 private:
  // TODO: b:380808338 - Make this configurable with command line flags.
  static constexpr absl::Duration kSampleInterval = absl::Seconds(1);

  struct SourceEntry {
    std::string name;
    std::unique_ptr<Source> source;
  };

  template <std::derived_from<Source> T>
  void TryAddSource(const std::string& name) {
    auto source = std::make_unique<T>();
    absl::Status avail = source->CheckAvailability();
    if (avail.ok()) {
      source_entries.push_back({.name = name, .source = std::move(source)});
    } else {
      VLOG(1) << name << " is not available: " << avail.message();
    }
  }

  std::vector<SourceEntry> source_entries;
};

}  // namespace tflite::cros

int main() {
  tflite::cros::Top().RunSyslogLoop();
  return 0;
}
