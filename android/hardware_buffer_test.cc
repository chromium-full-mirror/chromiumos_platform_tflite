/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "android/hardware_buffer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "android/native_handle_util.h"
#include "common/log.h"

namespace {}  // namespace

constexpr int kSize = 4096;
constexpr uint64_t kUsage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                            AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;

TEST(AHardwareBuffer, BasicOps) {
  const AHardwareBuffer_Desc desc = {
      .width = kSize,
      .height = 1,
      .layers = 1,
      .format = AHARDWAREBUFFER_FORMAT_BLOB,
      .usage = kUsage,
      .stride = kSize,
  };
  ASSERT_TRUE(AHardwareBuffer_isSupported(&desc));

  // Allocate a buffer.
  AHardwareBuffer* buffer = nullptr;
  ASSERT_EQ(AHardwareBuffer_allocate(&desc, &buffer), 0);
  ASSERT_NE(buffer, nullptr);

  // Describe it and do a simple comparasion with original description.
  AHardwareBuffer_Desc desc2 = {};
  AHardwareBuffer_describe(buffer, &desc2);
  EXPECT_EQ(desc.width, desc2.width);

  // Lock and write some data.
  void* addr = nullptr;
  ASSERT_EQ(AHardwareBuffer_lock(buffer, kUsage, /*fence=*/-1, /*rect=*/nullptr,
                                 &addr),
            0);
  ASSERT_NE(addr, nullptr);
  memset(addr, 0xFF, kSize);
  ASSERT_EQ(AHardwareBuffer_unlock(buffer, /*fence=*/nullptr), 0);
  addr = nullptr;

  // Play with reference counting.
  AHardwareBuffer_acquire(buffer);
  AHardwareBuffer_release(buffer);

  // Lock again, the data should be the previously written value.
  ASSERT_EQ(AHardwareBuffer_lock(buffer, kUsage, /*fence=*/-1, /*rect=*/nullptr,
                                 &addr),
            0);
  ASSERT_NE(addr, nullptr);
  EXPECT_EQ(std::count(static_cast<uint8_t*>(addr),
                       static_cast<uint8_t*>(addr) + kSize, 0xFF),
            kSize);
  ASSERT_EQ(AHardwareBuffer_unlock(buffer, /*fence=*/nullptr), 0);
  addr = nullptr;

  // Decrease the reference count to zero so lock() should fail.
  AHardwareBuffer_release(buffer);
  ASSERT_NE(AHardwareBuffer_lock(buffer, kUsage, /*fence=*/-1, /*rect=*/nullptr,
                                 &addr),
            0);
  ASSERT_EQ(addr, nullptr);
}

TEST(AHardwareBuffer, MultipleBuffers) {
  const int kNumBuffers = 10;
  AHardwareBuffer* buffers[kNumBuffers] = {};

  // Allocate some buffers.
  const AHardwareBuffer_Desc desc = {
      .width = kSize,
      .height = 1,
      .layers = 1,
      .format = AHARDWAREBUFFER_FORMAT_BLOB,
      .usage = kUsage,
      .stride = kSize,
  };
  for (auto& buffer : buffers) {
    ASSERT_EQ(AHardwareBuffer_allocate(&desc, &buffer), 0);
    ASSERT_NE(buffer, nullptr);
  }

  // Execrise and release each of them.
  for (auto& buffer : buffers) {
    void* addr = nullptr;
    ASSERT_EQ(AHardwareBuffer_lock(buffer, kUsage, /*fence=*/-1,
                                   /*rect=*/nullptr, &addr),
              0);
    ASSERT_NE(addr, nullptr);
    ASSERT_EQ(AHardwareBuffer_unlock(buffer, /*fence=*/nullptr), 0);
    addr = nullptr;

    AHardwareBuffer_release(buffer);
    ASSERT_NE(AHardwareBuffer_lock(buffer, kUsage, /*fence=*/-1,
                                   /*rect=*/nullptr, &addr),
              0);
    ASSERT_EQ(addr, nullptr);
  }
}

TEST(AHardwareBuffer, NativeHandleRegister) {
  const AHardwareBuffer_Desc desc = {
      .width = kSize,
      .height = 1,
      .layers = 1,
      .format = AHARDWAREBUFFER_FORMAT_BLOB,
      .usage = kUsage,
      .stride = kSize,
  };

  // Allocate a buffer.
  AHardwareBuffer* original = nullptr;
  ASSERT_EQ(AHardwareBuffer_allocate(&desc, &original), 0);
  ASSERT_NE(original, nullptr);

  // Lock and write some data on the original buffer.
  // TODO(shik): Consider writing random data instead of a fixed one, to avoid
  // left over from other test cases match accidentally.
  void* addr = nullptr;
  ASSERT_EQ(AHardwareBuffer_lock(original, kUsage, /*fence=*/-1,
                                 /*rect=*/nullptr, &addr),
            0);
  ASSERT_NE(addr, nullptr);
  memset(addr, 0xFF, kSize);
  ASSERT_EQ(AHardwareBuffer_unlock(original, /*fence=*/nullptr), 0);
  addr = nullptr;

  // Take fd out of the handle and release the original buffer. We now owns the
  // memory region with the taken fd.
  const native_handle_t* original_handle =
      AHardwareBuffer_getNativeHandle(original);
  ASSERT_NE(original_handle, nullptr);
  ASSERT_EQ(original_handle->numFds, 1);
  int fd = dup(original_handle->data[0]);
  ASSERT_NE(fd, -1);
  AHardwareBuffer_release(original);

  // Construct a new handle with the fd.
  auto handle = tflite::cros::OwnedNativeHandle(static_cast<native_handle_t*>(
      std::malloc(sizeof(native_handle_t) + sizeof(int))));
  *handle = {
      .version = sizeof(native_handle_t),
      .numFds = 1,
      .numInts = 0,
  };
  handle->data[0] = fd;

  // Create another buffer using the handle with the "register" method.
  AHardwareBuffer* registered = nullptr;
  ASSERT_EQ(
      AHardwareBuffer_createFromHandle(
          &desc, handle.get(),
          AHARDWAREBUFFER_CREATE_FROM_HANDLE_METHOD_REGISTER, &registered),
      0);
  ASSERT_NE(registered, nullptr);

  // Lock and read the registered buffer, the data should be the same.
  ASSERT_EQ(AHardwareBuffer_lock(registered, kUsage, /*fence=*/-1,
                                 /*rect=*/nullptr, &addr),
            0);
  ASSERT_NE(addr, nullptr);
  EXPECT_EQ(std::count(static_cast<uint8_t*>(addr),
                       static_cast<uint8_t*>(addr) + kSize, 0xFF),
            kSize);
  ASSERT_EQ(AHardwareBuffer_unlock(registered, /*fence=*/nullptr), 0);
  addr = nullptr;

  // Release the registered buffer. The fd should be closed now, so trying to
  // close it again is expected to fail.
  AHardwareBuffer_release(registered);
  EXPECT_EQ(close(fd), -1);
}

TEST(AHardwareBuffer, NativeHandleClone) {
  const AHardwareBuffer_Desc desc = {
      .width = kSize,
      .height = 1,
      .layers = 1,
      .format = AHARDWAREBUFFER_FORMAT_BLOB,
      .usage = kUsage,
      .stride = kSize,
  };

  // Allocate a buffer and get handle from it.
  AHardwareBuffer* original = nullptr;
  ASSERT_EQ(AHardwareBuffer_allocate(&desc, &original), 0);
  ASSERT_NE(original, nullptr);
  const native_handle_t* handle = AHardwareBuffer_getNativeHandle(original);
  ASSERT_NE(handle, nullptr);

  // Create another buffer using the handle with the "clone" method.
  AHardwareBuffer* cloned = nullptr;
  ASSERT_EQ(AHardwareBuffer_createFromHandle(
                &desc, handle, AHARDWAREBUFFER_CREATE_FROM_HANDLE_METHOD_CLONE,
                &cloned),
            0);
  ASSERT_NE(cloned, nullptr);

  // The cloned handle should have a different fd.
  const native_handle_t* cloned_handle =
      AHardwareBuffer_getNativeHandle(cloned);
  EXPECT_NE(cloned_handle->data[0], handle->data[0]);

  // Lock and write some data on the original buffer.
  void* addr = nullptr;
  ASSERT_EQ(AHardwareBuffer_lock(original, kUsage, /*fence=*/-1,
                                 /*rect=*/nullptr, &addr),
            0);
  ASSERT_NE(addr, nullptr);
  memset(addr, 0xFF, kSize);
  ASSERT_EQ(AHardwareBuffer_unlock(original, /*fence=*/nullptr), 0);
  addr = nullptr;

  // Lock and read the cloned buffer, the data should be the same.
  ASSERT_EQ(AHardwareBuffer_lock(cloned, kUsage, /*fence=*/-1,
                                 /*rect=*/nullptr, &addr),
            0);
  ASSERT_NE(addr, nullptr);
  EXPECT_EQ(std::count(static_cast<uint8_t*>(addr),
                       static_cast<uint8_t*>(addr) + kSize, 0xFF),
            kSize);
  ASSERT_EQ(AHardwareBuffer_unlock(cloned, /*fence=*/nullptr), 0);
  addr = nullptr;

  // Release the original buffer and check we can still describe the cloned one.
  AHardwareBuffer_release(original);
  AHardwareBuffer_Desc desc2 = {};
  AHardwareBuffer_describe(cloned, &desc2);
  EXPECT_EQ(desc.width, desc2.width);

  // Release the cloned buffer. The lock should fail.
  AHardwareBuffer_release(cloned);
  ASSERT_NE(AHardwareBuffer_lock(cloned, kUsage, /*fence=*/-1, /*rect=*/nullptr,
                                 &addr),
            0);
  ASSERT_EQ(addr, nullptr);
}

TEST(AHardwareBuffer, LockUsage) {
  // Allocate a read-only buffer.
  const AHardwareBuffer_Desc desc = {
      .width = kSize,
      .height = 1,
      .layers = 1,
      .format = AHARDWAREBUFFER_FORMAT_BLOB,
      .usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,
      .stride = kSize,
  };
  AHardwareBuffer* buffer = nullptr;
  ASSERT_EQ(AHardwareBuffer_allocate(&desc, &buffer), 0);
  ASSERT_NE(buffer, nullptr);

  // Can be locked for read.
  void* addr = nullptr;
  ASSERT_EQ(AHardwareBuffer_lock(buffer, AHARDWAREBUFFER_USAGE_CPU_READ_RARELY,
                                 /*fence=*/-1, /*rect=*/nullptr, &addr),
            0);
  ASSERT_NE(addr, nullptr);
  ASSERT_EQ(AHardwareBuffer_unlock(buffer, /*fence=*/nullptr), 0);
  addr = nullptr;

  // Cannot be locked for write.
  ASSERT_NE(AHardwareBuffer_lock(buffer, AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY,
                                 /*fence=*/-1, /*rect=*/nullptr, &addr),
            0);
  ASSERT_EQ(addr, nullptr);

  // Release the buffer. The lock should fail.
  AHardwareBuffer_release(buffer);
  ASSERT_NE(AHardwareBuffer_lock(buffer, AHARDWAREBUFFER_USAGE_CPU_READ_RARELY,
                                 /*fence=*/-1, /*rect=*/nullptr, &addr),
            0);
  ASSERT_EQ(addr, nullptr);
}

TEST(AHardwareBuffer, ManyBuffers) {
  auto get_vm_size = [] {
    constexpr char kProcSelfStatmPath[] = "/proc/self/statm";

    std::ifstream file(kProcSelfStatmPath);
    CHECK(file.is_open());

    // The content of `/proc/self/statm` is
    //
    // <size> <resident> <shared> <text> <lib> <data> <dt>
    //
    // The numbers represent the memory usage, measured in pages. For more
    // details, refer to `man 5 proc`.
    std::string line;
    CHECK(std::getline(file, line));

    std::vector<std::string_view> cols =
        absl::StrSplit(line, absl::ByChar(' '), absl::SkipWhitespace());
    CHECK_EQ(cols.size(), 7);

    uint64_t vm_size;
    CHECK(absl::SimpleAtoi(cols[0], &vm_size));
    return vm_size;
  };

  constexpr int kNumBuffers = 1024;

  // Each buffer is 4KiB (or 1 page in most cases)
  const AHardwareBuffer_Desc desc = {
      .width = kSize,
      .height = 1,
      .layers = 1,
      .format = AHARDWAREBUFFER_FORMAT_BLOB,
      .usage = kUsage,
      .stride = kSize,
  };

  // Allocate kNumBuffers buffers and check the memory usage does not increase.
  uint64_t memory_usage = 0;
  for (int i = 0; i < kNumBuffers; ++i) {
    AHardwareBuffer* buffer = nullptr;
    ASSERT_EQ(AHardwareBuffer_allocate(&desc, &buffer), 0);
    ASSERT_NE(buffer, nullptr);

    AHardwareBuffer_release(buffer);
    if (i == 0) {
      memory_usage = get_vm_size();
    } else {
      // Add a 1 page tolerance to accomodate ahwb allocator's internal memory
      // overhead.
      EXPECT_LE(get_vm_size(), memory_usage + 1);
    }
  }
}

TEST(AHardwareBuffer, GetId) {
  const AHardwareBuffer_Desc desc = {
      .width = kSize,
      .height = 1,
      .layers = 1,
      .format = AHARDWAREBUFFER_FORMAT_BLOB,
      .usage = kUsage,
      .stride = kSize,
  };

  AHardwareBuffer* buffer1 = nullptr;
  ASSERT_EQ(AHardwareBuffer_allocate(&desc, &buffer1), 0);
  ASSERT_NE(buffer1, nullptr);

  // Id should not change.
  uint64_t id1 = 0;
  ASSERT_EQ(AHardwareBuffer_getId(buffer1, &id1), 0);
  uint64_t id1_second_try = 0;
  ASSERT_EQ(AHardwareBuffer_getId(buffer1, &id1_second_try), 0);
  EXPECT_EQ(id1, id1_second_try);

  AHardwareBuffer* buffer2 = nullptr;
  ASSERT_EQ(AHardwareBuffer_allocate(&desc, &buffer2), 0);
  ASSERT_NE(buffer2, nullptr);

  // Id should not collide.
  uint64_t id2 = 0;
  ASSERT_EQ(AHardwareBuffer_getId(buffer2, &id2), 0);
  EXPECT_NE(id2, id1);

  AHardwareBuffer_release(buffer1);
  AHardwareBuffer_release(buffer2);

  // We should not be able to get the id from a released buffer.
  uint64_t id2_second_try = 0;
  ASSERT_NE(AHardwareBuffer_getId(buffer2, &id2_second_try), 0);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
