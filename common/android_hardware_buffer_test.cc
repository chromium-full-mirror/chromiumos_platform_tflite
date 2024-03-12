/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <gtest/gtest.h>

#include <algorithm>

#include "common/android_hardware_buffer.h"

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

  // Decrease the refernce count to zero so lock() should fail.
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

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
