/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "common/android_hardware_buffer.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <map>
#include <new>
#include <utility>

namespace {

class ScopedFd {
 public:
  static const int kInvalidFd = -1;

  explicit ScopedFd(int fd = kInvalidFd) : fd_(fd) {}

  ScopedFd(ScopedFd&& other) { *this = std::move(other); }

  ScopedFd& operator=(ScopedFd&& other) {
    reset(other.fd_);
    other.fd_ = kInvalidFd;
    return *this;
  }

  ~ScopedFd() { reset(); }

  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;

  void reset(int fd = kInvalidFd) {
    if (fd_ >= 0) {
      close(fd_);
    }
    fd_ = fd;
  }

  int get() const { return fd_; }

  operator int() const { return get(); }

  bool is_valid() { return fd_ != kInvalidFd; }

 private:
  int fd_;
};

class Allocator {
 public:
  static Allocator* GetInstance() {
    // Leaky singleton.
    alignas(Allocator) static uint8_t storage[sizeof(Allocator)];
    static Allocator* instance = new (storage) Allocator();
    return instance;
  }

  // TODO(shik): Use dma_heap or udmabuf to create a real dmabuf.
  int Allocate(const AHardwareBuffer_Desc* _Nonnull desc,
               AHardwareBuffer* _Nullable* _Nonnull out_buffer) {
    if (!IsSupported(desc)) {
      return -EINVAL;
    }
    uint32_t size = desc->width;
    // TODO(shik): Use a more descriptive name to make debugging easier.
    ScopedFd fd(memfd_create("ahwb", MFD_CLOEXEC | MFD_ALLOW_SEALING));
    if (!fd.is_valid()) {
      return -errno;
    }
    if (ftruncate64(fd, size) != 0) {
      return -errno;
    }
    if (fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW) != 0) {
      return -errno;
    }
    void* data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
      return -errno;
    }

    Buffer buffer = {
        .fd = std::move(fd),
        .data = data,
        .size = size,
        .ref_count = 1,
        .desc = *desc,
    };
    *out_buffer = reinterpret_cast<AHardwareBuffer*>(data);
    buffers_.emplace(*out_buffer, std::move(buffer));
    return 0;
  }

  void Acquire(AHardwareBuffer* _Nonnull buffer) {
    auto it = buffers_.find(buffer);
    if (it == buffers_.end()) {
      return;
    }
    it->second.ref_count++;
  }

  void Release(AHardwareBuffer* _Nonnull buffer) {
    auto it = buffers_.find(buffer);
    if (it == buffers_.end()) {
      return;
    }
    if (--it->second.ref_count == 0) {
      buffers_.erase(it);
    }
  }

  void Describe(const AHardwareBuffer* _Nonnull buffer,
                AHardwareBuffer_Desc* _Nonnull out_desc) {
    *out_desc = buffers_.at(buffer).desc;
  }

  int Lock(AHardwareBuffer* _Nonnull buffer,
           uint64_t usage,
           int32_t fence,
           const ARect* _Nullable rect,
           void* _Nullable* _Nonnull out_virtual_address) {
    const uint64_t kNonCpuUsageMask = ~(AHARDWAREBUFFER_USAGE_CPU_READ_MASK |
                                        AHARDWAREBUFFER_USAGE_CPU_WRITE_MASK);
    auto it = buffers_.find(buffer);
    // TODO(shik): Support fence.
    if (it == buffers_.end() || (usage & kNonCpuUsageMask) != 0 || fence >= 0 ||
        rect != nullptr) {
      return -EINVAL;
    }
    auto& buf = it->second;
    // TODO(shik): Call proper ioctl() with DMA_BUF_IOCTL_SYNC.
    *out_virtual_address = buf.data;
    return 0;
  }

  int Unlock(AHardwareBuffer* _Nonnull buffer, int32_t* _Nullable fence) {
    if (buffers_.find(buffer) == buffers_.end()) {
      return -EINVAL;
    }
    if (fence != nullptr) {
      *fence = -1;
    }
    // TODO(shik): Call proper ioctl() with DMA_BUF_IOCTL_SYNC.
    return 0;
  }

  bool IsSupported(const AHardwareBuffer_Desc* _Nonnull desc) {
    // TODO(shik): Check usage as well.
    return desc->format == AHARDWAREBUFFER_FORMAT_BLOB && desc->height == 1 &&
           desc->layers == 1 && desc->rfu0 == 0 && desc->rfu1 == 0;
  }

  const native_handle_t* GetNativeHandle(const AHardwareBuffer* buffer) {
    // TODO(shik): Implement this.
    return nullptr;
  }

  int CreateFromHandle(const AHardwareBuffer_Desc* _Nonnull desc,
                       const native_handle_t* _Nonnull handle,
                       int32_t method,
                       AHardwareBuffer* _Nullable* _Nonnull out_buffer) {
    // TODO(shik): Implement this.
    return -ENOSYS;
  }

 private:
  struct Buffer {
    ScopedFd fd;
    void* data;
    uint32_t size;
    int ref_count;
    AHardwareBuffer_Desc desc;
  };

  std::map<const AHardwareBuffer*, Buffer> buffers_;
};

}  // namespace

int AHardwareBuffer_allocate(const AHardwareBuffer_Desc* _Nonnull desc,
                             AHardwareBuffer* _Nullable* _Nonnull outBuffer) {
  return Allocator::GetInstance()->Allocate(desc, outBuffer);
}

void AHardwareBuffer_acquire(AHardwareBuffer* _Nonnull buffer) {
  return Allocator::GetInstance()->Acquire(buffer);
}

void AHardwareBuffer_release(AHardwareBuffer* _Nonnull buffer) {
  return Allocator::GetInstance()->Release(buffer);
}

void AHardwareBuffer_describe(const AHardwareBuffer* _Nonnull buffer,
                              AHardwareBuffer_Desc* _Nonnull outDesc) {
  return Allocator::GetInstance()->Describe(buffer, outDesc);
}

int AHardwareBuffer_lock(AHardwareBuffer* _Nonnull buffer,
                         uint64_t usage,
                         int32_t fence,
                         const ARect* _Nullable rect,
                         void* _Nullable* _Nonnull outVirtualAddress) {
  return Allocator::GetInstance()->Lock(buffer, usage, fence, rect,
                                        outVirtualAddress);
}

int AHardwareBuffer_unlock(AHardwareBuffer* _Nonnull buffer,
                           int32_t* _Nullable fence) {
  return Allocator::GetInstance()->Unlock(buffer, fence);
}

int AHardwareBuffer_isSupported(const AHardwareBuffer_Desc* _Nonnull desc) {
  return Allocator::GetInstance()->IsSupported(desc);
}

const native_handle_t* AHardwareBuffer_getNativeHandle(
    const AHardwareBuffer* _Nonnull buffer) {
  return Allocator::GetInstance()->GetNativeHandle(buffer);
}

int AHardwareBuffer_createFromHandle(
    const AHardwareBuffer_Desc* _Nonnull desc,
    const native_handle_t* _Nonnull handle,
    int32_t method,
    AHardwareBuffer* _Nullable* _Nonnull outBuffer) {
  return Allocator::GetInstance()->CreateFromHandle(desc, handle, method,
                                                    outBuffer);
}
