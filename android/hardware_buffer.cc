/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "android/hardware_buffer.h"

#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/udmabuf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <map>
#include <memory>
#include <new>
#include <utility>

#include "android/native_handle_util.h"
#include "common/log.h"
#include "common/scoped_fd.h"

namespace tflite::cros {
namespace {

bool IsDmaBuf(int fd) {
  // Do a no-op sync intentionally.
  dma_buf_sync sync = {.flags = 0};
  // TODO(shik): Handle EINTR.
  int ret = ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);

  // The ioctl() will get ENOTTY when the specified request does not apply to
  // the kind of object that the file descriptor references.
  bool not_applicable = ret == -1 && errno == ENOTTY;
  return !not_applicable;
}

bool IsUdmabufAvailable() {
  static bool avail = [] {
    ScopedFd fd(open("/dev/udmabuf", O_RDWR));
    if (!fd.is_valid()) {
      LOGF(INFO) << "/dev/udmabuf is not available";
    }
    return fd.is_valid();
  }();
  return avail;
}

ScopedFd AllocateWithMemfd(size_t size) {
  // TODO(shik): Use a more descriptive name to make debugging easier.
  ScopedFd fd(memfd_create("ahwb", MFD_CLOEXEC | MFD_ALLOW_SEALING));
  if (!fd.is_valid()) {
    PLOGF(ERROR) << "memfd_create() failed";
    return {};
  }

  if (ftruncate64(fd, size) != 0) {
    PLOGF(ERROR) << "ftruncate64() failed";
    return {};
  }

  if (fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW) != 0) {
    PLOGF(ERROR) << "fcntl() failed";
    return {};
  }

  if (!IsUdmabufAvailable()) {
    return fd;
  }

  ScopedFd udmabuf(open("/dev/udmabuf", O_RDWR));
  if (!udmabuf.is_valid()) {
    PLOGF(ERROR) << "open /dev/udmabuf failed";
    return {};
  }

  udmabuf_create create = {
      .memfd = static_cast<__u32>(fd.get()),
      .flags = UDMABUF_FLAGS_CLOEXEC,
      .offset = 0,
      .size = size,
  };
  ScopedFd dmabuf_fd(ioctl(udmabuf, UDMABUF_CREATE, &create));
  if (!dmabuf_fd.is_valid()) {
    PLOGF(ERROR) << "ioctl() for UDMABUF_CREATE failed";
    return {};
  }

  return dmabuf_fd;
}

uint64_t SyncFlagsFromUsageMask(uint64_t usage) {
  uint64_t flags = 0;
  if (usage & AHARDWAREBUFFER_USAGE_CPU_READ_MASK) {
    flags |= DMA_BUF_SYNC_READ;
  }
  if (usage & AHARDWAREBUFFER_USAGE_CPU_WRITE_MASK) {
    flags |= DMA_BUF_SYNC_WRITE;
  }
  return flags;
}

class Allocator {
 public:
  static Allocator* GetInstance() {
    // Leaky singleton.
    alignas(Allocator) static uint8_t storage[sizeof(Allocator)];
    static Allocator* instance = new (storage) Allocator();
    return instance;
  }

  // TODO(shik): Add another backend using dma_heap.
  int Allocate(const AHardwareBuffer_Desc* _Nonnull desc,
               AHardwareBuffer* _Nullable* _Nonnull out_buffer) {
    if (!IsSupported(desc)) {
      LOGF(ERROR) << "Unsupported desc";
      return -EINVAL;
    }

    // Ensure the allocated size is page-aligned.
    size_t page = sysconf(_SC_PAGESIZE);
    size_t size = (desc->width + page - 1) / page * page;

    ScopedFd fd = AllocateWithMemfd(size);
    if (!fd.is_valid()) {
      return -EINVAL;
    }

    void* data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
      PLOGF(ERROR) << "mmap() failed";
      return -errno;
    }

    auto handle = CreateNativeHandle(fd);
    bool is_dmabuf = IsDmaBuf(fd);
    Buffer buffer = {
        .fd = std::move(fd),
        .data = data,
        .size = size,
        .is_dmabuf = is_dmabuf,
        .ref_count = 1,
        .desc = *desc,
        .handle = std::move(handle),
        .locked_usage = 0,
        .id = GetUniqueId(),
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
      if (munmap(it->second.data, it->second.size) != 0) {
        PLOGF(ERROR) << "munmap() failed";
      }
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
    const uint64_t kCpuUsageMask = (AHARDWAREBUFFER_USAGE_CPU_READ_MASK |
                                    AHARDWAREBUFFER_USAGE_CPU_WRITE_MASK);
    auto it = buffers_.find(buffer);
    // TODO(shik): Support fence.
    if (it == buffers_.end() || fence >= 0 || rect != nullptr) {
      return -EINVAL;
    }
    auto& buf = it->second;

    if (buf.locked_usage != 0) {
      // TODO(shik): Support multiple concurrent locks if the usages are
      // compatible. The semantic is a little bit tricky and there is no use
      // case yet, so simply return an error for now.
      LOGF(ERROR) << "Buffer is alerady locked";
      return -EINVAL;
    }

    if ((usage & kCpuUsageMask) == 0 || (usage & ~kCpuUsageMask) != 0) {
      LOGF(ERROR) << "Invalid usage mask";
      return -EINVAL;
    }

    bool hasRead = (usage & AHARDWAREBUFFER_USAGE_CPU_READ_MASK) != 0;
    bool canRead = (buf.desc.usage & AHARDWAREBUFFER_USAGE_CPU_READ_MASK) != 0;
    bool hasWrite = (usage & AHARDWAREBUFFER_USAGE_CPU_WRITE_MASK) != 0;
    bool canWrite =
        (buf.desc.usage & AHARDWAREBUFFER_USAGE_CPU_WRITE_MASK) != 0;
    if ((hasRead && !canRead) || (hasWrite && !canWrite)) {
      LOGF(ERROR) << "Incompatible usage mask";
      return -EINVAL;
    }

    if (buf.is_dmabuf) {
      dma_buf_sync sync = {
          .flags = DMA_BUF_SYNC_START | SyncFlagsFromUsageMask(usage),
      };
      // TODO(shik): Handle EINTR.
      int ret = ioctl(buf.fd, DMA_BUF_IOCTL_SYNC, &sync);
      if (ret != 0) {
        PLOGF(ERROR) << "ioctl() for DMA_BUF_IOCTL_SYNC failed";
        return -EINVAL;
      }
    }

    buf.locked_usage = usage;
    *out_virtual_address = buf.data;
    return 0;
  }

  int Unlock(AHardwareBuffer* _Nonnull buffer, int32_t* _Nullable fence) {
    auto it = buffers_.find(buffer);
    if (it == buffers_.end()) {
      return -EINVAL;
    }
    auto& buf = it->second;

    // TODO(shik): Support fence.
    if (fence != nullptr) {
      *fence = -1;
    }

    if (buf.is_dmabuf) {
      dma_buf_sync sync = {
          .flags = DMA_BUF_SYNC_END | SyncFlagsFromUsageMask(buf.locked_usage),
      };
      // TODO(shik): Handle EINTR.
      int ret = ioctl(buf.fd, DMA_BUF_IOCTL_SYNC, &sync);
      if (ret != 0) {
        PLOGF(ERROR) << "ioctl() for DMA_BUF_IOCTL_SYNC failed";
        return -EINVAL;
      }
    }

    buf.locked_usage = 0;
    return 0;
  }

  bool IsSupported(const AHardwareBuffer_Desc* _Nonnull desc) {
    // TODO(shik): Check usage as well.
    return desc->format == AHARDWAREBUFFER_FORMAT_BLOB && desc->height == 1 &&
           desc->layers == 1 && desc->rfu0 == 0 && desc->rfu1 == 0;
  }

  const native_handle_t* GetNativeHandle(const AHardwareBuffer* buffer) {
    auto it = buffers_.find(buffer);
    if (it == buffers_.end()) {
      return nullptr;
    }
    return it->second.handle.get();
  }

  int CreateFromHandle(const AHardwareBuffer_Desc* _Nonnull desc,
                       const native_handle_t* _Nonnull handle,
                       int32_t method,
                       AHardwareBuffer* _Nullable* _Nonnull out_buffer) {
    // Although we have _Nonnull annotation, this nullptr checking behavior is
    // clearly specified in the header comment.
    if (desc == nullptr || handle == nullptr || out_buffer == nullptr) {
      return -EINVAL;
    }

    if (method != AHARDWAREBUFFER_CREATE_FROM_HANDLE_METHOD_REGISTER &&
        method != AHARDWAREBUFFER_CREATE_FROM_HANDLE_METHOD_CLONE) {
      return -EINVAL;
    }

    if (!IsSupported(desc) || handle->version != sizeof(native_handle_t) ||
        handle->numFds != 1) {
      return -EINVAL;
    }

    int fd = handle->data[0];
    auto owned_handle = CloneNativeHandle(handle);
    if (method == AHARDWAREBUFFER_CREATE_FROM_HANDLE_METHOD_CLONE) {
      // TODO(shik): Handle EINTR and set CLOEXEC.
      fd = dup(fd);
      if (fd == -1) {
        return -errno;
      }
      owned_handle->data[0] = fd;
    }

    size_t size = desc->width;
    void* data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
      return -errno;
    }

    bool is_dmabuf = IsDmaBuf(fd);
    Buffer buffer = {
        .fd = ScopedFd(fd),
        .data = data,
        .size = size,
        .is_dmabuf = is_dmabuf,
        .ref_count = 1,
        .desc = *desc,
        .handle = std::move(owned_handle),
        .locked_usage = 0,
        .id = GetUniqueId(),
    };
    *out_buffer = reinterpret_cast<AHardwareBuffer*>(data);
    buffers_.emplace(*out_buffer, std::move(buffer));
    return 0;
  }

  int GetId(const AHardwareBuffer* _Nonnull buffer, uint64_t* _Nonnull outId) {
    if (buffer == nullptr || outId == nullptr) {
      return -EINVAL;
    }

    auto it = buffers_.find(buffer);
    if (it == buffers_.end()) {
      return -EINVAL;
    }

    *outId = it->second.id;
    return 0;
  }

 private:
  struct Buffer {
    ScopedFd fd;
    void* data;
    size_t size;
    bool is_dmabuf;
    int ref_count;
    AHardwareBuffer_Desc desc;

    // The file descriptor (fd) in native_handle_t is the same one owned by
    // ScopedFd. Therefore, we don't need to close it separately when releasing
    // native_handle_t, nor do we need to duplicate it (dup) when cloning.
    OwnedNativeHandle handle;

    // The usage mask applied when the buffer is locked. The value would be 0
    // when the buffer is not locked.
    uint64_t locked_usage;

    // The system wide unique id of the buffer.
    uint64_t id;
  };

  std::map<const AHardwareBuffer*, Buffer> buffers_;

  static uint64_t GetUniqueId() {
    // TODO(ototot): This function is supposed to generate a system wide unique
    // ID. The current ID generation mechanism is prone to sandboxing and PID
    // namespaces, but it is compatible with our use cases right now. We still
    // need to find a way to generate real system wide unique IDs.
    static std::atomic<uint32_t> next_id = 0;
    uint64_t id = static_cast<uint64_t>(getpid()) << 32;
    id |= next_id++;
    return id;
  }
};

}  // namespace
}  // namespace tflite::cros

using Allocator = tflite::cros::Allocator;

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

const native_handle_t* _Nullable AHardwareBuffer_getNativeHandle(
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

int AHardwareBuffer_getId(const AHardwareBuffer* _Nonnull buffer,
                          uint64_t* _Nonnull outId) {
  return Allocator::GetInstance()->GetId(buffer, outId);
}
