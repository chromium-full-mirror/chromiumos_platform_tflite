/*
 * Copyright 2024 The ChromiumOS Authors
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef ANDROID_HARDWARE_BUFFER_H_
#define ANDROID_HARDWARE_BUFFER_H_

#include <stdint.h>
#include <sys/cdefs.h>

#include "android/native_handle.h"

/**
 * This is a minimum AHardwareBuffer API shim ported from Android header
 * frameworks/native/libs/nativewindow/include/android/hardware_buffer.h and
 * frameworks/native/libs/nativewindow/include/vndk/hardware_buffer.h.
 */

__BEGIN_DECLS

/**
 * Locking rectangle is not supported on ChromeOS, so we simply use an opaque
 * type here. The caller is expected to use NULL pointer as the value if the
 * argument type is ARect.
 */
typedef struct ARect ARect;

/**
 * Buffer pixel formats.
 */
enum AHardwareBuffer_Format {
  /**
   * Opaque binary blob format.
   * Must have height 1 and one layer, with width equal to the buffer
   * size in bytes. Corresponds to Vulkan buffers and OpenGL buffer
   * objects. Can be bound to the latter using GL_EXT_external_buffer.
   */
  AHARDWAREBUFFER_FORMAT_BLOB = 0x21,
};

/**
 * Buffer usage flags, specifying how the buffer will be accessed.
 */
enum AHardwareBuffer_UsageFlags {
  /**
   * The buffer will never be locked for direct CPU reads using the
   * AHardwareBuffer_lock() function. Note that reading the buffer
   * using OpenGL or Vulkan functions or memory mappings is still
   * allowed.
   */
  AHARDWAREBUFFER_USAGE_CPU_READ_NEVER = 0UL,
  /**
   * The buffer will sometimes be locked for direct CPU reads using
   * the AHardwareBuffer_lock() function. Note that reading the
   * buffer using OpenGL or Vulkan functions or memory mappings
   * does not require the presence of this flag.
   */
  AHARDWAREBUFFER_USAGE_CPU_READ_RARELY = 2UL,
  /**
   * The buffer will often be locked for direct CPU reads using
   * the AHardwareBuffer_lock() function. Note that reading the
   * buffer using OpenGL or Vulkan functions or memory mappings
   * does not require the presence of this flag.
   */
  AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN = 3UL,
  /** CPU read value mask. */
  AHARDWAREBUFFER_USAGE_CPU_READ_MASK = 0xFUL,

  /**
   * The buffer will never be locked for direct CPU writes using the
   * AHardwareBuffer_lock() function. Note that writing the buffer
   * using OpenGL or Vulkan functions or memory mappings is still
   * allowed.
   */
  AHARDWAREBUFFER_USAGE_CPU_WRITE_NEVER = 0UL << 4,
  /**
   * The buffer will sometimes be locked for direct CPU writes using
   * the AHardwareBuffer_lock() function. Note that writing the
   * buffer using OpenGL or Vulkan functions or memory mappings
   * does not require the presence of this flag.
   */
  AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY = 2UL << 4,
  /**
   * The buffer will often be locked for direct CPU writes using
   * the AHardwareBuffer_lock() function. Note that writing the
   * buffer using OpenGL or Vulkan functions or memory mappings
   * does not require the presence of this flag.
   */
  AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN = 3UL << 4,
  /** CPU write value mask. */
  AHARDWAREBUFFER_USAGE_CPU_WRITE_MASK = 0xFUL << 4,

  /**
   * The buffer will be used as a shader storage or uniform buffer object.
   * When this flag is present, the format must be AHARDWAREBUFFER_FORMAT_BLOB.
   */
  AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER = 1UL << 24,
};

/**
 * Buffer description. Used for allocating new buffers and querying
 * parameters of existing ones.
 */
typedef struct AHardwareBuffer_Desc {
  uint32_t width;   // Width in pixels.
  uint32_t height;  // Height in pixels.
  /**
   * Number of images in an image array. AHardwareBuffers with one
   * layer correspond to regular 2D textures. AHardwareBuffers with
   * more than layer correspond to texture arrays. If the layer count
   * is a multiple of 6 and the usage flag
   * AHARDWAREBUFFER_USAGE_GPU_CUBE_MAP is present, the buffer is
   * a cube map or a cube map array.
   */
  uint32_t layers;
  uint32_t format;  // One of AHardwareBuffer_Format.
  uint64_t usage;   // Combination of AHardwareBuffer_UsageFlags.
  uint32_t stride;  // Row stride in pixels,
                    // ignored for AHardwareBuffer_allocate().
  uint32_t rfu0;    // Initialize to zero, reserved for future use.
  uint64_t rfu1;    // Initialize to zero, reserved for future use.
} AHardwareBuffer_Desc;

/**
 * Opaque handle for a native hardware buffer.
 */
typedef struct AHardwareBuffer AHardwareBuffer;

/**
 * Allocates a buffer that matches the passed AHardwareBuffer_Desc.
 *
 * If allocation succeeds, the buffer can be used according to the
 * usage flags specified in its description. If a buffer is used in ways
 * not compatible with its usage flags, the results are undefined and
 * may include program termination.
 *
 * Return 0 on success, or an error number of the allocation fails for
 * any reason. The returned buffer has a reference count of 1.
 */
int AHardwareBuffer_allocate(const AHardwareBuffer_Desc* _Nonnull desc,
                             AHardwareBuffer* _Nullable* _Nonnull outBuffer);
/**
 * Acquire a reference on the given AHardwareBuffer object.
 *
 * This prevents the object from being deleted until the last reference
 * is removed.
 */
void AHardwareBuffer_acquire(AHardwareBuffer* _Nonnull buffer);

/**
 * Remove a reference that was previously acquired with
 * AHardwareBuffer_acquire() or AHardwareBuffer_allocate().
 */
void AHardwareBuffer_release(AHardwareBuffer* _Nonnull buffer);

/**
 * Return a description of the AHardwareBuffer in the passed
 * AHardwareBuffer_Desc struct.
 */
void AHardwareBuffer_describe(const AHardwareBuffer* _Nonnull buffer,
                              AHardwareBuffer_Desc* _Nonnull outDesc);

/**
 * Lock the AHardwareBuffer for direct CPU access.
 *
 * This function can lock the buffer for either reading or writing.
 * It may block if the hardware needs to finish rendering, if CPU caches
 * need to be synchronized, or possibly for other implementation-
 * specific reasons.
 *
 * The passed AHardwareBuffer must have one layer, otherwise the call
 * will fail.
 *
 * If a fence is not negative, it specifies a fence file descriptor on
 * which to wait before locking the buffer. If it's negative, the caller
 * is responsible for ensuring that writes to the buffer have completed
 * before calling this function.  Using this parameter is more efficient
 * than waiting on the fence and then calling this function.
 *
 * The a usage parameter may only specify AHARDWAREBUFFER_USAGE_CPU_*.
 * If set, then outVirtualAddress is filled with the address of the
 * buffer in virtual memory. The flags must also be compatible with
 * usage flags specified at buffer creation: if a read flag is passed,
 * the buffer must have been created with
 * AHARDWAREBUFFER_USAGE_CPU_READ_RARELY or
 * AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN. If a write flag is passed, it
 * must have been created with AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY or
 * AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN.
 *
 * If a rect is not NULL, the caller promises to modify only data in
 * the area specified by rect. If rect is NULL, the caller may modify
 * the contents of the entire buffer. The content of the buffer outside
 * of the specified rect is NOT modified by this call.
 *
 * It is legal for several different threads to lock a buffer for read
 * access; none of the threads are blocked.
 *
 * Locking a buffer simultaneously for write or read/write is undefined,
 * but will neither terminate the process nor block the caller.
 * AHardwareBuffer_lock may return an error or leave the buffer's
 * content in an indeterminate state.
 *
 * If the buffer has AHARDWAREBUFFER_FORMAT_BLOB, it is legal lock it
 * for reading and writing in multiple threads and/or processes
 * simultaneously, and the contents of the buffer behave like shared
 * memory.
 *
 * Return 0 on success. -EINVAL if a buffer is NULL, the usage flags
 * are not a combination of AHARDWAREBUFFER_USAGE_CPU_*, or the buffer
 * has more than one layer. Error number if the lock fails for any other
 * reason.
 */
int AHardwareBuffer_lock(AHardwareBuffer* _Nonnull buffer,
                         uint64_t usage,
                         int32_t fence,
                         const ARect* _Nullable rect,
                         void* _Nullable* _Nonnull outVirtualAddress);

/**
 * Unlock the AHardwareBuffer from direct CPU access.
 *
 * Must be called after all changes to the buffer are completed by the
 * caller.  If a fence is NULL, the function will block until all work
 * is completed.  Otherwise, a fence will be set either to a valid file
 * descriptor or to -1.  The file descriptor will become signaled once
 * the unlocking is complete and buffer contents are updated.
 * The caller is responsible for closing the file descriptor once it's
 * no longer needed.  The value -1 indicates that unlocking has already
 * completed before the function returned and no further operations are
 * necessary.
 *
 * Return 0 on success. -EINVAL if a buffer is NULL. Error number if
 * the unlock fails for any reason.
 */
int AHardwareBuffer_unlock(AHardwareBuffer* _Nonnull buffer,
                           int32_t* _Nullable fence);

/**
 * Test whether the given format and usage flag combination is
 * allocatable.
 *
 * If this function returns true, it means that a buffer with the given
 * description can be allocated on this implementation, unless resource
 * exhaustion occurs. If this function returns false, it means that the
 * allocation of the given description will never succeed.
 *
 * The return value of this function may depend on all fields in the
 * description, except stride, which is always ignored. For example,
 * some implementations have implementation-defined limits on texture
 * size and layer count.
 *
 * Return 1 if the format and usage flag combination is allocatable,
 *     0 otherwise.
 */
int AHardwareBuffer_isSupported(const AHardwareBuffer_Desc* _Nonnull desc);

/**
 * Get the native handle from an AHardwareBuffer.
 *
 * Return a non-NULL native handle on success, NULL if buffer is nullptr or
 * the operation fails for any reason.
 */
const native_handle_t* _Nullable AHardwareBuffer_getNativeHandle(
    const AHardwareBuffer* _Nonnull buffer);

enum CreateFromHandleMethod {
  AHARDWAREBUFFER_CREATE_FROM_HANDLE_METHOD_REGISTER = 2,
  AHARDWAREBUFFER_CREATE_FROM_HANDLE_METHOD_CLONE = 3,
};

/**
 * Create an AHardwareBuffer from a native handle.
 *
 * This function wraps a native handle in an AHardwareBuffer suitable for use by
 * applications or other parts of the system. The contents of desc will be
 * returned by AHardwareBuffer_describe().
 *
 * If method is AHARDWAREBUFFER_CREATE_FROM_HANDLE_METHOD_REGISTER, the handle
 * is assumed to be unregistered, and it will be registered/imported before
 * being wrapped in the AHardwareBuffer. If successful, the AHardwareBuffer will
 * own the handle.
 *
 * If method is AHARDWAREBUFFER_CREATE_FROM_HANDLE_METHOD_CLONE, the handle will
 * be cloned and the clone registered. The AHardwareBuffer will own the cloned
 * handle but not the original.
 *
 * Return 0 on success, -EINVAL if desc or handle or outBuffer is NULL, or an
 * error number if the operation fails for any reason.
 */
int AHardwareBuffer_createFromHandle(
    const AHardwareBuffer_Desc* _Nonnull desc,
    const native_handle_t* _Nonnull handle,
    int32_t method,
    AHardwareBuffer* _Nullable* _Nonnull outBuffer);

/**
 * Get the system wide unique id for an AHardwareBuffer.
 *
 * Return 0 on success, -EINVAL if buffer or outId is NULL, or an error number
 * if the operation fails for any reason.
 */
int AHardwareBuffer_getId(const AHardwareBuffer* _Nonnull buffer,
                          uint64_t* _Nonnull outId);
__END_DECLS

#endif  // ANDROID_HARDWARE_BUFFER_H_
