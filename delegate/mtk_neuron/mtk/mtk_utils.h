/*
 * Copyright (C) 2021 MediaTek Inc., this file is modified on 02/26/2021
 * by MediaTek Inc. based on MIT License .
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the ""Software""), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED ""AS IS"", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef DELEGATE_MTK_NEURON_MTK_MTK_UTILS_H_
#define DELEGATE_MTK_NEURON_MTK_MTK_UTILS_H_

#include <inttypes.h>
#include <stdlib.h>

namespace tflite {
namespace mtk {

/**
 * Identify the type and value of the OEM Operand
 * type: The actual type of this operand
 * typeLen: The size(byte) of type.
 * data: The operand value.
 * dataLen: The size(byte) of data.
 */
struct OemOperandValue {
  uint8_t* type;
  uint8_t typeLen;
  uint8_t* data;
  uint32_t dataLen;
};

bool PropertyGetBool(const char* key, bool default_value);
size_t PackOemScalarString(const char* str, uint8_t** out_buffer);

}  // namespace mtk
}  // namespace tflite

#endif  // DELEGATE_MTK_NEURON_MTK_MTK_UTILS_H_
