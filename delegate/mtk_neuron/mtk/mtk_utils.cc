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

#define LOG_TAG "MtkUtils"
#include "delegate/mtk_neuron/mtk/mtk_utils.h"

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "common/minimal_logging.h"
#include "tensorflow/lite/c/common.h"

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif  // __ANDROID__

#define MAX_OEM_OP_STRING_LEN 100

namespace tflite {
namespace mtk {

bool PropertyGetBool(const char* key, bool default_value) {
#ifdef __ANDROID__
  if (!key) {
    return default_value;
  }

  bool result = default_value;
  char buf[PROP_VALUE_MAX] = {'\0'};

  int len = __system_property_get(key, buf);
  if (len == 1) {
    char ch = buf[0];
    if (ch == '0' || ch == 'n') {
      result = false;
    } else if (ch == '1' || ch == 'y') {
      result = true;
    }
  } else if (len > 1) {
    if (!strcmp(buf, "no") || !strcmp(buf, "false") || !strcmp(buf, "off")) {
      result = false;
    } else if (!strcmp(buf, "yes") || !strcmp(buf, "true") ||
               !strcmp(buf, "on")) {
      result = true;
    }
  }
  return result;
#else
  return false;
#endif  // __ANDROID__
}
/*
 * The format of data will be:
 *  -------------------------------------------------------------------------------
 *  | 1 byte typeLen  | N bytes type     | 4 bytes dataLen  | N bytes data |
 *  -------------------------------------------------------------------------------
 */
static int EncodeOperandValue(OemOperandValue* operand, uint8_t* output) {
  size_t currPos = 0;

  // 1 byte for typeLen, 4 bytes for bufferLen
  if (output == nullptr) {
    return -1;
  }

  // Set length of type
  *output = operand->typeLen;
  currPos += sizeof(uint8_t);

  // Copy type to output
  memcpy(output + currPos, operand->type, operand->typeLen);
  currPos += operand->typeLen;

  // Set the length of buffer
  uint32_t* dataLen = reinterpret_cast<uint32_t*>(&output[currPos]);
  *dataLen = operand->dataLen;
  currPos += sizeof(uint32_t);

  // Copy  operand value to output
  memcpy(&output[currPos], operand->data, operand->dataLen);

  return 0;
}

size_t PackOemScalarString(const char* str, uint8_t** out_buffer) {
  if (str == nullptr) {
    return 0;
  }
  size_t out_len = 0;
  uint8_t type[] = {'s', 't', 'r', 'i', 'n', 'g'};
  OemOperandValue operand_value;

  operand_value.typeLen = sizeof(type);
  operand_value.type = type;
  operand_value.dataLen = strlen(str);
  if (operand_value.dataLen > MAX_OEM_OP_STRING_LEN) {
    return 0;
  }
  operand_value.data =
      reinterpret_cast<uint8_t*>(malloc(operand_value.dataLen));
  if (operand_value.data == nullptr) {
    return 0;
  }
  memcpy(operand_value.data, str, operand_value.dataLen);

  out_len =
      operand_value.typeLen + operand_value.dataLen + (sizeof(size_t) * 2);
  *out_buffer = reinterpret_cast<uint8_t*>(calloc(out_len, sizeof(uint8_t)));
  if (*out_buffer == nullptr) {
    free(operand_value.data);
    return 0;
  }
  EncodeOperandValue(&operand_value, *out_buffer);
  free(operand_value.data);

  TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                  "PackOemScalarString: %s, buffer size:%zu", str, out_len);
  return out_len;
}

}  // namespace mtk
}  // namespace tflite
