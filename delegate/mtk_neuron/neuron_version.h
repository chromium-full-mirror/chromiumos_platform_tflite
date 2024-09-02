/*
 * Copyright (C) 2022 MediaTek Inc., this file is modified on 02/26/2021
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

#ifndef DELEGATE_MTK_NEURON_NEURON_VERSION_H_
#define DELEGATE_MTK_NEURON_NEURON_VERSION_H_

#define NEURON_DELEGATE_MAJOR_VERSION 1
#define NEURON_DELEGATE_MINOR_VERSION 4
#define NEURON_DELEGATE_PATCH_VERSION 3

#define NEURON_DELEGATE_STR_HELPER(x) #x
#define NEURON_DELEGATE_STR(x) NEURON_DELEGATE_STR_HELPER(x)

// clang-format off
#define NEURON_DELEGATE_VERSION_STRING                     \
  (NEURON_DELEGATE_STR(NEURON_DELEGATE_MAJOR_VERSION) "."  \
    NEURON_DELEGATE_STR(NEURON_DELEGATE_MINOR_VERSION) "." \
    NEURON_DELEGATE_STR(NEURON_DELEGATE_PATCH_VERSION))
// clang-format on

#endif  // DELEGATE_MTK_NEURON_NEURON_VERSION_H_
