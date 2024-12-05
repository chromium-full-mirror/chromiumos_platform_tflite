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

#ifndef DELEGATE_MTK_NEURON_NEURON_DELEGATE_H_
#define DELEGATE_MTK_NEURON_NEURON_DELEGATE_H_

#include <farmhash.h>

#include <memory>
#include <string>
#include <vector>

#include "common/minimal_logging.h"
#include "common/simple_async_delegate.h"
#include "delegate/mtk_neuron/neuron_delegate_options.h"
#include "delegate/mtk_neuron/neuron_implementation.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace neuron {

static const char kNeuronStableDelegateName[] = "NeuronStableDelegate";
static const char kNeuronStableDelegateVersion[] = "1.0.0";

// Returns a structure with the default delegate options.
NeuronDelegateOptions TfLiteNeuronDelegateOptionsDefault();

class NeuronStableDelegate : public cros::SimpleAsyncDelegateInterface {
 public:
  explicit NeuronStableDelegate(const NeuronDelegateOptions& options)
      : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistrationExternal* registration,
                                 const TfLiteOpaqueNode* node,
                                 TfLiteOpaqueContext* context) const override;

  TfLiteStatus Initialize(TfLiteOpaqueContext* context) override;

  const char* Name() const override;

  std::unique_ptr<cros::SimpleAsyncDelegateKernelInterface>
  CreateDelegateKernelInterface() override;

 private:
  void GetPreOpCheckResult(TfLiteOpaqueContext* context);

  const NeuronApi* neuron_ = NeuronApiImplementation();
  NeuronDelegateOptions options_;
  std::vector<int> node_is_supported_;
  mutable int node_id_ = -1;
};

}  // namespace neuron
}  // namespace tflite

#endif  // DELEGATE_MTK_NEURON_NEURON_DELEGATE_H_
