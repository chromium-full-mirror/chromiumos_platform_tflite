# Copyright 2024 The ChromiumOS Authors
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

_real_build = '''
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "openvino",
    srcs = ["openvino/lib64/libopenvino.so"],
    includes = ["openvino/include/openvino"],
    hdrs = glob(["openvino/include/openvino/**"]),
    visibility = ["//visibility:public"],
)
'''

_stub_build = '''
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "openvino",
    visibility = ["//visibility:public"],
)
'''

def _impl(ctx):
    openvino_dir = ctx.os.environ.get("OPENVINO_DIR")
    if openvino_dir:
        ctx.symlink(openvino_dir, "openvino")
        ctx.file("BUILD", _real_build)
    else:
        ctx.file("BUILD", _stub_build)

openvino_repository = repository_rule(
    implementation = _impl,
    local = True,
    environ = ["OPENVINO_DIR"],
)
