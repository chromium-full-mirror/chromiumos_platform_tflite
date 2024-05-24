#!/usr/bin/env python3
# Copyright 2024 The ChromiumOS Authors
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""A helper script to generate compile_commands.json."""

import argparse
import functools
import json
import logging
from pathlib import Path
import shlex
import subprocess
import sys
from typing import List, Optional


@functools.lru_cache(1)
def get_workspace_root() -> Path:
    """Gets the root of tflite workspace."""
    root = Path(__file__).resolve().parent.parent
    assert root.name == "tflite" and (root / "WORKSPACE.bazel").exists()
    logging.debug("root = %s", root)
    return root


def shell_join(cmd: List[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def run(args: List[str]):
    logging.debug("$ %s", shell_join(args))
    subprocess.check_call(args, cwd=get_workspace_root())


def check_output(args: List[str]) -> str:
    logging.debug("$ %s", shell_join(args))
    return subprocess.check_output(args, text=True, cwd=get_workspace_root())


def rewrite_compilation_args(args: List[str]) -> List[str]:
    """Rewrites the compilation arguments to fix include paths."""

    # The remote (external) repositories are downloaded/symlinked into the
    # execRoot directory. We need to fix those include paths by prepending the
    # proper prefix. See https://bazel.build/remote/output-directories for more
    # details about the directory layout.
    prefix = "bazel-tflite/"

    new_args = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-isystem", "-iquote") and (
            args[i + 1].startswith("external/")
        ):
            new_args.append(arg)
            new_args.append(prefix + args[i + 1])
            i += 2
        elif arg.startswith("-Iexternal/"):
            new_args.append(f"-I{prefix}{arg[2:]}")
            i += 1
        else:
            new_args.append(arg)
            i += 1
    return new_args


def build(target: str):
    logging.info("Building...")

    cmd = ["bazel", "build", "--config=host_clang", target]
    run(cmd)


def generate_compdb(target: str, output_path: Path):
    logging.info("Generating...")

    cmd = [
        "bazel",
        "aquery",
        "--config=host_clang",
        f'mnemonic("CppCompile", deps({json.dumps(target)}))',
        "--output=jsonproto",
    ]
    output = check_output(cmd)
    actions = json.loads(output)["actions"]

    compdb = []
    directory = str(get_workspace_root())
    for action in actions:
        args = rewrite_compilation_args(action["arguments"])
        file = args[args.index("-c") + 1]
        entry = {
            "file": file,
            "arguments": args,
            "directory": directory,
        }
        compdb.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(compdb, f, indent=2)

    logging.info("Compilation database saved to %s", output_path)


def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--target",
        # All rule targets in packages in the main repository. Does not include
        # targets from external repositories.
        # https://bazel.build/run/build#specifying-build-targets
        default="//...",
        help="bazel target to generate",
    )
    parser.add_argument(
        "--build",
        action=argparse.BooleanOptionalAction,
        help="whether to build the target",
        default=False,
    )
    parser.add_argument(
        "--output",
        default="compile_commands.json",
        help="output path for generated compilation database",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> Optional[int]:
    parser = setup_argument_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.debug else logging.INFO
    log_format = "%(asctime)s - %(levelname)s - %(funcName)s: %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    logging.debug("args = %s", args)

    if args.build:
        build(args.target)

    generate_compdb(args.target, args.output)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
