#!/usr/bin/env python3
# Copyright 2024 The ChromiumOS Authors
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""A script to calculate and breakdown the installed packages size."""

import argparse
import functools
import heapq
import logging
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Dict, List, Literal, NoReturn, Optional, Tuple


# Use Dict[] directly instead of TypedDict to make type inference works better
# for .items().
PackageSize = Dict[
    Literal["rootfs", "stateful"],
    # List of (file path, size in bytes) pairs sorted decreasingly by size.
    List[Tuple[str, int]],
]

# TODO(shik): Extract common python functions into a utility module.


@functools.lru_cache(1)
def get_workspace_root() -> Path:
    """Gets the root of tflite workspace."""
    root = Path(__file__).resolve().parent.parent
    assert root.name == "tflite" and (root / "WORKSPACE.bazel").exists()
    logging.debug("root = %s", root)
    return root


def shell_join(cmd: List[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def run(args: List[str]) -> int:
    logging.debug("$ %s", shell_join(args))
    return subprocess.check_call(args, cwd=get_workspace_root())


def check_output(args: List[str]) -> str:
    logging.debug("$ %s", shell_join(args))
    return subprocess.check_output(args, text=True, cwd=get_workspace_root())


@functools.lru_cache(1)
def is_in_cros_sdk() -> bool:
    return Path("/etc/cros_chroot_version").exists()


def human_readble_size(num_bytes: int) -> str:
    units = [(2**30, "G"), (2**20, "M"), (2**10, "K")]
    scale, unit = next((u for u in units if num_bytes >= u[0]), (1, "B"))

    fmt = "%d" if unit == "B" else "%.1f"
    return fmt % (num_bytes / scale) + unit


def get_package_size(board: str, package: str) -> PackageSize:
    sysroot = Path(f"/build/{board}")
    if not sysroot.is_dir():
        raise ValueError(
            f"{sysroot} not found, "
            "please check board name and ensure packages are built"
        )

    cmd = [
        f"equery-{board}",
        "files",
        "--filter=obj",
        package,
    ]
    files = check_output(cmd).splitlines()
    package_size: PackageSize = {
        "rootfs": [],
        "stateful": [],
    }
    for file in files:
        # TODO(shik): Ignore files listed in install_mask.py instead of
        # hard-coding common masks here. Reference:
        # https://www.chromium.org/chromium-os/developer-library/guides/portage/ebuild-faq/#how-do-i-find-out-the-on-disk-package-size
        if re.search(r"^/usr/(include|lib/debug)/|\.(a|c|cc|h|hpp)$", file):
            continue

        path = sysroot / file.lstrip("/")
        size = path.stat().st_size
        key = "stateful" if file.startswith("/usr/local/") else "rootfs"
        package_size[key].append((file, size))

    return package_size


def print_package_size(package_size: PackageSize, top: int) -> None:
    for key, sizes in package_size.items():
        total = sum(size for _, size in sizes)
        print(f"  {key}: {human_readble_size(total)}")

        largest_files = heapq.nlargest(top, sizes, key=lambda x: x[1])
        for file, size in largest_files:
            print("    %7s %s" % (human_readble_size(size), file))


def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--board",
        required=True,
        help="board name",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="how many largest files to print for each package",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable debug logging",
    )
    parser.add_argument(
        "packages",
        metavar="package",
        nargs="*",
        default=["tensorflow"],
        help="package names",
    )
    return parser


def run_self_in_cros_sdk(argv: List[str]) -> NoReturn:
    logging.debug("Rerun self in cros_sdk")
    tflite = Path("/mnt/host/source/src/platform/tflite")
    script = tflite / Path(__file__).relative_to(get_workspace_root())
    cmd = [
        "cros_sdk",
        "--no-update",
        "--",
        script,
    ] + (argv or [])
    os.chdir(get_workspace_root())
    os.execvp(cmd[0], cmd)


def main(argv: Optional[List[str]] = None) -> Optional[int]:
    parser = setup_argument_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.debug else logging.INFO
    log_format = "%(asctime)s - %(levelname)s - %(funcName)s: %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    logging.debug("args = %s", args)

    if not is_in_cros_sdk():
        run_self_in_cros_sdk(argv or [])

    for i, pkg in enumerate(args.packages):
        package_size = get_package_size(args.board, pkg)
        if i > 0:
            print()
        print(f"{pkg}:")
        print_package_size(package_size, args.top)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
