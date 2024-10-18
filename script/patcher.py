#!/usr/bin/env python3
# Copyright 2024 The ChromiumOS Authors
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""A script to manage tensorflow patches."""

import argparse
import functools
import logging
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import List, Optional, Tuple
import xml.etree.ElementTree


# TODO(shik): Extract common utilities into a module.


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


def xml_get_value(el: xml.etree.ElementTree.Element, key="value") -> str:
    value = el.get(key)
    assert value is not None
    return value


class XMLTree:
    """Helper for parsing and querying XML using XPath expressions."""

    def __init__(self, xml_content: str):
        self.root = xml.etree.ElementTree.fromstring(xml_content)

    def get_all(self, xpath: str) -> List[str]:
        return [xml_get_value(el) for el in self.root.findall(xpath)]

    def get(self, xpath: str) -> str:
        el = self.root.find(xpath)
        assert el is not None
        return xml_get_value(el)


def get_url_and_patches() -> Tuple[str, List[str]]:
    """Gets the TensorFlow archive URL and patches from Bazel."""
    cmd = [
        "bazel",
        "query",
        "--output=xml",
        "deps(//external:org_tensorflow)",
    ]
    xml_content = check_output(cmd)
    tree = XMLTree(xml_content)

    rule = ".//rule[@class='http_archive']"
    url = tree.get(f"{rule}/string[@name='url']")

    patches = []
    for label in tree.get_all(f"{rule}/list[@name='patches']/label"):
        # The label looks like `//patch:${num}-some-descriptive-name.patch`
        m = re.match(r"^//patch:(.+\.patch)$", label)
        assert m is not None
        patches.append(m.group(1))

    return (url, patches)


def download_archive(url: str, dest: Path):
    logging.info("Download and extract %s into %s", url, dest)

    # It's easier and faster to download and extract the tarball by running the
    # shell command with a pipe. Having a Python implementation requires 10x
    # more lines of code, is actually slower, and consumes more memory.
    # Since it's a developer-facing script, we are okay to shell out here.
    assert url.endswith(".tar.gz")
    curl = shell_join(["curl", "-L", url])
    tar = shell_join(["tar", "-xz", "--strip-components=1", "-C", str(dest)])
    run(["sh", "-c", f"{curl} | {tar}"])


def cmd_eject(args: argparse.Namespace):
    del args

    # TODO(shik): Make the destination configurable.
    dest = get_workspace_root() / "tensorflow"
    if dest.exists():
        # TODO(shik): Add a --force flag to override the existing one
        logging.fatal("tensorflow/ is already ejected")
    dest.mkdir()

    url, patches = get_url_and_patches()
    logging.debug("url = %s", url)
    logging.debug("patches = %s", patches)
    download_archive(url, dest)
    # TODO(shik): Apply patches.


def cmd_seal(args: argparse.Namespace):
    del args
    logging.info("Not implemented yet")
    # TODO(shik): Implement seal.


def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable debug logging",
    )
    parser.set_defaults(func=lambda _: parser.print_help())
    subparsers = parser.add_subparsers()

    eject_parser = subparsers.add_parser(
        "eject",
        help="Eject tensorflow to a local git repo with patches as commits",
    )
    eject_parser.set_defaults(func=cmd_eject)

    seal_parser = subparsers.add_parser(
        "seal",
        help="Seal the local git repo and format the commits as patches",
    )
    seal_parser.set_defaults(func=cmd_seal)

    return parser


def main(argv: Optional[List[str]] = None) -> Optional[int]:
    parser = setup_argument_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.debug else logging.INFO
    log_format = "%(asctime)s - %(levelname)s - %(funcName)s: %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    logging.debug("args = %s", args)

    args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
