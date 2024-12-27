#!/usr/bin/env python3
# Copyright 2024 The ChromiumOS Authors
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""A script to manage tensorflow patches."""

import argparse
import functools
import json
import logging
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree


# Set a fixed random date to ensure a stable commit hash. The hash is derived
# from both the tree content and the commit object, which includes the author
# and committer dates.
# The commit object can be inspected with `git cat-file -p $COMMIT_HASH`.
INIT_COMMIT_DATE = "2024-10-01T00:00:00+08:00"


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


def run(
    args: List[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> int:
    logging.debug("$ %s", shell_join(args))
    if cwd is None:
        cwd = get_workspace_root()
    if env is not None:
        env = {**os.environ, **env}
    return subprocess.check_call(args, cwd=cwd, env=env)


def check_output(
    args: List[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> str:
    logging.debug("$ %s", shell_join(args))
    if cwd is None:
        cwd = get_workspace_root()
    if env is not None:
        env = {**os.environ, **env}
    return subprocess.check_output(args, text=True, cwd=cwd, env=env)


def format_path(path: Path) -> str:
    cwd = Path.cwd()
    if path.is_relative_to(cwd):
        path = path.relative_to(cwd)
    s = str(path)
    if path.is_dir():
        s += "/"
    return s


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


def init_git_repo(repo: Path):
    logging.info("Initialize Git repository in %s", str(repo))

    run(["git", "init"], cwd=repo)
    run(["git", "config", "user.name", "TFLite Patcher"], cwd=repo)
    run(["git", "config", "user.email", "tflite@localhost"], cwd=repo)
    run(["git", "add", "."], cwd=repo)
    run(
        [
            "git",
            "commit",
            "--quiet",
            "--message",
            "Initial commit from ejected repo",
        ],
        cwd=repo,
        env={
            # https://git-scm.com/docs/git-commit/2.24.0#_date_formats
            "GIT_COMMITTER_DATE": INIT_COMMIT_DATE,
            "GIT_AUTHOR_DATE": INIT_COMMIT_DATE,
        },
    )


def apply_patches(repo: Path, patches: List[str]):
    logging.info("Apply patches to the repository at %s", repo)

    # Note that this function uses `git am` to apply patches, which requires
    # each patch file to be in the format produced by `git format-patch`. This
    # format includes commit metadata such as the author, date, and commit
    # message in the header.
    #
    # This approach is stricter than Bazel's patch application process. Bazel
    # applies patches using the `patch -p1` command, which works with plain
    # diff patches that may not include commit metadata.
    #
    # By using `git am`, we can apply each patch as a separate commit,
    # preserving the original commit messages and author information. This
    # makes it easier to manage and track changes in the repository.
    for patch in patches:
        patch_path = get_workspace_root() / "patch" / patch
        run(
            [
                "git",
                "am",
                "--whitespace=nowarn",  # Silent warning from upstream code
                "--committer-date-is-author-date",
                str(patch_path),
            ],
            cwd=repo,
        )


def cmd_eject(args: argparse.Namespace):
    force: bool = args.force
    del args

    # TODO(shik): Make the destination configurable.
    dest = get_workspace_root() / "tensorflow"
    if dest.exists():
        if force:
            logging.info("Removing existing %s", dest)
            shutil.rmtree(dest)
        else:
            logging.fatal("tensorflow/ is already ejected")
    dest.mkdir()

    url, patches = get_url_and_patches()
    logging.debug("url = %s", url)
    logging.debug("patches = %s", patches)
    download_archive(url, dest)

    init_git_repo(dest)
    apply_patches(dest, patches)


def get_patch_commits(repo: Path) -> List[str]:
    initial_commit = check_output(
        ["git", "rev-list", "--max-parents=0", "HEAD"], cwd=repo
    ).strip()

    commits = check_output(
        ["git", "rev-list", "--reverse", f"{initial_commit}..HEAD"], cwd=repo
    ).splitlines()

    return commits


def get_patch_name(repo: Path, commit: str) -> Optional[str]:
    msg = check_output(["git", "show", "-s", "--format=%B", commit], cwd=repo)

    names = re.findall(r"^PATCH_NAME=([-\w]+)", msg, re.MULTILINE)
    if len(names) >= 2:
        logging.fatal("Found more than one PATCH_NAME tag")

    return names[0] if names else None


def generate_patch(repo: Path, commit: str, index: int, out_dir: Path) -> Path:
    patch = Path(
        check_output(
            [
                "git",
                "format-patch",
                "-1",
                f"--start-number={index}",
                "--no-numbered",
                f"--output-directory={out_dir}",
                "--no-signature",
                commit,
            ],
            cwd=repo,
        ).strip()
    )
    assert patch.is_file()

    # Check if there is a preferred name specified in commit message.
    patch_name = get_patch_name(repo, commit)
    if patch_name is not None:
        full_name = f"{index:04d}-{patch_name}.patch"
        patch = patch.rename(out_dir / full_name)

    return patch


def update_workspace():
    patch_dir = get_workspace_root() / "patch"
    assert patch_dir.is_dir()
    patches = sorted(f"//patch:{p.name}" for p in patch_dir.glob("*.patch"))

    workspace = get_workspace_root() / "WORKSPACE.bazel"
    new_bzl = re.sub(
        r'(name = "org_tensorflow".*patches\s*=\s*)(\[.*?\])',
        lambda m: m.group(1) + json.dumps(patches),
        workspace.read_text(),
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )
    workspace.write_text(new_bzl)

    run(["cros", "format", str(workspace)])
    logging.info("Updated %s", format_path(workspace))


def cmd_seal(args: argparse.Namespace):
    del args

    repo = get_workspace_root() / "tensorflow"
    if not (repo / ".git").exists():
        logging.fatal("%s is not a git repository", repo)
        sys.exit(1)

    commits = get_patch_commits(repo)

    with tempfile.TemporaryDirectory(prefix="tflite_") as tmp_dir:
        logging.debug("tmp_dir = %s", tmp_dir)
        tmp_dir = Path(tmp_dir)

        for idx, commit in enumerate(commits, start=1):
            patch = generate_patch(repo, commit, idx, tmp_dir)
            logging.debug("Generated patch %s", patch)

        # Update the patches.
        patch_dir = get_workspace_root() / "patch"
        assert patch_dir.is_dir()

        for patch in patch_dir.glob("*.patch"):
            patch.unlink()

        for patch in tmp_dir.glob("*.patch"):
            shutil.move(patch, patch_dir / patch.name)

    logging.info("Patches have been generated in %s", format_path(patch_dir))
    update_workspace()


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
    eject_parser.add_argument(
        "--force",
        action="store_true",
        help="force eject even if tensorflow/ already exists",
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
