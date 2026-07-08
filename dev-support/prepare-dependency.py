#!/usr/bin/env python3
"""Prepare a third-party source checkout from a pinned source manifest."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceSpec:
    name: str
    upstream: str
    base_commit: str
    patches: tuple[Path, ...]


class PrepError(Exception):
    """A user-actionable preparation failure."""


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        spec = _load_source(args.source)
        _prepare(spec, args.dest.expanduser())
    except PrepError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an external dependency by shallow-fetching a pinned commit "
            "and applying local patches."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to a source.toml manifest.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Checkout destination. Existing destinations are verified, not modified.",
    )
    return parser.parse_args(argv)


def _load_source(source: Path) -> SourceSpec:
    source_path = source.expanduser().resolve()
    if not source_path.is_file():
        raise PrepError(f"source manifest does not exist: {source_path}")

    try:
        with source_path.open("rb") as source_file:
            data = tomllib.load(source_file)
    except tomllib.TOMLDecodeError as exc:
        raise PrepError(f"invalid TOML in {source_path}: {exc}") from exc
    except OSError as exc:
        raise PrepError(f"failed to read {source_path}: {exc}") from exc

    repo = _expect_table(data, "repo", source_path)
    name = _expect_string(repo, "name", source_path)
    upstream = _expect_string(repo, "upstream", source_path)
    base_commit = _expect_string(repo, "base_commit", source_path)
    patch_names = _expect_string_list(repo, "patches", source_path)
    manifest_dir = source_path.parent

    patches = tuple(manifest_dir / patch_name for patch_name in patch_names)
    for patch in patches:
        if not patch.is_file():
            raise PrepError(f"patch listed in {source_path} does not exist: {patch}")

    return SourceSpec(
        name=name,
        upstream=upstream,
        base_commit=base_commit,
        patches=patches,
    )


def _expect_table(data: dict[str, Any], key: str, source: Path) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise PrepError(f"{source} must define a [{key}] table")
    return value


def _expect_string(data: dict[str, Any], key: str, source: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise PrepError(f"{source} must define a non-empty string value for {key}")
    return value


def _expect_string_list(
    data: dict[str, Any], key: str, source: Path
) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise PrepError(f"{source} must define {key} as a list of non-empty strings")
    return tuple(value)


def _prepare(spec: SourceSpec, dest: Path) -> None:
    _require_tool("git")
    _require_tool("patch")

    if dest.exists():
        _verify_existing_checkout(spec, dest)
        print(
            f"{spec.name}: existing checkout is already prepared at {dest}",
            file=sys.stderr,
        )
        return

    print(f"{spec.name}: creating checkout at {dest}", file=sys.stderr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.mkdir()
    _run(["git", "init"], cwd=dest)
    _run(["git", "remote", "add", "origin", spec.upstream], cwd=dest)
    _run(["git", "fetch", "--depth", "1", "origin", spec.base_commit], cwd=dest)
    _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=dest)
    _apply_patches(spec, dest)
    print(f"{spec.name}: prepared checkout at {dest}", file=sys.stderr)


def _require_tool(tool: str) -> None:
    if shutil.which(tool) is None:
        raise PrepError(f"required executable not found on PATH: {tool}")


def _verify_existing_checkout(spec: SourceSpec, dest: Path) -> None:
    if not dest.is_dir():
        raise PrepError(f"destination exists but is not a directory: {dest}")
    if not _is_git_checkout_root(dest):
        raise PrepError(f"destination is not a git checkout root: {dest}")

    head = _run(["git", "rev-parse", "HEAD"], cwd=dest).stdout.strip()
    if head.lower() != spec.base_commit.lower():
        raise PrepError(
            f"destination is at commit {head}, expected {spec.base_commit}: {dest}"
        )

    for patch in spec.patches:
        result = _run(
            ["patch", "-p1", "-t", "--dry-run", "--reverse", "-i", str(patch)],
            cwd=dest,
            check=False,
        )
        if result.returncode != 0:
            raise PrepError(
                f"patch does not appear to be applied in {dest}: {patch}\n"
                f"{_format_command_output(result)}"
            )


def _is_git_checkout_root(path: Path) -> bool:
    inside = _run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=path,
        check=False,
    )
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return False

    top_level = _run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path,
        check=False,
    )
    if top_level.returncode != 0:
        return False
    return Path(top_level.stdout.strip()).resolve() == path.resolve()


def _apply_patches(spec: SourceSpec, dest: Path) -> None:
    for patch in spec.patches:
        print(f"{spec.name}: applying {patch.name}", file=sys.stderr)
        _run(["patch", "-p1", "-t", "-i", str(patch)], cwd=dest)


def _run(
    command: list[str],
    *,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise PrepError(
            f"command failed in {cwd}: {' '.join(command)}\n"
            f"{_format_command_output(result)}"
        )
    return result


def _format_command_output(result: subprocess.CompletedProcess[str]) -> str:
    parts = []
    if result.stdout:
        parts.append(f"stdout:\n{result.stdout.rstrip()}")
    if result.stderr:
        parts.append(f"stderr:\n{result.stderr.rstrip()}")
    if not parts:
        return f"exit code: {result.returncode}"
    return "\n".join(parts)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
