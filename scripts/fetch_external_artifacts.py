#!/usr/bin/env python3
"""Fetch content-addressed external artifacts without storing them in Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "manifests" / "external_artifacts.json"
SCHEMA = "open-dreamer-external-artifacts-v1"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"invalid external-artifact schema in {path}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"external-artifact manifest is empty: {path}")
    return artifacts


def digest_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def verify(path: Path, artifact: dict[str, Any]) -> None:
    size, digest = digest_file(path)
    if size != artifact["bytes"] or digest != artifact["sha256"]:
        raise ValueError(
            f"artifact verification failed for {artifact['name']}: "
            f"bytes={size}, sha256={digest}"
        )


def destination(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"artifact path escapes destination root: {relative}")
    return path


def fetch_s3(
    uri: str,
    output: Path,
    *,
    profile: str | None,
    endpoint_url: str | None,
) -> None:
    command = ["aws"]
    if profile:
        command.extend(["--profile", profile])
    if endpoint_url:
        command.extend(["--endpoint-url", endpoint_url])
    command.extend(["s3", "cp", uri, str(output), "--only-show-errors"])
    subprocess.run(command, check=True)


def fetch_artifact(
    artifact: dict[str, Any],
    *,
    root: Path,
    profile: str | None,
    endpoint_url: str | None,
) -> Path:
    output = destination(root, str(artifact["path"]))
    if output.is_file():
        verify(output, artifact)
        print(f"verified {artifact['name']}: {output}")
        return output

    uri = str(artifact["uri"])
    if not uri.startswith("s3://"):
        raise ValueError(f"unsupported artifact URI: {uri}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        fetch_s3(
            uri,
            temporary,
            profile=profile,
            endpoint_url=endpoint_url,
        )
        verify(temporary, artifact)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"fetched {artifact['name']}: {output}")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--profile")
    parser.add_argument("--endpoint-url")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    requested = set(args.artifact)
    artifacts = load_manifest(args.manifest)
    selected = [
        artifact
        for artifact in artifacts
        if not requested or artifact.get("name") in requested
    ]
    found = {str(artifact.get("name")) for artifact in selected}
    missing = requested - found
    if missing:
        raise SystemExit(f"unknown artifacts: {', '.join(sorted(missing))}")
    for artifact in selected:
        fetch_artifact(
            artifact,
            root=args.root,
            profile=args.profile,
            endpoint_url=args.endpoint_url,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
