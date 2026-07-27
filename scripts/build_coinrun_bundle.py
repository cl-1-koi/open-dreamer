#!/usr/bin/env python3
"""Package /opt/coinrun from a verified runner image into a portable bundle.

Publishing a 26.5 GB OCI image is unnecessary: only /opt/coinrun (the uv
environment plus the warmed Procgen cache, ~6.9 GB) is not already present in
the pinned RunPod PyTorch base. This extracts exactly that subtree into a
content-addressed `.tar.zst`, and emits a small manifest that IS checked in.

The archive stays untracked under artifacts/coinrun_bundle/. The manifest is the
contract: scripts/runpod_coinrun.py validates it against the local uv.lock and
against the archive's size and SHA-256 before any paid mutation, and the pod
re-verifies the SHA after transfer.

RECONSTRUCTION LIMITS (see docs/ops/COINRUN_RUNNER_IMAGE.md). The manifest lets
anyone re-derive an equivalent bundle -- same Dockerfile SHA, same uv.lock SHA,
same pinned base digest -- but not a byte-identical one. tar metadata is
normalised (sorted names, mtime 0, uid/gid 0), yet the payload still contains
absolute paths, `UV_COMPILE_BYTECODE=1` .pyc files whose headers embed source
mtimes, and uv cache entries with build-time-dependent contents. Treat
reconstruction as semantic, not bit-exact: rebuild the image from the recorded
Dockerfile and lock, re-run this tool, and expect a different archive SHA with
the same behaviour.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = REPO_ROOT / "artifacts" / "coinrun_bundle"
MANIFEST_PATH = REPO_ROOT / "manifests" / "coinrun_runner_bundle.json"
TELEMETRY_PATH = BUNDLE_DIR / "package_telemetry.jsonl"
SCHEMA = "coinrun-runner-bundle-v1"
IMAGE_ROOT = "/opt/coinrun"
EXTRACT_PARENT = "/opt"
EXPECTED_TOP_LEVEL = "coinrun"
# Exported verbatim on the pod; the bundle carries no image ENV of its own.
CONTRACT_ENV_KEYS = (
    "COINRUN_IMAGE_CONTRACT",
    "COINRUN_SOURCE_COMMIT",
    "COINRUN_UV_LOCK_SHA256",
    "UV_PROJECT_ENVIRONMENT",
    "UV_CACHE_DIR",
    "UV_PYTHON_INSTALL_DIR",
    "UV_LINK_MODE",
    "UV_COMPILE_BYTECODE",
    "UV_PYTHON_DOWNLOADS",
)
LOCK_LABEL = "io.coinrun.uv.lock.sha256"


class BundleError(RuntimeError):
    """The bundle could not be built from a trustworthy input."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_text(command: Sequence[str]) -> str:
    result = subprocess.run(list(command), text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise BundleError(
            f"command failed ({' '.join(command[:3])}...): {result.stderr.strip()[:400]}"
        )
    return result.stdout.strip()


def tool_versions(docker: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name, command in (
        ("docker", [*docker, "--version"]),
        ("tar", ["tar", "--version"]),
        ("zstd", ["zstd", "--version"]),
    ):
        try:
            versions[name] = run_text(command).splitlines()[0]
        except (BundleError, OSError, IndexError):
            pass
    return versions


def inspect_image(docker: Sequence[str], reference: str) -> dict[str, Any]:
    raw = run_text([*docker, "image", "inspect", reference, "--format", "{{json .}}"])
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BundleError(f"docker image inspect returned invalid JSON: {exc}") from exc
    config = info.get("Config") or {}
    return {
        "id": info.get("Id", ""),
        "labels": config.get("Labels") or {},
        "env": dict(
            entry.split("=", 1) for entry in (config.get("Env") or []) if "=" in entry
        ),
        "repo_digests": info.get("RepoDigests") or [],
    }


def validate_source_image(info: dict[str, Any], lock_sha256: str) -> dict[str, str]:
    """Refuse to package an image that does not match this checkout's lock."""

    env = info["env"]
    if env.get("COINRUN_IMAGE_CONTRACT") != "1":
        raise BundleError(
            "source image does not declare COINRUN_IMAGE_CONTRACT=1; it is not a "
            "CoinRun runner image"
        )
    image_lock = (env.get("COINRUN_UV_LOCK_SHA256") or "").strip().lower()
    label_lock = (info["labels"].get(LOCK_LABEL) or "").strip().lower()
    if image_lock != label_lock:
        raise BundleError(
            f"image label {LOCK_LABEL}={label_lock!r} disagrees with "
            f"COINRUN_UV_LOCK_SHA256={image_lock!r}"
        )
    if image_lock != lock_sha256:
        raise BundleError(
            f"image was built for uv.lock {image_lock} but this checkout has "
            f"{lock_sha256}; rebuild the image before packaging"
        )
    contract = {key: env[key] for key in CONTRACT_ENV_KEYS if key in env}
    missing = [key for key in CONTRACT_ENV_KEYS if key not in contract]
    if missing:
        raise BundleError(f"image is missing contract env: {', '.join(missing)}")
    return contract


def archive_command(docker: Sequence[str], reference: str) -> list[str]:
    """Deterministic tar of /opt/coinrun, streamed to stdout.

    --sort=name fixes entry order, --mtime=@0 and numeric owner 0:0 strip
    host-varying metadata. The tar is produced inside the image so the payload
    is exactly what the runtime saw.
    """

    return [
        *docker, "run", "--rm", "--entrypoint", "tar", reference,
        "--sort=name", "--mtime=@0", "--owner=0", "--group=0", "--numeric-owner",
        "--format=gnu", "-C", EXTRACT_PARENT, "-cf", "-", EXPECTED_TOP_LEVEL,
    ]


def compress_command(level: int, threads: int, destination: Path) -> list[str]:
    if not 1 <= level <= 19:
        raise BundleError("zstd level must be in [1, 19]")
    return ["zstd", f"-{level}", f"-T{threads}", "-q", "-f", "-o", str(destination)]


def package(
    docker: Sequence[str],
    reference: str,
    destination: Path,
    *,
    level: int,
    threads: int,
) -> dict[str, Any]:
    """Stream the container tar through zstd, hashing the raw tar on the way."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    tar_digest = hashlib.sha256()
    tar_bytes = 0
    started = time.monotonic()
    producer = subprocess.Popen(
        archive_command(docker, reference), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    compressor = subprocess.Popen(
        compress_command(level, threads, destination),
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert producer.stdout is not None and compressor.stdin is not None
    try:
        while chunk := producer.stdout.read(1 << 22):
            tar_digest.update(chunk)
            tar_bytes += len(chunk)
            compressor.stdin.write(chunk)
    finally:
        compressor.stdin.close()
        producer.stdout.close()
    producer_error = (producer.stderr.read().decode() if producer.stderr else "")[:400]
    compressor_error = (compressor.stderr.read().decode() if compressor.stderr else "")[:400]
    if producer.wait() != 0:
        raise BundleError(f"tar inside the image failed: {producer_error}")
    if compressor.wait() != 0:
        raise BundleError(f"zstd failed: {compressor_error}")
    return {
        "tar_sha256": tar_digest.hexdigest(),
        "tar_bytes": tar_bytes,
        "package_seconds": round(time.monotonic() - started, 3),
    }


def build_manifest(
    *,
    info: dict[str, Any],
    contract_env: dict[str, str],
    lock_sha256: str,
    dockerfile_sha256: str,
    base_reference: str,
    base_digest: str,
    build_inputs: dict[str, str],
    archive: Path,
    archive_facts: dict[str, Any],
    reference: str,
    versions: dict[str, str],
    level: int,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "created_utc": isoformat(utc_now()),
        "archive": {
            "filename": archive.name,
            "sha256": sha256_file(archive),
            "bytes": archive.stat().st_size,
            "compression": "zstd",
            "compression_level": level,
            "uncompressed_tar_sha256": archive_facts["tar_sha256"],
            "uncompressed_bytes": archive_facts["tar_bytes"],
        },
        "extract": {
            "target": IMAGE_ROOT,
            "parent": EXTRACT_PARENT,
            "expected_top_level": EXPECTED_TOP_LEVEL,
        },
        "source": {
            "commit": contract_env["COINRUN_SOURCE_COMMIT"],
            "uv_lock_sha256": lock_sha256,
            "dockerfile_sha256": dockerfile_sha256,
            "image_reference": reference,
            "image_id": info["id"],
        },
        "base_image": {"reference": base_reference, "digest": base_digest},
        # Hashes of the files that determine what ends up inside /opt/coinrun.
        # scripts/coinrun_runner.py is shipped at /opt/coinrun/runner/, so the
        # launcher gates on it. The Dockerfile and generator are recorded as
        # rebuild triggers but not gated -- see the manifest section of
        # docs/ops/COINRUN_RUNNER_IMAGE.md.
        "build_inputs": build_inputs,
        "contract_env": contract_env,
        "build": {
            "command": " ".join(archive_command(["docker"], reference)),
            "tool_versions": versions,
            "determinism": (
                "tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner; "
                "payload still contains absolute paths and bytecode, so "
                "reconstruction is semantic, not bit-exact"
            ),
        },
    }


def resolve_base(docker: Sequence[str], reference: str) -> tuple[str, str]:
    info = inspect_image(docker, reference)
    digests = [d for d in info["repo_digests"] if "@sha256:" in d]
    if not digests:
        raise BundleError(
            f"base image {reference} has no repo digest locally; run "
            f"`docker pull {reference}` so the pinned digest can be recorded"
        )
    return reference, digests[0].split("@", 1)[1]


# Recorded in every manifest. Only coinrun_runner_sha256 gates compatibility in
# the launcher (that file is shipped inside the bundle); the others are rebuild
# triggers an operator should think about, not automatic failures.
BUILD_INPUTS = {
    "uv_lock_sha256": "uv.lock",
    "dockerfile_sha256": "Dockerfile",
    "generator_sha256": "dreamer/data/generate_coinrun_dataset.py",
    "coinrun_runner_sha256": "scripts/coinrun_runner.py",
}


def collect_build_inputs() -> dict[str, str]:
    return {key: sha256_file(REPO_ROOT / path) for key, path in BUILD_INPUTS.items()}


def append_telemetry(record: dict[str, Any]) -> None:
    TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TELEMETRY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Local runner image to package")
    parser.add_argument(
        "--base-image",
        default="runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
        help="Pinned RunPod base the pod will actually run",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--bundle-dir", type=Path, default=BUNDLE_DIR)
    parser.add_argument("--zstd-level", type=int, default=9)
    parser.add_argument("--zstd-threads", type=int, default=0, help="0 means all cores")
    parser.add_argument("--docker", default="sudo -n docker")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    docker = args.docker.split()
    lock_sha256 = sha256_file(REPO_ROOT / "uv.lock")
    dockerfile_sha256 = sha256_file(REPO_ROOT / "Dockerfile")
    started_utc, monotonic = utc_now(), time.monotonic()
    record: dict[str, Any] = {
        "schema": "coinrun-bundle-package-telemetry-v1",
        "started_utc": isoformat(started_utc),
        "image": args.image,
        "cpu_count": os.cpu_count(),
        "disk_free_bytes": shutil.disk_usage(REPO_ROOT).free,
    }
    try:
        info = inspect_image(docker, args.image)
        contract_env = validate_source_image(info, lock_sha256)
        base_reference, base_digest = resolve_base(docker, args.base_image)
        commit = contract_env["COINRUN_SOURCE_COMMIT"]
        archive = Path(args.bundle_dir) / f"coinrun-opt-{commit[:12]}.tar.zst"
        facts = package(
            docker, args.image, archive,
            level=args.zstd_level, threads=args.zstd_threads,
        )
        manifest = build_manifest(
            info=info, contract_env=contract_env, lock_sha256=lock_sha256,
            dockerfile_sha256=dockerfile_sha256, base_reference=base_reference,
            base_digest=base_digest, build_inputs=collect_build_inputs(),
            archive=archive, archive_facts=facts,
            reference=args.image, versions=tool_versions(docker),
            level=args.zstd_level,
        )
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        record.update({
            "returncode": 0,
            "archive_path": str(archive.relative_to(REPO_ROOT)),
            "archive_bytes": manifest["archive"]["bytes"],
            "archive_sha256": manifest["archive"]["sha256"],
            "uncompressed_bytes": facts["tar_bytes"],
            "package_seconds": facts["package_seconds"],
            "manifest_path": str(Path(args.manifest).relative_to(REPO_ROOT)),
        })
        print(
            f"bundle {archive.name}: {manifest['archive']['bytes']:,} B "
            f"(from {facts['tar_bytes']:,} B) in {facts['package_seconds']:.1f}s\n"
            f"sha256 {manifest['archive']['sha256']}\nmanifest {args.manifest}"
        )
        return 0
    except BundleError as exc:
        record.update({"returncode": 1, "error": str(exc)})
        print(f"build_coinrun_bundle: {exc}", file=sys.stderr)
        return 1
    finally:
        record["ended_utc"] = isoformat(utc_now())
        record["duration_seconds"] = round(time.monotonic() - monotonic, 3)
        append_telemetry(record)


if __name__ == "__main__":
    raise SystemExit(main())
