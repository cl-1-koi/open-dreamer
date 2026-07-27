#!/usr/bin/env python3
"""Thin container entrypoint for the prebuilt CoinRun runner image.

This is an adapter, not an orchestrator. It verifies that the pod is the image
and commit we intended, then hands off to code that already exists:

``smoke``       tiny random+scripted dataset via the existing generator,
                one JSON result, hard aggregate budget.
``experiment``  ``exec`` into scripts/run_coinrun_h100_experiment.sh so its own
                deadlines and signal handling apply unchanged.

RunPod lifecycle, artifact packing and log streaming stay in
scripts/runpod_coinrun.py. Scientific thresholds stay in the controller.
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
from pathlib import Path
from typing import Any, Sequence

IMAGE_CONTRACT = "1"
SMOKE_BUDGET_SECONDS = 300
SUPPORTED_GPUS = ("H100", "H200", "B200")
REQUIRED_BINARIES = ("git", "uv", "timeout", "grep", "ffmpeg", "nvidia-smi")
REQUIRED_IMPORTS = ("jax", "flax", "optax", "grain", "hydra", "imageio", "array_record")
# Two episodes per split proves the Procgen toolchain and record round-trip.
# Dataset scale for real runs lives in the experiment controller.
SMOKE_EPISODES = 2

GPU_PROBE = """
import json, jax
print(json.dumps([{"kind": d.device_kind, "platform": d.platform} for d in jax.devices()]))
"""


class ContractError(RuntimeError):
    """A precondition for doing paid work on this pod is not satisfied."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tail(text: str, limit: int = 2000) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "...[truncated]...\n" + text[-limit:]


def run(command: Sequence[str], *, cwd: Path, env: dict[str, str], timeout: float):
    return subprocess.run(
        list(command), cwd=str(cwd), env=env, timeout=timeout,
        text=True, capture_output=True, check=False,
    )


def child_env(checkout: Path) -> dict[str, str]:
    """The image already holds the environment; forbid re-resolving it here.

    PYTHONPATH points at the checkout because the image deliberately does not
    install the project -- otherwise a pod could import image code instead of
    the pushed commit.
    """
    env = dict(os.environ)
    env["UV_NO_SYNC"] = "1"
    env["UV_FROZEN"] = "1"
    env["UV_OFFLINE"] = "1"
    env.setdefault("UV_PYTHON_DOWNLOADS", "never")
    checkout_path = str(checkout)
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{checkout_path}:{prior}" if prior else checkout_path
    return env


def check_contract(checkout: Path, expect_commit: str | None) -> dict[str, Any]:
    """Immutable-image metadata, exact commit, and image-vs-checkout uv.lock."""
    if os.environ.get("COINRUN_IMAGE_CONTRACT") != IMAGE_CONTRACT:
        raise ContractError(
            "COINRUN_IMAGE_CONTRACT is not "
            f"{IMAGE_CONTRACT}; this entrypoint must run inside the prebuilt runner image"
        )
    image_lock = (os.environ.get("COINRUN_UV_LOCK_SHA256") or "").strip().lower()
    if len(image_lock) != 64:
        raise ContractError("image does not record COINRUN_UV_LOCK_SHA256")

    lock = checkout / "uv.lock"
    if not lock.is_file():
        raise ContractError(f"checkout has no uv.lock at {lock}")
    checkout_lock = sha256_file(lock)
    if checkout_lock != image_lock:
        raise ContractError(
            f"uv.lock mismatch: image={image_lock} checkout={checkout_lock}. The "
            "prebuilt environment does not match this commit; rebuild and "
            "republish the runner image instead of syncing on the pod."
        )

    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    )
    if head.returncode != 0:
        raise ContractError(f"cannot resolve HEAD in {checkout}: {tail(head.stderr, 300)}")
    commit = head.stdout.strip()
    if expect_commit and commit != expect_commit.strip().lower():
        raise ContractError(f"checkout HEAD {commit} != expected commit {expect_commit}")
    return {
        "uv_lock_sha256": checkout_lock,
        "commit": commit,
        "expected_commit": expect_commit,
        "image_source_commit": os.environ.get("COINRUN_SOURCE_COMMIT"),
    }


def check_tools(checkout: Path, env: dict[str, str]) -> dict[str, Any]:
    missing = [name for name in REQUIRED_BINARIES if shutil.which(name) is None]
    if missing:
        raise ContractError(f"missing required binaries: {', '.join(missing)}")
    program = "import importlib;" + "".join(
        f"importlib.import_module({name!r});" for name in REQUIRED_IMPORTS
    )
    result = run([sys.executable, "-c", program], cwd=checkout, env=env, timeout=180)
    if result.returncode != 0:
        raise ContractError(f"dependency import check failed: {tail(result.stderr)}")
    return {"binaries": list(REQUIRED_BINARIES), "imports": list(REQUIRED_IMPORTS)}


def check_gpu(checkout: Path, env: dict[str, str]) -> dict[str, Any]:
    """Exactly one attached CUDA device, of a supported model."""
    result = run([sys.executable, "-c", GPU_PROBE], cwd=checkout, env=env, timeout=180)
    if result.returncode != 0:
        raise ContractError(f"JAX GPU probe failed: {tail(result.stderr)}")
    devices = json.loads(result.stdout.strip().splitlines()[-1])
    cuda = [d for d in devices if str(d.get("platform", "")).lower() in ("gpu", "cuda")]
    if len(cuda) != 1:
        raise ContractError(f"expected exactly one CUDA device, JAX reported {devices}")
    kind = str(cuda[0].get("kind", ""))
    if not any(model.lower() in kind.lower() for model in SUPPORTED_GPUS):
        raise ContractError(f"GPU {kind!r} is not one of {', '.join(SUPPORTED_GPUS)}")
    return {"device_kind": kind, "device_count": len(cuda)}


def dataset_command(checkout: Path, collector: str, out: Path, seed: int) -> list[str]:
    """Same invocation shape the experiment controller uses, at smoke scale."""
    return [
        "uv", "run", "--isolated", "--script",
        str(checkout / "dreamer" / "data" / "generate_coinrun_dataset.py"),
        f"--collector={collector}", f"--output-dir={out}", f"--seed={seed}",
        f"--num-episodes-train={SMOKE_EPISODES}",
        f"--num-episodes-val={SMOKE_EPISODES}",
        f"--num-episodes-test={SMOKE_EPISODES}",
        "--max-episode-length=64", "--chunk-size=32", "--chunks-per-file=1",
        "--overwrite",
    ]


def run_smoke(args: argparse.Namespace, checkout: Path, env: dict[str, str]) -> dict[str, Any]:
    started = time.monotonic()
    commands: list[dict[str, Any]] = []
    workdir = Path(args.artifact_dir) / "runner" / "smoke-dataset"
    for collector in ("random", "scripted"):
        remaining = args.budget_seconds - (time.monotonic() - started)
        if remaining <= 5:
            raise ContractError(f"smoke budget of {args.budget_seconds}s exhausted")
        out = workdir / collector
        phase = time.monotonic()
        try:
            result = run(
                dataset_command(checkout, collector, out, args.seed),
                cwd=checkout, env=env, timeout=remaining,
            )
        except subprocess.TimeoutExpired as exc:
            raise ContractError(f"{collector} generation exceeded the smoke budget") from exc
        shards = sorted(out.rglob("*.array_record"))
        commands.append({
            "collector": collector,
            "returncode": result.returncode,
            "elapsed_seconds": round(time.monotonic() - phase, 3),
            "shard_count": len(shards),
            "stderr_tail": tail(result.stderr) if result.returncode else "",
        })
        if result.returncode != 0:
            raise ContractError(
                f"{collector} generation failed (rc={result.returncode})",
                details={"commands": commands},
            )
        if not shards:
            raise ContractError(
                f"{collector} generation wrote no .array_record shards",
                details={"commands": commands},
            )
    return {"commands": commands, "elapsed_seconds": round(time.monotonic() - started, 3)}


def exec_experiment(args: argparse.Namespace, checkout: Path, env: dict[str, str]) -> None:
    """Replace this process so the controller owns signals and its own deadline."""
    controller = checkout / "scripts" / "run_coinrun_h100_experiment.sh"
    if not controller.is_file():
        raise ContractError(f"experiment controller missing at {controller}")
    command = [
        "bash", str(controller),
        "--artifact-dir", str(Path(args.artifact_dir) / "experiment"),
        "--seed", str(args.seed), "--preset", args.preset,
    ]
    print(f"coinrun-runner: exec {' '.join(command)}", flush=True)
    os.chdir(checkout)
    os.execvpe("bash", command, env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("smoke", "experiment"):
        mode = sub.add_parser(name)
        mode.add_argument(
            "--checkout-root",
            default=os.environ.get("COINRUN_CHECKOUT_ROOT", "/workspace/open-dreamer"),
        )
        mode.add_argument(
            "--artifact-dir",
            default=os.environ.get("COINRUN_ARTIFACT_DIR", "/workspace/coinrun-artifacts"),
        )
        mode.add_argument("--expect-commit", default=os.environ.get("COINRUN_COMMIT_SHA") or None)
        mode.add_argument("--seed", type=int, default=20260726)
    sub.choices["smoke"].add_argument(
        "--budget-seconds", type=int, default=SMOKE_BUDGET_SECONDS
    )
    sub.choices["experiment"].add_argument(
        "--preset", choices=("initial", "fallback"), default="initial"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    checkout = Path(args.checkout_root).resolve()
    runner_dir = Path(args.artifact_dir).expanduser().resolve() / "runner"
    runner_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir = str(runner_dir.parent)
    result: dict[str, Any] = {"mode": args.mode, "status": "failed"}
    try:
        if getattr(args, "budget_seconds", 1) <= 0:
            raise ContractError("--budget-seconds must be positive")
        env = child_env(checkout)
        result["contract"] = check_contract(checkout, args.expect_commit)
        result["tools"] = check_tools(checkout, env)
        result["gpu"] = check_gpu(checkout, env)
        if args.mode == "experiment":
            exec_experiment(args, checkout, env)  # never returns
        result["smoke"] = run_smoke(args, checkout, env)
        result["status"] = "passed"
    except ContractError as exc:
        result["error"] = str(exc)
        if exc.details is not None:
            result["failure_details"] = exc.details
    except Exception as exc:  # noqa: BLE001 - never lose the diagnosis
        result["error"] = f"{type(exc).__name__}: {exc}"
    path = runner_dir / f"{args.mode}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "passed":
        print(f"coinrun-runner: {result['error']}", file=sys.stderr)
        print(f"coinrun-runner: diagnostics at {path}", file=sys.stderr, flush=True)
        return 1
    print(f"coinrun-runner: {args.mode} passed; result at {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
