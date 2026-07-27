#!/usr/bin/env python3
"""Bounded CoinRun reconstruction smoke test and telemetry collector.

The parent process orchestrates the existing dataset and training commands. An
internal child mode imports the official trainers and wraps their existing
``train_step`` call, so timing and metrics are captured without maintaining a
second training loop.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pickle
import platform
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_RUNTIME_SECONDS = 20 * 60
DEFAULT_STEPS = 3
COINRUN_ACTION_DIM = 15
COINRUN_NOOP_ACTION = 4
MAX_CHECKPOINT_INTERVAL_SECONDS = 15 * 60
DEFAULT_DATASET_MAX_EPISODE_LENGTH = 256
DEFAULT_LATENT_STAT_MAX_RECORDS = 4
DEFAULT_LATENT_STD_EPSILON = 1e-6
MIB = 1024 * 1024


class PreflightError(RuntimeError):
    """A clear, user-facing preflight failure."""


class HardRuntimeExceeded(PreflightError):
    """The global smoke deadline was reached."""


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    elapsed_seconds: float
    output: str
    gpu_samples_mb: list[float]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _run_quiet(command: Sequence[str], cwd: Path) -> str:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PreflightError(f"Command failed ({' '.join(command)}): {detail}")
    return completed.stdout.strip()


def collect_git_state(repo_root: Path) -> dict[str, Any]:
    status = _run_quiet(["git", "status", "--porcelain"], repo_root)
    return {
        "commit": _run_quiet(["git", "rev-parse", "HEAD"], repo_root),
        "branch": _run_quiet(["git", "branch", "--show-current"], repo_root),
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def query_gpu_inventory() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        return []

    inventory = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            inventory.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "memory_used_mb": float(fields[2]),
                    "memory_total_mb": float(fields[3]),
                }
            )
        except ValueError:
            continue
    return inventory


def _query_nvidia_memory_mb() -> float | None:
    inventory = query_gpu_inventory()
    if not inventory:
        return None
    return sum(float(device["memory_used_mb"]) for device in inventory)


class _GpuMonitor:
    def __init__(self, interval_seconds: float = 0.25):
        self.interval_seconds = interval_seconds
        self.samples_mb: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            return

        def sample() -> None:
            while not self._stop.is_set():
                value = _query_nvidia_memory_mb()
                if value is not None:
                    self.samples_mb.append(value)
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5)


def run_bounded_command(
    command: Sequence[str],
    *,
    cwd: Path,
    deadline: float,
    env: dict[str, str] | None = None,
    monitor_gpu: bool = True,
) -> CommandResult:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise HardRuntimeExceeded("Hard runtime bound reached before command launch")

    command_list = [str(part) for part in command]
    child_env = os.environ.copy()
    child_env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "WANDB_MODE": "offline",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    if env:
        child_env.update(env)

    started = time.monotonic()
    process = subprocess.Popen(
        command_list,
        cwd=cwd,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    monitor = _GpuMonitor()
    if monitor_gpu:
        monitor.start()
    try:
        try:
            output_bytes, _ = process.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            process.communicate()
            raise HardRuntimeExceeded(
                f"Hard runtime bound reached while running: {shlex.join(command_list)}"
            ) from None
    finally:
        monitor.stop()

    output = output_bytes.decode("utf-8", errors="replace")
    return CommandResult(
        command=command_list,
        returncode=process.returncode,
        elapsed_seconds=time.monotonic() - started,
        output=output,
        gpu_samples_mb=monitor.samples_mb,
    )


def _event_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def emit(kind: str, **payload: Any) -> None:
        event = {"kind": kind, "time": _utc_now(), **payload}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(event), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    return emit


def _jax_memory_snapshot() -> dict[str, Any]:
    try:
        import jax
    except ImportError:
        return {"available": False}

    devices = [device for device in jax.local_devices() if device.platform == "gpu"]
    snapshots = []
    for device in devices:
        stats = device.memory_stats() or {}
        snapshots.append(
            {
                "device": str(device),
                "bytes_in_use": stats.get("bytes_in_use"),
                "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
                "bytes_limit": stats.get("bytes_limit"),
            }
        )
    current_values = [
        item["bytes_in_use"] for item in snapshots if item["bytes_in_use"] is not None
    ]
    peak_values = [
        item["peak_bytes_in_use"]
        for item in snapshots
        if item["peak_bytes_in_use"] is not None
    ]
    return {
        "available": bool(snapshots),
        "source": "jax",
        "current_mb": sum(current_values) / MIB if current_values else None,
        "peak_mb": sum(peak_values) / MIB if peak_values else None,
        "devices": snapshots,
    }


def _block_until_ready(value: Any) -> None:
    import jax

    try:
        jax.block_until_ready(value)
    except (AttributeError, TypeError):
        leaves = jax.tree_util.tree_leaves(value)
        for leaf in leaves:
            block = getattr(leaf, "block_until_ready", None)
            if block is not None:
                block()


def _numeric_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"result": _jsonable(value)}
    metrics = {}
    for key, item in value.items():
        converted = _jsonable(item)
        if isinstance(converted, (int, float)):
            metrics[str(key)] = converted
        elif isinstance(converted, list) and len(converted) <= 32:
            metrics[str(key)] = converted
    return metrics


def fatal_artifact_failures(output: str) -> list[str]:
    markers = {
        "consolidated MP4 write failed": "rollout MP4 writer failed",
        "produced no rollout MP4": "rollout MP4 artifact is absent",
    }
    return [description for marker, description in markers.items() if marker in output]


def build_default_dataset_command(
    *,
    repo_root: Path,
    dataset_dir: Path,
    sequence_length: int,
    seed: int,
    uv_executable: str | None = None,
) -> list[str]:
    uv = uv_executable or shutil.which("uv")
    if not uv:
        raise PreflightError(
            "Default CoinRun generation requires the uv executable for its "
            "isolated PEP 723 environment"
        )
    generator = repo_root / "dreamer" / "data" / "generate_coinrun_dataset.py"
    if not generator.is_file():
        raise PreflightError(f"Missing CoinRun dataset generator: {generator}")
    return [
        uv,
        "run",
        "--isolated",
        "--script",
        str(generator),
        "--num-episodes-train=2",
        "--num-episodes-val=1",
        "--num-episodes-test=1",
        f"--output-dir={dataset_dir}",
        f"--min-episode-length={sequence_length}",
        f"--max-episode-length={DEFAULT_DATASET_MAX_EPISODE_LENGTH}",
        f"--chunk-size={DEFAULT_DATASET_MAX_EPISODE_LENGTH}",
        "--chunks-per-file=1",
        "--collector=scripted",
        "--keep-short-terminated",
        f"--seed={seed}",
        "--overwrite",
    ]


def default_collection_scope(sequence_length: int) -> dict[str, Any]:
    return {
        "arm": "scripted_plumbing",
        "collector": "scripted",
        "purpose": "bounded plumbing and reward-prevalence gate",
        "max_episode_length": DEFAULT_DATASET_MAX_EPISODE_LENGTH,
        "chunk_size": DEFAULT_DATASET_MAX_EPISODE_LENGTH,
        "minimum_training_window": sequence_length,
        "keep_short_terminated": True,
        "scientific_comparison": "random-versus-scripted remains follow-on",
    }


def _load_trainer_module(stage: str, trainer_path: Path):
    module_name = f"_coinrun_preflight_{stage}_trainer"
    spec = importlib.util.spec_from_file_location(module_name, trainer_path)
    if spec is None or spec.loader is None:
        raise PreflightError(f"Could not import official trainer: {trainer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def instrumented_trainer_main(argv: Sequence[str]) -> int:
    if len(argv) < 3:
        raise PreflightError(
            "Internal trainer mode requires STAGE EVENT_PATH TRAINER_PATH"
        )
    stage = argv[0]
    event_path = Path(argv[1]).resolve()
    trainer_path = Path(argv[2]).resolve()
    hydra_args = list(argv[3:])
    if stage not in {"tokenizer", "dynamics"}:
        raise PreflightError(f"Unknown instrumented trainer stage: {stage}")

    emit = _event_writer(event_path)
    emit("child_started", stage=stage, trainer=str(trainer_path), argv=hydra_args)
    try:
        module = _load_trainer_module(stage, trainer_path)

        original_count = module.count_parameters_by_component

        def count_parameters(model):
            counts = original_count(model)
            emit("parameter_counts", stage=stage, counts=counts)
            return counts

        module.count_parameters_by_component = count_parameters

        original_step = module.train_step
        step_call = 0

        def timed_train_step(*args, **kwargs):
            nonlocal step_call
            started = time.perf_counter()
            result = original_step(*args, **kwargs)
            _block_until_ready(result)
            elapsed = time.perf_counter() - started
            emit(
                "training_step",
                stage=stage,
                call_index=step_call,
                step=int(kwargs.get("step", step_call)),
                elapsed_seconds=elapsed,
                phase="compile_and_first_step" if step_call == 0 else "steady",
                metrics=_numeric_metrics(result),
                gpu_memory=_jax_memory_snapshot(),
            )
            step_call += 1
            return result

        module.train_step = timed_train_step

        if stage == "dynamics":
            import importlib.util as importlib_util
            import dreamer.training as dreamer_training

            original_imwrite = dreamer_training.iio.imwrite
            pyav_available = importlib_util.find_spec("av") is not None

            def write_rollout_video(*args, **kwargs):
                requested_plugin = kwargs.get("plugin")
                selected_plugin = requested_plugin
                if requested_plugin == "pyav" and not pyav_available:
                    selected_plugin = "FFMPEG"
                    kwargs["plugin"] = selected_plugin
                emit(
                    "video_backend",
                    stage=stage,
                    requested_plugin=requested_plugin,
                    selected_plugin=selected_plugin,
                    pyav_available=pyav_available,
                )
                return original_imwrite(*args, **kwargs)

            dreamer_training.iio.imwrite = write_rollout_video
            original_shift_actions = module.shift_actions

            def checked_shift_actions(*args, **kwargs):
                import jax
                import numpy as np

                shifted = original_shift_actions(*args, **kwargs)
                categorical = shifted.categorical
                if categorical is None:
                    raise PreflightError(
                        "CoinRun dynamics produced no categorical actions after shifting"
                    )
                first_actions = np.asarray(
                    jax.device_get(categorical[:, 0])
                ).reshape(-1)
                passed = bool(
                    first_actions.size
                    and np.all(first_actions == COINRUN_NOOP_ACTION)
                )
                emit(
                    "action_semantics",
                    stage=stage,
                    expected_noop=COINRUN_NOOP_ACTION,
                    observed_first_actions=first_actions.tolist(),
                    passed=passed,
                )
                if not passed:
                    raise PreflightError(
                        "Action shift did not prepend CoinRun no-op 4; "
                        f"observed {first_actions.tolist()}"
                    )
                return shifted

            module.shift_actions = checked_shift_actions

        original_run = module.run

        def instrumented_run(cfg):
            from omegaconf import OmegaConf

            emit(
                "resolved_config",
                stage=stage,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            if stage == "dynamics":
                gate = validate_dynamics_preconditions(
                    OmegaConf.to_container(cfg, resolve=True)
                )
                emit("dynamics_preconditions", stage=stage, **gate)
            return original_run(cfg)

        module.run = instrumented_run

        import dreamer.logging as dreamer_logging

        original_build_logger = dreamer_logging.build_logger

        class CapturingLogger:
            def __init__(self, delegate):
                object.__setattr__(self, "_delegate", delegate)

            def __getattr__(self, name):
                return getattr(self._delegate, name)

            def __setattr__(self, name, value):
                if name == "_delegate":
                    object.__setattr__(self, name, value)
                else:
                    setattr(self._delegate, name, value)

            def __enter__(self):
                self._delegate.__enter__()
                return self

            def __exit__(self, exc_type, exc_value, exc_tb):
                return self._delegate.__exit__(exc_type, exc_value, exc_tb)

            def should_log(self, step):
                return self._delegate.should_log(step)

            def log(self, step, metrics, *args, **kwargs):
                emit(
                    "logger_metrics",
                    stage=stage,
                    step=int(step),
                    prefix=kwargs.get("prefix", "train/"),
                    metrics=_numeric_metrics(metrics),
                )
                return self._delegate.log(step, metrics, *args, **kwargs)

            def log_metrics(self, step, metrics, prefix="train/"):
                emit(
                    "logger_metrics",
                    stage=stage,
                    step=int(step),
                    prefix=prefix,
                    metrics=_numeric_metrics(metrics),
                )
                return self._delegate.log_metrics(step, metrics, prefix)

            def log_image(self, step, key, image, **kwargs):
                emit("image", stage=stage, step=int(step), key=key)
                return self._delegate.log_image(step, key, image, **kwargs)

            def log_video(self, step, key, video_path, **kwargs):
                emit(
                    "video",
                    stage=stage,
                    step=int(step),
                    key=key,
                    path=str(Path(video_path).resolve()),
                )
                return self._delegate.log_video(step, key, video_path, **kwargs)

        def build_capturing_logger(*args, **kwargs):
            return CapturingLogger(original_build_logger(*args, **kwargs))

        module.build_logger = build_capturing_logger

        bundle_class = (
            module.TokenizerCheckpointBundle
            if stage == "tokenizer"
            else module.DynamicsCheckpointBundle
        )
        original_restore = bundle_class.restore

        def restore_with_event(self, checkpoint_manager, rng):
            start_step, bundle, restored_rng = original_restore(
                self, checkpoint_manager, rng
            )
            checkpoint_directory = getattr(checkpoint_manager, "directory", "")
            if callable(checkpoint_directory):
                checkpoint_directory = checkpoint_directory()
            emit(
                "checkpoint_restore",
                stage=stage,
                restored=start_step > 0,
                start_step=int(start_step),
                checkpoint_dir=str(Path(checkpoint_directory).resolve()),
            )
            return start_step, bundle, restored_rng

        bundle_class.restore = restore_with_event

        sys.argv = [str(trainer_path), *hydra_args]
        hydra_task = getattr(module.main, "__wrapped__", None)
        if hydra_task is not None:
            hydra_task.__module__ = "__main__"
        module.main()
        emit("child_completed", stage=stage, training_steps=step_call)
        return 0
    except BaseException as exc:
        emit(
            "child_failed",
            stage=stage,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise


def validate_measured_latent_stats(
    payload: dict[str, Any],
    *,
    expected_dim: int | None = None,
    std_epsilon: float = DEFAULT_LATENT_STD_EPSILON,
) -> dict[str, Any]:
    mean = payload.get("latent_mean")
    std = payload.get("latent_std")
    if not isinstance(mean, list) or not isinstance(std, list):
        raise PreflightError(
            "Held-out tokenizer probe must produce list-valued latent_mean/latent_std"
        )
    if not mean or len(mean) != len(std):
        raise PreflightError(
            "Held-out tokenizer latent_mean/latent_std must have equal nonzero lengths"
        )
    if expected_dim is not None and len(mean) != expected_dim:
        raise PreflightError(
            f"Held-out tokenizer latent dimension {len(mean)} does not match "
            f"expected d_bottleneck={expected_dim}"
        )
    if std_epsilon <= 0:
        raise PreflightError("Latent standard-deviation epsilon must be positive")

    nonfinite_mean = [
        index for index, value in enumerate(mean) if not math.isfinite(float(value))
    ]
    nonfinite_std = [
        index for index, value in enumerate(std) if not math.isfinite(float(value))
    ]
    near_zero_std = [
        index for index, value in enumerate(std) if float(value) <= std_epsilon
    ]
    if nonfinite_mean or nonfinite_std or near_zero_std:
        raise PreflightError(
            "Held-out tokenizer latent statistics failed closed: "
            f"nonfinite_mean_dims={nonfinite_mean}, "
            f"nonfinite_std_dims={nonfinite_std}, "
            f"std_at_or_below_{std_epsilon:g}={near_zero_std}"
        )

    source = payload.get("source")
    if not isinstance(source, dict):
        raise PreflightError("Held-out tokenizer latent statistics lack source metadata")
    if source.get("split") != "val":
        raise PreflightError(
            f"Latent statistics must come from held-out val records, got "
            f"{source.get('split')!r}"
        )
    if source.get("tokenizer_variant") != "online":
        raise PreflightError(
            "Latent statistics must come from the restored online tokenizer"
        )
    checkpoint_step = source.get("checkpoint_step")
    if not isinstance(checkpoint_step, int) or checkpoint_step < 0:
        raise PreflightError(
            "Latent statistics source must identify a restored checkpoint step"
        )
    level_seeds = source.get("level_seeds")
    if not isinstance(level_seeds, list) or not level_seeds:
        raise PreflightError(
            "Latent statistics source must identify held-out val level seeds"
        )
    latent_sample_count = payload.get("latent_sample_count")
    frame_count = payload.get("frame_count")
    record_count = payload.get("record_count")
    if not isinstance(latent_sample_count, int) or latent_sample_count <= 0:
        raise PreflightError("Latent statistics have no bottleneck samples")
    if not isinstance(frame_count, int) or frame_count <= 0:
        raise PreflightError("Latent statistics have no held-out frames")
    if not isinstance(record_count, int) or record_count <= 0:
        raise PreflightError("Latent statistics have no held-out records")

    return {
        "passed": True,
        "dimensions": len(mean),
        "minimum_std": min(float(value) for value in std),
        "maximum_std": max(float(value) for value in std),
        "std_epsilon": std_epsilon,
        "latent_sample_count": latent_sample_count,
        "frame_count": frame_count,
        "record_count": record_count,
        "source": source,
    }


def build_latent_hydra_overrides(payload: dict[str, Any]) -> list[str]:
    validate_measured_latent_stats(payload)
    mean = json.dumps(payload["latent_mean"], separators=(",", ":"))
    std = json.dumps(payload["latent_std"], separators=(",", ":"))
    return [
        f"dynamics.latent_mean={mean}",
        f"dynamics.latent_std={std}",
    ]


def compute_tokenizer_probe_metrics(
    *,
    latents: Any,
    reconstructions: Any,
    targets: Any,
    dataset_mean: Sequence[float],
) -> dict[str, Any]:
    import numpy as np

    latent_array = np.asarray(latents, dtype=np.float64)
    reconstruction = np.asarray(reconstructions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if latent_array.ndim != 4:
        raise PreflightError(
            f"Tokenizer latents must have shape (B,T,N,D), got {latent_array.shape}"
        )
    if reconstruction.shape != target.shape or target.ndim != 5:
        raise PreflightError(
            "Tokenizer reconstruction and target must share shape (B,T,H,W,C); "
            f"got {reconstruction.shape} and {target.shape}"
        )

    flattened_latents = latent_array.reshape(-1, latent_array.shape[-1])
    latent_mean = np.mean(flattened_latents, axis=0)
    latent_std = np.std(flattened_latents, axis=0)

    target_unit = target / 255.0
    reconstruction_unit = np.clip(reconstruction, 0.0, 255.0) / 255.0
    channel_mean = np.asarray(dataset_mean, dtype=np.float64)
    if channel_mean.shape != (target.shape[-1],):
        raise PreflightError(
            f"Dataset mean shape {channel_mean.shape} does not match "
            f"target channels {target.shape[-1]}"
        )
    mean_prediction = np.broadcast_to(channel_mean, target_unit.shape)

    def mse(left, right) -> float:
        return float(np.mean(np.square(left - right), dtype=np.float64))

    def psnr(mse_value: float) -> float:
        return float(-10.0 * math.log10(max(mse_value, 1e-12)))

    reconstruction_mse = mse(reconstruction_unit, target_unit)
    dataset_mean_mse = mse(mean_prediction, target_unit)
    if target.shape[1] > 1:
        reconstruction_transition_mse = mse(
            reconstruction_unit[:, 1:], target_unit[:, 1:]
        )
        copy_previous_mse = mse(target_unit[:, :-1], target_unit[:, 1:])
    else:
        reconstruction_transition_mse = None
        copy_previous_mse = None

    reconstruction_metrics = {
        "pixel_range": "[0,1]",
        "model_mse": reconstruction_mse,
        "model_psnr_db": psnr(reconstruction_mse),
        "dataset_mean_baseline_mse": dataset_mean_mse,
        "dataset_mean_baseline_psnr_db": psnr(dataset_mean_mse),
        "beats_dataset_mean_baseline": reconstruction_mse < dataset_mean_mse,
        "model_to_dataset_mean_mse_ratio": (
            reconstruction_mse / dataset_mean_mse
            if dataset_mean_mse > 0
            else None
        ),
        "model_transition_mse": reconstruction_transition_mse,
        "copy_previous_baseline_mse": copy_previous_mse,
        "copy_previous_baseline_psnr_db": (
            psnr(copy_previous_mse) if copy_previous_mse is not None else None
        ),
        "beats_copy_previous_baseline": (
            reconstruction_transition_mse < copy_previous_mse
            if reconstruction_transition_mse is not None
            and copy_previous_mse is not None
            else None
        ),
        "model_to_copy_previous_mse_ratio": (
            reconstruction_transition_mse / copy_previous_mse
            if reconstruction_transition_mse is not None
            and copy_previous_mse is not None
            and copy_previous_mse > 0
            else None
        ),
        "gate_mode": "diagnostic_only_for_bounded_plumbing_smoke",
    }
    numeric_metrics = [
        value
        for value in reconstruction_metrics.values()
        if isinstance(value, float)
    ]
    if not all(math.isfinite(value) for value in numeric_metrics):
        raise PreflightError(
            "Held-out tokenizer reconstruction metrics contain nonfinite values"
        )

    return {
        "latent_mean": latent_mean.tolist(),
        "latent_std": latent_std.tolist(),
        "latent_sample_count": int(flattened_latents.shape[0]),
        "d_bottleneck": int(latent_array.shape[-1]),
        "n_latents": int(latent_array.shape[-2]),
        "reconstruction": reconstruction_metrics,
    }


def tokenizer_probe_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Internal bounded held-out tokenizer latent/reconstruction probe."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-length", required=True, type=int)
    parser.add_argument(
        "--max-records", type=int, default=DEFAULT_LATENT_STAT_MAX_RECORDS
    )
    parser.add_argument(
        "--std-epsilon", type=float, default=DEFAULT_LATENT_STD_EPSILON
    )
    args = parser.parse_args(argv)
    if args.sequence_length <= 1:
        raise PreflightError("Tokenizer probe sequence length must be greater than 1")
    if args.max_records <= 0:
        raise PreflightError("Tokenizer probe max records must be positive")

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx

    from dreamer.checkpointing import TokenizerCheckpointBundle
    from dreamer.parallel import build_parallel

    dataset_dir = Path(args.dataset_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_path = Path(args.output).resolve()
    clips = []
    level_seeds = []
    skipped_short_records = 0
    for split, raw_record in iter_array_records(dataset_dir):
        if split != "val":
            continue
        record = pickle.loads(raw_record)
        sequence_length = int(record["sequence_length"])
        if sequence_length < args.sequence_length:
            skipped_short_records += 1
            continue
        frame_shape = tuple(
            int(value) for value in record.get("frame_shape", (64, 64, 3))
        )
        frames = np.frombuffer(record["raw_video"], dtype=np.uint8).reshape(
            sequence_length, *frame_shape
        )
        clips.append(frames[: args.sequence_length])
        if "level_seed" in record:
            level_seeds.append(int(record["level_seed"]))
        if len(clips) >= args.max_records:
            break
    if not clips:
        raise PreflightError(
            "Held-out tokenizer probe found no val record long enough for "
            f"sequence_length={args.sequence_length}; "
            f"skipped_short_records={skipped_short_records}"
        )

    videos_np = np.stack(clips)
    mesh, _, mesh_rules = build_parallel("data")
    started = time.perf_counter()
    with jax.set_mesh(mesh):
        bundle = TokenizerCheckpointBundle.from_pretrained(
            str(checkpoint_dir),
            mesh_rules=mesh_rules,
            model_names={"tokenizer"},
        )
        tokenizer = bundle.tokenizer

        @nnx.jit
        def probe_step(model, videos):
            latent_values, _, _ = model.encode(videos, deterministic=True)
            reconstructed, _ = model.decode(latent_values, deterministic=True)
            return latent_values, reconstructed

        latents, reconstructions = probe_step(
            tokenizer, jnp.asarray(videos_np)
        )
        latents, reconstructions = jax.device_get((latents, reconstructions))
    elapsed = time.perf_counter() - started

    metrics = compute_tokenizer_probe_metrics(
        latents=latents,
        reconstructions=reconstructions,
        targets=videos_np,
        dataset_mean=tuple(tokenizer.cfg.encoder.dataset_mean),
    )
    checkpoint_steps = [
        int(path.name)
        for path in checkpoint_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    payload = {
        "schema_version": 1,
        **metrics,
        "frame_count": int(videos_np.shape[0] * videos_np.shape[1]),
        "record_count": len(clips),
        "sequence_length": args.sequence_length,
        "skipped_short_records": skipped_short_records,
        "source": {
            "split": "val",
            "level_seeds": sorted(set(level_seeds)),
            "tokenizer_variant": "online",
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_step": max(checkpoint_steps) if checkpoint_steps else None,
            "dataset_dir": str(dataset_dir),
        },
        "elapsed_seconds": elapsed,
        "gpu_memory": _jax_memory_snapshot(),
    }
    payload["validation"] = validate_measured_latent_stats(
        payload,
        expected_dim=metrics["d_bottleneck"],
        std_epsilon=args.std_epsilon,
    )
    _write_json(output_path, payload)
    print(
        f"Held-out tokenizer probe wrote {output_path}: "
        f"{payload['frame_count']} frames, "
        f"{payload['latent_sample_count']} latent samples"
    )
    return 0


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise PreflightError(
                f"Malformed telemetry event at {path}:{line_number}: {exc}"
            ) from exc
    return events


def _nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def summarize_training_events(
    events: Sequence[dict[str, Any]],
    nvidia_samples_mb: Sequence[float] = (),
) -> dict[str, Any]:
    step_events = [event for event in events if event.get("kind") == "training_step"]
    config_events = [event for event in events if event.get("kind") == "resolved_config"]
    parameter_events = [
        event for event in events if event.get("kind") == "parameter_counts"
    ]
    restore_events = [
        event for event in events if event.get("kind") == "checkpoint_restore"
    ]
    logger_events = [event for event in events if event.get("kind") == "logger_metrics"]
    video_events = [event for event in events if event.get("kind") == "video"]
    video_backend_events = [
        event for event in events if event.get("kind") == "video_backend"
    ]

    config = config_events[-1].get("config", {}) if config_events else {}
    durations = [float(event["elapsed_seconds"]) for event in step_events]
    steady_durations = durations[1:]
    steady_seconds = statistics.median(steady_durations) if steady_durations else None
    first_seconds = durations[0] if durations else None
    compile_estimate = (
        max(0.0, first_seconds - steady_seconds)
        if first_seconds is not None and steady_seconds is not None
        else None
    )

    batch_size = _nested(config, "dataset", "dataloader_cfg", "B")
    sequence_length = _nested(config, "dataset", "dataloader_cfg", "long_T")
    examples_per_second = None
    frames_per_second = None
    if steady_seconds and batch_size:
        examples_per_second = float(batch_size) / steady_seconds
        if sequence_length:
            frames_per_second = (
                float(batch_size) * float(sequence_length) / steady_seconds
            )

    jax_memories = [
        event.get("gpu_memory", {})
        for event in step_events
        if event.get("gpu_memory", {}).get("available")
    ]
    jax_peaks = [
        float(memory["peak_mb"])
        for memory in jax_memories
        if memory.get("peak_mb") is not None
    ]
    jax_current = [
        float(memory["current_mb"])
        for memory in jax_memories[1:] or jax_memories
        if memory.get("current_mb") is not None
    ]
    nvidia_values = [float(value) for value in nvidia_samples_mb]
    gpu_memory = {
        "jax_peak_mb": max(jax_peaks) if jax_peaks else None,
        "jax_steady_mb": statistics.median(jax_current) if jax_current else None,
        "nvidia_smi_peak_mb": max(nvidia_values) if nvidia_values else None,
        "nvidia_smi_steady_mb": (
            statistics.median(nvidia_values[-5:]) if nvidia_values else None
        ),
        "sources": [
            source
            for source, present in (
                ("jax", bool(jax_memories)),
                ("nvidia-smi", bool(nvidia_values)),
            )
            if present
        ],
    }

    last_metrics = step_events[-1].get("metrics", {}) if step_events else {}
    losses = {
        key: value
        for key, value in last_metrics.items()
        if "loss" in key.lower() or "mse" in key.lower()
    }
    gradient_norms = {
        key: value
        for key, value in last_metrics.items()
        if "grad" in key.lower() and "norm" in key.lower()
    }
    evaluation_metrics = [
        event.get("metrics", {})
        for event in logger_events
        if event.get("prefix") == "eval/"
    ]
    evaluation_seconds = sum(
        float(value)
        for metrics in evaluation_metrics
        for key, value in metrics.items()
        if key.endswith("/eval_time") and isinstance(value, (int, float))
    )

    return {
        "resolved_config": config or None,
        "parameter_counts": parameter_events[-1].get("counts") if parameter_events else None,
        "timing": {
            "measured_steps": len(durations),
            "compile_and_first_step_seconds": first_seconds,
            "steady_step_seconds": steady_seconds,
            "jax_compile_seconds_estimate": compile_estimate,
            "all_step_seconds": durations,
        },
        "throughput": {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "examples_per_second": examples_per_second,
            "frames_per_second": frames_per_second,
        },
        "last_step_metrics": last_metrics or None,
        "losses": losses,
        "gradient_norms": gradient_norms,
        "gpu_memory": gpu_memory,
        "checkpoint_restore_events": restore_events,
        "logger_metrics": logger_events,
        "videos": video_events,
        "video_backends": video_backend_events,
        "evaluation": {
            "four_way_total_seconds": evaluation_seconds or None,
            "metrics": evaluation_metrics,
        },
    }


def find_coinrun_config(
    configs_dir: Path, stage: str, explicit: str | None = None
) -> str:
    if explicit:
        candidate = Path(explicit)
        if candidate.suffix == ".yaml":
            candidate = candidate.with_suffix("")
        config_path = configs_dir / candidate.with_suffix(".yaml")
        if not config_path.is_file():
            raise PreflightError(
                f"Missing {stage} config '{explicit}' at {config_path}"
            )
        return candidate.as_posix()

    candidates = []
    for path in configs_dir.rglob("*.yaml"):
        relative = path.relative_to(configs_dir).with_suffix("")
        normalized = relative.as_posix().lower()
        if "coinrun" in normalized and stage in normalized:
            candidates.append(relative.as_posix())

    if not candidates:
        raise PreflightError(
            f"Missing dedicated CoinRun {stage} config under {configs_dir}; "
            f"pass --{stage}-config once the owned config stage is present"
        )

    preferred_names = [
        f"coinrun_{stage}",
        f"{stage}_coinrun",
        f"coinrun/{stage}",
        f"{stage}/coinrun",
    ]
    for preferred in preferred_names:
        if preferred in candidates:
            return preferred
    return sorted(candidates)[0]


def load_coinrun_dataset_config(configs_dir: Path) -> dict[str, Any]:
    path = configs_dir / "dataset" / "coinrun.yaml"
    if not path.is_file():
        raise PreflightError(f"Missing CoinRun dataset config: {path}")
    try:
        from omegaconf import OmegaConf

        value = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    except Exception as exc:
        raise PreflightError(f"Could not load CoinRun dataset config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"CoinRun dataset config is not a mapping: {path}")
    return value


def validate_dataset_gates(
    stats: dict[str, Any],
    metadata: dict[str, Any],
    dataset_config: dict[str, Any],
    *,
    required_sequence_length: int | None = None,
) -> dict[str, Any]:
    errors = []
    metadata_action_dim = metadata.get("num_actions")
    config_action_dim = dataset_config.get("categorical_action_dim")
    metadata_noop = metadata.get("categorical_noop")
    config_noop = dataset_config.get("categorical_noop")

    if metadata_action_dim != COINRUN_ACTION_DIM:
        errors.append(
            f"dataset metadata num_actions={metadata_action_dim!r}, "
            f"expected {COINRUN_ACTION_DIM}"
        )
    if config_action_dim != COINRUN_ACTION_DIM:
        errors.append(
            f"dataset config categorical_action_dim={config_action_dim!r}, "
            f"expected {COINRUN_ACTION_DIM}"
        )
    if metadata_noop != COINRUN_NOOP_ACTION:
        errors.append(
            f"dataset metadata categorical_noop={metadata_noop!r}, "
            f"expected {COINRUN_NOOP_ACTION}"
        )
    if config_noop != COINRUN_NOOP_ACTION:
        errors.append(
            f"dataset config categorical_noop={config_noop!r}, "
            f"expected {COINRUN_NOOP_ACTION}"
        )

    histogram = stats.get("action_histogram", {})
    illegal_actions = sorted(
        int(action)
        for action in histogram
        if not 0 <= int(action) < COINRUN_ACTION_DIM
    )
    if illegal_actions:
        errors.append(f"dataset contains illegal action ids: {illegal_actions}")

    level_seeds = {
        split: set(values)
        for split, values in stats.get("level_seeds_by_split", {}).items()
    }
    required_splits = {"train", "val", "test"}
    missing_seed_splits = sorted(
        split for split in required_splits if not level_seeds.get(split)
    )
    if missing_seed_splits:
        errors.append(
            "level seeds are absent for splits: " + ", ".join(missing_seed_splits)
        )
    overlaps = {}
    split_pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left, right in split_pairs:
        overlap = sorted(level_seeds.get(left, set()) & level_seeds.get(right, set()))
        if overlap:
            overlaps[f"{left}:{right}"] = overlap
    if overlaps:
        errors.append(f"held-out level seed sets overlap: {overlaps}")

    reward_bias = float(dataset_config.get("p_include_reward", 0.0) or 0.0)
    reward_frames = int(stats.get("reward_nonzero_frames", 0))
    if reward_bias > 0 and reward_frames == 0:
        errors.append(
            f"p_include_reward={reward_bias} but retained nonzero-reward frames are zero"
        )

    if metadata.get("action_alignment") != "action_applied_after_frame":
        errors.append("dataset metadata does not declare action alignment")
    if metadata.get("reward_alignment") != "reward_resulting_from_action":
        errors.append("dataset metadata does not declare reward alignment")
    if int(stats.get("episode_start_frames", 0)) == 0:
        errors.append("dataset records contain no episode start boundary")
    if int(stats.get("episode_end_frames", 0)) == 0:
        errors.append("dataset records contain no episode end boundary")
    minimum_record_length = stats.get("minimum_record_length")
    minimum_by_split = stats.get("minimum_record_length_by_split", {})
    if required_sequence_length is not None:
        for split in ("train", "val"):
            split_minimum = minimum_by_split.get(split)
            if (
                not isinstance(split_minimum, int)
                or split_minimum < required_sequence_length
            ):
                errors.append(
                    f"minimum {split} record length {split_minimum!r} is shorter "
                    f"than training window {required_sequence_length}"
                )

    if errors:
        raise PreflightError(
            "CoinRun dataset gates failed closed:\n- " + "\n- ".join(errors)
        )
    return {
        "action_dim": COINRUN_ACTION_DIM,
        "categorical_noop": COINRUN_NOOP_ACTION,
        "held_out_level_seeds_disjoint": True,
        "level_seeds_by_split": {
            split: sorted(values) for split, values in sorted(level_seeds.items())
        },
        "reward_bias_probability": reward_bias,
        "reward_nonzero_frames": reward_frames,
        "reward_prevalence": stats.get("reward_prevalence"),
        "minimum_record_length": minimum_record_length,
        "minimum_record_length_by_split": minimum_by_split,
        "required_sequence_length": required_sequence_length,
    }


def validate_dynamics_preconditions(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config.get("dataset")
    dynamics = config.get("dynamics")
    if not isinstance(dataset, dict) or not isinstance(dynamics, dict):
        raise PreflightError(
            "Resolved dynamics config must contain dataset and dynamics mappings"
        )

    errors = []
    action_dim = dataset.get("categorical_action_dim")
    model_action_dim = dynamics.get("categorical_action_dim")
    noop = dataset.get("categorical_noop")
    if action_dim != COINRUN_ACTION_DIM or model_action_dim != COINRUN_ACTION_DIM:
        errors.append(
            "resolved dataset/model categorical action dimensions must both be 15 "
            f"(got {action_dim!r}/{model_action_dim!r})"
        )
    if noop != COINRUN_NOOP_ACTION:
        errors.append(
            f"resolved dataset categorical_noop must be 4 (got {noop!r})"
        )

    latent_mean = dynamics.get("latent_mean")
    latent_std = dynamics.get("latent_std")
    if dataset.get("data_type") == "video" and (
        latent_mean is None or latent_std is None
    ):
        errors.append(
            "video dynamics requires explicit held-out latent_mean and latent_std"
        )
    bottleneck_dim = dynamics.get("d_bottleneck")
    if latent_mean is not None and (
        not isinstance(latent_mean, (list, tuple))
        or len(latent_mean) != bottleneck_dim
    ):
        errors.append(
            f"latent_mean length must match d_bottleneck={bottleneck_dim!r}"
        )
    if latent_std is not None:
        if not isinstance(latent_std, (list, tuple)) or len(latent_std) != bottleneck_dim:
            errors.append(
                f"latent_std length must match d_bottleneck={bottleneck_dim!r}"
            )
        else:
            try:
                if any(float(value) <= 0 for value in latent_std):
                    errors.append("all latent_std values must be positive")
            except (TypeError, ValueError):
                errors.append("latent_std values must be numeric")

    if errors:
        raise PreflightError(
            "Dynamics preconditions failed closed:\n- " + "\n- ".join(errors)
        )
    return {
        "passed": True,
        "action_dim": action_dim,
        "categorical_noop": noop,
        "latent_stats_source": "resolved_config",
        "latent_dimensions": len(latent_mean) if latent_mean is not None else 0,
    }


def validate_checkpoint_cadence(
    training_summary: dict[str, Any],
    checkpoint_paths: Sequence[str],
    *,
    max_interval_seconds: float = MAX_CHECKPOINT_INTERVAL_SECONDS,
) -> dict[str, Any]:
    config = training_summary.get("resolved_config") or {}
    save_interval_steps = _nested(config, "ckpt", "save_interval_steps")
    steady_step_seconds = _nested(
        training_summary, "timing", "steady_step_seconds"
    )
    if not isinstance(save_interval_steps, int) or save_interval_steps <= 0:
        raise PreflightError(
            f"Invalid checkpoint save_interval_steps={save_interval_steps!r}"
        )
    if not isinstance(steady_step_seconds, (int, float)) or steady_step_seconds <= 0:
        raise PreflightError(
            f"Cannot project checkpoint cadence from steady step time "
            f"{steady_step_seconds!r}"
        )
    projected_seconds = save_interval_steps * float(steady_step_seconds)
    if projected_seconds > max_interval_seconds:
        raise PreflightError(
            f"Projected checkpoint interval is {projected_seconds:.1f}s, "
            f"exceeding the {max_interval_seconds:.0f}s bound"
        )
    if len(checkpoint_paths) < 2:
        raise PreflightError(
            "Smoke must retain at least two checkpoints to prove periodic and final saves"
        )
    return {
        "save_interval_steps": save_interval_steps,
        "projected_interval_seconds": projected_seconds,
        "max_interval_seconds": max_interval_seconds,
        "checkpoint_count": len(checkpoint_paths),
        "passed": True,
    }


def _action_values(actions: Any) -> list[int]:
    if isinstance(actions, dict):
        for key in ("categorical", "action", "actions"):
            if key in actions:
                return _action_values(actions[key])
        return []
    try:
        import numpy as np

        array = np.asarray(actions)
        if array.ndim > 1 and array.shape[-1] == 1:
            array = array[..., 0]
        return [int(value) for value in array.reshape(-1)]
    except (TypeError, ValueError):
        return []


def summarize_dataset_records(
    records: Iterable[tuple[str, bytes]],
    *,
    image_h: int = 64,
    image_w: int = 64,
    image_c: int = 3,
    deadline: float | None = None,
) -> dict[str, Any]:
    import numpy as np

    records_by_split: Counter[str] = Counter()
    frames_by_split: Counter[str] = Counter()
    action_histogram: Counter[int] = Counter()
    collector_histogram: Counter[str] = Counter()
    seeds: set[int] = set()
    level_seeds_by_split: dict[str, set[int]] = {}
    episode_outcomes: Counter[str] = Counter()
    observed_episode_outcomes: set[str] = set()
    episode_start_frames = 0
    episode_end_frames = 0
    reward_frames = 0
    positive_reward_frames = 0
    total_rewards = 0
    total_records = 0
    total_frames = 0
    decoded_value_min = 255
    decoded_value_max = 0
    record_lengths: list[int] = []
    record_lengths_by_split: dict[str, list[int]] = {}

    for split, raw_record in records:
        if deadline is not None and time.monotonic() >= deadline:
            raise HardRuntimeExceeded("Hard runtime bound reached while decoding dataset")
        try:
            record = pickle.loads(raw_record)
        except Exception as exc:
            raise PreflightError(f"Could not decode {split} dataset record: {exc}") from exc

        missing = [
            key
            for key in ("raw_video", "sequence_length", "actions", "rewards")
            if key not in record
        ]
        if missing:
            raise PreflightError(
                f"Decoded {split} record is missing required fields: {', '.join(missing)}"
            )

        sequence_length = int(record["sequence_length"])
        expected_bytes = sequence_length * image_h * image_w * image_c
        video_bytes = record["raw_video"]
        if len(video_bytes) != expected_bytes:
            raise PreflightError(
                f"Decoded {split} record has {len(video_bytes)} video bytes; "
                f"expected {expected_bytes} for {sequence_length}x"
                f"{image_h}x{image_w}x{image_c}"
            )
        frames = np.frombuffer(video_bytes, dtype=np.uint8).reshape(
            sequence_length, image_h, image_w, image_c
        )

        actions = _action_values(record["actions"])
        rewards = np.asarray(record["rewards"]).reshape(-1)
        if len(actions) != sequence_length:
            raise PreflightError(
                f"Decoded {split} record action length {len(actions)} does not "
                f"match frame count {sequence_length}"
            )
        if len(rewards) != sequence_length:
            raise PreflightError(
                f"Decoded {split} record reward length {len(rewards)} does not "
                f"match frame count {sequence_length}"
            )

        total_records += 1
        total_frames += sequence_length
        record_lengths.append(sequence_length)
        record_lengths_by_split.setdefault(split, []).append(sequence_length)
        records_by_split[split] += 1
        frames_by_split[split] += sequence_length
        action_histogram.update(actions)
        total_rewards += len(rewards)
        reward_frames += int(np.count_nonzero(rewards))
        positive_reward_frames += int(np.count_nonzero(rewards > 0))
        decoded_value_min = min(decoded_value_min, int(frames.min()))
        decoded_value_max = max(decoded_value_max, int(frames.max()))

        collector = next(
            (
                record[key]
                for key in ("collector", "collector_id", "collector_identity")
                if key in record
            ),
            None,
        )
        if collector is not None:
            collector_histogram[str(collector)] += sequence_length
        for key in (
            "seed",
            "episode_seed",
            "level_seed",
            "environment_seed",
            "action_seed",
            "dataset_seed",
        ):
            if key in record:
                try:
                    seeds.add(int(record[key]))
                except (TypeError, ValueError):
                    pass
        if "level_seed" in record:
            try:
                level_seeds_by_split.setdefault(split, set()).add(
                    int(record["level_seed"])
                )
            except (TypeError, ValueError):
                pass
        if "episode_starts" in record:
            episode_start_frames += int(
                np.count_nonzero(np.asarray(record["episode_starts"]))
            )
        if "episode_ends" in record:
            episode_end_frames += int(
                np.count_nonzero(np.asarray(record["episode_ends"]))
            )
        if bool(record.get("is_last_chunk", True)):
            episode_id = str(
                record.get("episode_id", f"{split}:record:{total_records}")
            )
            if episode_id not in observed_episode_outcomes:
                observed_episode_outcomes.add(episode_id)
                if record.get("terminated"):
                    episode_outcomes["terminated"] += 1
                elif record.get("truncated"):
                    episode_outcomes["timeout_or_truncated"] += 1
                else:
                    episode_outcomes["unknown"] += 1

    if total_records == 0:
        raise PreflightError("Dataset decode stage found zero ArrayRecord records")

    return {
        "records": total_records,
        "records_by_split": dict(sorted(records_by_split.items())),
        "frames": total_frames,
        "frames_by_split": dict(sorted(frames_by_split.items())),
        "reward_nonzero_frames": reward_frames,
        "reward_positive_frames": positive_reward_frames,
        "reward_prevalence": reward_frames / total_rewards if total_rewards else 0.0,
        "positive_reward_prevalence": (
            positive_reward_frames / total_rewards if total_rewards else 0.0
        ),
        "action_histogram": {
            str(key): value for key, value in sorted(action_histogram.items())
        },
        "collector_frame_histogram": dict(sorted(collector_histogram.items())),
        "unique_seeds": len(seeds),
        "seeds": sorted(seeds),
        "level_seeds_by_split": {
            split: sorted(values)
            for split, values in sorted(level_seeds_by_split.items())
        },
        "episode_outcomes": dict(sorted(episode_outcomes.items())),
        "episode_start_frames": episode_start_frames,
        "episode_end_frames": episode_end_frames,
        "decoded_pixel_min": decoded_value_min,
        "decoded_pixel_max": decoded_value_max,
        "minimum_record_length": min(record_lengths),
        "maximum_record_length": max(record_lengths),
        "minimum_record_length_by_split": {
            split: min(lengths)
            for split, lengths in sorted(record_lengths_by_split.items())
        },
        "maximum_record_length_by_split": {
            split: max(lengths)
            for split, lengths in sorted(record_lengths_by_split.items())
        },
    }


def iter_array_records(dataset_dir: Path) -> Iterable[tuple[str, bytes]]:
    try:
        from array_record.python.array_record_data_source import (
            ArrayRecordDataSource,
        )
    except ImportError as exc:
        raise PreflightError(
            "Dataset decode requires the project environment with array-record installed"
        ) from exc

    shards = sorted(dataset_dir.rglob("*.array_record"))
    if not shards:
        raise PreflightError(
            f"Dataset generation produced no .array_record shards under {dataset_dir}"
        )
    grouped: dict[str, list[Path]] = {}
    for shard in shards:
        relative = shard.relative_to(dataset_dir)
        split = relative.parts[0] if len(relative.parts) > 1 else "unknown"
        grouped.setdefault(split, []).append(shard)

    for split, split_shards in sorted(grouped.items()):
        source = ArrayRecordDataSource([str(path) for path in split_shards])
        for index in range(len(source)):
            yield split, source[index]


def discover_checkpoints(checkpoint_dir: Path) -> list[str]:
    if not checkpoint_dir.is_dir():
        raise PreflightError(f"Missing checkpoint directory: {checkpoint_dir}")
    checkpoint_paths = sorted(
        path.resolve()
        for path in checkpoint_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if not checkpoint_paths:
        raise PreflightError(f"No saved checkpoints found in {checkpoint_dir}")
    return [str(path) for path in checkpoint_paths]


def validate_rollout_mp4(
    dynamics_run_dir: Path, *, expected_sequence_length: int | None = None
) -> dict[str, Any]:
    videos = sorted(dynamics_run_dir.rglob("*.mp4"))
    if not videos:
        raise PreflightError(
            f"Dynamics stage produced no rollout MP4 under {dynamics_run_dir}"
        )
    path = videos[-1]
    if path.stat().st_size == 0:
        raise PreflightError(f"Generated rollout MP4 is empty: {path}")

    try:
        import imageio.v3 as iio
        import numpy as np

        frame_count = 0
        pixel_min = 255
        pixel_max = 0
        changed_transitions = 0
        absolute_change_sum = 0.0
        previous_frame = None
        for frame in iio.imiter(path, plugin="FFMPEG"):
            frame_array = np.asarray(frame)
            frame_count += 1
            pixel_min = min(pixel_min, int(frame_array.min()))
            pixel_max = max(pixel_max, int(frame_array.max()))
            if previous_frame is not None:
                absolute_change = float(
                    np.mean(
                        np.abs(
                            frame_array.astype(np.int16)
                            - previous_frame.astype(np.int16)
                        )
                    )
                )
                absolute_change_sum += absolute_change
                if absolute_change > 0:
                    changed_transitions += 1
            previous_frame = frame_array
    except Exception as exc:
        raise PreflightError(f"Could not decode generated rollout MP4 {path}: {exc}") from exc

    if frame_count == 0:
        raise PreflightError(f"Generated rollout MP4 has no frames: {path}")
    if pixel_min == pixel_max:
        raise PreflightError(
            f"Generated rollout MP4 is blank (constant value {pixel_min}): {path}"
        )
    if frame_count > 1 and changed_transitions == 0:
        raise PreflightError(
            f"Generated rollout MP4 is temporally constant across {frame_count} frames: "
            f"{path}"
        )
    if expected_sequence_length is not None and frame_count != expected_sequence_length:
        raise PreflightError(
            f"Generated rollout MP4 has {frame_count} frames; "
            f"expected declared sequence length {expected_sequence_length}"
        )
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "frames": frame_count,
        "pixel_min": pixel_min,
        "pixel_max": pixel_max,
        "changed_frame_transitions": changed_transitions,
        "mean_absolute_frame_change": (
            absolute_change_sum / (frame_count - 1) if frame_count > 1 else 0.0
        ),
        "nonblank": True,
        "temporally_nonconstant": True,
    }


class Preflight:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = Path(args.repo_root).resolve()
        self.output_dir = Path(args.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = self.output_dir / "telemetry.json"
        self.deadline = time.monotonic() + args.runtime_seconds
        self.report: dict[str, Any] = {
            "schema_version": 1,
            "status": "running",
            "started_at": _utc_now(),
            "hard_runtime_bound_seconds": args.runtime_seconds,
            "output_dir": str(self.output_dir),
            "invocation": [sys.executable, *sys.argv],
            "git": collect_git_state(self.repo_root),
            "host": {
                "hostname": platform.node(),
                "platform": platform.platform(),
                "python": sys.version,
                "gpu_inventory": query_gpu_inventory(),
            },
            "stages": [],
        }
        self._flush()

    def _flush(self) -> None:
        self.report["elapsed_seconds"] = self.args.runtime_seconds - max(
            0.0, self.deadline - time.monotonic()
        )
        _write_json(self.report_path, self.report)

    def _stage_command(
        self,
        name: str,
        command: Sequence[str],
        *,
        event_path: Path | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        stage: dict[str, Any] = {
            "name": name,
            "status": "running",
            "started_at": _utc_now(),
            "command": [str(part) for part in command],
        }
        self.report["stages"].append(stage)
        self._flush()
        result: CommandResult | None = None
        try:
            result = run_bounded_command(
                command, cwd=self.repo_root, deadline=self.deadline
            )
            log_path = self.output_dir / "logs" / f"{name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(result.output, encoding="utf-8")
            stage.update(
                {
                    "returncode": result.returncode,
                    "elapsed_seconds": result.elapsed_seconds,
                    "log_path": str(log_path),
                    "gpu_samples_mb": result.gpu_samples_mb,
                }
            )
            events = read_events(event_path) if event_path is not None else []
            if event_path is not None:
                stage["event_path"] = str(event_path)
            if result.returncode != 0:
                tail = "\n".join(result.output.strip().splitlines()[-20:])
                raise PreflightError(
                    f"Stage '{name}' failed with exit code {result.returncode}. "
                    f"See {log_path}.\n{tail}"
                )
            artifact_failures = fatal_artifact_failures(result.output)
            if artifact_failures:
                raise PreflightError(
                    f"Stage '{name}' reported an artifact failure despite exit code 0: "
                    + "; ".join(artifact_failures)
                    + f". See {log_path}."
                )
            stage["status"] = "passed"
            return stage, events
        except BaseException as exc:
            stage["status"] = "failed"
            stage["error"] = f"{type(exc).__name__}: {exc}"
            if result is not None:
                stage["returncode"] = result.returncode
            raise
        finally:
            stage["finished_at"] = _utc_now()
            self._flush()

    def _record_python_stage(self, name: str, operation) -> Any:
        stage: dict[str, Any] = {
            "name": name,
            "status": "running",
            "started_at": _utc_now(),
        }
        self.report["stages"].append(stage)
        self._flush()
        try:
            if time.monotonic() >= self.deadline:
                raise HardRuntimeExceeded(
                    f"Hard runtime bound reached before stage '{name}'"
                )
            result = operation()
            stage["result"] = result
            stage["status"] = "passed"
            return result
        except BaseException as exc:
            stage["status"] = "failed"
            stage["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            stage["finished_at"] = _utc_now()
            self._flush()

    def _trainer_command(
        self,
        *,
        stage: str,
        config_name: str,
        event_path: Path,
        run_dir: Path,
        dataset_dir: Path,
        tokenizer_checkpoint: Path | None = None,
        latent_stats: dict[str, Any] | None = None,
    ) -> list[str]:
        trainer_path = self.repo_root / "scripts" / f"train_{stage}.py"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_instrumented_trainer",
            stage,
            str(event_path),
            str(trainer_path),
            f"--config-name={config_name}",
            f"hydra.run.dir={run_dir}",
            f"run_name=coinrun-preflight-{stage}",
            f"dataset.array_record_path={dataset_dir / 'train'}",
            f"dataset.dataloader_cfg.B={self.args.batch_size}",
            f"dataset.dataloader_cfg.short_T={self.args.sequence_length}",
            f"dataset.dataloader_cfg.long_T={self.args.sequence_length}",
            "dataset.dataloader_cfg.num_workers=0",
            f"max_steps={self.args.steps}",
            f"lr_schedule.max_steps={self.args.steps}",
            f"ckpt.max_steps={self.args.steps}",
            "ckpt.max_to_keep=2",
            "ckpt.save_interval_steps=1",
            "logger.log_every=1",
            "logger.use_wandb=false",
            "use_wandb=false",
        ]
        if stage == "tokenizer":
            command.extend(
                [
                    "logger.log_gradients=true",
                    "lpips_weight=0",
                    "visualize_every=0",
                ]
            )
        else:
            if tokenizer_checkpoint is None:
                raise PreflightError("Dynamics stage requires a tokenizer checkpoint")
            if latent_stats is None:
                raise PreflightError(
                    "Dynamics stage requires measured held-out tokenizer latent stats"
                )
            command.extend(
                [
                    f"tokenizer_ckpt={tokenizer_checkpoint}",
                    f"bootstrap_start={self.args.steps}",
                    "bootstrap_fraction=0",
                    "image_fraction=0",
                    "ot.enabled=false",
                    "dynamics.k_max=4",
                    "write_video_every=0",
                ]
            )
            command.extend(build_latent_hydra_overrides(latent_stats))
        return command

    def run(self) -> None:
        configs_dir = self.repo_root / "configs"
        tokenizer_config = find_coinrun_config(
            configs_dir, "tokenizer", self.args.tokenizer_config
        )
        dynamics_config = find_coinrun_config(
            configs_dir, "dynamics", self.args.dynamics_config
        )
        self.report["selected_configs"] = {
            "tokenizer": tokenizer_config,
            "dynamics": dynamics_config,
        }
        self._flush()

        dataset_dir = self.output_dir / "dataset"
        if self.args.dataset_command:
            dataset_command = [
                part.format(dataset_dir=str(dataset_dir))
                for part in shlex.split(self.args.dataset_command)
            ]
            collection_scope = {
                "arm": "custom",
                "purpose": "caller-supplied dataset command",
                "scientific_comparison": "not inferred by preflight",
            }
        else:
            dataset_command = build_default_dataset_command(
                repo_root=self.repo_root,
                dataset_dir=dataset_dir,
                sequence_length=self.args.sequence_length,
                seed=self.args.seed,
            )
            dataset_command.extend(self.args.dataset_arg)
            collection_scope = default_collection_scope(self.args.sequence_length)
        self.report["collection_scope"] = collection_scope
        self._flush()
        self._stage_command("dataset_generation", dataset_command)

        dataset_stats = self._record_python_stage(
            "dataset_decode",
            lambda: summarize_dataset_records(
                iter_array_records(dataset_dir),
                deadline=self.deadline,
            ),
        )
        metadata_path = dataset_dir / "metadata.json"
        if not metadata_path.is_file():
            raise PreflightError(
                f"Dataset generation did not produce required metadata: {metadata_path}"
            )
        try:
            dataset_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PreflightError(f"Invalid dataset metadata {metadata_path}: {exc}") from exc
        dataset_config = load_coinrun_dataset_config(configs_dir)
        self.report["dataset"] = {
            "path": str(dataset_dir),
            "stats": dataset_stats,
            "metadata_path": str(metadata_path),
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "metadata": dataset_metadata,
            "config": dataset_config,
        }
        self._flush()
        dataset_gates = self._record_python_stage(
            "dataset_gates",
            lambda: validate_dataset_gates(
                dataset_stats,
                dataset_metadata,
                dataset_config,
                required_sequence_length=self.args.sequence_length,
            ),
        )
        self.report["dataset"]["gates"] = dataset_gates
        self._flush()

        tokenizer_run = self.output_dir / "tokenizer"
        tokenizer_events = self.output_dir / "events" / "tokenizer.jsonl"
        tokenizer_command = self._trainer_command(
            stage="tokenizer",
            config_name=tokenizer_config,
            event_path=tokenizer_events,
            run_dir=tokenizer_run,
            dataset_dir=dataset_dir,
        )
        tokenizer_stage, tokenizer_event_data = self._stage_command(
            "tokenizer_train", tokenizer_command, event_path=tokenizer_events
        )
        tokenizer_summary = summarize_training_events(
            tokenizer_event_data,
            tokenizer_stage.get("gpu_samples_mb", []),
        )
        if tokenizer_summary["timing"]["measured_steps"] < 2:
            raise PreflightError(
                "Tokenizer stage did not emit enough timed optimizer steps "
                "to distinguish compile and steady-state timing"
            )
        if not tokenizer_summary["losses"] or not tokenizer_summary["gradient_norms"]:
            raise PreflightError(
                "Tokenizer stage did not emit required loss and gradient norm metrics"
            )
        tokenizer_checkpoints = discover_checkpoints(tokenizer_run / "checkpoints")
        tokenizer_summary["checkpoints"] = tokenizer_checkpoints
        tokenizer_summary["checkpoint_cadence"] = validate_checkpoint_cadence(
            tokenizer_summary, tokenizer_checkpoints
        )
        tokenizer_stage["telemetry"] = tokenizer_summary
        self._flush()

        tokenizer_restore_events = self.output_dir / "events" / "tokenizer_restore.jsonl"
        tokenizer_restore_command = self._trainer_command(
            stage="tokenizer",
            config_name=tokenizer_config,
            event_path=tokenizer_restore_events,
            run_dir=tokenizer_run,
            dataset_dir=dataset_dir,
        )
        restore_stage, restore_event_data = self._stage_command(
            "tokenizer_restore",
            tokenizer_restore_command,
            event_path=tokenizer_restore_events,
        )
        if not any(
            event.get("kind") == "checkpoint_restore" and event.get("restored")
            for event in restore_event_data
        ):
            raise PreflightError(
                "Tokenizer restore stage completed without restoring a checkpoint"
            )
        restore_stage["checkpoint_paths"] = tokenizer_checkpoints
        self._flush()

        tokenizer_probe_path = self.output_dir / "tokenizer_heldout_probe.json"
        tokenizer_probe_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_tokenizer_probe",
            f"--checkpoint-dir={tokenizer_run / 'checkpoints'}",
            f"--dataset-dir={dataset_dir}",
            f"--output={tokenizer_probe_path}",
            f"--sequence-length={self.args.sequence_length}",
            f"--max-records={self.args.latent_stat_max_records}",
            f"--std-epsilon={self.args.latent_std_epsilon}",
        ]
        tokenizer_probe_stage, _ = self._stage_command(
            "tokenizer_heldout_probe", tokenizer_probe_command
        )
        if not tokenizer_probe_path.is_file():
            raise PreflightError(
                f"Held-out tokenizer probe did not produce {tokenizer_probe_path}"
            )
        try:
            tokenizer_probe = json.loads(
                tokenizer_probe_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise PreflightError(
                f"Invalid held-out tokenizer probe artifact "
                f"{tokenizer_probe_path}: {exc}"
            ) from exc
        validate_measured_latent_stats(
            tokenizer_probe,
            expected_dim=tokenizer_probe.get("d_bottleneck"),
            std_epsilon=self.args.latent_std_epsilon,
        )
        tokenizer_probe_stage["telemetry"] = tokenizer_probe
        self.report["tokenizer_heldout_probe"] = {
            "path": str(tokenizer_probe_path),
            "sha256": hashlib.sha256(tokenizer_probe_path.read_bytes()).hexdigest(),
            **tokenizer_probe,
        }
        self._flush()

        dynamics_run = self.output_dir / "dynamics"
        dynamics_events = self.output_dir / "events" / "dynamics.jsonl"
        dynamics_command = self._trainer_command(
            stage="dynamics",
            config_name=dynamics_config,
            event_path=dynamics_events,
            run_dir=dynamics_run,
            dataset_dir=dataset_dir,
            tokenizer_checkpoint=tokenizer_run / "checkpoints",
            latent_stats=tokenizer_probe,
        )
        dynamics_stage, dynamics_event_data = self._stage_command(
            "dynamics_train", dynamics_command, event_path=dynamics_events
        )
        dynamics_summary = summarize_training_events(
            dynamics_event_data,
            dynamics_stage.get("gpu_samples_mb", []),
        )
        if dynamics_summary["timing"]["measured_steps"] < 2:
            raise PreflightError(
                "Dynamics stage did not emit enough timed optimizer steps "
                "to distinguish compile and steady-state timing"
            )
        if not dynamics_summary["losses"] or not dynamics_summary["gradient_norms"]:
            raise PreflightError(
                "Dynamics stage did not emit required loss and gradient norm metrics"
            )
        dynamics_checkpoints = discover_checkpoints(dynamics_run / "checkpoints")
        dynamics_summary["checkpoints"] = dynamics_checkpoints
        dynamics_summary["checkpoint_cadence"] = validate_checkpoint_cadence(
            dynamics_summary, dynamics_checkpoints
        )
        action_gate_events = [
            event
            for event in dynamics_event_data
            if event.get("kind") == "action_semantics" and event.get("passed")
        ]
        if not action_gate_events:
            raise PreflightError(
                "Dynamics stage did not prove that shifted actions prepend no-op 4"
            )
        dynamics_summary["action_semantics"] = action_gate_events[-1]
        dynamics_stage["telemetry"] = dynamics_summary
        self._flush()

        dynamics_restore_events = self.output_dir / "events" / "dynamics_restore.jsonl"
        dynamics_restore_command = self._trainer_command(
            stage="dynamics",
            config_name=dynamics_config,
            event_path=dynamics_restore_events,
            run_dir=dynamics_run,
            dataset_dir=dataset_dir,
            tokenizer_checkpoint=tokenizer_run / "checkpoints",
            latent_stats=tokenizer_probe,
        )
        dynamics_restore_stage, dynamics_restore_data = self._stage_command(
            "dynamics_restore",
            dynamics_restore_command,
            event_path=dynamics_restore_events,
        )
        if not any(
            event.get("kind") == "checkpoint_restore" and event.get("restored")
            for event in dynamics_restore_data
        ):
            raise PreflightError(
                "Dynamics restore stage completed without restoring a checkpoint"
            )
        dynamics_restore_stage["checkpoint_paths"] = dynamics_checkpoints
        self._flush()

        eval_metrics = [
            event
            for event in dynamics_event_data
            if event.get("kind") == "logger_metrics"
            and event.get("prefix") == "eval/"
        ]
        if not eval_metrics:
            raise PreflightError(
                "Dynamics stage did not emit rollout evaluation metrics"
            )
        horizon = int(eval_metrics[-1].get("metrics", {}).get("horizon", 0))
        if horizon <= 0:
            raise PreflightError("Dynamics rollout declared a non-positive horizon")
        rollout = self._record_python_stage(
            "rollout_validation",
            lambda: validate_rollout_mp4(
                dynamics_run, expected_sequence_length=self.args.sequence_length
            ),
        )
        rollout["context_length"] = self.args.sequence_length - horizon
        rollout["horizon"] = horizon
        self.report["rollout"] = rollout

        gpu_sources = []
        for stage in self.report["stages"]:
            sources = (
                stage.get("telemetry", {})
                .get("gpu_memory", {})
                .get("sources", [])
            )
            gpu_sources.extend(sources)
        if not gpu_sources and not self.args.allow_cpu:
            raise PreflightError(
                "No GPU memory telemetry was available from JAX or nvidia-smi"
            )

        self.report["status"] = "passed"
        self.report["finished_at"] = _utc_now()
        self._flush()


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description="Run a bounded, instrumented CoinRun reconstruction smoke test."
    )
    parser.add_argument("--repo-root", default=str(repo_root))
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "artifacts" / "coinrun_preflight" / timestamp),
    )
    parser.add_argument(
        "--runtime-seconds", type=float, default=DEFAULT_RUNTIME_SECONDS
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--latent-stat-max-records",
        type=int,
        default=DEFAULT_LATENT_STAT_MAX_RECORDS,
    )
    parser.add_argument(
        "--latent-std-epsilon",
        type=float,
        default=DEFAULT_LATENT_STD_EPSILON,
    )
    parser.add_argument("--tokenizer-config")
    parser.add_argument("--dynamics-config")
    parser.add_argument(
        "--dataset-command",
        help=(
            "Override dataset generation; {dataset_dir} is expanded before launch"
        ),
    )
    parser.add_argument(
        "--dataset-arg",
        action="append",
        default=[],
        help="Append one argument to the default dataset generator command",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow a smoke run without GPU memory telemetry",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.runtime_seconds <= 0:
        raise PreflightError("--runtime-seconds must be positive")
    if args.steps < 2:
        raise PreflightError("--steps must be at least 2 for steady-state timing")
    if args.batch_size <= 0 or args.sequence_length <= 5:
        raise PreflightError(
            "--batch-size must be positive and --sequence-length must be greater than 5"
        )
    if args.latent_stat_max_records <= 0:
        raise PreflightError("--latent-stat-max-records must be positive")
    if args.latent_std_epsilon <= 0:
        raise PreflightError("--latent-std-epsilon must be positive")

    preflight: Preflight | None = None
    try:
        preflight = Preflight(args)
        preflight.run()
    except BaseException as exc:
        if preflight is not None:
            preflight.report["status"] = (
                "timed_out" if isinstance(exc, HardRuntimeExceeded) else "failed"
            )
            preflight.report["finished_at"] = _utc_now()
            preflight.report["error"] = f"{type(exc).__name__}: {exc}"
            preflight._flush()
            print(
                f"CoinRun preflight failed: {exc}\nTelemetry: {preflight.report_path}",
                file=sys.stderr,
            )
        else:
            print(f"CoinRun preflight failed: {exc}", file=sys.stderr)
        return 1

    print(f"CoinRun preflight passed. Telemetry: {preflight.report_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_instrumented_trainer":
        raise SystemExit(instrumented_trainer_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "_tokenizer_probe":
        raise SystemExit(tokenizer_probe_main(sys.argv[2:]))
    raise SystemExit(main())
