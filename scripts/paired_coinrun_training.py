"""Fail-closed launcher for paired RGB CoinRun tokenizer and dynamics training.

This path is intentionally separate from the legacy CoinRun H200 launcher. It
never collects data: both converted corpus splits and their manifest digest
must be supplied explicitly before trainer commands can be produced or run.
"""

from __future__ import annotations

import argparse
import functools
import json
import operator
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamer.data.paired_rgb_adapter import (
    ADAPTER_SCHEMA_VERSION,
    COINRUN_NOOP_ACTION,
    COINRUN_NUM_ACTIONS,
    SOURCE_SHARDS,
    sha256_file,
)
from scripts.coinrun_preflight import (
    DEFAULT_LATENT_STAT_MAX_RECORDS,
    DEFAULT_LATENT_STD_EPSILON,
    PreflightError,
    build_latent_hydra_overrides,
    validate_measured_latent_stats,
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_CONTRACT_FILENAME = "paired_run_contract.json"
MANIFEST_COPY_FILENAME = "paired_manifest.json"
LATENT_STATS_FILENAME = "tokenizer_heldout_probe.json"


class PairedTrainingError(RuntimeError):
    """The paired corpus or requested training launch violated its contract."""


@dataclass(frozen=True)
class ValidatedCorpus:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    train_dir: Path
    validation_dir: Path
    split_summaries: dict[str, dict[str, Any]]
    action_space: dict[str, Any]


@dataclass(frozen=True)
class TrainingPlan:
    repo_root: Path
    run_dir: Path
    corpus: ValidatedCorpus
    tokenizer_config_name: str
    dynamics_config_name: str
    tokenizer_command: tuple[str, ...]
    tokenizer_probe_command: tuple[str, ...]
    tokenizer_config: dict[str, Any]
    dynamics_preflight_config: dict[str, Any]
    dynamics_user_overrides: tuple[str, ...]
    dynamics_required_overrides: tuple[str, ...]
    python_executable: str
    latent_stat_max_records: int
    latent_std_epsilon: float
    latent_dimension: int


@dataclass(frozen=True)
class RuntimeDynamics:
    command: tuple[str, ...]
    config: dict[str, Any]
    stats_path: Path
    stats_sha256: str
    stats: dict[str, Any]
    validation: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _require_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairedTrainingError(f"{description} must be a JSON object.")
    return value


def _require_int(mapping: Mapping[str, Any], name: str) -> int:
    value = mapping.get(name)
    if type(value) is not int:
        raise PairedTrainingError(f"Manifest {name} must be an integer.")
    return value


def _validate_action_space(manifest: Mapping[str, Any]) -> dict[str, Any]:
    action_space = _require_mapping(
        manifest.get("action_space"),
        "Manifest action_space",
    )
    categorical_dim = _require_int(action_space, "categorical_action_dim")
    categorical_noop = _require_int(action_space, "categorical_noop")
    num_binary = _require_int(action_space, "num_binary_actions")
    continuous_dim = _require_int(action_space, "continuous_action_dim")
    action_type = action_space.get("type")

    if categorical_dim == 15:
        raise PairedTrainingError(
            "Paired corpus manifest declares the legacy 15-action config; "
            "paired V2 training requires categorical_action_dim=9."
        )
    if categorical_dim != COINRUN_NUM_ACTIONS:
        raise PairedTrainingError(
            "Paired corpus manifest categorical_action_dim must be "
            f"{COINRUN_NUM_ACTIONS}; got {categorical_dim}."
        )
    if categorical_noop != COINRUN_NOOP_ACTION:
        raise PairedTrainingError(
            "Paired corpus manifest categorical_noop must be "
            f"{COINRUN_NOOP_ACTION}; got {categorical_noop}."
        )
    if num_binary != 0 or continuous_dim != 0:
        raise PairedTrainingError(
            "Paired corpus manifest must declare categorical-only actions."
        )
    if action_type != "procgen_discrete":
        raise PairedTrainingError(
            "Paired corpus manifest action_space.type must be 'procgen_discrete'."
        )
    return dict(action_space)


def _safe_output_path(corpus_root: Path, relative: Any, split: str) -> Path:
    if not isinstance(relative, str):
        raise PairedTrainingError("Manifest output shard path must be a string.")
    pure_path = PurePosixPath(relative)
    if (
        pure_path.is_absolute()
        or ".." in pure_path.parts
        or not pure_path.parts
        or pure_path.parts[0] != split
        or pure_path.suffix != ".array_record"
    ):
        raise PairedTrainingError(
            f"Manifest output shard path is invalid for split {split!r}: {relative!r}."
        )
    path = (corpus_root / Path(*pure_path.parts)).resolve()
    expected_parent = (corpus_root / split).resolve()
    if path.parent != expected_parent:
        raise PairedTrainingError(
            f"Manifest output shard must be directly under {split}: {relative!r}."
        )
    return path


def _validate_source_manifests(
    manifest: Mapping[str, Any],
    corpus_root: Path,
) -> dict[str, dict[str, Any]]:
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise PairedTrainingError("Manifest sources must be a JSON array.")

    expected = {spec.filename: (spec.split, spec.source) for spec in SOURCE_SHARDS}
    actual_by_filename: dict[str, Mapping[str, Any]] = {}
    for source_value in sources:
        source = _require_mapping(source_value, "Manifest source entry")
        filename = source.get("filename")
        if not isinstance(filename, str) or filename in actual_by_filename:
            raise PairedTrainingError(
                "Manifest source filenames must be unique strings."
            )
        actual_by_filename[filename] = source
    if set(actual_by_filename) != set(expected):
        raise PairedTrainingError(
            "Manifest source shards do not match the four paired V2 inputs."
        )

    split_summaries = {
        "train": {
            "array_record_shards": [],
            "num_episodes": 0,
            "num_frames": 0,
            "sources": [],
        },
        "val": {
            "array_record_shards": [],
            "num_episodes": 0,
            "num_frames": 0,
            "sources": [],
        },
    }
    seen_paths: set[Path] = set()
    for filename, (expected_split, expected_source) in expected.items():
        source = actual_by_filename[filename]
        if (
            source.get("split") != expected_split
            or source.get("source") != expected_source
        ):
            raise PairedTrainingError(
                f"Manifest source identity mismatch for {filename}."
            )
        source_sha256 = source.get("sha256")
        if not isinstance(source_sha256, str) or not SHA256_RE.fullmatch(source_sha256):
            raise PairedTrainingError(
                f"Manifest source SHA-256 is invalid for {filename}."
            )
        num_episodes = source.get("num_episodes")
        num_frames = source.get("num_frames")
        if (
            type(num_episodes) is not int
            or num_episodes <= 0
            or type(num_frames) is not int
            or num_frames <= 0
        ):
            raise PairedTrainingError(
                f"Manifest source counts are invalid for {filename}."
            )
        episodes = source.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != num_episodes:
            raise PairedTrainingError(
                f"Manifest episode list does not match num_episodes for {filename}."
            )
        output_shards = source.get("output_shards")
        if not isinstance(output_shards, list) or not output_shards:
            raise PairedTrainingError(f"Manifest has no output shards for {filename}.")

        recorded_shards = []
        for output_value in output_shards:
            output = _require_mapping(
                output_value,
                f"Manifest output shard for {filename}",
            )
            path = _safe_output_path(
                corpus_root,
                output.get("path"),
                expected_split,
            )
            if not path.name.startswith(f"{expected_source}-"):
                raise PairedTrainingError(
                    f"ArrayRecord filename does not preserve source identity "
                    f"{expected_source!r}: {path.name!r}."
                )
            if path in seen_paths:
                raise PairedTrainingError(
                    f"Manifest output shard is listed more than once: {path}."
                )
            seen_paths.add(path)
            if not path.is_file():
                raise PairedTrainingError(f"Manifest output shard is missing: {path}.")
            expected_size = output.get("size_bytes")
            if type(expected_size) is not int or path.stat().st_size != expected_size:
                raise PairedTrainingError(
                    f"ArrayRecord size does not match manifest: {path}."
                )
            expected_sha256 = output.get("sha256")
            if (
                not isinstance(expected_sha256, str)
                or not SHA256_RE.fullmatch(expected_sha256)
                or sha256_file(path) != expected_sha256
            ):
                raise PairedTrainingError(
                    f"ArrayRecord SHA-256 does not match manifest: {path}."
                )
            recorded_shards.append(
                {
                    "path": str(path),
                    "sha256": expected_sha256,
                    "size_bytes": expected_size,
                }
            )

        summary = split_summaries[expected_split]
        summary["num_episodes"] += num_episodes
        summary["num_frames"] += num_frames
        summary["array_record_shards"].extend(recorded_shards)
        summary["sources"].append(
            {
                "filename": filename,
                "num_episodes": num_episodes,
                "num_frames": num_frames,
                "source": expected_source,
                "source_sha256": source_sha256,
            }
        )

    actual_paths = {
        path.resolve()
        for split in ("train", "val")
        for path in (corpus_root / split).rglob("*.array_record")
    }
    if actual_paths != seen_paths:
        missing = sorted(str(path) for path in seen_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - seen_paths)
        raise PairedTrainingError(
            f"ArrayRecord file set does not match manifest; "
            f"missing={missing}, extra={extra}."
        )
    return split_summaries


def _validate_split_summaries(
    manifest: Mapping[str, Any],
    split_summaries: Mapping[str, Mapping[str, Any]],
) -> None:
    manifest_splits = _require_mapping(
        manifest.get("splits"),
        "Manifest splits",
    )
    if set(manifest_splits) != {"train", "val"}:
        raise PairedTrainingError(
            "Manifest must contain exactly train and val split summaries."
        )
    for split, actual in split_summaries.items():
        recorded = _require_mapping(
            manifest_splits.get(split),
            f"Manifest {split} split",
        )
        if (
            recorded.get("num_episodes") != actual["num_episodes"]
            or recorded.get("num_frames") != actual["num_frames"]
        ):
            raise PairedTrainingError(
                f"Manifest {split} aggregate counts do not match its sources."
            )
        expected_sources = [source["source"] for source in actual["sources"]]
        if recorded.get("sources") != expected_sources:
            raise PairedTrainingError(
                f"Manifest {split} source identities do not match its sources."
            )
    total_episodes = sum(
        int(summary["num_episodes"]) for summary in split_summaries.values()
    )
    total_frames = sum(
        int(summary["num_frames"]) for summary in split_summaries.values()
    )
    if (
        manifest.get("num_episodes") != total_episodes
        or manifest.get("num_frames") != total_frames
    ):
        raise PairedTrainingError(
            "Manifest global counts do not match train and val sources."
        )


def validate_paired_corpus(
    *,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
    train_dir: Path | str,
    validation_dir: Path | str,
) -> ValidatedCorpus:
    manifest_path = Path(manifest_path).expanduser().resolve()
    train_dir = Path(train_dir).expanduser().resolve()
    validation_dir = Path(validation_dir).expanduser().resolve()
    if not SHA256_RE.fullmatch(expected_manifest_sha256):
        raise PairedTrainingError(
            "Expected paired manifest SHA-256 must be 64 lowercase hex characters."
        )
    if not manifest_path.is_file():
        raise PairedTrainingError(
            f"Required paired corpus manifest is missing: {manifest_path}."
        )
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise PairedTrainingError(
            "Paired corpus manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, got {actual_manifest_sha256}."
        )
    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PairedTrainingError(
            f"Paired corpus manifest is not valid JSON: {manifest_path}."
        ) from exc
    manifest = dict(_require_mapping(manifest_value, "Paired corpus manifest"))
    if manifest.get("adapter_schema_version") != ADAPTER_SCHEMA_VERSION:
        raise PairedTrainingError(
            "Paired corpus manifest adapter schema does not match this launcher."
        )

    corpus_root = manifest_path.parent.resolve()
    expected_train = (corpus_root / "train").resolve()
    expected_validation = (corpus_root / "val").resolve()
    if train_dir != expected_train:
        raise PairedTrainingError(
            f"Explicit train directory {train_dir} does not match manifest "
            f"split {expected_train}."
        )
    if validation_dir != expected_validation:
        raise PairedTrainingError(
            f"Explicit validation directory {validation_dir} does not match "
            f"manifest split {expected_validation}."
        )
    if train_dir == validation_dir:
        raise PairedTrainingError(
            "Train and validation ArrayRecord directories must be distinct."
        )
    if not train_dir.is_dir() or not validation_dir.is_dir():
        raise PairedTrainingError(
            "Train and validation ArrayRecord directories must both exist."
        )

    action_space = _validate_action_space(manifest)
    split_summaries = _validate_source_manifests(manifest, corpus_root)
    _validate_split_summaries(manifest, split_summaries)
    return ValidatedCorpus(
        manifest_path=manifest_path,
        manifest_sha256=actual_manifest_sha256,
        manifest=manifest,
        train_dir=train_dir,
        validation_dir=validation_dir,
        split_summaries=split_summaries,
        action_space=action_space,
    )


PROTECTED_OVERRIDES = {
    "dataset",
    "dataset.array_record_path",
    "dataset.categorical_action_dim",
    "dataset.categorical_noop",
    "dataset.continuous_action_dim",
    "dataset.num_binary_actions",
    "dynamics.categorical_action_dim",
    "dynamics.continuous_action_dim",
    "dynamics.latent_mean",
    "dynamics.latent_std",
    "dynamics.num_binary_actions",
    "hydra.run.dir",
    "run_name",
    "tokenizer_ckpt",
}


def _override_key(override: str) -> str:
    key = override.split("=", 1)[0]
    return key.lstrip("+~")


def _validate_user_overrides(stage: str, overrides: Sequence[str]) -> None:
    for override in overrides:
        key = _override_key(override)
        if not key or any(
            key == protected or protected.startswith(f"{key}.")
            for protected in PROTECTED_OVERRIDES
        ):
            raise PairedTrainingError(
                f"{stage} override may not replace paired contract field "
                f"{key!r}: {override!r}."
            )


def _hydra_string(value: Path | str) -> str:
    return json.dumps(str(value))


def _register_config_resolvers() -> None:
    resolvers: dict[str, Callable[..., Any]] = {
        "mul": lambda *values: functools.reduce(operator.mul, values),
        "sum": lambda *values: sum(values),
        "floordiv": lambda left, right: left // right,
        "max": max,
        "min": min,
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)


def _compose_config(
    repo_root: Path,
    config_name: str,
    overrides: Sequence[str],
) -> dict[str, Any]:
    _register_config_resolvers()
    with initialize_config_dir(
        config_dir=str((repo_root / "configs").resolve()),
        version_base=None,
    ):
        try:
            config = compose(config_name=config_name, overrides=list(overrides))
            resolved = OmegaConf.to_container(
                config,
                resolve=True,
                throw_on_missing=True,
            )
        except Exception as exc:
            raise PairedTrainingError(
                f"Could not compose paired training config {config_name!r}: {exc}"
            ) from exc
    if not isinstance(resolved, dict):
        raise PairedTrainingError(f"Resolved {config_name} config is not a mapping.")
    return resolved


def _required_overrides(
    *,
    stage: str,
    corpus: ValidatedCorpus,
    run_dir: Path,
) -> list[str]:
    action = corpus.action_space
    stage_run_dir = run_dir / stage
    overrides = [
        f"run_name={_hydra_string(f'paired-coinrun-{stage}')}",
        f"dataset.array_record_path={_hydra_string(corpus.train_dir)}",
        (f"dataset.categorical_action_dim={action['categorical_action_dim']}"),
        f"dataset.categorical_noop={action['categorical_noop']}",
        f"dataset.num_binary_actions={action['num_binary_actions']}",
        (f"dataset.continuous_action_dim={action['continuous_action_dim']}"),
    ]
    if stage == "dynamics":
        overrides.extend(
            [
                (f"dynamics.categorical_action_dim={action['categorical_action_dim']}"),
                f"dynamics.num_binary_actions={action['num_binary_actions']}",
                (f"dynamics.continuous_action_dim={action['continuous_action_dim']}"),
                (
                    "tokenizer_ckpt="
                    f"{_hydra_string(run_dir / 'tokenizer' / 'checkpoints')}"
                ),
            ]
        )
    overrides.append(f"hydra.run.dir={_hydra_string(stage_run_dir)}")
    return overrides


def _validate_resolved_config(
    *,
    stage: str,
    config: Mapping[str, Any],
    corpus: ValidatedCorpus,
    tokenizer_checkpoint: Path,
    latent_stats: Mapping[str, Any] | None = None,
) -> None:
    dataset = _require_mapping(
        config.get("dataset"),
        f"Resolved {stage} dataset config",
    )
    action = corpus.action_space
    categorical_dim = dataset.get("categorical_action_dim")
    if categorical_dim == 15:
        raise PairedTrainingError(
            f"Resolved {stage} config still uses the legacy 15-action space."
        )
    expected_fields = {
        "categorical_action_dim": action["categorical_action_dim"],
        "categorical_noop": action["categorical_noop"],
        "continuous_action_dim": action["continuous_action_dim"],
        "num_binary_actions": action["num_binary_actions"],
    }
    for field, expected in expected_fields.items():
        if dataset.get(field) != expected:
            raise PairedTrainingError(
                f"Resolved {stage} dataset.{field}={dataset.get(field)!r}; "
                f"paired manifest requires {expected!r}."
            )
    if dataset.get("name") != "coinrun" or dataset.get("data_type") != "video":
        raise PairedTrainingError(
            f"Resolved {stage} config must use the CoinRun pixel data path."
        )
    configured_path = Path(str(dataset.get("array_record_path"))).resolve()
    if configured_path != corpus.train_dir:
        raise PairedTrainingError(
            f"Resolved {stage} config does not use the explicit train split."
        )

    if stage == "dynamics":
        dynamics = _require_mapping(
            config.get("dynamics"),
            "Resolved dynamics model config",
        )
        for field in (
            "categorical_action_dim",
            "continuous_action_dim",
            "num_binary_actions",
        ):
            if dynamics.get(field) != expected_fields[field]:
                raise PairedTrainingError(
                    f"Resolved dynamics model {field} does not match the "
                    "paired manifest."
                )
        configured_checkpoint = Path(str(config.get("tokenizer_ckpt"))).resolve()
        if configured_checkpoint != tokenizer_checkpoint.resolve():
            raise PairedTrainingError(
                "Resolved dynamics config does not use this paired run's "
                "tokenizer checkpoint."
            )
        latent_mean = dynamics.get("latent_mean")
        latent_std = dynamics.get("latent_std")
        if latent_stats is None:
            if dataset.get("data_type") == "video" and (
                latent_mean is not None or latent_std is not None
            ):
                raise PairedTrainingError(
                    "Paired video dynamics preflight may not use default latent "
                    "statistics; held-out tokenizer measurements are required."
                )
        else:
            expected_mean = latent_stats["latent_mean"]
            expected_std = latent_stats["latent_std"]
            if latent_mean != expected_mean or latent_std != expected_std:
                raise PairedTrainingError(
                    "Runtime dynamics latent normalization does not exactly "
                    "match the validated held-out tokenizer statistics."
                )


def build_training_plan(
    *,
    repo_root: Path | str,
    run_dir: Path | str,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
    train_dir: Path | str,
    validation_dir: Path | str,
    tokenizer_config_name: str = "coinrun_tokenizer",
    dynamics_config_name: str = "coinrun_dynamics",
    tokenizer_overrides: Sequence[str] = (),
    dynamics_overrides: Sequence[str] = (),
    python_executable: Path | str = sys.executable,
    latent_stat_max_records: int = DEFAULT_LATENT_STAT_MAX_RECORDS,
    latent_std_epsilon: float = DEFAULT_LATENT_STD_EPSILON,
) -> TrainingPlan:
    repo_root = Path(repo_root).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve()
    if not (repo_root / "scripts" / "train_tokenizer.py").is_file():
        raise PairedTrainingError(f"Invalid repository root: {repo_root}.")
    if run_dir.exists():
        raise PairedTrainingError(f"Paired run directory already exists: {run_dir}.")
    if latent_stat_max_records <= 0:
        raise PairedTrainingError("latent_stat_max_records must be positive.")
    if latent_std_epsilon <= 0:
        raise PairedTrainingError("latent_std_epsilon must be positive.")
    _validate_user_overrides("tokenizer", tokenizer_overrides)
    _validate_user_overrides("dynamics", dynamics_overrides)
    corpus = validate_paired_corpus(
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        train_dir=train_dir,
        validation_dir=validation_dir,
    )

    tokenizer_required = _required_overrides(
        stage="tokenizer",
        corpus=corpus,
        run_dir=run_dir,
    )
    dynamics_required = _required_overrides(
        stage="dynamics",
        corpus=corpus,
        run_dir=run_dir,
    )
    tokenizer_compose_overrides = [
        *tokenizer_overrides,
        *tokenizer_required[:-1],
    ]
    dynamics_compose_overrides = [
        *dynamics_overrides,
        *dynamics_required[:-1],
    ]
    tokenizer_config = _compose_config(
        repo_root,
        tokenizer_config_name,
        tokenizer_compose_overrides,
    )
    dynamics_config = _compose_config(
        repo_root,
        dynamics_config_name,
        dynamics_compose_overrides,
    )
    tokenizer_checkpoint = run_dir / "tokenizer" / "checkpoints"
    _validate_resolved_config(
        stage="tokenizer",
        config=tokenizer_config,
        corpus=corpus,
        tokenizer_checkpoint=tokenizer_checkpoint,
    )
    tokenizer_model = _require_mapping(
        tokenizer_config.get("tokenizer"),
        "Resolved tokenizer model config",
    )
    tokenizer_encoder = _require_mapping(
        tokenizer_model.get("encoder"),
        "Resolved tokenizer encoder config",
    )
    tokenizer_dimension = tokenizer_encoder.get("d_bottleneck")
    dynamics_model = _require_mapping(
        dynamics_config.get("dynamics"),
        "Resolved dynamics model config",
    )
    dynamics_dimension = dynamics_model.get("d_bottleneck")
    if (
        type(tokenizer_dimension) is not int
        or tokenizer_dimension <= 0
        or dynamics_dimension != tokenizer_dimension
    ):
        raise PairedTrainingError(
            "Tokenizer and dynamics d_bottleneck dimensions must match before "
            "held-out latent statistics can be measured."
        )
    _validate_resolved_config(
        stage="dynamics",
        config=dynamics_config,
        corpus=corpus,
        tokenizer_checkpoint=tokenizer_checkpoint,
    )

    # Do not resolve this path. Virtual-environment Python executables are
    # commonly symlinks to a base interpreter; dereferencing that symlink drops
    # the venv's site-packages when the child process starts.
    python_path = Path(python_executable).expanduser()
    if not python_path.is_absolute():
        located_python = shutil.which(str(python_path))
        if located_python is None:
            raise PairedTrainingError(
                f"Python executable is not available on PATH: {python_path}."
            )
        python_path = Path(located_python)
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise PairedTrainingError(
            f"Python executable is missing or not executable: {python_path}."
        )
    python_executable = str(python_path.absolute())
    tokenizer_command = (
        python_executable,
        str(repo_root / "scripts" / "train_tokenizer.py"),
        f"--config-name={tokenizer_config_name}",
        *tokenizer_overrides,
        *tokenizer_required,
    )
    tokenizer_sequence_length = tokenizer_config["dataset"]["dataloader_cfg"]["long_T"]
    if type(tokenizer_sequence_length) is not int or tokenizer_sequence_length <= 1:
        raise PairedTrainingError(
            "Tokenizer sequence length must be greater than one for the "
            "held-out latent-stat probe."
        )
    tokenizer_probe_command = (
        python_executable,
        str(repo_root / "scripts" / "coinrun_preflight.py"),
        "_tokenizer_probe",
        f"--checkpoint-dir={tokenizer_checkpoint}",
        f"--dataset-dir={corpus.manifest_path.parent}",
        f"--output={run_dir / LATENT_STATS_FILENAME}",
        f"--sequence-length={tokenizer_sequence_length}",
        f"--max-records={latent_stat_max_records}",
        f"--std-epsilon={latent_std_epsilon}",
    )
    return TrainingPlan(
        repo_root=repo_root,
        run_dir=run_dir,
        corpus=corpus,
        tokenizer_config_name=tokenizer_config_name,
        dynamics_config_name=dynamics_config_name,
        tokenizer_command=tokenizer_command,
        tokenizer_probe_command=tokenizer_probe_command,
        tokenizer_config=tokenizer_config,
        dynamics_preflight_config=dynamics_config,
        dynamics_user_overrides=tuple(dynamics_overrides),
        dynamics_required_overrides=tuple(dynamics_required),
        python_executable=python_executable,
        latent_stat_max_records=latent_stat_max_records,
        latent_std_epsilon=latent_std_epsilon,
        latent_dimension=tokenizer_dimension,
    )


def _validation_level_ids(corpus: ValidatedCorpus) -> set[int]:
    level_ids: set[int] = set()
    sources = corpus.manifest["sources"]
    for source in sources:
        if source["split"] != "val":
            continue
        for episode in source["episodes"]:
            level_id = episode.get("level_id")
            if type(level_id) is not int:
                raise PairedTrainingError(
                    "Paired manifest val episodes must identify integer level IDs."
                )
            level_ids.add(level_id)
    if not level_ids:
        raise PairedTrainingError("Paired manifest contains no validation level IDs.")
    return level_ids


def _validate_latent_stats_artifact(
    plan: TrainingPlan,
    stats_path: Path | str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    stats_path = Path(stats_path).expanduser().resolve()
    expected_path = (plan.run_dir / LATENT_STATS_FILENAME).resolve()
    if stats_path != expected_path:
        raise PairedTrainingError(
            f"Latent statistics must be written to {expected_path}; got {stats_path}."
        )
    if not stats_path.is_file() or stats_path.stat().st_size == 0:
        raise PairedTrainingError(
            f"Held-out tokenizer latent-stat artifact is missing: {stats_path}."
        )
    try:
        value = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PairedTrainingError(
            f"Held-out tokenizer latent-stat artifact is malformed: {stats_path}."
        ) from exc
    if not isinstance(value, dict):
        raise PairedTrainingError(
            "Held-out tokenizer latent-stat artifact must be a JSON object."
        )
    try:
        validation = validate_measured_latent_stats(
            value,
            expected_dim=plan.latent_dimension,
            std_epsilon=plan.latent_std_epsilon,
        )
    except (PreflightError, TypeError, ValueError, OverflowError) as exc:
        raise PairedTrainingError(
            f"Held-out tokenizer latent statistics are invalid: {exc}"
        ) from exc

    if value.get("d_bottleneck") != plan.latent_dimension:
        raise PairedTrainingError(
            "Latent-stat artifact d_bottleneck does not match the selected "
            "tokenizer/dynamics profile."
        )
    source = _require_mapping(
        value.get("source"),
        "Latent-stat source",
    )
    expected_dataset_dir = plan.corpus.manifest_path.parent.resolve()
    if Path(str(source.get("dataset_dir"))).resolve() != expected_dataset_dir:
        raise PairedTrainingError(
            "Latent statistics were not measured from this paired corpus."
        )
    expected_checkpoint_dir = (plan.run_dir / "tokenizer" / "checkpoints").resolve()
    if Path(str(source.get("checkpoint_dir"))).resolve() != expected_checkpoint_dir:
        raise PairedTrainingError(
            "Latent statistics were not measured from this run's tokenizer checkpoint."
        )
    checkpoint_step = source.get("checkpoint_step")
    if not (expected_checkpoint_dir / str(checkpoint_step)).is_dir():
        raise PairedTrainingError(
            "Latent-stat source checkpoint_step does not exist in this run's "
            "tokenizer checkpoint directory."
        )

    expected_collectors = {
        source_summary["source"]
        for source_summary in plan.corpus.split_summaries["val"]["sources"]
    }
    for field in ("collector_records_seen", "collector_records_selected"):
        counts = source.get(field)
        if not isinstance(counts, dict):
            raise PairedTrainingError(f"Latent-stat source is missing {field}.")
        missing = [
            collector
            for collector in sorted(expected_collectors)
            if type(counts.get(collector)) is not int or counts[collector] <= 0
        ]
        if missing:
            raise PairedTrainingError(
                f"Latent-stat {field} lacks paired validation sources: {missing}."
            )

    measured_levels = source.get("level_seeds")
    if (
        not isinstance(measured_levels, list)
        or not measured_levels
        or any(type(level_id) is not int for level_id in measured_levels)
    ):
        raise PairedTrainingError(
            "Latent-stat source must contain integer validation level IDs."
        )
    unknown_levels = set(measured_levels) - _validation_level_ids(plan.corpus)
    if unknown_levels:
        raise PairedTrainingError(
            "Latent statistics include levels outside the paired validation "
            f"split: {sorted(unknown_levels)}."
        )
    return value, validation, sha256_file(stats_path)


def materialize_runtime_dynamics(
    plan: TrainingPlan,
    stats_path: Path | str,
) -> RuntimeDynamics:
    stats, validation, stats_sha256 = _validate_latent_stats_artifact(
        plan,
        stats_path,
    )
    try:
        stats_overrides = build_latent_hydra_overrides(stats)
    except (PreflightError, TypeError, ValueError, OverflowError) as exc:
        raise PairedTrainingError(
            f"Could not build latent normalization overrides: {exc}"
        ) from exc
    compose_overrides = [
        *plan.dynamics_user_overrides,
        *plan.dynamics_required_overrides[:-1],
        *stats_overrides,
    ]
    config = _compose_config(
        plan.repo_root,
        plan.dynamics_config_name,
        compose_overrides,
    )
    tokenizer_checkpoint = plan.run_dir / "tokenizer" / "checkpoints"
    _validate_resolved_config(
        stage="dynamics",
        config=config,
        corpus=plan.corpus,
        tokenizer_checkpoint=tokenizer_checkpoint,
        latent_stats=stats,
    )
    command = (
        plan.python_executable,
        str(plan.repo_root / "scripts" / "train_dynamics.py"),
        f"--config-name={plan.dynamics_config_name}",
        *plan.dynamics_user_overrides,
        *plan.dynamics_required_overrides[:-1],
        *stats_overrides,
        plan.dynamics_required_overrides[-1],
    )
    return RuntimeDynamics(
        command=command,
        config=config,
        stats_path=Path(stats_path).resolve(),
        stats_sha256=stats_sha256,
        stats=stats,
        validation=validation,
    )


def _config_summary(
    config: Mapping[str, Any],
    stage: str,
    config_name: str,
) -> dict[str, Any]:
    dataset = _require_mapping(config["dataset"], f"{stage} dataset config")
    summary = {
        "array_record_path": dataset["array_record_path"],
        "categorical_action_dim": dataset["categorical_action_dim"],
        "categorical_noop": dataset["categorical_noop"],
        "config_name": config_name,
        "data_type": dataset["data_type"],
        "dataset_name": dataset["name"],
        "run_name": config["run_name"],
    }
    if stage == "dynamics":
        dynamics = _require_mapping(config["dynamics"], "dynamics config")
        summary["model_categorical_action_dim"] = dynamics["categorical_action_dim"]
        summary["tokenizer_ckpt"] = config["tokenizer_ckpt"]
        summary["latent_mean"] = dynamics["latent_mean"]
        summary["latent_std"] = dynamics["latent_std"]
    return summary


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def publish_preflight_artifacts(plan: TrainingPlan) -> Path:
    if plan.run_dir.exists():
        raise PairedTrainingError(
            f"Paired run directory already exists: {plan.run_dir}."
        )
    inputs_dir = plan.run_dir / "inputs"
    commands_dir = plan.run_dir / "commands"
    configs_dir = plan.run_dir / "configs"
    inputs_dir.mkdir(parents=True)
    commands_dir.mkdir()
    configs_dir.mkdir()
    manifest_copy = inputs_dir / MANIFEST_COPY_FILENAME
    shutil.copyfile(plan.corpus.manifest_path, manifest_copy)
    if sha256_file(manifest_copy) != plan.corpus.manifest_sha256:
        raise PairedTrainingError("Copied paired manifest hash changed.")

    command_artifacts = {}
    for stage, command in (
        ("tokenizer", plan.tokenizer_command),
        ("latent_stats", plan.tokenizer_probe_command),
    ):
        command_path = commands_dir / f"{stage}.command"
        command_path.write_text(
            shlex.join(command) + "\n",
            encoding="utf-8",
        )
        command_artifacts[stage] = {
            "path": str(command_path),
            "sha256": sha256_file(command_path),
        }

    config_artifacts = {}
    for name, config in (
        ("tokenizer_preflight", plan.tokenizer_config),
        ("dynamics_preflight", plan.dynamics_preflight_config),
    ):
        config_path = configs_dir / f"{name}.json"
        _write_json_atomic(config_path, config)
        config_artifacts[name] = {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        }

    contract = {
        "action_space": plan.corpus.action_space,
        "artifacts": {
            "commands": command_artifacts,
            "configs": config_artifacts,
        },
        "commands": {
            "dynamics": None,
            "latent_stats": list(plan.tokenizer_probe_command),
            "tokenizer": list(plan.tokenizer_command),
        },
        "configs": {
            "dynamics_preflight": _config_summary(
                plan.dynamics_preflight_config,
                "dynamics",
                plan.dynamics_config_name,
            ),
            "tokenizer": _config_summary(
                plan.tokenizer_config,
                "tokenizer",
                plan.tokenizer_config_name,
            ),
        },
        "created_at": _utc_now(),
        "paired_manifest": {
            "artifact_copy": str(manifest_copy),
            "path": str(plan.corpus.manifest_path),
            "sha256": plan.corpus.manifest_sha256,
        },
        "schema_version": "paired-coinrun-training-run-v1",
        "splits": {
            "train": {
                **plan.corpus.split_summaries["train"],
                "array_record_dir": str(plan.corpus.train_dir),
                "manifest_split": "train",
            },
            "validation": {
                **plan.corpus.split_summaries["val"],
                "array_record_dir": str(plan.corpus.validation_dir),
                "manifest_split": "val",
            },
        },
        "stages": {
            "dynamics": {
                "reason": "waiting for validated held-out latent statistics",
                "status": "blocked",
            },
            "latent_stats": {
                "artifact_path": str(plan.run_dir / LATENT_STATS_FILENAME),
                "max_records": plan.latent_stat_max_records,
                "status": "pending",
                "std_epsilon": plan.latent_std_epsilon,
            },
            "tokenizer": {"status": "pending"},
        },
        "status": "preflight_passed",
    }
    contract_path = plan.run_dir / RUN_CONTRACT_FILENAME
    _write_json_atomic(contract_path, contract)
    return contract_path


def _set_run_status(
    contract_path: Path,
    status: str,
    **details: Any,
) -> None:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["status"] = status
    contract.update(details)
    _write_json_atomic(contract_path, contract)


def _set_stage_status(
    contract_path: Path,
    stage: str,
    status: str,
    **details: Any,
) -> None:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    stage_payload = contract["stages"][stage]
    stage_payload["status"] = status
    stage_payload.update(details)
    contract["active_stage"] = None if status == "passed" else stage
    _write_json_atomic(contract_path, contract)


def _record_runtime_dynamics(
    plan: TrainingPlan,
    runtime: RuntimeDynamics,
    contract_path: Path,
) -> None:
    config_path = plan.run_dir / "configs" / "dynamics_runtime.json"
    command_path = plan.run_dir / "commands" / "dynamics.command"
    _write_json_atomic(config_path, runtime.config)
    command_path.write_text(
        shlex.join(runtime.command) + "\n",
        encoding="utf-8",
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["commands"]["dynamics"] = list(runtime.command)
    contract["configs"]["dynamics_runtime"] = _config_summary(
        runtime.config,
        "dynamics",
        plan.dynamics_config_name,
    )
    contract["latent_stats"] = {
        "path": str(runtime.stats_path),
        "sha256": runtime.stats_sha256,
        "source": runtime.stats["source"],
        "validation": runtime.validation,
    }
    contract["artifacts"]["commands"]["dynamics"] = {
        "path": str(command_path),
        "sha256": sha256_file(command_path),
    }
    contract["artifacts"]["configs"]["dynamics_runtime"] = {
        "path": str(config_path),
        "sha256": sha256_file(config_path),
    }
    contract["stages"]["latent_stats"].update(
        {
            "sha256": runtime.stats_sha256,
            "status": "passed",
            "validation": runtime.validation,
        }
    )
    contract["stages"]["dynamics"] = {
        "reason": None,
        "status": "ready",
    }
    contract["status"] = "latent_stats_passed"
    contract["active_stage"] = None
    _write_json_atomic(contract_path, contract)


def _revalidate_plan_corpus(plan: TrainingPlan) -> None:
    validate_paired_corpus(
        manifest_path=plan.corpus.manifest_path,
        expected_manifest_sha256=plan.corpus.manifest_sha256,
        train_dir=plan.corpus.train_dir,
        validation_dir=plan.corpus.validation_dir,
    )


def _run_stage_command(
    plan: TrainingPlan,
    contract_path: Path,
    stage: str,
    command: Sequence[str],
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> Path:
    try:
        _revalidate_plan_corpus(plan)
        _set_run_status(
            contract_path,
            f"running_{stage}",
            active_stage=stage,
            launch_started_at=_utc_now(),
        )
        _set_stage_status(
            contract_path,
            stage,
            "running",
            started_at=_utc_now(),
        )
        log_path = plan.run_dir / f"{stage}.log"
        with log_path.open("wb") as log:
            completed = runner(
                list(command),
                cwd=plan.repo_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise PairedTrainingError(
                f"{stage} stage exited with {completed.returncode}; see {log_path}."
            )
        return log_path
    except BaseException as exc:
        _set_stage_status(
            contract_path,
            stage,
            "failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=_utc_now(),
        )
        _set_run_status(
            contract_path,
            "failed",
            active_stage=stage,
            error=f"{type(exc).__name__}: {exc}",
            finished_at=_utc_now(),
        )
        raise


def execute_training(
    plan: TrainingPlan,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> None:
    contract_path = plan.run_dir / RUN_CONTRACT_FILENAME
    if not contract_path.is_file():
        raise PairedTrainingError(
            "Paired preflight artifacts must be published before launch."
        )

    tokenizer_log = _run_stage_command(
        plan,
        contract_path,
        "tokenizer",
        plan.tokenizer_command,
        runner,
    )
    checkpoint_dir = plan.run_dir / "tokenizer" / "checkpoints"
    if not checkpoint_dir.is_dir() or not any(checkpoint_dir.iterdir()):
        error = "tokenizer trainer produced no checkpoint"
        _set_stage_status(
            contract_path,
            "tokenizer",
            "failed",
            error=error,
            finished_at=_utc_now(),
        )
        _set_run_status(
            contract_path,
            "failed",
            active_stage="tokenizer",
            error=error,
            finished_at=_utc_now(),
        )
        raise PairedTrainingError(
            "Tokenizer trainer succeeded but produced no checkpoint; "
            "latent-stat and dynamics launch are blocked."
        )
    _set_stage_status(
        contract_path,
        "tokenizer",
        "passed",
        checkpoint_dir=str(checkpoint_dir),
        finished_at=_utc_now(),
        log_path=str(tokenizer_log),
        log_sha256=sha256_file(tokenizer_log),
    )

    stats_log = _run_stage_command(
        plan,
        contract_path,
        "latent_stats",
        plan.tokenizer_probe_command,
        runner,
    )
    try:
        runtime = materialize_runtime_dynamics(
            plan,
            plan.run_dir / LATENT_STATS_FILENAME,
        )
        _record_runtime_dynamics(plan, runtime, contract_path)
        _set_stage_status(
            contract_path,
            "latent_stats",
            "passed",
            finished_at=_utc_now(),
            log_path=str(stats_log),
            log_sha256=sha256_file(stats_log),
            sha256=runtime.stats_sha256,
            validation=runtime.validation,
        )
    except BaseException as exc:
        _set_stage_status(
            contract_path,
            "latent_stats",
            "failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=_utc_now(),
        )
        _set_run_status(
            contract_path,
            "failed",
            active_stage="latent_stats",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=_utc_now(),
        )
        raise

    try:
        if sha256_file(runtime.stats_path) != runtime.stats_sha256:
            raise PairedTrainingError(
                "Held-out latent-stat artifact changed before dynamics launch."
            )
        runtime = materialize_runtime_dynamics(plan, runtime.stats_path)
    except BaseException as exc:
        _set_stage_status(
            contract_path,
            "dynamics",
            "failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=_utc_now(),
        )
        _set_run_status(
            contract_path,
            "failed",
            active_stage="dynamics",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=_utc_now(),
        )
        raise

    dynamics_log = _run_stage_command(
        plan,
        contract_path,
        "dynamics",
        runtime.command,
        runner,
    )
    _set_stage_status(
        contract_path,
        "dynamics",
        "passed",
        finished_at=_utc_now(),
        log_path=str(dynamics_log),
        log_sha256=sha256_file(dynamics_log),
    )
    _set_run_status(
        contract_path,
        "completed",
        active_stage=None,
        finished_at=_utc_now(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or launch paired RGB CoinRun tokenizer and dynamics "
            "training without dataset collection."
        )
    )
    parser.add_argument("mode", choices=("preflight", "launch"))
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--paired-manifest", type=Path, required=True)
    parser.add_argument("--paired-manifest-sha256", required=True)
    parser.add_argument("--train-array-record-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-array-record-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tokenizer-config-name",
        default="coinrun_tokenizer",
    )
    parser.add_argument(
        "--dynamics-config-name",
        default="coinrun_dynamics",
    )
    parser.add_argument(
        "--tokenizer-override",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--dynamics-override",
        action="append",
        default=[],
    )
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = build_training_plan(
            repo_root=args.repo_root,
            run_dir=args.run_dir,
            manifest_path=args.paired_manifest,
            expected_manifest_sha256=args.paired_manifest_sha256,
            train_dir=args.train_array_record_dir,
            validation_dir=args.validation_array_record_dir,
            tokenizer_config_name=args.tokenizer_config_name,
            dynamics_config_name=args.dynamics_config_name,
            tokenizer_overrides=args.tokenizer_override,
            dynamics_overrides=args.dynamics_override,
            latent_stat_max_records=args.latent_stat_max_records,
            latent_std_epsilon=args.latent_std_epsilon,
        )
        contract_path = publish_preflight_artifacts(plan)
        if args.mode == "launch":
            execute_training(plan)
    except PairedTrainingError as exc:
        print(f"Paired CoinRun training rejected: {exc}", file=sys.stderr)
        return 2
    print(f"Paired CoinRun {args.mode} passed: {contract_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
