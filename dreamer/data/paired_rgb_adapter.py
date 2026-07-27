"""Convert Transparent CoinRun paired V2 NPZ shards to CoinRun ArrayRecords.

The resulting records use the existing pixel pipeline's pickle contract. Each
ArrayRecord entry contains one complete source episode, so episode and split
boundaries are not introduced or merged by the adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt

PAIRED_SCHEMA_VERSION = "transparent-coinrun-compact-paired-trajectory-v2"
PAIRED_CONTRACT = "transparent-coinrun-v1"
ADAPTER_SCHEMA_VERSION = "open-dreamer-transparent-coinrun-paired-rgb-v1"
COINRUN_NUM_ACTIONS = 9
COINRUN_NOOP_ACTION = 4


@dataclass(frozen=True)
class SourceShard:
    filename: str
    split: str
    source: str


SOURCE_SHARDS = (
    SourceShard("train_random.npz", "train", "random"),
    SourceShard(
        "train_scripted.npz",
        "train",
        "scripted_forward_v0",
    ),
    SourceShard("val_random.npz", "val", "random"),
    SourceShard(
        "val_scripted.npz",
        "val",
        "scripted_forward_v0",
    ),
)

REQUIRED_FIELDS = (
    "schema_version",
    "contract",
    "source",
    "episode_index",
    "provenance_json",
    "trajectory_level_ids",
    "trajectory_offsets",
    "trajectory_lengths",
    "effective_action",
    "reward",
    "first",
    "done",
    "terminal_cause",
    "timestep",
    "next_state_valid",
    "rgb64",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: npt.NDArray[Any]) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _scalar_string(arrays: Mapping[str, npt.NDArray[Any]], name: str) -> str:
    if name not in arrays:
        raise ValueError(f"Paired shard is missing {name!r}.")
    value = np.asarray(arrays[name])
    if value.shape != ():
        raise ValueError(f"Paired shard field {name!r} must be scalar.")
    return str(value.item())


def _scalar_int(arrays: Mapping[str, npt.NDArray[Any]], name: str) -> int:
    if name not in arrays:
        raise ValueError(f"Paired shard is missing {name!r}.")
    value = np.asarray(arrays[name])
    if value.shape != () or value.dtype.kind not in "iu":
        raise ValueError(f"Paired shard field {name!r} must be a scalar integer.")
    return int(value.item())


def _require_array(
    arrays: Mapping[str, npt.NDArray[Any]],
    name: str,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int | None, ...],
) -> npt.NDArray[Any]:
    if name not in arrays:
        raise ValueError(f"Paired shard is missing {name!r}.")
    value = np.asarray(arrays[name])
    if value.dtype != dtype:
        raise ValueError(
            f"Paired shard field {name!r} must have dtype {dtype}; got {value.dtype}."
        )
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape, strict=True)
    ):
        raise ValueError(
            f"Paired shard field {name!r} must have shape {shape}; got {value.shape}."
        )
    return value


def validate_paired_shard(
    arrays: Mapping[str, npt.NDArray[Any]],
    spec: SourceShard,
) -> None:
    """Validate fields that define pixel/action alignment and boundaries."""

    if _scalar_string(arrays, "schema_version") != PAIRED_SCHEMA_VERSION:
        raise ValueError(
            f"{spec.filename} is not a Transparent CoinRun paired V2 shard."
        )
    if _scalar_string(arrays, "contract") != PAIRED_CONTRACT:
        raise ValueError(f"{spec.filename} has an unexpected contract.")
    actual_source = _scalar_string(arrays, "source")
    if actual_source != spec.source:
        raise ValueError(
            f"{spec.filename} source is {actual_source!r}, expected {spec.source!r}."
        )
    _scalar_int(arrays, "episode_index")

    level_ids = _require_array(
        arrays,
        "trajectory_level_ids",
        dtype=np.dtype(np.int32),
        shape=(None,),
    )
    offsets = _require_array(
        arrays,
        "trajectory_offsets",
        dtype=np.dtype(np.int64),
        shape=(level_ids.size,),
    )
    lengths = _require_array(
        arrays,
        "trajectory_lengths",
        dtype=np.dtype(np.int64),
        shape=(level_ids.size,),
    )
    if level_ids.size == 0 or np.unique(level_ids).size != level_ids.size:
        raise ValueError(f"{spec.filename} must contain unique level IDs.")
    if np.any(lengths <= 0):
        raise ValueError(f"{spec.filename} has a non-positive episode length.")
    expected_offsets = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.cumsum(lengths[:-1], dtype=np.int64),
        )
    )
    if not np.array_equal(offsets, expected_offsets):
        raise ValueError(f"{spec.filename} trajectory offsets are not contiguous.")

    total = int(lengths.sum(dtype=np.int64))
    rgb64 = _require_array(
        arrays,
        "rgb64",
        dtype=np.dtype(np.uint8),
        shape=(total, 64, 64, 3),
    )
    actions = _require_array(
        arrays,
        "effective_action",
        dtype=np.dtype(np.int8),
        shape=(total,),
    )
    rewards = _require_array(
        arrays,
        "reward",
        dtype=np.dtype(np.float32),
        shape=(total,),
    )
    first = _require_array(
        arrays,
        "first",
        dtype=np.dtype(np.bool_),
        shape=(total,),
    )
    done = _require_array(
        arrays,
        "done",
        dtype=np.dtype(np.bool_),
        shape=(total,),
    )
    next_state_valid = _require_array(
        arrays,
        "next_state_valid",
        dtype=np.dtype(np.bool_),
        shape=(total,),
    )
    timestep = _require_array(
        arrays,
        "timestep",
        dtype=np.dtype(np.int32),
        shape=(total,),
    )
    _require_array(
        arrays,
        "terminal_cause",
        dtype=np.dtype(np.uint8),
        shape=(total,),
    )

    if rgb64.shape[0] != actions.shape[0] or actions.shape != rewards.shape:
        raise ValueError(f"{spec.filename} frame/action/reward rows are misaligned.")
    if np.any((actions < 0) | (actions >= COINRUN_NUM_ACTIONS)):
        raise ValueError(f"{spec.filename} contains an invalid CoinRun action.")
    if not np.isfinite(rewards).all():
        raise ValueError(f"{spec.filename} contains a non-finite reward.")
    if not np.array_equal(next_state_valid, ~done):
        raise ValueError(
            f"{spec.filename} next_state_valid must be the inverse of done."
        )

    for offset_value, length_value in zip(offsets, lengths, strict=True):
        start = int(offset_value)
        length = int(length_value)
        end = start + length
        expected_first = np.zeros(length, dtype=np.bool_)
        expected_first[0] = True
        expected_done = np.zeros(length, dtype=np.bool_)
        expected_done[-1] = True
        if not np.array_equal(first[start:end], expected_first):
            raise ValueError(
                f"{spec.filename} has invalid first labels at row {start}."
            )
        if not np.array_equal(done[start:end], expected_done):
            raise ValueError(f"{spec.filename} has invalid done labels at row {start}.")
        if not np.array_equal(
            timestep[start:end],
            np.arange(length, dtype=np.int32),
        ):
            raise ValueError(f"{spec.filename} has invalid timesteps at row {start}.")


class _ArrayRecordShardWriter:
    def __init__(
        self,
        output_dir: Path,
        prefix: str,
        records_per_shard: int,
    ) -> None:
        if records_per_shard <= 0:
            raise ValueError("records_per_shard must be positive.")
        try:
            from array_record.python.array_record_module import ArrayRecordWriter
        except ImportError as exc:
            raise ImportError(
                "array-record is required to convert paired RGB shards."
            ) from exc

        self._writer_type = ArrayRecordWriter
        self.output_dir = output_dir
        self.prefix = prefix
        self.records_per_shard = records_per_shard
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._writer = None
        self._records_in_shard = 0
        self._shard_index = 0
        self.paths: list[Path] = []

    def _open_shard(self) -> None:
        if self._writer is not None:
            self._writer.close()
        path = self.output_dir / f"{self.prefix}-{self._shard_index:05d}.array_record"
        self._writer = self._writer_type(str(path), "group_size:1")
        self.paths.append(path)
        self._shard_index += 1
        self._records_in_shard = 0

    def write(self, payload: bytes) -> None:
        if self._writer is None or self._records_in_shard >= self.records_per_shard:
            self._open_shard()
        self._writer.write(payload)
        self._records_in_shard += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _episode_slices(
    offsets: npt.NDArray[np.int64],
    lengths: npt.NDArray[np.int64],
) -> Iterator[tuple[int, int, int]]:
    for trajectory_index, (offset, length) in enumerate(
        zip(offsets, lengths, strict=True)
    ):
        start = int(offset)
        yield trajectory_index, start, start + int(length)


def _record_for_episode(
    arrays: Mapping[str, npt.NDArray[Any]],
    spec: SourceShard,
    source_sha256: str,
    trajectory_index: int,
    start: int,
    end: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frames = np.ascontiguousarray(arrays["rgb64"][start:end])
    actions = np.asarray(arrays["effective_action"][start:end]).copy()
    rewards = np.asarray(arrays["reward"][start:end]).copy()
    first = np.asarray(arrays["first"][start:end]).copy()
    done = np.asarray(arrays["done"][start:end]).copy()
    terminal_cause = np.asarray(arrays["terminal_cause"][start:end]).copy()
    timestep = np.asarray(arrays["timestep"][start:end]).copy()
    next_state_valid = np.asarray(arrays["next_state_valid"][start:end]).copy()
    level_id = int(arrays["trajectory_level_ids"][trajectory_index])
    episode_index = _scalar_int(arrays, "episode_index")
    episode_id = f"{spec.split}:{spec.source}:level-{level_id}:episode-{episode_index}"
    terminal_cause_value = int(terminal_cause[-1])

    frame_sha256 = sha256_array(frames)
    action_sha256 = sha256_array(actions)
    reward_sha256 = sha256_array(rewards)
    record = {
        "schema_version": 1,
        "serialization_format": "pickle",
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "source_schema_version": PAIRED_SCHEMA_VERSION,
        "source_contract": PAIRED_CONTRACT,
        "source_shard": spec.filename,
        "source_shard_sha256": source_sha256,
        "raw_video": frames.tobytes(order="C"),
        "frame_shape": np.asarray(frames.shape[1:], dtype=np.int32),
        "sequence_length": end - start,
        "actions": actions,
        "rewards": rewards,
        "episode_starts": first,
        "episode_ends": done,
        "split": spec.split,
        "source": spec.source,
        "collector": spec.source,
        "collector_identity": spec.source,
        "episode_id": episode_id,
        "episode_index": episode_index,
        "trajectory_index": trajectory_index,
        "episode_length": end - start,
        "chunk_index": 0,
        "chunk_start": 0,
        "chunk_end": end - start,
        "is_first_chunk": True,
        "is_last_chunk": True,
        "level_id": level_id,
        "level_seed": level_id,
        "source_row_start": start,
        "source_row_end": end,
        "terminated": terminal_cause_value != 3,
        "truncated": terminal_cause_value == 3,
        "terminal_cause": terminal_cause,
        "timestep": timestep,
        "next_state_valid": next_state_valid,
        "action_space": "procgen_discrete",
        "num_actions": COINRUN_NUM_ACTIONS,
        "categorical_noop": COINRUN_NOOP_ACTION,
        "action_alignment": "action_applied_after_frame",
        "reward_alignment": "reward_resulting_from_action",
        "paired_frame_sha256": frame_sha256,
        "action_sha256": action_sha256,
        "reward_sha256": reward_sha256,
        "source_provenance_json": _scalar_string(
            arrays,
            "provenance_json",
        ),
    }
    manifest_episode = {
        "episode_id": episode_id,
        "episode_index": episode_index,
        "frame_sha256": frame_sha256,
        "action_sha256": action_sha256,
        "reward_sha256": reward_sha256,
        "length": end - start,
        "level_id": level_id,
        "source_row_end": end,
        "source_row_start": start,
        "trajectory_index": trajectory_index,
    }
    return record, manifest_episode


def _convert_source_shard(
    source_path: Path,
    staging_dir: Path,
    spec: SourceShard,
    records_per_shard: int,
) -> dict[str, Any]:
    source_sha256 = sha256_file(source_path)
    with np.load(source_path, allow_pickle=False) as loaded:
        arrays = {
            name: loaded[name] for name in REQUIRED_FIELDS if name in loaded.files
        }
    validate_paired_shard(arrays, spec)

    offsets = np.asarray(arrays["trajectory_offsets"])
    lengths = np.asarray(arrays["trajectory_lengths"])
    episodes: list[dict[str, Any]] = []
    split_dir = staging_dir / spec.split
    writer = _ArrayRecordShardWriter(
        split_dir,
        prefix=spec.source,
        records_per_shard=records_per_shard,
    )
    with writer:
        for trajectory_index, start, end in _episode_slices(offsets, lengths):
            record, manifest_episode = _record_for_episode(
                arrays,
                spec,
                source_sha256,
                trajectory_index,
                start,
                end,
            )
            payload = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
            manifest_episode["record_sha256"] = hashlib.sha256(payload).hexdigest()
            writer.write(payload)
            episodes.append(manifest_episode)

    output_shards = [
        {
            "path": path.relative_to(staging_dir).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in writer.paths
    ]
    return {
        "episodes": episodes,
        "filename": spec.filename,
        "num_episodes": len(episodes),
        "num_frames": int(lengths.sum(dtype=np.int64)),
        "output_shards": output_shards,
        "sha256": source_sha256,
        "size_bytes": source_path.stat().st_size,
        "source": spec.source,
        "split": spec.split,
    }


def convert_paired_rgb_corpus(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    records_per_shard: int = 100,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert all four fixed paired V2 source shards and return the manifest."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if records_per_shard <= 0:
        raise ValueError("records_per_shard must be positive.")
    missing = [
        spec.filename
        for spec in SOURCE_SHARDS
        if not (input_path / spec.filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing required paired RGB shards in {input_path}: {missing}"
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass overwrite=True to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.tmp-",
            dir=output_path.parent,
        )
    )
    try:
        source_manifests = [
            _convert_source_shard(
                input_path / spec.filename,
                staging_path,
                spec,
                records_per_shard,
            )
            for spec in SOURCE_SHARDS
        ]
        manifest = {
            "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
            "action_alignment": "action_applied_after_frame",
            "action_space": {
                "categorical_action_dim": COINRUN_NUM_ACTIONS,
                "categorical_noop": COINRUN_NOOP_ACTION,
                "continuous_action_dim": 0,
                "num_binary_actions": 0,
                "type": "procgen_discrete",
            },
            "input_contract": PAIRED_CONTRACT,
            "input_schema_version": PAIRED_SCHEMA_VERSION,
            "num_episodes": sum(source["num_episodes"] for source in source_manifests),
            "num_frames": sum(source["num_frames"] for source in source_manifests),
            "records_per_shard": records_per_shard,
            "reward_alignment": "reward_resulting_from_action",
            "sources": source_manifests,
            "splits": {
                split: {
                    "num_episodes": sum(
                        source["num_episodes"]
                        for source in source_manifests
                        if source["split"] == split
                    ),
                    "num_frames": sum(
                        source["num_frames"]
                        for source in source_manifests
                        if source["split"] == split
                    ),
                    "sources": [
                        source["source"]
                        for source in source_manifests
                        if source["split"] == split
                    ],
                }
                for split in ("train", "val")
            },
        }
        manifest_path = staging_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if output_path.exists():
            shutil.rmtree(output_path)
        os.replace(staging_path, output_path)
        return manifest
    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Transparent CoinRun paired V2 NPZ shards into the "
            "existing CoinRun pixel pipeline's ArrayRecord format."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records-per-shard", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = convert_paired_rgb_corpus(
        args.input_dir,
        args.output_dir,
        records_per_shard=args.records_per_shard,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "manifest": str(args.output_dir / "manifest.json"),
                "num_episodes": manifest["num_episodes"],
                "num_frames": manifest["num_frames"],
                "train_path": str(args.output_dir / "train"),
                "val_path": str(args.output_dir / "val"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
