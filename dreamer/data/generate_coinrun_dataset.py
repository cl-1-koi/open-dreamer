# /// script
# requires-python = "==3.11.*"
# dependencies = [
#   "array-record>=0.8.3",
#   "gym3==0.3.3",
#   "numpy>=1.25,<2",
#   "procgen @ git+https://github.com/openai/procgen.git@5e1dbf341d291eff40d1f9e0c0a0d5003643aebf",
#   "tyro>=1.0.0",
# ]
#
# [tool.uv.extra-build-dependencies]
# procgen = ["gym3==0.3.3", "setuptools>=61", "wheel"]
# ///

"""Generate deterministic, bounded CoinRun ArrayRecord datasets.

Run this file with ``uv run --isolated`` so Procgen's NumPy 1.x environment is
separate from the NumPy 2.x JAX training environment.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np

CollectorName = Literal["random", "scripted"]

COINRUN_NUM_ACTIONS = 15
COINRUN_NOOP_ACTION = 4
COINRUN_RIGHT_ACTION = 7
COINRUN_RIGHT_JUMP_ACTION = 8
COINRUN_BEHAVIORAL_NOOP_ACTIONS = (4, 9, 10, 11, 12, 13, 14)
COINRUN_DISTINCT_MOVE_ACTIONS = tuple(range(9))
MAX_PROCGEN_SEED = 2**31 - 1
SPLIT_IDS = {"train": 0, "val": 1, "test": 2}
LEVEL_SEED_RANGE_SIZE = MAX_PROCGEN_SEED // len(SPLIT_IDS)


@dataclass(frozen=True)
class Args:
    num_episodes_train: int = 10000
    num_episodes_val: int = 500
    num_episodes_test: int = 500
    output_dir: str = "datasets/coinrun_episodes"
    min_episode_length: int = 64
    max_episode_length: int = 1000
    chunk_size: int = 160
    chunks_per_file: int = 100
    seed: int = 0
    collector: CollectorName = "random"
    max_attempts_per_split: int = 0
    keep_short_terminated: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class CollectedEpisode:
    frames: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: bool


class PickleShardWriter:
    """ArrayRecord writer for the pickle format consumed by CoinRun transforms."""

    def __init__(self, output_dir: Path | str, records_per_shard: int):
        if records_per_shard <= 0:
            raise ValueError("records_per_shard must be positive.")
        try:
            from array_record.python.array_record_module import ArrayRecordWriter
        except ImportError as exc:
            raise ImportError(
                "array-record is required to generate CoinRun datasets."
            ) from exc

        self._writer_type = ArrayRecordWriter
        self.output_dir = Path(output_dir)
        self.records_per_shard = records_per_shard
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._writer = None
        self._shard_index = 0
        self.records_in_shard = 0
        self.total_records = 0

    @property
    def num_shards(self) -> int:
        return self._shard_index

    def _open_shard(self) -> None:
        if self._writer is not None:
            self._writer.close()
        path = self.output_dir / f"shard-{self._shard_index:05d}.array_record"
        self._writer = self._writer_type(str(path), "group_size:1")
        self._shard_index += 1
        self.records_in_shard = 0

    def write(self, record: dict[str, Any]) -> None:
        if (
            self._writer is None
            or self.records_in_shard >= self.records_per_shard
        ):
            self._open_shard()
        self._writer.write(pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
        self.records_in_shard += 1
        self.total_records += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def validate_args(args: Args) -> None:
    episode_counts = (
        args.num_episodes_train,
        args.num_episodes_val,
        args.num_episodes_test,
    )
    if any(count < 0 for count in episode_counts):
        raise ValueError("Episode counts must be non-negative.")
    if args.min_episode_length <= 0:
        raise ValueError("min_episode_length must be positive.")
    if args.max_episode_length < args.min_episode_length:
        raise ValueError(
            "max_episode_length must be greater than or equal to "
            "min_episode_length."
        )
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if args.chunks_per_file <= 0:
        raise ValueError("chunks_per_file must be positive.")
    if args.max_attempts_per_split < 0:
        raise ValueError("max_attempts_per_split must be non-negative.")
    if args.seed < 0:
        raise ValueError("seed must be non-negative.")
    if args.collector not in ("random", "scripted"):
        raise ValueError(f"Unknown collector: {args.collector!r}.")


def _split_seed(dataset_seed: int, split: str) -> int:
    seed_sequence = np.random.SeedSequence([dataset_seed, SPLIT_IDS[split]])
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def _collector_identity(collector: CollectorName) -> str:
    if collector == "random":
        return "procgen_uniform_random_v1"
    return "coinrun_run_right_seeded_jump_v1"


def _level_seed_range(split: str) -> tuple[int, int]:
    start = SPLIT_IDS[split] * LEVEL_SEED_RANGE_SIZE
    return start, start + LEVEL_SEED_RANGE_SIZE


def _select_action(
    collector: CollectorName,
    action_rng: np.random.Generator,
    step: int,
    num_actions: int,
    scripted_jump_phase: int,
) -> int:
    if collector == "random":
        return int(action_rng.integers(0, num_actions))

    if num_actions <= COINRUN_RIGHT_JUMP_ACTION:
        raise ValueError(
            f"The scripted CoinRun collector requires at least "
            f"{COINRUN_RIGHT_JUMP_ACTION + 1} actions; got {num_actions}."
        )
    jump_period = 12
    if (step + scripted_jump_phase) % jump_period < 2:
        return COINRUN_RIGHT_JUMP_ACTION
    return COINRUN_RIGHT_ACTION


def collect_episode(
    env: Any,
    *,
    collector: CollectorName,
    action_rng: np.random.Generator,
    max_episode_length: int,
) -> CollectedEpisode:
    """Collect aligned ``(frame, action, resulting reward)`` transitions."""

    _, observation, _ = env.observe()
    current_frame = np.asarray(observation["rgb"])[0]
    num_actions = int(env.ac_space.eltype.n)
    scripted_jump_phase = int(action_rng.integers(0, 12))

    frames: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    terminated = False

    for step in range(max_episode_length):
        action = _select_action(
            collector,
            action_rng,
            step,
            num_actions,
            scripted_jump_phase,
        )
        env.act(np.asarray([action], dtype=np.int32))
        reward, next_observation, first = env.observe()

        frames.append(np.asarray(current_frame, dtype=np.uint8).copy())
        actions.append(action)
        rewards.append(float(np.asarray(reward).reshape(-1)[0]))

        if bool(np.asarray(first).reshape(-1)[0]):
            terminated = True
            break
        current_frame = np.asarray(next_observation["rgb"])[0]

    return CollectedEpisode(
        frames=np.stack(frames).astype(np.uint8, copy=False),
        actions=np.asarray(actions, dtype=np.int32),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=terminated,
    )


def episode_records(
    episode: CollectedEpisode,
    *,
    split: str,
    episode_id: str,
    dataset_seed: int,
    split_seed: int,
    level_seed: int,
    environment_seed: int,
    action_seed: int,
    collector: CollectorName,
    num_actions: int,
    chunk_size: int,
) -> Iterator[dict[str, Any]]:
    """Yield lossless chunk records with episode boundary metadata."""

    episode_length = int(episode.frames.shape[0])
    if not (
        episode.actions.shape == (episode_length,)
        and episode.rewards.shape == (episode_length,)
    ):
        raise ValueError("Frame, action, and reward lengths must match.")

    for chunk_index, start in enumerate(range(0, episode_length, chunk_size)):
        end = min(start + chunk_size, episode_length)
        frames = episode.frames[start:end]
        actions = episode.actions[start:end].copy()
        rewards = episode.rewards[start:end].copy()
        sequence_length = end - start
        episode_starts = np.zeros(sequence_length, dtype=np.bool_)
        episode_ends = np.zeros(sequence_length, dtype=np.bool_)
        if start == 0:
            episode_starts[0] = True
        if end == episode_length:
            episode_ends[-1] = True

        yield {
            "schema_version": 1,
            "serialization_format": "pickle",
            "raw_video": frames.tobytes(),
            "frame_shape": np.asarray(frames.shape[1:], dtype=np.int32),
            "sequence_length": sequence_length,
            "actions": actions,
            "rewards": rewards,
            "episode_starts": episode_starts,
            "episode_ends": episode_ends,
            "split": split,
            "episode_id": episode_id,
            "episode_length": episode_length,
            "chunk_index": chunk_index,
            "chunk_start": start,
            "chunk_end": end,
            "is_first_chunk": start == 0,
            "is_last_chunk": end == episode_length,
            "terminated": episode.terminated,
            "truncated": not episode.terminated,
            "dataset_seed": dataset_seed,
            "split_seed": split_seed,
            "level_seed": level_seed,
            "environment_seed": environment_seed,
            "action_seed": action_seed,
            "collector": collector,
            "collector_identity": _collector_identity(collector),
            "action_space": "procgen_discrete",
            "num_actions": num_actions,
            "categorical_noop": COINRUN_NOOP_ACTION,
            "action_alignment": "action_applied_after_frame",
            "reward_alignment": "reward_resulting_from_action",
            "behavioral_noop_actions": np.asarray(
                COINRUN_BEHAVIORAL_NOOP_ACTIONS,
                dtype=np.int32,
            ),
            "behaviorally_distinct_move_actions": np.asarray(
                COINRUN_DISTINCT_MOVE_ACTIONS,
                dtype=np.int32,
            ),
        }


def _default_env_factory(level_seed: int, environment_seed: int) -> Any:
    try:
        from procgen import ProcgenGym3Env
    except ImportError as exc:
        raise ImportError(
            "Procgen is required to generate CoinRun data. Run this file with "
            "`uv run dreamer/data/generate_coinrun_dataset.py` to use its "
            "isolated collector environment."
        ) from exc

    return ProcgenGym3Env(
        num=1,
        env_name="coinrun",
        start_level=level_seed,
        num_levels=1,
        rand_seed=environment_seed,
        num_threads=0,
    )


def _prepare_split_directory(path: Path, overwrite: bool) -> None:
    existing = sorted(path.glob("shard-*.array_record")) if path.exists() else []
    if existing and not overwrite:
        raise FileExistsError(
            f"{path} already contains ArrayRecord shards; pass --overwrite to "
            "replace them."
        )
    if overwrite:
        for shard in existing:
            shard.unlink()


def generate_split(
    args: Args,
    *,
    split: str,
    num_episodes: int,
    env_factory: Callable[[int, int], Any] = _default_env_factory,
    writer_factory: Callable[[Path, int], Any] = PickleShardWriter,
) -> dict[str, Any]:
    """Generate one split and return JSON-serializable aggregate metadata."""

    output_dir = Path(args.output_dir) / split
    _prepare_split_directory(output_dir, args.overwrite)
    split_seed = _split_seed(args.seed, split)
    split_rng = np.random.default_rng(split_seed)
    max_attempts = args.max_attempts_per_split or max(10 * num_episodes, 1)
    if max_attempts > LEVEL_SEED_RANGE_SIZE:
        raise ValueError(
            f"max_attempts_per_split={max_attempts} exceeds the disjoint level "
            f"seed range size {LEVEL_SEED_RANGE_SIZE}."
        )
    level_seed_start, level_seed_end = _level_seed_range(split)
    level_seed_offset = int(split_rng.integers(0, LEVEL_SEED_RANGE_SIZE))

    episodes: list[dict[str, Any]] = []
    total_frames = 0
    total_positive_rewards = 0
    total_reward = 0.0
    action_histogram: np.ndarray | None = None
    attempts = 0

    with writer_factory(output_dir, args.chunks_per_file) as writer:
        while len(episodes) < num_episodes and attempts < max_attempts:
            attempts += 1
            level_seed = level_seed_start + (
                level_seed_offset + attempts - 1
            ) % LEVEL_SEED_RANGE_SIZE
            environment_seed = int(split_rng.integers(0, MAX_PROCGEN_SEED))
            action_seed = int(split_rng.integers(0, MAX_PROCGEN_SEED))
            env = env_factory(level_seed, environment_seed)
            try:
                num_actions = int(env.ac_space.eltype.n)
                if num_actions != COINRUN_NUM_ACTIONS:
                    raise ValueError(
                        f"CoinRun must expose {COINRUN_NUM_ACTIONS} actions; "
                        f"got {num_actions}."
                    )
                episode = collect_episode(
                    env,
                    collector=args.collector,
                    action_rng=np.random.default_rng(action_seed),
                    max_episode_length=args.max_episode_length,
                )
            finally:
                close = getattr(env, "close", None)
                if close is not None:
                    close()

            episode_length = int(episode.frames.shape[0])
            keep_short_termination = (
                episode.terminated and args.keep_short_terminated
            )
            if (
                episode_length < args.min_episode_length
                and not keep_short_termination
            ):
                print(
                    f"[{split}] rejected attempt {attempts}: "
                    f"{episode_length} < {args.min_episode_length} frames"
                )
                continue

            if action_histogram is None:
                action_histogram = np.zeros(num_actions, dtype=np.int64)
            elif action_histogram.shape != (num_actions,):
                raise ValueError("Procgen action-space size changed within a split.")

            episode_index = len(episodes)
            episode_id = f"{split}-{episode_index:06d}"
            records = list(
                episode_records(
                    episode,
                    split=split,
                    episode_id=episode_id,
                    dataset_seed=args.seed,
                    split_seed=split_seed,
                    level_seed=level_seed,
                    environment_seed=environment_seed,
                    action_seed=action_seed,
                    collector=args.collector,
                    num_actions=num_actions,
                    chunk_size=args.chunk_size,
                )
            )
            for record in records:
                writer.write(record)

            counts = np.bincount(episode.actions, minlength=num_actions)
            action_histogram += counts
            positive_rewards = int(np.count_nonzero(episode.rewards > 0))
            episode_reward = float(np.sum(episode.rewards, dtype=np.float64))
            total_frames += episode_length
            total_positive_rewards += positive_rewards
            total_reward += episode_reward
            episodes.append(
                {
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "attempt": attempts,
                    "length": episode_length,
                    "num_records": len(records),
                    "short_termination_retained": (
                        episode_length < args.min_episode_length
                        and keep_short_termination
                    ),
                    "terminated": episode.terminated,
                    "truncated": not episode.terminated,
                    "dataset_seed": args.seed,
                    "split_seed": split_seed,
                    "level_seed": level_seed,
                    "environment_seed": environment_seed,
                    "action_seed": action_seed,
                    "collector": args.collector,
                    "collector_identity": _collector_identity(args.collector),
                    "total_reward": episode_reward,
                    "positive_reward_frames": positive_rewards,
                    "action_histogram": counts.astype(int).tolist(),
                }
            )
            print(
                f"[{split}] episode {episode_index + 1}/{num_episodes}: "
                f"{episode_length} frames, {len(records)} records"
            )

        num_records = writer.total_records
        num_shards = writer.num_shards

    if len(episodes) != num_episodes:
        raise RuntimeError(
            f"[{split}] collected {len(episodes)}/{num_episodes} episodes after "
            f"the bounded limit of {max_attempts} attempts. Lower "
            "min_episode_length, raise max_attempts_per_split, or use the "
            "scripted collector."
        )

    histogram = (
        action_histogram.astype(int).tolist()
        if action_histogram is not None
        else []
    )
    return {
        "split": split,
        "split_seed": split_seed,
        "level_seed_range": [level_seed_start, level_seed_end],
        "requested_episodes": num_episodes,
        "num_episodes": len(episodes),
        "attempts": attempts,
        "max_attempts": max_attempts,
        "num_records": num_records,
        "num_shards": num_shards,
        "num_frames": total_frames,
        "average_episode_length": (
            total_frames / len(episodes) if episodes else None
        ),
        "total_reward": total_reward,
        "positive_reward_frames": total_positive_rewards,
        "positive_reward_prevalence": (
            total_positive_rewards / total_frames if total_frames else 0.0
        ),
        "action_histogram": histogram,
        "episodes": episodes,
    }


def generate_dataset(
    args: Args,
    *,
    env_factory: Callable[[int, int], Any] = _default_env_factory,
    writer_factory: Callable[[Path, int], Any] = PickleShardWriter,
) -> dict[str, Any]:
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{metadata_path} already exists; pass --overwrite to replace it."
        )

    requested = {
        "train": args.num_episodes_train,
        "val": args.num_episodes_val,
        "test": args.num_episodes_test,
    }
    splits = {
        split: generate_split(
            args,
            split=split,
            num_episodes=num_episodes,
            env_factory=env_factory,
            writer_factory=writer_factory,
        )
        for split, num_episodes in requested.items()
    }

    action_sizes = {
        len(summary["action_histogram"])
        for summary in splits.values()
        if summary["action_histogram"]
    }
    if len(action_sizes) > 1:
        raise ValueError("Procgen action-space size differs between splits.")
    num_actions = next(iter(action_sizes), 0)

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "serialization_format": "pickle",
        "env": "coinrun",
        "dataset_seed": args.seed,
        "collector": args.collector,
        "collector_identity": _collector_identity(args.collector),
        "num_actions": num_actions,
        "categorical_noop": COINRUN_NOOP_ACTION,
        "behavioral_noop_actions": list(COINRUN_BEHAVIORAL_NOOP_ACTIONS),
        "behaviorally_distinct_move_actions": list(
            COINRUN_DISTINCT_MOVE_ACTIONS
        ),
        "level_seed_ranges": {
            split: list(_level_seed_range(split)) for split in SPLIT_IDS
        },
        "action_alignment": "action_applied_after_frame",
        "reward_alignment": "reward_resulting_from_action",
        "args": asdict(args),
        "splits": splits,
    }
    for split, summary in splits.items():
        metadata[f"num_episodes_{split}"] = summary["num_episodes"]
        metadata[f"avg_episode_len_{split}"] = summary[
            "average_episode_length"
        ]
        metadata[f"episode_metadata_{split}"] = summary["episodes"]

    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main(args: Args | None = None) -> None:
    if args is None:
        import tyro

        args = tyro.cli(Args)
    metadata = generate_dataset(args)
    total_records = sum(
        split["num_records"] for split in metadata["splits"].values()
    )
    total_frames = sum(
        split["num_frames"] for split in metadata["splits"].values()
    )
    print(
        f"Done generating CoinRun dataset: {total_records} records, "
        f"{total_frames} frames."
    )


if __name__ == "__main__":
    main()
