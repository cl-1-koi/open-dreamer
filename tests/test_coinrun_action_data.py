from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import grain
import jax.numpy as jnp
import numpy as np

from dreamer.actions import Actions, shift_actions
from dreamer.data.generate_coinrun_dataset import (
    COINRUN_BEHAVIORAL_NOOP_ACTIONS,
    COINRUN_DISTINCT_MOVE_ACTIONS,
    COINRUN_NUM_ACTIONS,
    Args,
    CollectedEpisode,
    PickleShardWriter,
    collect_episode,
    episode_records,
    generate_dataset,
    generate_split,
)
from dreamer.data.transforms import EpisodeLengthFilter, ProcessEpisodeAndSlice


def _sample_episode() -> CollectedEpisode:
    frames = np.arange(5 * 2 * 3, dtype=np.uint8).reshape(5, 2, 3, 1)
    return CollectedEpisode(
        frames=frames,
        actions=np.asarray([4, 7, 8, 7, 4], dtype=np.int32),
        rewards=np.asarray([0.0, 0.0, 1.0, 0.0, -1.0], dtype=np.float32),
        terminated=True,
    )


def _sample_records() -> list[dict]:
    return list(
        episode_records(
            _sample_episode(),
            split="train",
            episode_id="train-000000",
            dataset_seed=11,
            split_seed=12,
            level_seed=13,
            environment_seed=14,
            action_seed=15,
            collector="scripted",
            num_actions=15,
            chunk_size=2,
        )
    )


def _read_pickled_records(paths: list[Path]) -> list[tuple[bytes, dict]]:
    records = []
    for path in paths:
        source = grain.sources.ArrayRecordDataSource([str(path)])
        records.extend(
            (source[index], pickle.loads(source[index]))
            for index in range(len(source))
        )
    return records


class _FakeCoinRunEnv:
    def __init__(
        self,
        level_seed: int,
        environment_seed: int,
        length: int = 5,
        *,
        terminates: bool = True,
    ):
        self.ac_space = SimpleNamespace(eltype=SimpleNamespace(n=15))
        self.level_seed = level_seed
        self.environment_seed = environment_seed
        self.length = length
        self.terminates = terminates
        self.step = 0
        self.last_action = 4
        self.closed = False

    def _observation(self) -> dict[str, np.ndarray]:
        value = (
            255
            if self.step >= self.length
            else (self.level_seed + self.environment_seed + self.step) % 255
        )
        return {"rgb": np.full((1, 2, 3, 1), value, dtype=np.uint8)}

    def observe(self):
        reward = np.asarray(
            [1.0 if self.last_action == 8 else 0.0],
            dtype=np.float32,
        )
        first = np.asarray(
            [self.terminates and self.step >= self.length],
            dtype=np.bool_,
        )
        return reward, self._observation(), first

    def act(self, action: np.ndarray) -> None:
        self.last_action = int(action[0])
        self.step += 1

    def close(self) -> None:
        self.closed = True


def _fake_env_factory(level_seed: int, environment_seed: int) -> _FakeCoinRunEnv:
    return _FakeCoinRunEnv(level_seed, environment_seed)


class CoinRunActionDataTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.tmp_path = Path(temporary_directory.name)

    def test_categorical_only_actions_shift_with_explicit_coinrun_noop(self):
        actions = Actions.from_dict(
            {
                "categorical": jnp.asarray(
                    [[1, 2, 8], [7, 4, 3]],
                    dtype=jnp.int32,
                )
            }
        )

        shifted = shift_actions(
            actions,
            categorical_action_dim=15,
            categorical_noop=4,
        )

        np.testing.assert_array_equal(
            shifted.categorical,
            np.asarray([[4, 1, 2], [4, 7, 4]], dtype=np.int32),
        )
        self.assertIsNone(shifted.binary)
        self.assertIsNone(shifted.continuous)

    def test_action_shift_keeps_legacy_vpt_camera_center(self):
        actions = Actions(
            binary=jnp.ones((1, 3, 2), dtype=jnp.int32),
            categorical=jnp.asarray([[12, 13, 14]], dtype=jnp.int32),
            continuous=jnp.ones((1, 3, 2), dtype=jnp.float32),
        )

        shifted = shift_actions(
            actions,
            categorical_action_dim=121,
            categorical_noop=60,
        )

        np.testing.assert_array_equal(shifted.categorical, [[60, 12, 13]])
        np.testing.assert_array_equal(shifted.binary[:, 0], [[0, 0]])
        np.testing.assert_array_equal(shifted.continuous[:, 0], [[0.0, 0.0]])

    def test_pickle_arrayrecord_roundtrip_keeps_trailing_chunk_and_counters(
        self,
    ):
        records = _sample_records()
        self.assertEqual(
            [record["sequence_length"] for record in records],
            [2, 2, 1],
        )

        writer = PickleShardWriter(self.tmp_path, records_per_shard=2)
        with writer:
            for record in records:
                writer.write(record)
            self.assertEqual(writer.total_records, 3)
            self.assertEqual(writer.num_shards, 2)

        shard_paths = sorted(self.tmp_path.glob("shard-*.array_record"))
        self.assertEqual(len(shard_paths), 2)
        decoded = _read_pickled_records(shard_paths)
        self.assertEqual(len(decoded), 3)

        payloads, roundtripped = zip(*decoded, strict=True)
        roundtripped = sorted(
            roundtripped,
            key=lambda record: record["chunk_index"],
        )
        np.testing.assert_array_equal(
            np.concatenate([record["actions"] for record in roundtripped]),
            _sample_episode().actions,
        )
        np.testing.assert_array_equal(
            np.concatenate([record["rewards"] for record in roundtripped]),
            _sample_episode().rewards,
        )
        self.assertEqual(
            sum(record["sequence_length"] for record in roundtripped),
            5,
        )
        self.assertEqual(
            roundtripped[0]["episode_starts"].tolist(),
            [True, False],
        )
        self.assertEqual(roundtripped[-1]["episode_ends"].tolist(), [True])
        self.assertEqual(roundtripped[-1]["level_seed"], 13)
        self.assertEqual(roundtripped[-1]["serialization_format"], "pickle")
        self.assertEqual(
            roundtripped[-1]["reward_alignment"],
            "reward_resulting_from_action",
        )
        self.assertEqual(
            roundtripped[-1]["collector_identity"],
            "coinrun_run_right_seeded_jump_v1",
        )

        episode_filter = EpisodeLengthFilter(
            seq_len=2,
            format_hint="coinrun",
            print_filter_warnings=False,
        )
        self.assertTrue(episode_filter.filter(payloads[0]))
        processed = ProcessEpisodeAndSlice(
            seq_len=2,
            image_h=2,
            image_w=3,
            image_c=1,
        ).random_map(payloads[0], np.random.default_rng(0))
        np.testing.assert_array_equal(
            processed["videos"],
            _sample_episode().frames[:2],
        )
        np.testing.assert_array_equal(
            processed["actions"].categorical,
            _sample_episode().actions[:2],
        )
        np.testing.assert_array_equal(
            processed["rewards"],
            _sample_episode().rewards[:2],
        )

    def test_collection_excludes_reset_frame_and_aligns_resulting_reward(self):
        env = _FakeCoinRunEnv(10, 20, length=3)
        episode = collect_episode(
            env,
            collector="scripted",
            action_rng=np.random.default_rng(0),
            max_episode_length=5,
        )

        self.assertTrue(episode.terminated)
        self.assertEqual(episode.frames.shape[0], 3)
        self.assertFalse(np.any(episode.frames == 255))
        np.testing.assert_array_equal(
            episode.rewards,
            (episode.actions == 8).astype(np.float32),
        )

    def test_generation_is_deterministic_and_persists_episode_metadata(self):
        common = {
            "num_episodes_train": 1,
            "num_episodes_val": 0,
            "num_episodes_test": 0,
            "min_episode_length": 2,
            "max_episode_length": 5,
            "chunk_size": 3,
            "chunks_per_file": 1,
            "seed": 123,
            "collector": "scripted",
            "max_attempts_per_split": 2,
        }
        first = generate_dataset(
            Args(output_dir=str(self.tmp_path / "first"), **common),
            env_factory=_fake_env_factory,
        )
        second = generate_dataset(
            Args(output_dir=str(self.tmp_path / "second"), **common),
            env_factory=_fake_env_factory,
        )

        self.assertEqual(first["splits"], second["splits"])
        train = first["splits"]["train"]
        self.assertEqual(train["num_records"], 2)
        self.assertEqual(train["num_shards"], 2)
        self.assertEqual(train["num_frames"], 5)
        self.assertEqual(sum(train["action_histogram"]), 5)
        episode = train["episodes"][0]
        self.assertEqual(episode["dataset_seed"], 123)
        self.assertEqual(episode["collector"], "scripted")
        self.assertEqual(
            episode["collector_identity"],
            "coinrun_run_right_seeded_jump_v1",
        )
        self.assertEqual(episode["length"], 5)
        self.assertEqual(first["categorical_noop"], 4)
        self.assertEqual(first["num_actions"], COINRUN_NUM_ACTIONS)
        self.assertEqual(
            first["behavioral_noop_actions"],
            list(COINRUN_BEHAVIORAL_NOOP_ACTIONS),
        )
        self.assertEqual(
            first["behaviorally_distinct_move_actions"],
            list(COINRUN_DISTINCT_MOVE_ACTIONS),
        )

        level_ranges = first["level_seed_ranges"]
        self.assertLessEqual(
            level_ranges["train"][1],
            level_ranges["val"][0],
        )
        self.assertLessEqual(
            level_ranges["val"][1],
            level_ranges["test"][0],
        )
        for split, summary in first["splits"].items():
            with self.subTest(split=split):
                start, end = level_ranges[split]
                self.assertEqual(summary["level_seed_range"], [start, end])
                self.assertTrue(
                    all(
                        start <= item["level_seed"] < end
                        for item in summary["episodes"]
                    )
                )

        first_records = _read_pickled_records(
            sorted(
                (self.tmp_path / "first" / "train").glob(
                    "shard-*.array_record"
                )
            )
        )
        second_records = _read_pickled_records(
            sorted(
                (self.tmp_path / "second" / "train").glob(
                    "shard-*.array_record"
                )
            )
        )
        self.assertEqual(
            [payload for payload, _ in first_records],
            [payload for payload, _ in second_records],
        )

    def test_generation_stops_at_attempt_limit(self):
        attempts = []

        def short_env_factory(
            level_seed: int,
            environment_seed: int,
        ) -> _FakeCoinRunEnv:
            attempts.append((level_seed, environment_seed))
            return _FakeCoinRunEnv(level_seed, environment_seed, length=1)

        args = Args(
            num_episodes_train=1,
            num_episodes_val=0,
            num_episodes_test=0,
            output_dir=str(self.tmp_path),
            min_episode_length=2,
            max_episode_length=2,
            max_attempts_per_split=3,
            keep_short_terminated=False,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "bounded limit of 3 attempts",
        ):
            generate_split(
                args,
                split="train",
                num_episodes=1,
                env_factory=short_env_factory,
            )

        self.assertEqual(len(attempts), 3)

    def test_short_natural_termination_is_retained(self):
        args = Args(
            num_episodes_train=1,
            num_episodes_val=0,
            num_episodes_test=0,
            output_dir=str(self.tmp_path),
            min_episode_length=64,
            max_episode_length=64,
            max_attempts_per_split=1,
        )

        summary = generate_split(
            args,
            split="train",
            num_episodes=1,
            env_factory=lambda level_seed, environment_seed: _FakeCoinRunEnv(
                level_seed,
                environment_seed,
                length=1,
            ),
        )

        self.assertEqual(summary["num_episodes"], 1)
        self.assertTrue(summary["episodes"][0]["terminated"])
        self.assertTrue(summary["episodes"][0]["short_termination_retained"])

    def test_default_minimum_accepts_config_length_windows(self):
        self.assertEqual(Args().min_episode_length, 64)


if __name__ == "__main__":
    unittest.main()
