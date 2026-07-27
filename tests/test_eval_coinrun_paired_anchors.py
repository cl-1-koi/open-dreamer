"""Focused tests for the exact paired CoinRun pixel evaluator."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import eval_coinrun_paired_anchors as paired  # noqa: E402


class TestSlidingWindowRollout(unittest.TestCase):
    def test_window_never_exceeds_context_and_predictions_feed_back(self):
        seen: list[tuple[np.ndarray, np.ndarray]] = []

        def fake_step(values, actions, next_action, _step, _seed):
            seen.append((values.copy(), actions.copy()))
            return values[-1] + next_action

        rollout = paired.sliding_window_rollout(
            np.arange(4, dtype=np.float32)[:, None],
            np.asarray([10, 11, 12, 13], dtype=np.int8),
            np.asarray([1, 2, 3], dtype=np.int8),
            max_context=4,
            benchmark_seed=17,
            anchor_id="anchor-feedback",
            step_fn=fake_step,
        )

        self.assertTrue(all(values.shape[0] <= 4 for values, _ in seen))
        self.assertTrue(all(actions.shape[0] <= 4 for _, actions in seen))
        np.testing.assert_array_equal(rollout[:, 0], [4.0, 6.0, 9.0])
        self.assertEqual(float(seen[1][0][-1, 0]), 4.0)
        self.assertEqual(float(seen[2][0][-1, 0]), 6.0)
        self.assertEqual(int(seen[1][1][-1]), 1)
        self.assertEqual(int(seen[2][1][-1]), 2)

    def test_h1_h8_are_prefixes_of_one_h32_rollout(self):
        rollout = np.arange(32 * 3, dtype=np.float32).reshape(32, 3)
        prefixes = paired.rollout_prefixes(rollout)

        np.testing.assert_array_equal(prefixes[1], rollout[:1])
        np.testing.assert_array_equal(prefixes[8], rollout[:8])
        np.testing.assert_array_equal(prefixes[32], rollout)
        rollout[0, 0] = -1
        self.assertNotEqual(float(prefixes[1][0, 0]), -1.0)

    def test_per_anchor_rng_is_deterministic(self):
        def stochastic_step(values, _actions, _next_action, _step, seed):
            rng = np.random.default_rng(seed)
            return values[-1] + rng.normal(size=values.shape[1:])

        kwargs = {
            "initial_values": np.zeros((4, 2), dtype=np.float32),
            "context_actions": np.zeros(4, dtype=np.int8),
            "future_actions": np.ones(8, dtype=np.int8),
            "max_context": 4,
            "benchmark_seed": 29,
            "step_fn": stochastic_step,
        }
        first = paired.sliding_window_rollout(
            anchor_id="deterministic-a",
            **kwargs,
        )
        second = paired.sliding_window_rollout(
            anchor_id="deterministic-a",
            **kwargs,
        )
        other_anchor = paired.sliding_window_rollout(
            anchor_id="deterministic-b",
            **kwargs,
        )

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other_anchor))

    def test_future_truth_cannot_leak_into_rollout(self):
        def fake_step(values, _actions, next_action, _step, _seed):
            return values[-1] * np.float32(1.5) + next_action

        sequence_a = np.arange(12, dtype=np.float32)[:, None]
        sequence_b = sequence_a.copy()
        sequence_b[4:] = 10000.0
        kwargs = {
            "context_actions": np.zeros(4, dtype=np.int8),
            "future_actions": np.asarray([1, 2, 3, 4], dtype=np.int8),
            "max_context": 4,
            "benchmark_seed": 3,
            "anchor_id": "no-truth-input",
            "step_fn": fake_step,
        }

        first = paired.sliding_window_rollout(
            sequence_a[:4],
            **kwargs,
        )
        second = paired.sliding_window_rollout(
            sequence_b[:4],
            **kwargs,
        )

        np.testing.assert_array_equal(first, second)


class TestArgumentParsing(unittest.TestCase):
    def test_anchor_shard_arguments(self):
        args = paired._parse_args(
            [
                "--anchor-manifest",
                "/tmp/anchors.json",
                "--array-record-root",
                "/tmp/records",
                "--dynamics-ckpt",
                "/tmp/checkpoints",
                "--anchor-shard-index",
                "1",
                "--anchor-shard-count",
                "2",
            ]
        )

        self.assertEqual(args.anchor_shard_index, 1)
        self.assertEqual(args.anchor_shard_count, 2)


def _anchor_and_episode() -> tuple[paired.Anchor, dict, bytes]:
    length = 70
    trajectory_source_row_start = 500
    context_start = 2
    prediction_start = context_start + paired.CONTEXT
    action_start = prediction_start - 1
    sequence_stop = prediction_start + paired.MAX_HORIZON
    action_stop = action_start + paired.MAX_HORIZON

    frames = np.arange(length * 2 * 2 * 3, dtype=np.uint8).reshape(
        length, 2, 2, 3
    )
    actions = (np.arange(length) % paired.NUM_ACTIONS).astype(np.int8)
    rewards = np.zeros(length, dtype=np.float32)
    first = np.zeros(length, dtype=np.bool_)
    first[0] = True
    done = np.zeros(length, dtype=np.bool_)
    done[-1] = True
    terminal_cause = np.zeros(length, dtype=np.uint8)
    terminal_cause[-1] = 3
    timestep = np.arange(length, dtype=np.int32)
    next_state_valid = ~done

    frame_hash = paired.sha256_array(frames)
    action_hash = paired.sha256_array(actions)
    reward_hash = paired.sha256_array(rewards)
    record = {
        "raw_video": frames.tobytes(order="C"),
        "frame_shape": np.asarray([2, 2, 3], dtype=np.int32),
        "sequence_length": length,
        "actions": actions,
        "rewards": rewards,
        "episode_starts": first,
        "episode_ends": done,
        "terminal_cause": terminal_cause,
        "timestep": timestep,
        "next_state_valid": next_state_valid,
        "split": "val",
        "source": "random",
        "collector_identity": "random",
        "episode_id": "val:random:level-10000:episode-0",
        "episode_index": 0,
        "trajectory_index": 7,
        "level_id": 10000,
        "source_row_start": trajectory_source_row_start,
        "source_row_end": trajectory_source_row_start + length,
        "num_actions": paired.NUM_ACTIONS,
        "categorical_noop": paired.NOOP_ACTION,
        "action_alignment": "action_applied_after_frame",
        "reward_alignment": "reward_resulting_from_action",
        "paired_frame_sha256": frame_hash,
        "action_sha256": action_hash,
        "reward_sha256": reward_hash,
    }
    payload = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
    future_actions = np.ascontiguousarray(actions[action_start:action_stop])
    target = np.ascontiguousarray(frames[prediction_start:sequence_stop])
    sequence = np.ascontiguousarray(frames[context_start:sequence_stop])
    anchor_value = {
        "id": "a" * 64,
        "source": "random",
        "corpus_source": "random",
        "trajectory_index": 7,
        "level_id": 10000,
        "episode_index": 0,
        "episode_id": "val:random:level-10000:episode-0",
        "trajectory_source_row_start": trajectory_source_row_start,
        "source_row_start": trajectory_source_row_start + context_start,
        "context_start": context_start,
        "context_length": paired.CONTEXT,
        "prediction_start": prediction_start,
        "prediction_source_row_start": (
            trajectory_source_row_start + prediction_start
        ),
        "action_start": action_start,
        "action_source_row_start": trajectory_source_row_start + action_start,
        "max_horizon": paired.MAX_HORIZON,
        "future_actions": future_actions.astype(int).tolist(),
        "future_actions_sha256": paired.sha256_array(future_actions),
        "future_action_prefix_sha256": {
            str(horizon): paired.sha256_array(future_actions[:horizon])
            for horizon in paired.EVALUATION_HORIZONS
        },
        "terminal_reset_valid_mask": [True] * paired.MAX_HORIZON,
        "rgb_sequence_sha256": paired.sha256_array(sequence),
        "rgb_target_sha256": paired.sha256_array(target),
        "rgb_target_prefix_sha256": {
            str(horizon): paired.sha256_array(target[:horizon])
            for horizon in paired.EVALUATION_HORIZONS
        },
        "open_dreamer_episode_record_sha256": paired._sha256_bytes(payload),
        "open_dreamer_episode_frame_sha256": frame_hash,
        "open_dreamer_episode_action_sha256": action_hash,
        "open_dreamer_episode_reward_sha256": reward_hash,
    }
    anchor = paired.parse_anchor(anchor_value, index=0)
    return anchor, record, payload


class TestAnchorVerification(unittest.TestCase):
    def test_exact_anchor_is_accepted(self):
        anchor, record, payload = _anchor_and_episode()

        window = paired.verify_anchor_episode(anchor, record, payload=payload)

        self.assertEqual(window.context_rgb.shape, (32, 2, 2, 3))
        self.assertEqual(window.target_rgb.shape, (32, 2, 2, 3))
        np.testing.assert_array_equal(window.future_actions, anchor.future_actions)
        self.assertTrue(window.next_state_valid.all())

    def test_anchor_action_mismatch_is_rejected(self):
        anchor, record, payload = _anchor_and_episode()
        mismatched = dict(record)
        mismatched_actions = record["actions"].copy()
        mismatched_actions[anchor.action_start] = (
            int(mismatched_actions[anchor.action_start]) + 1
        ) % paired.NUM_ACTIONS
        mismatched["actions"] = mismatched_actions
        mismatched["action_sha256"] = paired.sha256_array(mismatched_actions)
        mismatched_payload = pickle.dumps(
            mismatched,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        with self.assertRaisesRegex(
            paired.PairedAnchorError,
            "SHA256 mismatch",
        ):
            paired.verify_anchor_episode(
                anchor,
                mismatched,
                payload=mismatched_payload,
            )

        # The original still verifies, proving the rejection is the mismatch.
        paired.verify_anchor_episode(anchor, record, payload=payload)


if __name__ == "__main__":
    unittest.main()
