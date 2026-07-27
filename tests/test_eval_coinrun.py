"""Tests for scripts/eval_coinrun.py.

Pure-helper tests run anywhere. The end-to-end test builds a tiny
tokenizer + dynamics pair, saves a real Orbax checkpoint bundle, generates a
synthetic pickle-serialized CoinRun dataset, and runs the full evaluator.

Run with:
    uv run python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import eval_coinrun as ec  # noqa: E402


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestRollPermutation(unittest.TestCase):
    def test_every_window_gets_foreign_actions(self):
        for b in (2, 3, 8):
            perm = ec.roll_permutation(b)
            self.assertEqual(sorted(perm.tolist()), list(range(b)))
            for i in range(b):
                self.assertNotEqual(i, int(perm[i]))

    def test_roll_by_one(self):
        self.assertEqual(ec.roll_permutation(4).tolist(), [1, 2, 3, 0])

    def test_batch_size_one_rejected(self):
        with self.assertRaises(ValueError):
            ec.roll_permutation(1)


class TestEvenlySpacedPositions(unittest.TestCase):
    def test_includes_first_and_last(self):
        pos = ec.evenly_spaced_positions(horizon=8, max_positions=4)
        self.assertEqual(pos[0], 0)
        self.assertEqual(pos[-1], 7)
        self.assertLessEqual(len(pos), 4)
        self.assertEqual(pos, sorted(set(pos)))

    def test_more_positions_than_horizon(self):
        self.assertEqual(ec.evenly_spaced_positions(3, 10), [0, 1, 2])

    def test_horizon_one(self):
        self.assertEqual(ec.evenly_spaced_positions(1, 4), [0])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            ec.evenly_spaced_positions(0, 4)
        with self.assertRaises(ValueError):
            ec.evenly_spaced_positions(4, 0)


class TestMetrics(unittest.TestCase):
    def test_psnr_from_mse(self):
        self.assertEqual(ec.psnr_from_mse(0.0), ec.PSNR_CAP_DB)
        self.assertAlmostEqual(ec.psnr_from_mse(255.0**2), 0.0, places=5)

    def test_frame_metrics_identical(self):
        a = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
        m = ec.frame_metrics(a, a.copy())
        self.assertEqual(m["pixel_mse"], 0.0)
        self.assertEqual(m["pixel_mae"], 0.0)
        self.assertEqual(m["psnr"], ec.PSNR_CAP_DB)

    def test_frame_metrics_known_offset(self):
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.full((4, 4, 3), 10, dtype=np.uint8)
        m = ec.frame_metrics(a, b)
        self.assertAlmostEqual(m["pixel_mse"], 100.0)
        self.assertAlmostEqual(m["pixel_mae"], 10.0)

    def test_frame_metrics_shape_mismatch(self):
        with self.assertRaises(ValueError):
            ec.frame_metrics(np.zeros((4, 4, 3)), np.zeros((4, 4, 2)))

    def test_latent_metrics(self):
        a = np.zeros((2, 3, 4))
        b = np.ones((2, 3, 4))
        self.assertAlmostEqual(ec.latent_metrics(a, b)["latent_mse"], 1.0)


class TestPairedEffectSize(unittest.TestCase):
    def test_known_values(self):
        deltas = np.array([1.0, 1.0, 1.0, 1.0])
        eff = ec.paired_effect_size(deltas)
        self.assertEqual(eff["mean_delta"], 1.0)
        self.assertEqual(eff["std_delta"], 0.0)
        self.assertIsNone(eff["effect_size_dz"])  # undefined at zero variance
        self.assertEqual(eff["frac_positive"], 1.0)
        self.assertEqual(eff["n"], 4)

    def test_mean_over_std(self):
        deltas = np.array([1.0, 3.0])
        eff = ec.paired_effect_size(deltas)
        self.assertAlmostEqual(eff["mean_delta"], 2.0)
        self.assertAlmostEqual(eff["std_delta"], np.sqrt(2.0))
        self.assertAlmostEqual(eff["effect_size_dz"], 2.0 / np.sqrt(2.0))
        self.assertAlmostEqual(eff["frac_positive"], 1.0)

    def test_negative_fraction(self):
        eff = ec.paired_effect_size(np.array([1.0, -1.0, 2.0]))
        self.assertAlmostEqual(eff["frac_positive"], 2.0 / 3.0)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            ec.paired_effect_size(np.array([]))


class TestCoinRunActionSemantics(unittest.TestCase):
    """Review G3: no-op is index 4; permutations stay within distinct 0-8."""

    def test_shift_prepends_noop_index_4(self):
        from dreamer.actions import Actions

        actions = Actions(categorical=np.array([[7, 1, 5], [3, 3, 0]], dtype=np.int32))
        shifted = ec.shift_actions_coinrun(actions)
        got = np.asarray(shifted.categorical)
        self.assertEqual(got.shape, (2, 3))
        # First action is the CoinRun no-op (4), the rest shift right by one.
        np.testing.assert_array_equal(got[:, 0], [4, 4])
        np.testing.assert_array_equal(got[0], [4, 7, 1])
        np.testing.assert_array_equal(got[1], [4, 3, 3])

    def test_shift_handles_none_leaves(self):
        from dreamer.actions import Actions

        actions = Actions(binary=None, categorical=np.array([[0]], dtype=np.int32), continuous=None)
        shifted = ec.shift_actions_coinrun(actions)
        self.assertIsNone(shifted.binary)
        self.assertIsNone(shifted.continuous)
        np.testing.assert_array_equal(np.asarray(shifted.categorical), [[4]])

    def test_action_index_permutation_is_derangement(self):
        perm = ec.ACTION_INDEX_PERMUTATION
        self.assertEqual(sorted(perm), list(range(9)))
        for x, y in enumerate(perm):
            self.assertNotEqual(x, y, "permutation must not fix an action")

    def test_permute_action_indices_maps_only_distinct(self):
        from dreamer.actions import Actions

        cat = np.array([[0, 4, 8, 9, 14]], dtype=np.int32)
        out = ec.permute_action_indices(Actions(categorical=cat))
        got = np.asarray(out.categorical)
        expected = [ec.ACTION_INDEX_PERMUTATION[0], ec.ACTION_INDEX_PERMUTATION[4],
                    ec.ACTION_INDEX_PERMUTATION[8], 9, 14]
        np.testing.assert_array_equal(got[0], expected)

    def test_permute_action_indices_preserves_distribution_support(self):
        from dreamer.actions import Actions

        rng = np.random.default_rng(0)
        cat = rng.integers(0, 9, (4, 12), dtype=np.int32)
        out = np.asarray(ec.permute_action_indices(Actions(categorical=cat)).categorical)
        # Support is unchanged: the permutation is a bijection on 0-8.
        self.assertEqual(
            sorted(np.unique(cat).tolist()), sorted(np.unique(out).tolist())
        )


class TestBuildEvalVideo(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.T, self.H, self.W, self.ctx = 6, 32, 32, 2
        self.gt = rng.integers(0, 256, (self.T, self.H, self.W, 3), dtype=np.uint8)
        self.pred = rng.integers(0, 256, (self.T, self.H, self.W, 3), dtype=np.uint8)

    def test_layout_and_seam(self):
        video = ec.build_eval_video(self.gt, self.pred, self.ctx)
        gap = ec.SEAM_GAP_ROWS
        self.assertEqual(video.shape, (self.T, 2 * self.H + gap, self.W, 3))
        # Rows preserved.
        np.testing.assert_array_equal(video[:, : self.H], self.gt)
        np.testing.assert_array_equal(video[:, self.H + gap :], self.pred)
        # Seam bar declares the context/horizon split.
        for t in range(self.T):
            seam = video[t, self.H + 4, 0]
            expected = ec.SEAM_GREEN if t < self.ctx else ec.SEAM_RED
            self.assertEqual(tuple(seam), expected)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            ec.build_eval_video(self.gt, self.pred[:, :4], self.ctx)

    def test_context_bounds(self):
        with self.assertRaises(ValueError):
            ec.build_eval_video(self.gt, self.pred, 0)
        with self.assertRaises(ValueError):
            ec.build_eval_video(self.gt, self.pred, self.T)


class TestClassifySeam(unittest.TestCase):
    def test_colors(self):
        self.assertEqual(ec.classify_seam(np.array(ec.SEAM_GREEN)), "green")
        self.assertEqual(ec.classify_seam(np.array(ec.SEAM_RED)), "red")
        self.assertEqual(ec.classify_seam(np.array((120, 120, 120))), "unknown")


class TestVerifyMp4(unittest.TestCase):
    def _write_video(self, path, video, fps=5):
        import imageio.v3 as iio

        iio.imwrite(str(path), video, fps=fps, codec="libx264")

    def _random_frames(self, seed, shape):
        return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)

    def test_valid_video_passes(self):
        T, H, W, ctx = 5, 32, 32, 3
        gt = self._random_frames(1, (T, H, W, 3))
        pred = self._random_frames(2, (T, H, W, 3))
        video = ec.build_eval_video(gt, pred, ctx)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.mp4"
            self._write_video(path, video)
            stats = ec.verify_mp4(path, expected_frames=T, context=ctx)
        self.assertEqual(stats["frames_readback"], T)
        self.assertTrue(stats["seam_marker_ok"])
        self.assertTrue(stats["nontrivial"])
        self.assertGreater(stats["frame_std"], 1.0)
        self.assertGreater(stats["horizon_region_std"], ec.CONSTANT_STD_THRESHOLD)
        self.assertGreater(stats["copy_last_context_mad_max"], ec.COPY_MAD_THRESHOLD)
        self.assertGreater(stats["frozen_mad_mean"], ec.FROZEN_MAD_THRESHOLD)

    def test_wrong_frame_count_fails(self):
        T, H, W, ctx = 5, 32, 32, 3
        gt = self._random_frames(2, (T, H, W, 3))
        video = ec.build_eval_video(gt, gt, ctx)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.mp4"
            self._write_video(path, video)
            with self.assertRaises(AssertionError):
                ec.verify_mp4(path, expected_frames=T + 1, context=ctx)

    def test_blank_video_fails(self):
        T, H, W, ctx = 5, 32, 32, 3
        gt = np.full((T, H, W, 3), 127, dtype=np.uint8)
        video = ec.build_eval_video(gt, gt, ctx)
        # Zero the seam bar as well so the whole frame is constant.
        video[:, H:H + ec.SEAM_GAP_ROWS] = 127
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.mp4"
            self._write_video(path, video)
            with self.assertRaises(AssertionError):
                ec.verify_mp4(path, expected_frames=T, context=ctx)

    def test_constant_generated_region_rejected(self):
        """Review G8: a frozen/blank rollout must be rejected even when the
        context and GT rows are normal."""
        T, H, W, ctx = 5, 32, 32, 3
        gt = self._random_frames(3, (T, H, W, 3))
        pred = self._random_frames(4, (T, H, W, 3))
        pred[ctx:] = 100  # constant horizon
        video = ec.build_eval_video(gt, pred, ctx)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.mp4"
            self._write_video(path, video)
            with self.assertRaisesRegex(AssertionError, "constant"):
                ec.verify_mp4(path, expected_frames=T, context=ctx)

    def test_copy_last_context_rollout_rejected(self):
        """Review G8: repeating the last context frame must be rejected."""
        T, H, W, ctx = 5, 32, 32, 3
        gt = self._random_frames(5, (T, H, W, 3))
        pred = self._random_frames(6, (T, H, W, 3))
        pred[ctx:] = pred[ctx - 1]  # copy last context frame across the horizon
        video = ec.build_eval_video(gt, pred, ctx)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.mp4"
            self._write_video(path, video)
            with self.assertRaisesRegex(AssertionError, "copies the last context frame"):
                ec.verify_mp4(path, expected_frames=T, context=ctx)


class TestResolveCheckpointDir(unittest.TestCase):
    def test_empty_path(self):
        with self.assertRaises(FileNotFoundError):
            ec.resolve_checkpoint_dir("", label="dynamics")

    def test_missing_path(self):
        with self.assertRaises(FileNotFoundError):
            ec.resolve_checkpoint_dir("/nonexistent/ckpt", label="dynamics")

    def test_plain_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(ec.resolve_checkpoint_dir(tmp, label="dynamics"), tmp)

    def test_checkpoints_suffix_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub = Path(tmp) / "checkpoints"
            sub.mkdir()
            self.assertEqual(
                ec.resolve_checkpoint_dir(tmp, label="dynamics"), str(sub)
            )


class TestValidateArgs(unittest.TestCase):
    def _args(self, **kw):
        base = dict(dynamics_ckpt="x")
        base.update(kw)
        return ec.Args(**base)

    def test_defaults_ok(self):
        ec.validate_args(self._args())

    def test_batch_size_one_rejected(self):
        with self.assertRaises(ValueError):
            ec.validate_args(self._args(batch_size=1))

    def test_horizon_zero_rejected(self):
        with self.assertRaises(ValueError):
            ec.validate_args(self._args(horizon=0))

    def test_bad_model_name_rejected(self):
        with self.assertRaises(ValueError):
            ec.validate_args(self._args(dynamics_model="ema"))

    def test_bad_reward_probability_rejected(self):
        with self.assertRaises(ValueError):
            ec.validate_args(self._args(p_include_reward=1.5))

    def test_training_split_refused(self):
        """Review G4: evaluation must not draw from the training split."""
        with self.assertRaisesRegex(ValueError, "training split"):
            ec.validate_args(self._args(array_record_path="datasets/coinrun_episodes/train"))

    def test_val_split_accepted(self):
        ec.validate_args(self._args(array_record_path="datasets/coinrun_episodes/val"))


class TestValidateModelConfigs(unittest.TestCase):
    """Fail-closed config gates (review G2 latent stats, G3 action space)."""

    def _cfgs(self, **dyn_overrides):
        tok_cfg, dyn_cfg = _tiny_configs()
        for key, value in dyn_overrides.items():
            setattr(dyn_cfg, key, value)
        dataset_info = {
            "first_record_sequence_length": 24,
            "first_record_raw_video_nbytes": 24 * 32 * 32 * 3,
            "action_max": 14,
        }
        return dyn_cfg, tok_cfg, dataset_info

    def test_valid_configs_pass(self):
        dyn_cfg, tok_cfg, info = self._cfgs()
        ec.validate_model_configs(dyn_cfg, tok_cfg, info)

    def test_action_dim_16_refused(self):
        dyn_cfg, tok_cfg, info = self._cfgs(categorical_action_dim=16)
        with self.assertRaisesRegex(ValueError, "15"):
            ec.validate_model_configs(dyn_cfg, tok_cfg, info)

    def test_missing_latent_stats_refused(self):
        dyn_cfg, tok_cfg, info = self._cfgs(latent_mean=None, latent_std=None)
        with self.assertRaisesRegex(ValueError, "latent"):
            ec.validate_model_configs(dyn_cfg, tok_cfg, info)

    def test_latent_stats_length_mismatch_refused(self):
        dyn_cfg, tok_cfg, info = self._cfgs(latent_mean=(0.0,) * 4)
        with self.assertRaisesRegex(ValueError, "d_bottleneck"):
            ec.validate_model_configs(dyn_cfg, tok_cfg, info)

    def test_frame_byte_mismatch_refused(self):
        dyn_cfg, tok_cfg, info = self._cfgs()
        info["first_record_raw_video_nbytes"] = 12345
        with self.assertRaisesRegex(ValueError, "bytes"):
            ec.validate_model_configs(dyn_cfg, tok_cfg, info)

    def test_dataset_action_out_of_range_refused(self):
        dyn_cfg, tok_cfg, info = self._cfgs()
        info["action_max"] = 15
        with self.assertRaisesRegex(ValueError, "action space"):
            ec.validate_model_configs(dyn_cfg, tok_cfg, info)


# ---------------------------------------------------------------------------
# Dataset preflight + artifact failure modes
# ---------------------------------------------------------------------------

def _write_coinrun_records(
    data_dir: Path,
    num_records: int,
    record_len: int,
    h: int = 32,
    w: int = 32,
    nonzero_rewards: bool = True,
) -> None:
    """Write pickle-serialized CoinRun chunk records (the format the Grain
    CoinRun transforms decode) to .array_record files."""
    from array_record.python.array_record_module import ArrayRecordWriter

    rng = np.random.default_rng(0)
    data_dir.mkdir(parents=True, exist_ok=True)
    writer = ArrayRecordWriter(str(data_dir / "shard-00000.array_record"), "group_size:1")
    for i in range(num_records):
        frames = rng.integers(0, 256, (record_len, h, w, 3), dtype=np.uint8)
        rewards = np.zeros((record_len,), dtype=np.float32)
        if nonzero_rewards:
            rewards[i % record_len] = 10.0
        record = {
            "raw_video": frames.tobytes(),
            "sequence_length": record_len,
            "actions": rng.integers(0, 15, (record_len,), dtype=np.int32),
            "rewards": rewards,
        }
        writer.write(pickle.dumps(record))
    writer.close()


class TestPreflightDataset(unittest.TestCase):
    def test_missing_dir(self):
        with self.assertRaises(FileNotFoundError):
            ec.preflight_dataset("/nonexistent/data", seq_len=5, batch_size=2, max_scan=8)

    def test_valid_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "val"
            _write_coinrun_records(data_dir, num_records=6, record_len=24)
            info = ec.preflight_dataset(str(data_dir), seq_len=5, batch_size=2, max_scan=8)
        self.assertEqual(info["num_records"], 6)
        self.assertEqual(info["usable_records_in_scan"], 6)
        self.assertEqual(info["first_record_sequence_length"], 24)
        self.assertEqual(
            info["first_record_raw_video_nbytes"], 24 * 32 * 32 * 3
        )
        self.assertEqual(info["action_min"], 0)
        self.assertLess(info["action_max"], 15)
        self.assertEqual(info["nonzero_reward_frames_in_scan"], 6)
        self.assertEqual(info["scanned_frames"], 6 * 24)
        self.assertIsNone(info["level_seed_key"])

    def test_all_records_too_short(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "val"
            _write_coinrun_records(data_dir, num_records=4, record_len=3)
            with self.assertRaisesRegex(ValueError, "sequence_length"):
                ec.preflight_dataset(str(data_dir), seq_len=16, batch_size=2, max_scan=8)

    def test_fewer_usable_than_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "val"
            _write_coinrun_records(data_dir, num_records=2, record_len=24)
            with self.assertRaisesRegex(ValueError, "batch_size"):
                ec.preflight_dataset(str(data_dir), seq_len=5, batch_size=4, max_scan=8)

    def test_non_pickle_record_rejected(self):
        from array_record.python.array_record_module import ArrayRecordWriter

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "val"
            data_dir.mkdir(parents=True)
            writer = ArrayRecordWriter(
                str(data_dir / "shard-00000.array_record"), "group_size:1"
            )
            writer.write(b"this is not a pickle")
            writer.close()
            with self.assertRaisesRegex(ValueError, "pickle"):
                ec.preflight_dataset(str(data_dir), seq_len=5, batch_size=1, max_scan=8)

    def test_illegal_action_index_rejected(self):
        from array_record.python.array_record_module import ArrayRecordWriter

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "val"
            data_dir.mkdir(parents=True)
            writer = ArrayRecordWriter(
                str(data_dir / "shard-00000.array_record"), "group_size:1"
            )
            frames = np.zeros((24, 32, 32, 3), dtype=np.uint8)
            record = {
                "raw_video": frames.tobytes(),
                "sequence_length": 24,
                "actions": np.full((24,), 15, dtype=np.int32),  # 15 is illegal (0-14)
                "rewards": np.zeros((24,), dtype=np.float32),
            }
            writer.write(pickle.dumps(record))
            writer.close()
            with self.assertRaisesRegex(ValueError, "15"):
                ec.preflight_dataset(str(data_dir), seq_len=5, batch_size=1, max_scan=8)

    def test_reward_bias_refused_without_rewards(self):
        """Review G6: refuse the reward-biased arm when rewards are identically zero."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "val"
            _write_coinrun_records(
                data_dir, num_records=4, record_len=24, nonzero_rewards=False
            )
            with self.assertRaisesRegex(ValueError, "reward"):
                ec.preflight_dataset(
                    str(data_dir), seq_len=5, batch_size=2, max_scan=8,
                    p_include_reward=0.5,
                )


class TestRunMissingArtifacts(unittest.TestCase):
    """The evaluator must fail clearly, never manufacture artifacts."""

    def _args(self, tmp, **kw):
        base = dict(
            dynamics_ckpt=str(Path(tmp) / "no_such_ckpt"),
            array_record_path=str(Path(tmp) / "val"),
            out_dir=str(Path(tmp) / "out"),
            context=3,
            horizon=2,
            num_windows=2,
            batch_size=2,
            num_workers=0,
        )
        base.update(kw)
        return ec.Args(**base)

    def test_missing_dataset_fails(self):
        args = ec.Args(
            dynamics_ckpt="/nonexistent/ckpt",
            array_record_path="/nonexistent/dataset",
            batch_size=2,
        )
        with self.assertRaises(FileNotFoundError):
            ec.run(args)

    def test_missing_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_coinrun_records(Path(tmp) / "val", num_records=4, record_len=24)
            with self.assertRaises(FileNotFoundError):
                ec.run(self._args(tmp))

    def test_empty_checkpoint_dir_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_coinrun_records(Path(tmp) / "val", num_records=4, record_len=24)
            (Path(tmp) / "no_such_ckpt").mkdir()
            with self.assertRaises(FileNotFoundError):
                ec.run(self._args(tmp))

    def test_training_split_fails_before_any_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(tmp, array_record_path=str(Path(tmp) / "train"))
            with self.assertRaisesRegex(ValueError, "training split"):
                ec.run(args)


# ---------------------------------------------------------------------------
# End-to-end: tiny real checkpoint + synthetic dataset
# ---------------------------------------------------------------------------

def _tiny_configs():
    from dreamer.configs import (
        DecoderModelConfig,
        DynamicsModelConfig,
        EncoderModelConfig,
        TokenizerModelConfig,
    )

    enc = EncoderModelConfig(
        n_latents=8,
        d_bottleneck=8,
        depth=1,
        d_model=64,
        n_heads=4,
        n_kv_heads=1,
        patch_size=8,
        dropout_rate=0.0,  # production configs use 0.0; nnx.remat lifts `deterministic`
        dtype="float32",
        param_dtype="float32",
    )
    dec = DecoderModelConfig(
        n_latents=8,
        d_bottleneck=8,
        depth=1,
        d_model=64,
        n_heads=4,
        n_kv_heads=1,
        patch_size=8,
        d_patch=8 * 8 * 3,
        H=32,
        W=32,
        dropout_rate=0.0,
        dtype="float32",
        param_dtype="float32",
    )
    tok_cfg = TokenizerModelConfig(encoder=enc, decoder=dec)
    dyn_cfg = DynamicsModelConfig(
        d_bottleneck=8,
        depth=1,
        d_model=64,
        n_heads=4,
        n_kv_heads=1,
        packing_factor=2,
        n_register=2,
        k_max=4,
        context_length=32,
        categorical_action_dim=15,  # Procgen CoinRun has 15 actions (review G3)
        latent_mean=(0.0,) * 8,     # review G2: explicit scale stats
        latent_std=(1.0,) * 8,
        dtype="float32",
        param_dtype="float32",
    )
    return tok_cfg, dyn_cfg


def _randomize_linear(linear, rng, scale=0.05):
    """Break zero-init heads so an untrained model emits non-constant output
    (stands in for a trained model for artifact-verification purposes)."""
    import jax

    rng, sub = jax.random.split(rng)
    linear.kernel.value = jax.random.normal(sub, linear.kernel.value.shape) * scale
    return rng


def _save_tiny_bundle(ckpt_dir: Path) -> None:
    """Build tiny tokenizer+dynamics and save a real Orbax checkpoint bundle
    through the same machinery the training scripts use."""
    import jax
    import optax
    from flax import nnx

    from dreamer.checkpointing import (
        DynamicsCheckpointBundle,
        build_checkpoint_manager,
    )
    from dreamer.configs import CheckpointConfig
    from dreamer.models import Dynamics, Tokenizer
    from dreamer.parallel import build_parallel

    tok_cfg, dyn_cfg = _tiny_configs()
    mesh, _sharding, mesh_rules = build_parallel("data")
    with jax.set_mesh(mesh):
        rngs = nnx.Rngs(0)
        tokenizer = Tokenizer(tok_cfg, mesh_rules=mesh_rules, rngs=rngs)
        dynamics = Dynamics(dyn_cfg, mesh_rules=mesh_rules, rngs=rngs)
        dynamics_ema = Dynamics(dyn_cfg, mesh_rules=mesh_rules, rngs=nnx.Rngs(1))

        rng = jax.random.PRNGKey(0)
        rng = _randomize_linear(tokenizer.decoder.patch_head, rng)
        rng = _randomize_linear(dynamics.flow_x_head, rng)
        _randomize_linear(dynamics_ema.flow_x_head, rng)

        bundle = DynamicsCheckpointBundle(
            dynamics=dynamics,
            dynamics_ema=dynamics_ema,
            tokenizer=tokenizer,
            dynamics_optimizer=nnx.Optimizer(dynamics, optax.adamw(1e-4), wrt=nnx.Param),
        )
        manager = build_checkpoint_manager(
            CheckpointConfig(max_to_keep=1, save_interval_steps=1, max_steps=1),
            ckpt_dir,
            item_names=DynamicsCheckpointBundle.get_item_names(),
        )
        with manager:
            bundle.maybe_save(manager, 0, jax.random.PRNGKey(0))


class TestEndToEnd(unittest.TestCase):
    def test_full_eval_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_dir = tmp / "val"
            _write_coinrun_records(data_dir, num_records=6, record_len=24)
            ckpt_dir = tmp / "run" / "checkpoints"
            _save_tiny_bundle(ckpt_dir)
            out_dir = tmp / "eval_out"

            args = ec.Args(
                # Pass the parent dir on purpose: exercises '/checkpoints' resolution.
                dynamics_ckpt=str(ckpt_dir.parent),
                array_record_path=str(data_dir),
                out_dir=str(out_dir),
                context=3,
                horizon=2,
                num_windows=2,
                batch_size=2,
                denoise_steps=2,
                one_step_positions=2,
                num_videos=1,
                fps=5,
                seed=0,
                num_workers=0,
                preflight_scan_records=8,
            )
            results = ec.run(args)

            # Top-level schema.
            self.assertEqual(results["schema_version"], "coinrun-eval/2")
            self.assertEqual(results["status"], "completed")
            self.assertTrue(results["git_commit"])

            # Setup echo.
            setup = results["setup"]
            self.assertEqual(setup["context"], 3)
            self.assertEqual(setup["horizon"], 2)
            self.assertEqual(setup["seq_len"], 5)
            self.assertEqual(setup["one_step_positions"], [0, 1])
            self.assertEqual(setup["num_windows_evaluated"], 2)
            self.assertEqual(
                setup["rollout_arms"], ["true", "episode_permuted", "action_permuted"]
            )

            # Artifacts resolved from the real checkpoint.
            artifacts = results["artifacts"]
            self.assertEqual(artifacts["dynamics_step"], 0)
            self.assertEqual(artifacts["tokenizer_source"], "dynamics_bundle")
            self.assertTrue(artifacts["dynamics_ckpt"].endswith("checkpoints"))
            self.assertEqual(artifacts["dataset"]["num_records"], 6)

            # Model facts.
            self.assertEqual(results["model"]["categorical_action_dim"], 15)
            self.assertEqual(results["model"]["coinrun_noop_index"], 4)

            # Metric shapes and finiteness.
            m = results["metrics"]
            for arm in ("true", "episode_permuted", "action_permuted"):
                block = m[f"open_loop_{arm}"]
                for name in ("psnr", "latent_mse", "pixel_mse", "pixel_mae"):
                    self.assertEqual(len(block[name]), 2, f"{arm}.{name}")
            self.assertEqual(m["one_step"]["positions"], [0, 1])
            self.assertEqual(len(m["one_step"]["psnr"]), 2)

            # Sensitivity blocks with paired effect sizes.
            for gate in ("episode_permutation", "action_permutation"):
                sens = m["sensitivity"][gate]
                self.assertEqual(len(sens["psnr_drop"]["per_horizon"]), 2)
                self.assertEqual(sens["psnr_drop"]["overall"]["n"], 2)
                self.assertEqual(len(sens["latent_mse_increase"]["per_horizon"]), 2)
            self.assertEqual(
                m["sensitivity"]["action_permutation_mapping"], [5, 6, 7, 8, 0, 1, 2, 3, 4]
            )

            # Baselines (G1), latent scale (G2), reward prevalence (G6).
            self.assertIn("dataset_mean", m["baselines"])
            self.assertIn("copy_previous", m["baselines"])
            self.assertEqual(len(m["baselines"]["copy_previous_horizon"]["psnr"]), 2)
            self.assertEqual(len(m["latent_scale"]["per_dim_mean"]), 8)
            self.assertEqual(len(m["latent_scale"]["per_dim_std"]), 8)
            self.assertGreater(m["reward"]["frame_fraction_nonzero"], 0.0)

            for block in (
                m["recon"], m["one_step"], m["baselines"], m["latent_scale"],
                m["open_loop_true"], m["summary"],
            ):
                for value in self._flatten(block):
                    self.assertTrue(np.isfinite(value), f"non-finite metric in {block}")
            # Random data + tiny model: recon PSNR should be sane (>= 0 dB).
            self.assertGreaterEqual(m["recon"]["psnr"], 0.0)

            # Per-window rows.
            self.assertEqual(len(results["windows"]), 2)
            for row in results["windows"]:
                self.assertIn("reward_mean", row)
                self.assertIn("open_loop_true_psnr_mean", row)
                self.assertIn("open_loop_episode_permuted_psnr_mean", row)
                self.assertIn("open_loop_action_permuted_psnr_mean", row)

            # MP4 artifacts: one window x three arms.
            videos = results["videos"]
            self.assertEqual(len(videos), 3)
            self.assertEqual(
                {v["arm"] for v in videos},
                {"true", "episode_permuted", "action_permuted"},
            )
            for v in videos:
                self.assertTrue(Path(v["path"]).exists())
                self.assertIn("ctx3_hor2", Path(v["path"]).name)
                self.assertEqual(v["frames_readback"], 5)
                self.assertTrue(v["seam_marker_ok"])
                self.assertTrue(v["nontrivial"])

            # results.json round-trips.
            with open(out_dir / "results.json") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["schema_version"], "coinrun-eval/2")
            self.assertEqual(
                loaded["metrics"]["summary"].keys(), m["summary"].keys()
            )

    @staticmethod
    def _flatten(block):
        for value in block.values():
            if isinstance(value, dict):
                yield from TestEndToEnd._flatten(value)
            elif isinstance(value, (list, tuple)):
                for v in value:
                    if isinstance(v, (int, float)):
                        yield float(v)
            elif isinstance(value, (int, float)):
                yield float(value)


if __name__ == "__main__":
    unittest.main()
