from __future__ import annotations

import pickle
import sys
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from scripts.coinrun_preflight import (
    HardRuntimeExceeded,
    Preflight,
    PreflightError,
    build_default_dataset_command,
    build_latent_hydra_overrides,
    compute_tokenizer_probe_metrics,
    default_collection_scope,
    discover_checkpoints,
    fatal_artifact_failures,
    find_coinrun_config,
    run_bounded_command,
    select_tokenizer_probe_records,
    summarize_dataset_records,
    summarize_training_events,
    validate_checkpoint_cadence,
    validate_dataset_gates,
    validate_dynamics_preconditions,
    validate_measured_latent_stats,
)


class TrainingTelemetryTests(unittest.TestCase):
    def test_summarizes_compile_steady_throughput_metrics_and_memory(self):
        events = [
            {
                "kind": "resolved_config",
                "config": {
                    "dataset": {
                        "dataloader_cfg": {"B": 2, "long_T": 16}
                    }
                },
            },
            {"kind": "parameter_counts", "counts": {"total": 1234}},
            {
                "kind": "training_step",
                "elapsed_seconds": 5.0,
                "metrics": {
                    "loss_total": 2.0,
                    "grad/global_norm": 3.0,
                },
                "gpu_memory": {
                    "available": True,
                    "peak_mb": 900.0,
                    "current_mb": 700.0,
                },
            },
            {
                "kind": "training_step",
                "elapsed_seconds": 1.0,
                "metrics": {
                    "loss_total": 1.5,
                    "grad/global_norm": 2.5,
                },
                "gpu_memory": {
                    "available": True,
                    "peak_mb": 1000.0,
                    "current_mb": 800.0,
                },
            },
            {
                "kind": "training_step",
                "elapsed_seconds": 1.2,
                "metrics": {
                    "loss_total": 1.0,
                    "grad/global_norm": 2.0,
                },
                "gpu_memory": {
                    "available": True,
                    "peak_mb": 1100.0,
                    "current_mb": 820.0,
                },
            },
            {
                "kind": "logger_metrics",
                "prefix": "eval/",
                "metrics": {
                    "online_diffusion/eval_time": 10.0,
                    "ema_diffusion/eval_time": 9.0,
                    "online_shortcut/eval_time": 8.0,
                    "ema_shortcut/eval_time": 7.0,
                },
            },
        ]

        summary = summarize_training_events(events, [100.0, 250.0, 200.0])

        self.assertEqual(summary["parameter_counts"]["total"], 1234)
        self.assertAlmostEqual(summary["timing"]["steady_step_seconds"], 1.1)
        self.assertAlmostEqual(
            summary["timing"]["jax_compile_seconds_estimate"], 3.9
        )
        self.assertAlmostEqual(
            summary["throughput"]["frames_per_second"], 32 / 1.1
        )
        self.assertEqual(summary["losses"], {"loss_total": 1.0})
        self.assertEqual(
            summary["gradient_norms"], {"grad/global_norm": 2.0}
        )
        self.assertEqual(summary["gpu_memory"]["jax_peak_mb"], 1100.0)
        self.assertEqual(summary["gpu_memory"]["nvidia_smi_peak_mb"], 250.0)
        self.assertEqual(summary["evaluation"]["four_way_total_seconds"], 34.0)


class DatasetTelemetryTests(unittest.TestCase):
    def test_tokenizer_probe_balances_collectors_despite_shard_order(self):
        def record(collector, seed, length=16):
            return (
                "val",
                pickle.dumps(
                    {
                        "collector": collector,
                        "level_seed": seed,
                        "sequence_length": length,
                    }
                ),
            )

        records = [
            *(record("random", seed) for seed in range(8)),
            *(record("scripted", 100 + seed) for seed in range(8)),
            record("scripted", 999, length=4),
        ]

        selected, skipped, seen, selected_counts = (
            select_tokenizer_probe_records(
                records,
                sequence_length=16,
                max_records=6,
            )
        )

        self.assertEqual(
            [item["collector"] for item in selected],
            ["random", "scripted"] * 3,
        )
        self.assertEqual(skipped, 1)
        self.assertEqual(seen, {"random": 8, "scripted": 9})
        self.assertEqual(selected_counts, {"random": 3, "scripted": 3})

    def test_default_generator_uses_isolated_pep723_scripted_arm(self):
        repo_root = Path(__file__).resolve().parents[1]
        dataset_dir = Path("/tmp/coinrun-preflight-dataset")

        command = build_default_dataset_command(
            repo_root=repo_root,
            dataset_dir=dataset_dir,
            sequence_length=16,
            seed=7,
            uv_executable="/opt/uv",
        )

        self.assertEqual(
            command[:5],
            [
                "/opt/uv",
                "run",
                "--isolated",
                "--script",
                str(
                    repo_root
                    / "dreamer"
                    / "data"
                    / "generate_coinrun_dataset.py"
                ),
            ],
        )
        self.assertIn("--collector=scripted", command)
        self.assertIn("--min-episode-length=16", command)
        self.assertIn("--max-episode-length=256", command)
        self.assertIn("--chunk-size=256", command)
        self.assertIn("--keep-short-terminated", command)
        self.assertIn("--seed=7", command)
        self.assertNotIn(sys.executable, command)
        scope = default_collection_scope(16)
        self.assertEqual(scope["arm"], "scripted_plumbing")
        self.assertIn("follow-on", scope["scientific_comparison"])

    def test_decodes_and_counts_required_coinrun_fields(self):
        frames = np.arange(2 * 64 * 64 * 3, dtype=np.uint8).reshape(
            2, 64, 64, 3
        )
        record = {
            "raw_video": frames.tobytes(),
            "sequence_length": 2,
            "actions": np.asarray([7, 8], dtype=np.int32),
            "rewards": np.asarray([0.0, 1.0], dtype=np.float32),
            "collector": "scripted",
            "level_seed": 11,
            "action_seed": 12,
            "episode_starts": np.asarray([True, False]),
            "episode_ends": np.asarray([False, True]),
        }

        stats = summarize_dataset_records(
            [("train", pickle.dumps(record))]
        )

        self.assertEqual(stats["records"], 1)
        self.assertEqual(stats["frames"], 2)
        self.assertEqual(stats["reward_prevalence"], 0.5)
        self.assertEqual(stats["action_histogram"], {"7": 1, "8": 1})
        self.assertEqual(stats["collector_frame_histogram"], {"scripted": 2})
        self.assertEqual(stats["seeds"], [11, 12])
        self.assertEqual(stats["episode_start_frames"], 1)
        self.assertEqual(stats["episode_end_frames"], 1)
        self.assertGreater(stats["decoded_pixel_max"], stats["decoded_pixel_min"])

    def test_fails_clearly_when_record_artifact_is_incomplete(self):
        record = {
            "raw_video": bytes(64 * 64 * 3),
            "sequence_length": 1,
            "actions": np.asarray([0]),
        }

        with self.assertRaisesRegex(PreflightError, "missing required fields: rewards"):
            summarize_dataset_records([("train", pickle.dumps(record))])


class FailClosedGateTests(unittest.TestCase):
    def setUp(self):
        self.stats = {
            "action_histogram": {"0": 4, "14": 2},
            "level_seeds_by_split": {
                "train": [1, 2],
                "val": [3],
                "test": [4],
            },
            "reward_nonzero_frames": 1,
            "reward_prevalence": 0.01,
            "episode_start_frames": 4,
            "episode_end_frames": 4,
            "minimum_record_length": 16,
            "minimum_record_length_by_split": {
                "train": 16,
                "val": 16,
                "test": 8,
            },
        }
        self.metadata = {
            "num_actions": 15,
            "categorical_noop": 4,
            "action_alignment": "action_applied_after_frame",
            "reward_alignment": "reward_resulting_from_action",
        }
        self.dataset_config = {
            "categorical_action_dim": 15,
            "categorical_noop": 4,
            "p_include_reward": 0.5,
        }

    def test_dataset_gates_accept_declared_disjoint_rewarded_contract(self):
        result = validate_dataset_gates(
            self.stats,
            self.metadata,
            self.dataset_config,
            required_sequence_length=16,
        )

        self.assertTrue(result["held_out_level_seeds_disjoint"])
        self.assertEqual(result["action_dim"], 15)
        self.assertEqual(result["categorical_noop"], 4)

    def test_dataset_gates_report_action_seed_and_reward_failures_together(self):
        self.metadata["num_actions"] = 16
        self.dataset_config["categorical_noop"] = None
        self.stats["level_seeds_by_split"]["val"] = [2]
        self.stats["reward_nonzero_frames"] = 0
        self.stats["minimum_record_length_by_split"]["val"] = 15

        with self.assertRaises(PreflightError) as context:
            validate_dataset_gates(
                self.stats,
                self.metadata,
                self.dataset_config,
                required_sequence_length=16,
            )

        message = str(context.exception)
        self.assertIn("num_actions=16", message)
        self.assertIn("categorical_noop=None", message)
        self.assertIn("level seed sets overlap", message)
        self.assertIn("nonzero-reward frames are zero", message)
        self.assertIn("minimum val record length 15", message)

    def test_dynamics_requires_action_contract_and_explicit_latent_stats(self):
        good = {
            "dataset": {
                "data_type": "video",
                "categorical_action_dim": 15,
                "categorical_noop": 4,
            },
            "dynamics": {
                "categorical_action_dim": 15,
                "d_bottleneck": 2,
                "latent_mean": [0.1, -0.1],
                "latent_std": [0.9, 1.1],
            },
        }
        self.assertTrue(validate_dynamics_preconditions(good)["passed"])

        bad = {
            "dataset": {
                "data_type": "video",
                "categorical_action_dim": 16,
            },
            "dynamics": {
                "categorical_action_dim": 16,
                "d_bottleneck": 2,
                "latent_mean": None,
                "latent_std": None,
            },
        }
        with self.assertRaises(PreflightError) as context:
            validate_dynamics_preconditions(bad)
        self.assertIn("dimensions must both be 15", str(context.exception))
        self.assertIn("categorical_noop must be 4", str(context.exception))
        self.assertIn("explicit held-out latent_mean", str(context.exception))

    def test_checkpoint_cadence_is_projected_from_measured_step_time(self):
        summary = {
            "resolved_config": {"ckpt": {"save_interval_steps": 10}},
            "timing": {"steady_step_seconds": 2.0},
        }
        result = validate_checkpoint_cadence(summary, ["1", "2"])
        self.assertEqual(result["projected_interval_seconds"], 20.0)

        summary["resolved_config"]["ckpt"]["save_interval_steps"] = 500
        with self.assertRaisesRegex(
            PreflightError, "exceeding the 900s bound"
        ):
            validate_checkpoint_cadence(summary, ["1", "2"])

        summary["resolved_config"]["ckpt"]["save_interval_steps"] = 10
        with self.assertRaisesRegex(PreflightError, "at least two checkpoints"):
            validate_checkpoint_cadence(summary, ["2"])

    def test_latent_stats_validate_and_build_exact_hydra_lists(self):
        payload = {
            "latent_mean": [0.125, -0.25],
            "latent_std": [0.5, 0.75],
            "latent_sample_count": 64,
            "frame_count": 16,
            "record_count": 1,
            "source": {
                "split": "val",
                "tokenizer_variant": "online",
                "checkpoint_step": 2,
                "level_seeds": [123],
            },
        }

        validation = validate_measured_latent_stats(
            payload, expected_dim=2, std_epsilon=1e-6
        )
        self.assertTrue(validation["passed"])
        self.assertEqual(
            build_latent_hydra_overrides(payload),
            [
                "dynamics.latent_mean=[0.125,-0.25]",
                "dynamics.latent_std=[0.5,0.75]",
            ],
        )

    def test_latent_stats_fail_on_nonfinite_or_near_zero_std(self):
        payload = {
            "latent_mean": [0.0, float("nan")],
            "latent_std": [1e-8, 1.0],
            "latent_sample_count": 2,
            "frame_count": 1,
            "record_count": 1,
            "source": {
                "split": "val",
                "tokenizer_variant": "online",
                "checkpoint_step": 2,
                "level_seeds": [123],
            },
        }
        with self.assertRaises(PreflightError) as context:
            validate_measured_latent_stats(
                payload, expected_dim=2, std_epsilon=1e-6
            )
        self.assertIn("nonfinite_mean_dims=[1]", str(context.exception))
        self.assertIn("std_at_or_below_1e-06=[0]", str(context.exception))

    def test_tokenizer_probe_metrics_include_reconstruction_baselines(self):
        targets = np.asarray(
            [
                [
                    [[[0, 0, 0], [0, 0, 0]]],
                    [[[255, 255, 255], [255, 255, 255]]],
                ]
            ],
            dtype=np.uint8,
        )
        latents = np.asarray(
            [[[[0.0, -1.0]], [[1.0, 1.0]]]], dtype=np.float32
        )

        metrics = compute_tokenizer_probe_metrics(
            latents=latents,
            reconstructions=targets,
            targets=targets,
            dataset_mean=[0.5, 0.5, 0.5],
        )

        self.assertEqual(metrics["latent_sample_count"], 2)
        self.assertEqual(metrics["d_bottleneck"], 2)
        self.assertEqual(metrics["reconstruction"]["model_mse"], 0.0)
        self.assertTrue(
            metrics["reconstruction"]["beats_dataset_mean_baseline"]
        )
        self.assertTrue(
            metrics["reconstruction"]["beats_copy_previous_baseline"]
        )

    def test_dynamics_command_injects_measured_latent_stats(self):
        preflight = object.__new__(Preflight)
        preflight.repo_root = Path(__file__).resolve().parents[1]
        preflight.args = Namespace(batch_size=2, sequence_length=16, steps=3)
        payload = {
            "latent_mean": [0.125, -0.25],
            "latent_std": [0.5, 0.75],
            "latent_sample_count": 64,
            "frame_count": 16,
            "record_count": 1,
            "source": {
                "split": "val",
                "tokenizer_variant": "online",
                "checkpoint_step": 2,
                "level_seeds": [123],
            },
        }

        command = preflight._trainer_command(
            stage="dynamics",
            config_name="coinrun_dynamics",
            event_path=Path("/tmp/dynamics-events.jsonl"),
            run_dir=Path("/tmp/dynamics-run"),
            dataset_dir=Path("/tmp/dataset"),
            tokenizer_checkpoint=Path("/tmp/tokenizer/checkpoints"),
            latent_stats=payload,
        )

        self.assertIn("dynamics.latent_mean=[0.125,-0.25]", command)
        self.assertIn("dynamics.latent_std=[0.5,0.75]", command)


class BoundAndArtifactTests(unittest.TestCase):
    def test_logged_mp4_write_failure_is_fatal_even_without_nonzero_exit(self):
        output = (
            "[eval] consolidated MP4 write failed: "
            "The pyav plugin is not installed"
        )
        self.assertEqual(
            fatal_artifact_failures(output),
            ["rollout MP4 writer failed"],
        )

    def test_subprocess_obeys_deadline(self):
        started = time.monotonic()
        with self.assertRaises(HardRuntimeExceeded):
            run_bounded_command(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                cwd=Path.cwd(),
                deadline=time.monotonic() + 0.1,
                monitor_gpu=False,
            )
        self.assertLess(time.monotonic() - started, 2.0)

    def test_missing_checkpoint_fails_with_artifact_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoints"
            with self.assertRaisesRegex(PreflightError, str(path)):
                discover_checkpoints(path)

    def test_config_auto_discovery_and_missing_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            configs = Path(directory)
            (configs / "coinrun_tokenizer.yaml").write_text(
                "defaults: []\n", encoding="utf-8"
            )
            self.assertEqual(
                find_coinrun_config(configs, "tokenizer"),
                "coinrun_tokenizer",
            )
            with self.assertRaisesRegex(
                PreflightError, "Missing dedicated CoinRun dynamics config"
            ):
                find_coinrun_config(configs, "dynamics")


if __name__ == "__main__":
    unittest.main()
