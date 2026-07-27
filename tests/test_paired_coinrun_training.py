from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from dreamer.data.paired_rgb_adapter import (
    ADAPTER_SCHEMA_VERSION,
    SOURCE_SHARDS,
    sha256_file,
)
from scripts.paired_coinrun_training import (
    MANIFEST_COPY_FILENAME,
    RUN_CONTRACT_FILENAME,
    PairedTrainingError,
    build_training_plan,
    execute_training,
    publish_preflight_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(root: Path, *, action_dim: int = 9) -> tuple[Path, str]:
    sources = []
    split_counts = {
        "train": {"num_episodes": 0, "num_frames": 0, "sources": []},
        "val": {"num_episodes": 0, "num_frames": 0, "sources": []},
    }
    for index, spec in enumerate(SOURCE_SHARDS):
        split_dir = root / spec.split
        split_dir.mkdir(parents=True, exist_ok=True)
        shard_path = split_dir / f"{spec.source}-00000.array_record"
        if not shard_path.exists():
            shard_path.write_bytes(f"{spec.filename}-record".encode())
        episode = {
            "episode_id": f"{spec.filename}-episode",
            "length": index + 1,
        }
        source = {
            "episodes": [episode],
            "filename": spec.filename,
            "num_episodes": 1,
            "num_frames": index + 1,
            "output_shards": [
                {
                    "path": shard_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(shard_path),
                    "size_bytes": shard_path.stat().st_size,
                }
            ],
            "sha256": hashlib.sha256(spec.filename.encode()).hexdigest(),
            "size_bytes": 1,
            "source": spec.source,
            "split": spec.split,
        }
        sources.append(source)
        split_counts[spec.split]["num_episodes"] += 1
        split_counts[spec.split]["num_frames"] += index + 1
        split_counts[spec.split]["sources"].append(spec.source)

    manifest = {
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "action_alignment": "action_applied_after_frame",
        "action_space": {
            "categorical_action_dim": action_dim,
            "categorical_noop": 4,
            "continuous_action_dim": 0,
            "num_binary_actions": 0,
            "type": "procgen_discrete",
        },
        "input_contract": "transparent-coinrun-v1",
        "input_schema_version": ("transparent-coinrun-compact-paired-trajectory-v2"),
        "num_episodes": 4,
        "num_frames": 10,
        "records_per_shard": 1,
        "reward_alignment": "reward_resulting_from_action",
        "sources": sources,
        "splits": split_counts,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, sha256_file(manifest_path)


class PairedCoinRunTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.tmp_path = Path(temporary_directory.name)
        self.corpus_root = self.tmp_path / "corpus"
        self.corpus_root.mkdir()
        self.manifest_path, self.manifest_sha256 = _write_manifest(self.corpus_root)
        self.run_dir = self.tmp_path / "run"

    def _plan(self, **overrides):
        arguments = {
            "repo_root": REPO_ROOT,
            "run_dir": self.run_dir,
            "manifest_path": self.manifest_path,
            "expected_manifest_sha256": self.manifest_sha256,
            "train_dir": self.corpus_root / "train",
            "validation_dir": self.corpus_root / "val",
        }
        arguments.update(overrides)
        return build_training_plan(**arguments)

    def test_plan_sets_manifest_action_contract_on_both_trainers(self):
        plan = self._plan()

        self.assertEqual(
            plan.tokenizer_config["dataset"]["categorical_action_dim"],
            9,
        )
        self.assertEqual(
            plan.dynamics_config["dataset"]["categorical_action_dim"],
            9,
        )
        self.assertEqual(
            plan.dynamics_config["dynamics"]["categorical_action_dim"],
            9,
        )
        self.assertEqual(
            plan.dynamics_config["dataset"]["categorical_noop"],
            4,
        )
        self.assertIn(
            "dataset.categorical_action_dim=9",
            plan.tokenizer_command,
        )
        self.assertIn(
            "dynamics.categorical_action_dim=9",
            plan.dynamics_command,
        )
        self.assertFalse(
            any(
                "categorical_action_dim=15" in argument
                for command in (
                    plan.tokenizer_command,
                    plan.dynamics_command,
                )
                for argument in command
            )
        )

    def test_preflight_artifacts_record_manifest_hash_and_source_splits(self):
        plan = self._plan()
        contract_path = publish_preflight_artifacts(plan)
        contract = json.loads(contract_path.read_text(encoding="utf-8"))

        self.assertEqual(contract["status"], "preflight_passed")
        self.assertEqual(
            contract["paired_manifest"]["sha256"],
            self.manifest_sha256,
        )
        copied_manifest = self.run_dir / "inputs" / MANIFEST_COPY_FILENAME
        self.assertEqual(copied_manifest.read_bytes(), self.manifest_path.read_bytes())
        self.assertEqual(
            contract["splits"]["train"]["manifest_split"],
            "train",
        )
        self.assertEqual(
            contract["splits"]["validation"]["manifest_split"],
            "val",
        )
        self.assertEqual(
            {source["source"] for source in contract["splits"]["train"]["sources"]},
            {"random", "scripted_forward_v0"},
        )
        self.assertEqual(
            contract["action_space"]["categorical_action_dim"],
            9,
        )

    def test_missing_mismatched_or_wrong_split_manifest_is_rejected(self):
        with (
            self.subTest("missing"),
            self.assertRaisesRegex(
                PairedTrainingError,
                "manifest is missing",
            ),
        ):
            self._plan(manifest_path=self.tmp_path / "missing.json")

        with (
            self.subTest("digest"),
            self.assertRaisesRegex(
                PairedTrainingError,
                "SHA-256 mismatch",
            ),
        ):
            self._plan(expected_manifest_sha256="0" * 64)

        with (
            self.subTest("split"),
            self.assertRaisesRegex(
                PairedTrainingError,
                "does not match manifest split",
            ),
        ):
            self._plan(train_dir=self.corpus_root / "val")

    def test_mutated_arrayrecord_is_rejected(self):
        shard_path = self.corpus_root / "train" / "random-00000.array_record"
        shard_path.write_bytes(b"mutated")

        with self.assertRaisesRegex(
            PairedTrainingError,
            "(size|SHA-256) does not match manifest",
        ):
            self._plan()

    def test_legacy_manifest_and_protected_15_action_override_are_rejected(self):
        self.manifest_path, self.manifest_sha256 = _write_manifest(
            self.corpus_root,
            action_dim=15,
        )
        with self.assertRaisesRegex(
            PairedTrainingError,
            "legacy 15-action config",
        ):
            self._plan()

        self.manifest_path, self.manifest_sha256 = _write_manifest(
            self.corpus_root,
            action_dim=9,
        )
        with self.assertRaisesRegex(
            PairedTrainingError,
            "may not replace paired contract field",
        ):
            self._plan(
                dynamics_overrides=[
                    "dataset.categorical_action_dim=15",
                ]
            )

    def test_missing_action_space_is_rejected(self):
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        del manifest["action_space"]
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.manifest_sha256 = sha256_file(self.manifest_path)

        with self.assertRaisesRegex(
            PairedTrainingError,
            "action_space must be a JSON object",
        ):
            self._plan()

    def test_launch_orders_trainers_and_requires_tokenizer_checkpoint(self):
        plan = self._plan()
        publish_preflight_artifacts(plan)
        calls = []

        def successful_runner(command, **kwargs):
            calls.append(command)
            if "train_tokenizer.py" in command[1]:
                checkpoint_dir = self.run_dir / "tokenizer" / "checkpoints"
                checkpoint_dir.mkdir(parents=True)
                (checkpoint_dir / "step").write_text("1", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0)

        execute_training(plan, runner=successful_runner)
        self.assertIn("train_tokenizer.py", calls[0][1])
        self.assertIn("train_dynamics.py", calls[1][1])
        contract = json.loads(
            (self.run_dir / RUN_CONTRACT_FILENAME).read_text(encoding="utf-8")
        )
        self.assertEqual(contract["status"], "completed")

    def test_dynamics_is_blocked_when_tokenizer_produces_no_checkpoint(self):
        plan = self._plan()
        publish_preflight_artifacts(plan)
        calls = []

        def runner_without_checkpoint(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0)

        with self.assertRaisesRegex(
            PairedTrainingError,
            "produced no checkpoint",
        ):
            execute_training(plan, runner=runner_without_checkpoint)
        self.assertEqual(len(calls), 1)
        contract = json.loads(
            (self.run_dir / RUN_CONTRACT_FILENAME).read_text(encoding="utf-8")
        )
        self.assertEqual(contract["status"], "failed")

    def test_post_preflight_corpus_mutation_blocks_all_trainers(self):
        plan = self._plan()
        publish_preflight_artifacts(plan)
        shard_path = self.corpus_root / "train" / "random-00000.array_record"
        shard_path.write_bytes(b"changed-after-preflight")
        calls = []

        def runner(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0)

        with self.assertRaisesRegex(
            PairedTrainingError,
            "(size|SHA-256) does not match manifest",
        ):
            execute_training(plan, runner=runner)
        self.assertEqual(calls, [])
        contract = json.loads(
            (self.run_dir / RUN_CONTRACT_FILENAME).read_text(encoding="utf-8")
        )
        self.assertEqual(contract["status"], "failed")


if __name__ == "__main__":
    unittest.main()
