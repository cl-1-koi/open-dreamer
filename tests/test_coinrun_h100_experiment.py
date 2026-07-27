from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_coinrun_h100_experiment.sh"


class CoinRunH100ExperimentScriptTests(unittest.TestCase):
    def run_script(self, *args: str, env: dict[str, str] | None = None):
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=full_env,
            check=False,
        )

    def test_dry_run_builds_prewarmed_collection_commands_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory) / "artifacts"
            result = self.run_script(
                "--dry-run", "--artifact-dir", str(artifact_dir), "--seed", "7"
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((artifact_dir / "final_manifest.json").read_text())
            self.assertEqual(manifest["status"], "dry_run")
            self.assertEqual(manifest["hard_total_deadline_seconds"], 13500)
            self.assertFalse(manifest["oom_fallback_used"])

            tokenizer = (artifact_dir / "commands" / "tokenizer_train_initial.command").read_text()
            self.assertIn("train_tokenizer.py", tokenizer)
            self.assertIn("dataset.dataloader_cfg.B=128", tokenizer)
            self.assertIn("tokenizer.encoder.d_model=512", tokenizer)
            self.assertIn("ckpt.save_interval_steps=1000", tokenizer)
            self.assertNotIn("runpod", tokenizer.lower())

            collection = (artifact_dir / "commands" / "collection.command").read_text()
            self.assertIn("coinrun-dataset-python", collection)
            self.assertNotIn("uv run --isolated", collection)
            self.assertIn("--collector=random|scripted", collection)

            dynamics = (artifact_dir / "commands" / "dynamics_train_initial.command").read_text()
            self.assertIn("dynamics.latent_mean=", dynamics)
            self.assertIn("dynamics.latent_std=", dynamics)

            script_text = SCRIPT.read_text()
            self.assertIn('"--dynamics-ckpt=$DYNAMICS_DIR/checkpoints"', script_text)
            self.assertIn('"--tokenizer-ckpt=$TOKENIZER_DIR/checkpoints"', script_text)
            self.assertIn('"--array-record-path=$DATASET_DIR/val"', script_text)
            self.assertIn("--p-include-reward=0.25", script_text)
            self.assertNotIn("--checkpoint-dir=$DYNAMICS_DIR", script_text)

            self.assertIn("collector_records_seen", script_text)
            self.assertIn("collector_records_selected", script_text)
            self.assertIn('("random", "scripted")', script_text)
            self.assertIn("'H100|H200|B200'", script_text)
            self.assertNotIn("rg -", script_text)

    def test_dry_run_fallback_renders_one_smaller_preset(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory) / "artifacts"
            result = self.run_script(
                "--dry-run", "--preset", "fallback", "--artifact-dir", str(artifact_dir)
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            command = (artifact_dir / "commands" / "tokenizer_train_initial.command").read_text()
            self.assertIn("dataset.dataloader_cfg.B=64", command)
            self.assertIn("tokenizer.encoder.d_model=384", command)
            manifest = json.loads((artifact_dir / "final_manifest.json").read_text())
            self.assertEqual(manifest["active_preset"], "fallback")

    def test_stage_functions_do_not_expand_locals_before_assignment(self):
        script_text = SCRIPT.read_text()
        for stage in ("tokenizer", "dynamics"):
            unsafe = (
                f'local preset="$1" '
                f'run_dir="$ARTIFACT_DIR/{stage}_$preset"'
            )
            safe = (
                'local preset="$1"\n'
                f'  local run_dir="$ARTIFACT_DIR/{stage}_$preset"'
            )
            self.assertNotIn(unsafe, script_text)
            self.assertIn(safe, script_text)

    def test_rejects_missing_artifact_directory_without_claiming_success(self):
        result = self.run_script("--dry-run", env={"COINRUN_ARTIFACT_DIR": ""})

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("COINRUN_ARTIFACT_DIR", result.stderr)
        self.assertNotIn("passed", result.stdout.lower())

    def test_rejects_invalid_preset_before_any_gpu_or_procgen_work(self):
        result = self.run_script("--dry-run", "--preset", "unsafe", "--artifact-dir", "/tmp/coinrun-invalid")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("preset must be initial or fallback", result.stderr)

    def test_dry_run_configuration_failure_writes_failed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory) / "artifacts"
            result = self.run_script(
                "--dry-run",
                "--artifact-dir",
                str(artifact_dir),
                env={"COINRUN_TOKENIZER_STEPS": "0"},
            )

            self.assertNotEqual(result.returncode, 0)
            manifest = json.loads((artifact_dir / "final_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("step overrides", manifest["failure"])


if __name__ == "__main__":
    unittest.main()
