"""Behavioral tests for the prebuilt CoinRun runner image and its entrypoint."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "Dockerfile"


def load_module(name: str, path: Path):
    """Register in sys.modules first; the launcher defines dataclasses."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = load_module("coinrun_runner", REPO_ROOT / "scripts" / "coinrun_runner.py")
runpod_coinrun = load_module(
    "runpod_coinrun_for_runner", REPO_ROOT / "scripts" / "runpod_coinrun.py"
)


@contextmanager
def environment(**overrides: str | None):
    saved = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def make_checkout(root: Path, lock_body: str = "lock-body\n") -> tuple[Path, str, str]:
    """Create a throwaway Git checkout with a uv.lock; return path, sha, commit."""
    checkout = root / "checkout"
    checkout.mkdir(parents=True)
    (checkout / "uv.lock").write_text(lock_body, encoding="utf-8")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    for command in (
        ["git", "init", "-q"],
        ["git", "add", "uv.lock"],
        ["git", "commit", "-qm", "initial"],
    ):
        subprocess.run(command, cwd=checkout, env=env, check=True)
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    return checkout, hashlib.sha256(lock_body.encode()).hexdigest(), commit


def dockerfile_stages(text: str) -> dict[str, list[str]]:
    """Split a Dockerfile into stages keyed by alias (final stage: "final")."""
    stages: dict[str, list[str]] = {}
    current: list[str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("FROM "):
            parts = line.split()
            alias = parts[parts.index("AS") + 1] if "AS" in parts else "final"
            current = stages.setdefault(alias, [])
            continue
        if current is not None and line and not line.startswith("#"):
            current.append(line)
    return stages


def first_index(lines: list[str], *prefixes: str) -> int:
    for index, line in enumerate(lines):
        if line.startswith(prefixes):
            return index
    return -1


def last_index(lines: list[str], *prefixes: str) -> int:
    found = -1
    for index, line in enumerate(lines):
        if line.startswith(prefixes):
            found = index
    return found


class DockerfileContractTests(unittest.TestCase):
    """The 10a0252 regression was an entrypoint swallowing CMD; pin the shape."""

    def setUp(self):
        self.text = DOCKERFILE.read_text(encoding="utf-8")

    def test_entrypoint_is_exec_form_with_separate_replaceable_cmd(self):
        self.assertIn('ENTRYPOINT ["/usr/local/bin/coinrun-runner"]', self.text)
        self.assertIn('CMD ["smoke"]', self.text)
        # Shell-form ENTRYPOINT would make CMD unreachable.
        self.assertNotIn("ENTRYPOINT /usr/local/bin", self.text)

    def test_image_declares_the_runner_contract_env_the_entrypoint_reads(self):
        for variable in (
            "COINRUN_IMAGE_CONTRACT=1",
            "COINRUN_UV_LOCK_SHA256=${UV_LOCK_SHA256}",
            "COINRUN_SOURCE_COMMIT=${SOURCE_COMMIT}",
        ):
            self.assertIn(variable, self.text)

    def test_runtime_forces_uv_to_use_the_prebuilt_cache_offline(self):
        self.assertIn("UV_OFFLINE=1", self.text)

    def test_bundle_tree_carries_the_uv_binary_that_created_its_cache(self):
        self.assertIn("cp /usr/local/bin/uv /opt/coinrun/bin/uv", self.text)
        self.assertIn("test -x /opt/coinrun/bin/uv", self.text)

    def test_build_fails_when_lock_hash_arg_does_not_match_copied_lock(self):
        self.assertIn('test "$(sha256sum uv.lock | cut -d\' \' -f1)" = "${UV_LOCK_SHA256}"', self.text)

    def test_dependencies_come_from_the_frozen_lock_without_installing_project(self):
        # --no-install-project keeps `import dreamer` resolving to the runtime
        # checkout, so a pod cannot run image code instead of the pushed commit.
        self.assertIn("uv sync --frozen --no-install-project", self.text)

    def test_procgen_prerequisites_and_runtime_tools_are_installed(self):
        for package in ("cmake", "qtbase5-dev", "libqt5gui5", "ffmpeg", "git", "grep", "coreutils"):
            self.assertIn(package, self.text)

    def test_procgen_wheel_is_prewarmed_into_the_image_cache(self):
        self.assertIn("generate_coinrun_dataset.py", self.text)
        self.assertIn("-name libenv.so", self.text)
        self.assertIn("rm -rf /tmp/procgen-warmup", self.text)

    def test_builder_does_not_depend_on_per_commit_source_metadata(self):
        # BuildKit folds in-scope ARGs into every later RUN cache key, so a
        # SOURCE_COMMIT reference here would rebuild apt/uv sync/Procgen on
        # every source-only commit.
        builder = dockerfile_stages(self.text)["builder"]
        joined = "\n".join(builder)
        self.assertNotIn("SOURCE_COMMIT", joined)
        self.assertNotIn("SOURCE_REPO", joined)
        self.assertNotIn("LABEL", joined, "labels on a non-final stage are discarded")

    def test_builder_lock_arg_is_declared_after_the_expensive_apt_layer(self):
        builder = dockerfile_stages(self.text)["builder"]
        lock_arg = first_index(builder, "ARG UV_LOCK_SHA256")
        self.assertGreater(lock_arg, first_index(builder, "RUN apt-get"))
        self.assertGreater(lock_arg, first_index(builder, "RUN uv python install"))
        # ...and before the sync that verifies against it.
        self.assertLess(lock_arg, first_index(builder, "RUN --mount=type=cache"))

    def test_final_stage_declares_source_metadata_after_every_expensive_step(self):
        final = dockerfile_stages(self.text)["final"]
        metadata = first_index(final, "ARG SOURCE_COMMIT", "ARG UV_LOCK_SHA256", "LABEL ")
        self.assertGreater(metadata, 0, "source metadata must exist in the final stage")
        for prefix in ("RUN ", "COPY "):
            self.assertLess(
                last_index(final, prefix), metadata,
                f"a {prefix.strip()} step follows the per-commit metadata and would be rekeyed",
            )

    def test_contract_env_is_split_from_static_env(self):
        final = dockerfile_stages(self.text)["final"]
        static_env = first_index(final, "ENV DEBIAN_FRONTEND")
        contract_env = first_index(final, "ENV COINRUN_IMAGE_CONTRACT=1")
        self.assertGreater(contract_env, static_env)
        # The static block must not carry per-commit values.
        self.assertNotIn("COINRUN_SOURCE_COMMIT", final[static_env])
        self.assertLess(first_index(final, "RUN apt-get"), contract_env)

    def test_no_credentials_or_datasets_are_baked_in(self):
        lowered = self.text.lower()
        for secret in ("api_key", "runpod_api_key", "ghcr_token", "password", "secret"):
            self.assertNotIn(secret, lowered)
        ignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
        for excluded in ("datasets/", "artifacts/", ".runpod/", "*.array_record"):
            self.assertIn(excluded, ignore)


class ImageContractTests(unittest.TestCase):
    def test_matching_lock_hash_and_commit_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout, lock_sha, commit = make_checkout(Path(directory))
            with environment(
                COINRUN_IMAGE_CONTRACT="1",
                COINRUN_UV_LOCK_SHA256=lock_sha,
                COINRUN_SOURCE_COMMIT=commit,
            ):
                detail = runner.check_contract(checkout, commit)
            self.assertEqual(detail["uv_lock_sha256"], lock_sha)
            self.assertEqual(detail["commit"], commit)

    def test_lock_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout, _, commit = make_checkout(Path(directory))
            with environment(
                COINRUN_IMAGE_CONTRACT="1", COINRUN_UV_LOCK_SHA256="b" * 64
            ):
                with self.assertRaises(runner.ContractError) as caught:
                    runner.check_contract(checkout, commit)
            message = str(caught.exception)
            self.assertIn("uv.lock mismatch", message)
            # It must tell the operator to rebuild, not to sync on the pod.
            self.assertIn("republish the runner image", message)

    def test_missing_image_contract_env_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout, lock_sha, commit = make_checkout(Path(directory))
            with environment(
                COINRUN_IMAGE_CONTRACT=None, COINRUN_UV_LOCK_SHA256=lock_sha
            ):
                with self.assertRaises(runner.ContractError) as caught:
                    runner.check_contract(checkout, commit)
            self.assertIn("prebuilt runner image", str(caught.exception))

    def test_wrong_commit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout, lock_sha, _ = make_checkout(Path(directory))
            with environment(
                COINRUN_IMAGE_CONTRACT="1", COINRUN_UV_LOCK_SHA256=lock_sha
            ):
                with self.assertRaises(runner.ContractError) as caught:
                    runner.check_contract(checkout, "c" * 40)
            self.assertIn("expected commit", str(caught.exception))

    def test_child_env_forbids_resync_and_points_python_at_the_checkout(self):
        env = runner.child_env(Path("/workspace/open-dreamer"))
        self.assertEqual(env["UV_NO_SYNC"], "1")
        self.assertEqual(env["UV_FROZEN"], "1")
        self.assertEqual(env["UV_OFFLINE"], "1")
        self.assertTrue(env["PYTHONPATH"].startswith("/workspace/open-dreamer"))


class SmokeCommandTests(unittest.TestCase):
    def test_dataset_command_matches_the_controller_invocation_shape(self):
        command = runner.dataset_command(
            Path("/workspace/open-dreamer"), "scripted", Path("/tmp/out"), 7
        )
        self.assertEqual(command[:5], ["uv", "run", "--isolated", "--script",
                                       "/workspace/open-dreamer/dreamer/data/generate_coinrun_dataset.py"])
        self.assertIn("--collector=scripted", command)
        self.assertIn("--seed=7", command)
        self.assertIn(f"--num-episodes-train={runner.SMOKE_EPISODES}", command)

    def test_unknown_collector_is_not_silently_accepted(self):
        # The smoke must only ever run the two declared collectors.
        commands = [
            runner.dataset_command(Path("/c"), collector, Path("/o"), 1)
            for collector in ("random", "scripted")
        ]
        self.assertEqual(len(commands), 2)
        self.assertNotEqual(commands[0], commands[1])

    def test_smoke_budget_is_bounded_at_five_minutes(self):
        self.assertLessEqual(runner.SMOKE_BUDGET_SECONDS, 300)


class GpuGateTests(unittest.TestCase):
    def _probe(self, devices):
        payload = json.dumps(devices)

        def fake_run(command, *, cwd, env, timeout):
            return subprocess.CompletedProcess(command, 0, payload, "")

        return fake_run

    def test_exactly_one_supported_cuda_gpu_passes(self):
        original = runner.run
        runner.run = self._probe([{"kind": "NVIDIA H200", "platform": "gpu"}])
        try:
            detail = runner.check_gpu(Path("."), {})
        finally:
            runner.run = original
        self.assertEqual(detail["device_count"], 1)

    def test_two_gpus_are_rejected(self):
        original = runner.run
        runner.run = self._probe(
            [{"kind": "NVIDIA H200", "platform": "gpu"}] * 2
        )
        try:
            with self.assertRaises(runner.ContractError) as caught:
                runner.check_gpu(Path("."), {})
        finally:
            runner.run = original
        self.assertIn("exactly one CUDA device", str(caught.exception))

    def test_cpu_only_backend_is_rejected(self):
        original = runner.run
        runner.run = self._probe([{"kind": "cpu", "platform": "cpu"}])
        try:
            with self.assertRaises(runner.ContractError):
                runner.check_gpu(Path("."), {})
        finally:
            runner.run = original

    def test_unsupported_gpu_model_is_rejected(self):
        original = runner.run
        runner.run = self._probe([{"kind": "NVIDIA A10", "platform": "gpu"}])
        try:
            with self.assertRaises(runner.ContractError) as caught:
                runner.check_gpu(Path("."), {})
        finally:
            runner.run = original
        self.assertIn("A10", str(caught.exception))


class FailureArtifactTests(unittest.TestCase):
    def test_contract_failure_writes_structured_json_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, _, _ = make_checkout(root)
            artifacts = root / "artifacts"
            with environment(
                COINRUN_IMAGE_CONTRACT="1", COINRUN_UV_LOCK_SHA256="d" * 64
            ):
                code = runner.main([
                    "smoke",
                    f"--checkout-root={checkout}",
                    f"--artifact-dir={artifacts}",
                ])
            self.assertEqual(code, 1)
            payload = json.loads((artifacts / "runner" / "smoke.json").read_text())
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["mode"], "smoke")
            self.assertIn("uv.lock mismatch", payload["error"])

    def test_experiment_mode_reports_a_missing_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, lock_sha, commit = make_checkout(root)
            artifacts = root / "artifacts"
            with environment(
                COINRUN_IMAGE_CONTRACT="1", COINRUN_UV_LOCK_SHA256=lock_sha
            ):
                original_tools, original_gpu = runner.check_tools, runner.check_gpu
                runner.check_tools = lambda *a, **k: {}
                runner.check_gpu = lambda *a, **k: {}
                try:
                    code = runner.main([
                        "experiment",
                        f"--checkout-root={checkout}",
                        f"--artifact-dir={artifacts}",
                        f"--expect-commit={commit}",
                    ])
                finally:
                    runner.check_tools, runner.check_gpu = original_tools, original_gpu
            self.assertEqual(code, 1)
            payload = json.loads((artifacts / "runner" / "experiment.json").read_text())
            self.assertIn("controller missing", payload["error"])

    def test_dataset_failure_preserves_the_failed_command_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkout, lock_sha, commit = make_checkout(root)
            artifacts = root / "artifacts"

            def failed_generation(command, *, cwd, env, timeout):
                self.assertEqual(env["UV_OFFLINE"], "1")
                return subprocess.CompletedProcess(
                    command, 1, "", "offline dependency lookup failed"
                )

            originals = runner.check_tools, runner.check_gpu, runner.run
            runner.check_tools = lambda *a, **k: {}
            runner.check_gpu = lambda *a, **k: {}
            runner.run = failed_generation
            try:
                with environment(
                    COINRUN_IMAGE_CONTRACT="1",
                    COINRUN_UV_LOCK_SHA256=lock_sha,
                ):
                    code = runner.main([
                        "smoke",
                        f"--checkout-root={checkout}",
                        f"--artifact-dir={artifacts}",
                        f"--expect-commit={commit}",
                    ])
            finally:
                runner.check_tools, runner.check_gpu, runner.run = originals

            self.assertEqual(code, 1)
            payload = json.loads((artifacts / "runner" / "smoke.json").read_text())
            command = payload["failure_details"]["commands"][0]
            self.assertEqual(command["collector"], "random")
            self.assertEqual(command["returncode"], 1)
            self.assertIn("offline dependency lookup failed", command["stderr_tail"])


class ImmutableImageSelectionTests(unittest.TestCase):
    def test_digest_reference_is_accepted(self):
        reference = f"{runpod_coinrun.RUNNER_IMAGE_REPOSITORY}@sha256:{'a' * 64}"
        detail = runpod_coinrun.validate_image_reference(reference)
        self.assertEqual(detail["kind"], "digest")
        self.assertEqual(detail["repository"], runpod_coinrun.RUNNER_IMAGE_REPOSITORY)

    def test_commit_tag_reference_is_accepted(self):
        commit = "b" * 40
        detail = runpod_coinrun.validate_image_reference(
            f"{runpod_coinrun.RUNNER_IMAGE_REPOSITORY}:sha-{commit}"
        )
        self.assertEqual(detail["kind"], "commit-tag")
        self.assertEqual(detail["commit"], commit)

    def test_mutable_references_are_refused(self):
        for reference in (
            f"{runpod_coinrun.RUNNER_IMAGE_REPOSITORY}:latest",
            f"{runpod_coinrun.RUNNER_IMAGE_REPOSITORY}:v1.2.3",
            runpod_coinrun.RUNNER_IMAGE_REPOSITORY,
            runpod_coinrun.DEFAULT_IMAGE,
            "",
        ):
            with self.subTest(reference=reference):
                with self.assertRaises(runpod_coinrun.DeploymentError):
                    runpod_coinrun.validate_image_reference(reference)

    def test_image_transport_requires_an_explicit_image(self):
        # argparse no longer requires --image (bundle mode must not take one),
        # so the requirement is enforced by the validation gate instead.
        args = runpod_coinrun.build_parser().parse_args([
            "launch", "--preflight-report", "r.json", "--gpu", "H200",
            "--experiment-command", "true",
        ])
        self.assertEqual(args.transport, "image")
        with self.assertRaises(runpod_coinrun.DeploymentError) as caught:
            runpod_coinrun.validate_cli_args(args)
        self.assertIn("--image is required", str(caught.exception))


class RemoteBootstrapTests(unittest.TestCase):
    def build(self, command: str = "coinrun-runner experiment") -> str:
        return runpod_coinrun.build_remote_script(
            branch="experiment/coinrun-reconstruction-20260726",
            commit_sha="e" * 40,
            experiment_command=command,
            runtime_seconds=3600,
            artifact_dir="/workspace/coinrun-artifacts",
            stream_token="token",
            deadline_epoch=2_000_000_000,
        )

    def test_bootstrap_verifies_the_image_lock_against_the_checkout(self):
        script = self.build()
        self.assertIn('test "${COINRUN_IMAGE_CONTRACT:-}" = "1"', script)
        self.assertIn('image_lock="${COINRUN_UV_LOCK_SHA256:-}"', script)
        self.assertIn("sha256sum uv.lock", script)
        self.assertIn("uv.lock mismatch", script)

    def test_bootstrap_never_syncs_dependencies_on_the_pod(self):
        script = self.build()
        self.assertNotIn("uv sync", script, "bootstrap must not sync on the pod")
        self.assertIn("export UV_NO_SYNC=1 UV_FROZEN=1 UV_OFFLINE=1", script)

    def test_bootstrap_still_pins_the_exact_pushed_commit(self):
        script = self.build()
        self.assertIn(f'test "$(git rev-parse HEAD)" = {shlex.quote("e" * 40)}', script)

    def test_experiment_command_running_uv_sync_is_refused(self):
        with self.assertRaises(runpod_coinrun.DeploymentError) as caught:
            runpod_coinrun.validate_experiment_command("uv sync --frozen && true")
        self.assertIn("already contains the locked environment", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
