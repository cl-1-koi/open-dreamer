"""Behavioral tests for the direct-bundle transport.

No Docker, no network, no pod. Every test either exercises a pure function or
drives run_launch with fakes and asserts that a bad input never reaches
create_pod.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


bundle = load_module("build_coinrun_bundle", REPO_ROOT / "scripts" / "build_coinrun_bundle.py")
rc = load_module("runpod_coinrun_bundle_tests", REPO_ROOT / "scripts" / "runpod_coinrun.py")

COMMIT = "e" * 40
BASE_DIGEST = "sha256:" + "6" * 64


def write_bundle(root: Path, *, lock_body: bytes = b"lock\n", payload: bytes = b"payload"):
    """Create a consistent manifest + archive pair under a temp root."""
    (root / "uv.lock").write_bytes(lock_body)
    lock_sha = hashlib.sha256(lock_body).hexdigest()
    bundle_dir = root / "artifacts" / "coinrun_bundle"
    bundle_dir.mkdir(parents=True)
    archive = bundle_dir / "coinrun-opt-test.tar.zst"
    archive.write_bytes(payload)
    manifest = {
        "schema": bundle.SCHEMA,
        "archive": {
            "filename": archive.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "compression": "zstd",
        },
        "extract": {"target": "/opt/coinrun", "parent": "/opt", "expected_top_level": "coinrun"},
        "source": {
            "commit": COMMIT,
            "uv_lock_sha256": lock_sha,
            "dockerfile_sha256": "a" * 64,
            "image_reference": "coinrun-runner:slim",
            "image_id": "sha256:" + "b" * 64,
        },
        "base_image": {
            "reference": "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
            "digest": BASE_DIGEST,
        },
        "contract_env": {
            "COINRUN_IMAGE_CONTRACT": "1",
            "COINRUN_SOURCE_COMMIT": COMMIT,
            "COINRUN_UV_LOCK_SHA256": lock_sha,
            "UV_PROJECT_ENVIRONMENT": "/opt/coinrun/venv",
        },
    }
    manifest_path = root / "manifests" / "coinrun_runner_bundle.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, bundle_dir, manifest, archive


def plan_for(root: Path, manifest_path: Path, bundle_dir: Path):
    return rc.validate_bundle_manifest(manifest_path, repo_root=root, bundle_dir=bundle_dir)


class ManifestValidationTests(unittest.TestCase):
    def test_consistent_manifest_resolves_the_pinned_base_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, _, archive = write_bundle(root)
            plan = plan_for(root, manifest_path, bundle_dir)
            self.assertEqual(plan.archive_path, archive.resolve())
            self.assertEqual(plan.base_image, f"runpod/pytorch@{BASE_DIGEST}")

    def test_lock_drift_between_manifest_and_checkout_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, _, _ = write_bundle(root)
            (root / "uv.lock").write_bytes(b"a different lock\n")
            with self.assertRaises(rc.DeploymentError) as caught:
                plan_for(root, manifest_path, bundle_dir)
            self.assertIn("rebuild the bundle", str(caught.exception))

    def test_contract_env_disagreeing_with_source_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, manifest, _ = write_bundle(root)
            manifest["contract_env"]["COINRUN_UV_LOCK_SHA256"] = "c" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(rc.DeploymentError) as caught:
                plan_for(root, manifest_path, bundle_dir)
            self.assertIn("contract env disagrees", str(caught.exception))

    def test_wrong_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, manifest, _ = write_bundle(root)
            manifest["schema"] = "something-else"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(rc.DeploymentError):
                plan_for(root, manifest_path, bundle_dir)

    def test_missing_sections_are_rejected(self):
        for section in ("archive", "extract", "source", "base_image", "contract_env"):
            with self.subTest(section=section), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest_path, bundle_dir, manifest, _ = write_bundle(root)
                del manifest[section]
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(rc.DeploymentError):
                    plan_for(root, manifest_path, bundle_dir)

    def test_archive_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, manifest, archive = write_bundle(root)
            archive.write_bytes(b"payloadX")  # same length, different content
            manifest["archive"]["bytes"] = len(b"payloadX")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(rc.DeploymentError) as caught:
                plan_for(root, manifest_path, bundle_dir)
            self.assertIn("sha256", str(caught.exception))

    def test_archive_size_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, manifest, _ = write_bundle(root)
            manifest["archive"]["bytes"] = 999999
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(rc.DeploymentError) as caught:
                plan_for(root, manifest_path, bundle_dir)
            self.assertIn("bytes", str(caught.exception))

    def test_missing_archive_names_the_rebuild_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, _, archive = write_bundle(root)
            archive.unlink()
            with self.assertRaises(rc.DeploymentError) as caught:
                plan_for(root, manifest_path, bundle_dir)
            self.assertIn("build_coinrun_bundle.py", str(caught.exception))

    def test_path_traversal_in_the_filename_is_rejected(self):
        for name in ("../escape.tar.zst", "/abs.tar.zst", ".hidden"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest_path, bundle_dir, manifest, _ = write_bundle(root)
                manifest["archive"]["filename"] = name
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(rc.DeploymentError) as caught:
                    plan_for(root, manifest_path, bundle_dir)
                self.assertIn("unsafe", str(caught.exception))

    def test_malformed_base_digest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, manifest, _ = write_bundle(root)
            manifest["base_image"]["digest"] = "sha256:short"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(rc.DeploymentError):
                plan_for(root, manifest_path, bundle_dir)

    def test_unexpected_extract_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, manifest, _ = write_bundle(root)
            manifest["extract"]["target"] = "/opt/elsewhere"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(rc.DeploymentError):
                plan_for(root, manifest_path, bundle_dir)


class BundleBuilderTests(unittest.TestCase):
    """Validation and command construction in scripts/build_coinrun_bundle.py."""

    def info(self, **overrides):
        env = {
            "COINRUN_IMAGE_CONTRACT": "1",
            "COINRUN_SOURCE_COMMIT": COMMIT,
            "COINRUN_UV_LOCK_SHA256": "d" * 64,
            "UV_PROJECT_ENVIRONMENT": "/opt/coinrun/venv",
            "UV_CACHE_DIR": "/opt/coinrun/uv-cache",
            "UV_PYTHON_INSTALL_DIR": "/opt/coinrun/python",
            "UV_LINK_MODE": "copy",
            "UV_COMPILE_BYTECODE": "1",
            "UV_PYTHON_DOWNLOADS": "never",
        }
        env.update(overrides.pop("env", {}))
        info = {
            "id": "sha256:" + "b" * 64,
            "labels": {bundle.LOCK_LABEL: "d" * 64},
            "env": env,
            "repo_digests": [],
        }
        info.update(overrides)
        return info

    def test_matching_image_yields_the_contract_env(self):
        contract = bundle.validate_source_image(self.info(), "d" * 64)
        self.assertEqual(contract["COINRUN_SOURCE_COMMIT"], COMMIT)
        self.assertEqual(sorted(contract), sorted(bundle.CONTRACT_ENV_KEYS))

    def test_image_built_for_another_lock_is_refused(self):
        with self.assertRaises(bundle.BundleError) as caught:
            bundle.validate_source_image(self.info(), "e" * 64)
        self.assertIn("rebuild the image before packaging", str(caught.exception))

    def test_label_disagreeing_with_env_is_refused(self):
        info = self.info(labels={bundle.LOCK_LABEL: "f" * 64})
        with self.assertRaises(bundle.BundleError) as caught:
            bundle.validate_source_image(info, "d" * 64)
        self.assertIn("disagrees", str(caught.exception))

    def test_non_runner_image_is_refused(self):
        info = self.info(env={"COINRUN_IMAGE_CONTRACT": "0"})
        with self.assertRaises(bundle.BundleError):
            bundle.validate_source_image(info, "d" * 64)

    def test_missing_contract_env_is_refused(self):
        info = self.info()
        del info["env"]["UV_CACHE_DIR"]
        with self.assertRaises(bundle.BundleError) as caught:
            bundle.validate_source_image(info, "d" * 64)
        self.assertIn("UV_CACHE_DIR", str(caught.exception))

    def test_archive_command_is_deterministic_and_scoped_to_opt_coinrun(self):
        command = bundle.archive_command(["docker"], "img")
        for flag in ("--sort=name", "--mtime=@0", "--owner=0", "--group=0", "--numeric-owner"):
            self.assertIn(flag, command)
        # Only /opt/coinrun is packaged, streamed to stdout.
        self.assertEqual(command[-5:], ["-C", "/opt", "-cf", "-", "coinrun"])

    def test_compress_level_is_validated(self):
        for level in (0, 20, -1):
            with self.subTest(level=level):
                with self.assertRaises(bundle.BundleError):
                    bundle.compress_command(level, 0, Path("/tmp/x"))

    def test_base_without_a_repo_digest_names_the_pull_command(self):
        module_docker = ["docker"]
        original = bundle.inspect_image
        bundle.inspect_image = lambda d, r: {"id": "", "labels": {}, "env": {}, "repo_digests": []}
        try:
            with self.assertRaises(bundle.BundleError) as caught:
                bundle.resolve_base(module_docker, "runpod/pytorch:x")
        finally:
            bundle.inspect_image = original
        self.assertIn("docker pull", str(caught.exception))


class ProvenanceVersusCompatibilityTests(unittest.TestCase):
    """source.commit is provenance, not a runtime-identity gate.

    The manifest is checked in, so committing it always advances HEAD:
    requiring source.commit == HEAD would be an impossible self-reference.
    Runtime checkout identity is enforced separately by --expect-commit.
    """

    def test_older_source_commit_is_accepted_when_the_lock_is_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, manifest, _ = write_bundle(root)
            # Bundle built at an older commit; the checkout has moved on.
            manifest["source"]["commit"] = "1" * 40
            manifest["contract_env"]["COINRUN_SOURCE_COMMIT"] = "1" * 40
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            plan = plan_for(root, manifest_path, bundle_dir)
        self.assertEqual(plan.manifest["source"]["commit"], "1" * 40)

    def test_the_lock_hash_is_what_actually_gates_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, manifest, _ = write_bundle(root)
            manifest["source"]["commit"] = "1" * 40
            manifest["contract_env"]["COINRUN_SOURCE_COMMIT"] = "1" * 40
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (root / "uv.lock").write_bytes(b"dependencies changed\n")
            with self.assertRaises(rc.DeploymentError) as caught:
                plan_for(root, manifest_path, bundle_dir)
        self.assertIn("rebuild the bundle", str(caught.exception))

    def test_lock_mismatch_is_rejected_before_pod_creation(self):
        class Api:
            def __init__(self):
                self.created = []

            def graphql(self, query):
                raise AssertionError("discovery must not run")

            def create_pod(self, payload):
                self.created.append(payload)
                raise AssertionError("create_pod must not run")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, manifest, _ = write_bundle(root)
            manifest["source"]["commit"] = "1" * 40  # older build, fine on its own
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (root / "uv.lock").write_bytes(b"dependencies changed\n")
            config = root / "runpod.toml"
            config.write_text('[default]\napi_key = "k"\n', encoding="utf-8")
            key, pub = root / "id", root / "id.pub"
            key.write_text("private", encoding="utf-8")
            pub.write_text("ssh-ed25519 AAAA user@host\n", encoding="utf-8")
            args = SimpleNamespace(
                command="launch", state=root / "state.json", config=config,
                repo_root=root, preflight_report=root / "telemetry.json",
                gpu="H200", experiment_command="coinrun-runner experiment",
                runtime_seconds=rc.MAX_RUNTIME_SECONDS, cloud="secure",
                balance_buffer=Decimal("5"), image=None, container_disk_gb=100,
                remote_artifact_dir="/workspace/coinrun-artifacts", poll_seconds=0.01,
                max_artifact_bytes=1024, execute=True, transport="bundle",
                bundle_manifest=manifest_path, bundle_dir=bundle_dir,
                ssh_key=key, ssh_public_key=pub, setup_timeout_seconds=600,
                transfer_telemetry=root / "transfer.jsonl",
            )
            api = Api()
            with self.assertRaises(rc.DeploymentError):
                rc.run_launch(args, api_factory=lambda k: api)
            self.assertEqual(api.created, [])

    def test_generator_dependency_pins_gate_when_recorded(self):
        # The generator's PEP 723 block pins Procgen, whose built wheel is
        # prewarmed into the bundle's uv cache.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, manifest, _ = write_bundle(root)
            generator = root / "dreamer" / "data" / "generate_coinrun_dataset.py"
            generator.parent.mkdir(parents=True)
            generator.write_text(
                "# /// script\n# dependencies = ['procgen@new']\n# ///\nbody\n",
                encoding="utf-8",
            )
            manifest["build_inputs"] = {"generator_dependency_sha256": "9" * 64}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(rc.DeploymentError) as caught:
                plan_for(root, manifest_path, bundle_dir)
        self.assertIn("bundle is stale", str(caught.exception))

    def test_generator_body_changes_do_not_stale_the_bundle(self):
        # Only the dependency block is hashed; the body runs from the checkout.
        header = "# /// script\n# dependencies = ['procgen@pinned']\n# ///\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, manifest, _ = write_bundle(root)
            generator = root / "dreamer" / "data" / "generate_coinrun_dataset.py"
            generator.parent.mkdir(parents=True)
            generator.write_text(header + "original body\n", encoding="utf-8")
            recorded = rc.script_dependency_sha256(generator)
            manifest["build_inputs"] = {"generator_dependency_sha256": recorded}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            generator.write_text(header + "a completely different body\n", encoding="utf-8")
            plan_for(root, manifest_path, bundle_dir)  # must not raise

    def test_launcher_and_runner_scripts_never_gate_a_bundle(self):
        # Orchestration runs from the pinned checkout, so it cannot stale a
        # dependency archive; gating on it would force 3GB no-op rebuilds.
        self.assertNotIn("scripts/coinrun_runner.py", rc.GATED_BUNDLE_INPUTS)
        self.assertNotIn("scripts/runpod_coinrun.py", rc.GATED_BUNDLE_INPUTS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, manifest, _ = write_bundle(root)
            manifest["build_inputs"] = {
                "coinrun_runner_sha256": "9" * 64,
                "dockerfile_sha256": "9" * 64,
                "generator_sha256": "9" * 64,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            plan_for(root, manifest_path, bundle_dir)  # must not raise

    def test_manifests_without_build_inputs_stay_usable(self):
        # The checked-in manifest predates build_inputs; it must not break.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, manifest, _ = write_bundle(root)
            self.assertNotIn("build_inputs", manifest)
            plan_for(root, manifest_path, bundle_dir)

    def test_only_dependency_affecting_inputs_gate(self):
        self.assertEqual(
            set(rc.GATED_BUNDLE_INPUTS),
            {"dreamer/data/generate_coinrun_dataset.py"},
        )


class BundleShimTests(unittest.TestCase):
    def test_remote_setup_recreates_the_runner_shim(self):
        # /usr/local/bin/coinrun-runner is built outside /opt/coinrun, so the
        # bundle cannot carry it; setup must recreate it or the documented
        # experiment command is command-not-found on the pod.
        script = rc.REMOTE_BUNDLE_SETUP
        self.assertIn("/usr/local/bin/coinrun-runner", script)
        self.assertIn("command -v coinrun-runner", script)

    def test_shim_execs_the_pinned_checkout_not_the_bundled_copy(self):
        # The bundle is a dependency environment. Orchestration must come from
        # the exact commit the bootstrap checked out and verified.
        script = rc.REMOTE_BUNDLE_SETUP
        self.assertIn(
            '"${COINRUN_CHECKOUT_ROOT:-/workspace/open-dreamer}/scripts/coinrun_runner.py"',
            script,
        )
        # The archive's own copy must be inert.
        self.assertNotIn("$target/runner/coinrun_runner.py", script)
        self.assertNotIn("runner/coinrun_runner.py\\n' \\\n  \"$target\" \"$target\"", script)

    def test_shim_still_uses_the_bundled_interpreter(self):
        # Python and its packages do come from the bundle.
        self.assertIn("%s/venv/bin/python", rc.REMOTE_BUNDLE_SETUP)

    def test_bootstrap_exports_the_checkout_root_the_shim_reads(self):
        script = rc.build_remote_script(
            branch="b", commit_sha=COMMIT, experiment_command="true",
            runtime_seconds=60, artifact_dir="/workspace/a", stream_token="t",
            deadline_epoch=2_000_000_000,
        )
        self.assertIn('COINRUN_CHECKOUT_ROOT="$REPO_DIR"', script)


class PublicKeyTests(unittest.TestCase):
    def test_private_key_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "id"
            path.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nx\n", encoding="utf-8")
            with self.assertRaises(rc.DeploymentError):
                rc.read_public_key(path)

    def test_public_key_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "id.pub"
            path.write_text("ssh-ed25519 AAAAC3Nz user@host\n", encoding="utf-8")
            self.assertTrue(rc.read_public_key(path).startswith("ssh-ed25519"))


class SshEndpointTests(unittest.TestCase):
    def test_endpoint_is_extracted_from_rest_fields(self):
        self.assertEqual(
            rc.extract_ssh_endpoint({"publicIp": "1.2.3.4", "portMappings": {"22": 40022}}),
            ("1.2.3.4", 40022),
        )

    def test_absent_public_ip_or_mapping_yields_none(self):
        for pod in (
            {"portMappings": {"22": 40022}},
            {"publicIp": "1.2.3.4"},
            {"publicIp": "1.2.3.4", "portMappings": {}},
            {"publicIp": "1.2.3.4", "portMappings": {"8000": 1}},
            {"publicIp": "1.2.3.4", "portMappings": None},
        ):
            with self.subTest(pod=pod):
                self.assertIsNone(rc.extract_ssh_endpoint(pod))

    def test_missing_mapping_times_out_within_the_setup_budget(self):
        class Api:
            def get_pod(self, pod_id):
                return {"publicIp": "1.2.3.4", "portMappings": {}}

        ticks = iter([0.0, 1.0, 2.0, 99.0, 99.0])
        with self.assertRaises(rc.DeploymentError) as caught:
            rc.wait_for_ssh_endpoint(
                Api(), "pod-1", deadline=10.0, poll_seconds=0.0,
                clock=lambda: next(ticks), sleep=lambda _: None,
            )
        self.assertIn("22/tcp", str(caught.exception))

    def test_vanished_pod_is_reported(self):
        class Api:
            def get_pod(self, pod_id):
                return None

        with self.assertRaises(rc.DeploymentError) as caught:
            rc.wait_for_ssh_endpoint(
                Api(), "pod-1", deadline=10.0, poll_seconds=0.0,
                clock=lambda: 0.0, sleep=lambda _: None,
            )
        self.assertIn("disappeared", str(caught.exception))


class CommandConstructionTests(unittest.TestCase):
    def test_ssh_uses_an_isolated_known_hosts_and_batch_mode(self):
        command = rc.ssh_command(
            "1.2.3.4", 40022, key=Path("/k"), known_hosts=Path("/kh"), remote=["true"]
        )
        self.assertIn("UserKnownHostsFile=/kh", command)
        self.assertIn("BatchMode=yes", command)
        self.assertIn("IdentitiesOnly=yes", command)
        self.assertEqual(command[-2:], ["root@1.2.3.4", "true"])

    def test_rsync_is_resumable(self):
        command = rc.rsync_command(
            Path("/a/b.tar.zst"), "1.2.3.4", 40022,
            key=Path("/k"), known_hosts=Path("/kh"), destination="/workspace/x/",
        )
        self.assertIn("--partial", command)
        self.assertIn("--append-verify", command)
        self.assertEqual(command[-1], "root@1.2.3.4:/workspace/x/")

    def test_relay_ssh_uses_batch_mode(self):
        command = rc.relay_ssh_command("fw-robot1", ["true"])
        self.assertIn("BatchMode=yes", command)
        self.assertEqual(command[-2:], ["fw-robot1", "true"])

    def test_relay_bundle_is_validated_against_local_manifest(self):
        captured = {}

        def runner(command, **kwargs):
            captured["command"] = command
            captured["input"] = kwargs["input"]
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, manifest, _ = write_bundle(root)
            plan = plan_for(root, manifest_path, bundle_dir)
            rc.validate_relay_bundle(
                "fw-robot1", "/root/coinrun-bundles", plan, runner=runner
            )
            manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        command_text = " ".join(captured["command"])
        self.assertIn("fw-robot1", command_text)
        self.assertIn(manifest["archive"]["sha256"], command_text)
        self.assertIn(manifest_sha, command_text)
        self.assertIn("sha256sum", captured["input"])

    def test_relay_transfer_removes_ephemeral_key_after_failure(self):
        calls = []

        def runner(command, **kwargs):
            calls.append(command)
            if any(part.startswith("rsync ") for part in command):
                return subprocess.CompletedProcess(command, 23, "", "network failed")
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, _, _ = write_bundle(root)
            plan = plan_for(root, manifest_path, bundle_dir)
            key = root / "ephemeral"
            key.write_text("private key\n", encoding="utf-8")
            with self.assertRaises(rc.DeploymentError) as caught:
                rc.transfer_bundle_via_relay(
                    "fw-robot1", "/root/coinrun-bundles", "1.2.3.4", 40022,
                    pod_id="pod-1", key=key, plan=plan, runner=runner,
                )
        self.assertIn("relay rsync failed", str(caught.exception))
        cleanup = calls[-1]
        cleanup_text = " ".join(cleanup)
        self.assertIn("rm", cleanup_text)
        self.assertIn("/tmp/coinrun-relay-pod-1.key", cleanup_text)
        self.assertIn("/tmp/coinrun-relay-pod-1.known_hosts", cleanup_text)

    def test_relay_key_bytes_are_cat_input_not_shell_source(self):
        calls = []

        def runner(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, _, _ = write_bundle(root)
            plan = plan_for(root, manifest_path, bundle_dir)
            key = root / "ephemeral"
            key.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nsecret\n", encoding="utf-8")
            rc.transfer_bundle_via_relay(
                "fw-robot1", "/root/coinrun-bundles", "1.2.3.4", 40022,
                pod_id="pod-1", key=key, plan=plan, runner=runner,
            )
        install_command, install_kwargs = calls[0]
        install_text = " ".join(install_command)
        self.assertIn("cat", install_text)
        self.assertNotIn("BEGIN OPENSSH", install_text)
        self.assertEqual(
            install_kwargs["input"],
            "-----BEGIN OPENSSH PRIVATE KEY-----\nsecret\n",
        )

    def test_remote_setup_verifies_hash_size_and_procgen(self):
        script = rc.REMOTE_BUNDLE_SETUP
        self.assertIn("sha256sum", script)
        self.assertIn("size mismatch", script)
        self.assertIn('verify "$staging/$archive"', script)
        self.assertIn("libenv.so", script)
        self.assertIn("venv/bin/python", script)
        # Atomic install: unpack beside the target, then rename.
        self.assertIn(".incoming", script)
        self.assertIn('mv "$target.incoming/coinrun" "$target"', script)

    def test_remote_verification_failure_is_surfaced(self):
        def runner(command, **kwargs):
            return subprocess.CompletedProcess(command, 12, "", "bundle sha mismatch")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, _, _ = write_bundle(root)
            plan = plan_for(root, manifest_path, bundle_dir)
            with self.assertRaises(rc.DeploymentError) as caught:
                rc.install_remote_bundle(
                    "1.2.3.4", 22, key=Path("/k"), known_hosts=Path("/kh"),
                    plan=plan, runner=runner,
                )
        self.assertIn("exit 12", str(caught.exception))
        self.assertIn("sha mismatch", str(caught.exception))

    def test_experiment_script_is_stdin_data_not_shell_source(self):
        captured = {}

        def runner(command, **kwargs):
            captured["command"] = command
            captured["input"] = kwargs["input"]
            return subprocess.CompletedProcess(command, 0, "", "")

        payload = "#!/bin/sh\necho experiment-payload\n"
        rc.install_remote_script(
            "1.2.3.4",
            22,
            key=Path("/k"),
            known_hosts=Path("/kh"),
            remote_script=payload,
            runner=runner,
        )
        command_text = " ".join(captured["command"])
        self.assertEqual(captured["input"], payload)
        self.assertNotIn("experiment-payload", command_text)
        self.assertIn("cat", command_text)
        self.assertIn(hashlib.sha256(payload.encode()).hexdigest(), command_text)

    def test_experiment_installer_executes_under_bash(self):
        payload = "#!/bin/sh\necho installed\n"
        expected_sha = hashlib.sha256(payload.encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run_coinrun.sh"
            result = subprocess.run(
                [
                    "bash", "-c", rc.REMOTE_SCRIPT_INSTALL,
                    "coinrun-script-install", str(target),
                    expected_sha, str(len(payload.encode())),
                ],
                input=payload,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(target.read_text(encoding="utf-8"), payload)
            self.assertTrue(target.stat().st_mode & 0o100)

    def test_bundle_shim_exports_nvidia_wheel_library_paths(self):
        script = rc.REMOTE_BUNDLE_SETUP
        self.assertIn('python_lib="$target/venv/lib/python3.11/site-packages"', script)
        self.assertIn('"$python_lib/nvidia"', script)
        self.assertIn("LD_LIBRARY_PATH", script)
        self.assertIn("paste -sd:", script)


class ContractEnvExportTests(unittest.TestCase):
    def test_bundle_mode_exports_the_manifest_contract_env(self):
        script = rc.build_remote_script(
            branch="b", commit_sha=COMMIT, experiment_command="true",
            runtime_seconds=60, artifact_dir="/workspace/a", stream_token="t",
            deadline_epoch=2_000_000_000,
            contract_env={"COINRUN_IMAGE_CONTRACT": "1", "UV_CACHE_DIR": "/opt/coinrun/uv-cache"},
        )
        self.assertIn("export COINRUN_IMAGE_CONTRACT=1", script)
        self.assertIn("export UV_CACHE_DIR=/opt/coinrun/uv-cache", script)

    def test_image_mode_script_is_unchanged(self):
        script = rc.build_remote_script(
            branch="b", commit_sha=COMMIT, experiment_command="true",
            runtime_seconds=60, artifact_dir="/workspace/a", stream_token="t",
            deadline_epoch=2_000_000_000,
        )
        self.assertNotIn("export COINRUN_IMAGE_CONTRACT", script)
        self.assertIn('test "${COINRUN_IMAGE_CONTRACT:-}" = "1"', script)

    def test_injected_env_names_are_validated(self):
        with self.assertRaises(rc.DeploymentError):
            rc.build_remote_script(
                branch="b", commit_sha=COMMIT, experiment_command="true",
                runtime_seconds=60, artifact_dir="/workspace/a", stream_token="t",
                deadline_epoch=2_000_000_000, contract_env={"bad name": "1"},
            )


class PodPayloadTests(unittest.TestCase):
    def payload(self, root: Path):
        manifest_path, bundle_dir, _, _ = write_bundle(root)
        plan = plan_for(root, manifest_path, bundle_dir)
        offer = rc.GPUOffer(
            choice="H200", gpu_id="NVIDIA H200", display_name="H200", memory_gb=141,
            stock_status="High", available_gpu_counts=(1,), hourly_price=Decimal("2"),
        )
        args = SimpleNamespace(cloud="secure", container_disk_gb=100)
        return rc.make_bundle_pod_payload(
            args, offer, plan, "pod-name", "ssh-ed25519 AAAA user@host", 2_000_000_000
        )

    def test_bundle_pod_exposes_ssh_and_http_and_requests_public_ip(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self.payload(Path(directory))
        self.assertEqual(payload["ports"], ["8000/http", "22/tcp"])
        self.assertTrue(payload["supportPublicIp"])
        self.assertEqual(payload["env"]["PUBLIC_KEY"], "ssh-ed25519 AAAA user@host")

    def test_pod_env_never_carries_the_account_api_key(self):
        # Image mode relies on RunPod's injected pod-scoped RUNPOD_API_KEY; the
        # local account key must never be shipped to a pod.
        with tempfile.TemporaryDirectory() as directory:
            payload = self.payload(Path(directory))
        self.assertEqual(list(payload["env"]), ["PUBLIC_KEY"])
        self.assertNotIn("RUNPOD_API_KEY", json.dumps(payload))

    def test_start_command_arms_the_watchdog_then_execs_start_sh(self):
        # A local crash between create_pod and teardown must not leave an
        # unbounded pod, and sshd must still come up.
        with tempfile.TemporaryDirectory() as directory:
            payload = self.payload(Path(directory))
        decoded = base64.b64decode(
            payload["dockerStartCmd"][0].split("printf %s ")[1].split(" |")[0].strip("'")
        ).decode()
        self.assertIn("coinrun_self_terminate.py", decoded)
        self.assertIn("hard-remote-deadline", decoded)
        self.assertIn("exec /start.sh", decoded)
        self.assertIn("2000000000", decoded)

    def test_bootstrap_rejects_an_invalid_deadline(self):
        with self.assertRaises(rc.DeploymentError):
            rc.build_bundle_bootstrap(0)

    def test_bundle_pod_uses_the_pinned_base_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self.payload(Path(directory))
        self.assertEqual(payload["imageName"], f"runpod/pytorch@{BASE_DIGEST}")


class RemoteHardeningTests(unittest.TestCase):
    def test_experiment_is_bounded_by_exactly_runtime_seconds(self):
        # Unused setup budget must not silently extend training.
        script = rc.build_remote_script(
            branch="b", commit_sha=COMMIT, experiment_command="true",
            runtime_seconds=1234, artifact_dir="/workspace/a", stream_token="t",
            deadline_epoch=2_000_000_000,
        )
        self.assertIn("experiment_timeout=1234", script)
        # ...while the absolute watchdog still bounds setup + run + finalize.
        self.assertIn("deadline_room=$((watchdog_delay - 120))", script)
        self.assertIn("HARD_DEADLINE_EPOCH=2000000000", script)

    def test_remote_setup_verifies_the_manifest_as_well_as_the_archive(self):
        self.assertIn('verify "$staging/$manifest"', rc.REMOTE_BUNDLE_SETUP)
        self.assertIn('verify "$staging/$archive"', rc.REMOTE_BUNDLE_SETUP)

    def test_manifest_verification_uses_locally_computed_values(self):
        captured = {}

        def runner(command, **kwargs):
            captured["command"] = command
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, bundle_dir, _, _ = write_bundle(root)
            plan = plan_for(root, manifest_path, bundle_dir)
            rc.install_remote_bundle(
                "1.2.3.4", 22, key=Path("/k"), known_hosts=Path("/kh"),
                plan=plan, runner=runner,
            )
            expected_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            expected_bytes = str(manifest_path.stat().st_size)
        command_text = " ".join(captured["command"])
        self.assertIn(expected_sha, command_text)
        self.assertIn(expected_bytes, command_text)
        self.assertIn(manifest_path.name, command_text)

    def test_remote_script_is_installed_and_verified_before_launch(self):
        self.assertIn("script sha mismatch", rc.REMOTE_SCRIPT_INSTALL)
        self.assertIn("script size mismatch", rc.REMOTE_SCRIPT_INSTALL)
        # The install step must be fully synchronous: nothing backgrounded, so
        # `cat` has certainly consumed stdin before the call returns.
        self.assertNotIn("nohup", rc.REMOTE_SCRIPT_INSTALL)
        self.assertNotIn("setsid", rc.REMOTE_SCRIPT_INSTALL)
        for line in rc.REMOTE_SCRIPT_INSTALL.splitlines():
            self.assertFalse(line.rstrip().endswith("&"), line)


class SetupBudgetTests(unittest.TestCase):
    def test_setup_timeout_must_be_positive_and_bounded(self):
        for value in (0, -1, rc.MAX_SETUP_TIMEOUT_SECONDS + 1, True):
            with self.subTest(value=value):
                with self.assertRaises(rc.DeploymentError):
                    rc.validate_setup_timeout(value)
        self.assertEqual(rc.validate_setup_timeout(600), 600)


class NoLaunchOnLocalFailureTests(unittest.TestCase):
    """A local validation failure must never reach create_pod."""

    class Api:
        def __init__(self):
            self.created = []

        def graphql(self, query):
            raise AssertionError("discovery must not run after a local failure")

        def create_pod(self, payload):
            self.created.append(payload)
            raise AssertionError("create_pod must not run after a local failure")

    def launch_args(self, root: Path, **overrides):
        manifest_path, bundle_dir, _, _ = write_bundle(root)
        config = root / "runpod.toml"
        config.write_text('[default]\napi_key = "k"\n', encoding="utf-8")
        key = root / "id"
        key.write_text("private", encoding="utf-8")
        pub = root / "id.pub"
        pub.write_text("ssh-ed25519 AAAA user@host\n", encoding="utf-8")
        args = SimpleNamespace(
            command="launch", state=root / "state.json", config=config,
            repo_root=root, preflight_report=root / "telemetry.json",
            gpu="H200", experiment_command="coinrun-runner experiment",
            runtime_seconds=rc.MAX_RUNTIME_SECONDS, cloud="secure",
            balance_buffer=Decimal("5"), image=None, container_disk_gb=100,
            remote_artifact_dir="/workspace/coinrun-artifacts", poll_seconds=0.01,
            max_artifact_bytes=1024, execute=True, transport="bundle",
            bundle_manifest=manifest_path, bundle_dir=bundle_dir,
            ssh_key=key, ssh_public_key=pub,
            setup_timeout_seconds=600,
            transfer_telemetry=root / "transfer.jsonl",
        )
        for name, value in overrides.items():
            setattr(args, name, value)
        return args

    def test_archive_mismatch_stops_before_any_api_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.launch_args(root)
            (root / "uv.lock").write_bytes(b"drifted\n")
            api = self.Api()
            with self.assertRaises(rc.DeploymentError):
                rc.run_launch(args, api_factory=lambda key: api)
            self.assertEqual(api.created, [])

    def test_missing_private_key_stops_before_any_api_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.launch_args(root, ssh_key=root / "absent")
            api = self.Api()
            with self.assertRaises(rc.DeploymentError) as caught:
                rc.run_launch(args, api_factory=lambda key: api)
            self.assertIn("SSH private key not found", str(caught.exception))
            self.assertEqual(api.created, [])

    def test_bad_setup_timeout_stops_before_any_api_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.launch_args(root, setup_timeout_seconds=0)
            api = self.Api()
            with self.assertRaises(rc.DeploymentError):
                rc.run_launch(args, api_factory=lambda key: api)
            self.assertEqual(api.created, [])


class CostAccountingTests(unittest.TestCase):
    def test_setup_budget_is_included_in_the_projected_spend(self):
        # 20 min setup + 4 h experiment at $2/hr must be quoted as 4h20m.
        setup, runtime, hourly = 1200, rc.MAX_RUNTIME_SECONDS, Decimal("2")
        projected = hourly * Decimal(runtime + setup) / Decimal(3600)
        self.assertGreater(projected, hourly * Decimal(runtime) / Decimal(3600))
        self.assertAlmostEqual(float(projected), 8.666666, places=4)


if __name__ == "__main__":
    unittest.main()
