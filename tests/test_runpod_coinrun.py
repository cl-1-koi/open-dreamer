from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "runpod_coinrun.py"
SPEC = importlib.util.spec_from_file_location("runpod_coinrun", SCRIPT_PATH)
runpod_coinrun = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runpod_coinrun
assert SPEC.loader is not None
SPEC.loader.exec_module(runpod_coinrun)


COMMIT = "a" * 40
BRANCH = "experiment/coinrun-reconstruction-20260726"


def completed(command, stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(
        command, returncode, stdout=stdout, stderr=stderr
    )


def git_runner(*, dirty=False, remote_sha=COMMIT):
    def run(command, **kwargs):
        args = tuple(command[1:])
        values = {
            ("status", "--porcelain"): " M tracked.py" if dirty else "",
            ("branch", "--show-current"): BRANCH,
            ("rev-parse", "HEAD"): COMMIT,
            ("remote", "get-url", "origin"): (
                "https://github.com/cl-1-koi/open-dreamer.git"
            ),
            (
                "ls-remote",
                runpod_coinrun.GITHUB_REPO_URL,
                f"refs/heads/{BRANCH}",
            ): f"{remote_sha}\trefs/heads/{BRANCH}",
        }
        if args not in values:
            return completed(command, stderr=f"unexpected command: {args}", returncode=2)
        return completed(command, stdout=values[args])

    return mock.Mock(side_effect=run)


def passed_report(path: Path, commit=COMMIT, status="passed"):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": status,
                "git": {
                    "commit": commit,
                    "branch": BRANCH,
                    "dirty": False,
                },
                "stages": [{"name": "bounded_smoke", "status": "passed"}],
            }
        ),
        encoding="utf-8",
    )


def discovery_payload(
    *,
    balance=100.0,
    pods=None,
    stock="High",
    price=2.0,
):
    data = {
        "myself": {
            "clientBalance": balance,
            "currentSpendPerHr": 0,
            "pods": pods or [],
        }
    }
    memory = {"H100": 80, "H200": 141, "B200": 180}
    for choice, gpu_id in runpod_coinrun.GPU_IDS.items():
        data[choice.lower()] = [
            {
                "id": gpu_id,
                "displayName": choice,
                "memoryInGb": memory[choice],
                "lowestPrice": {
                    "stockStatus": stock,
                    "uninterruptablePrice": price,
                    "availableGpuCounts": [1],
                },
            }
        ]
    return data


class FakeAPI:
    def __init__(
        self,
        *,
        discovery=None,
        pod=None,
        create_error=None,
    ):
        self.discovery = discovery or discovery_payload()
        self.pod = pod
        self.create_error = create_error
        self.create_calls = []
        self.delete_calls = []
        self.graphql_calls = []

    def graphql(self, query):
        self.graphql_calls.append(query)
        return self.discovery

    def create_pod(self, payload):
        self.create_calls.append(payload)
        if self.create_error is not None:
            raise self.create_error
        self.pod = {
            "id": "pod-123",
            "name": payload["name"],
            "desiredStatus": "RUNNING",
            "costPerHr": 2.0,
        }
        return dict(self.pod)

    def get_pod(self, pod_id):
        if self.pod is None or self.pod.get("id") != pod_id:
            return None
        return dict(self.pod)

    def delete_pod(self, pod_id):
        self.delete_calls.append(pod_id)
        self.pod = None
        return True


def launch_args(root: Path, *, execute=False):
    config = root / "runpod.toml"
    config.write_text('[default]\napi_key = "top-secret"\n', encoding="utf-8")
    report = root / "telemetry.json"
    passed_report(report)
    return SimpleNamespace(
        command="launch",
        state=root / "state.json",
        config=config,
        repo_root=root,
        preflight_report=report,
        gpu="H200",
        experiment_command=(
            "uv sync --frozen && uv run scripts/coinrun_preflight.py "
            "--output-dir=$COINRUN_ARTIFACT_DIR/run"
        ),
        runtime_seconds=runpod_coinrun.MAX_RUNTIME_SECONDS,
        cloud="secure",
        balance_buffer=Decimal("5"),
        image=runpod_coinrun.DEFAULT_IMAGE,
        container_disk_gb=100,
        remote_artifact_dir="/workspace/coinrun-artifacts",
        poll_seconds=0.01,
        max_artifact_bytes=1024 * 1024,
        execute=execute,
    )


class FakeResponse:
    def __init__(self, body, headers=None):
        self.body = body
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, size=-1):
        if size < 0:
            value, self.body = self.body, b""
            return value
        value, self.body = self.body[:size], self.body[size:]
        return value


class CredentialAndHTTPTests(unittest.TestCase):
    def test_reads_default_toml_key_without_printing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                '[default]\napi_key = "do-not-log-me"\n',
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                key = runpod_coinrun.load_api_key(path)

            self.assertEqual(key, "do-not-log-me")
            self.assertNotIn(key, output.getvalue())

    def test_http_failure_redacts_api_key(self):
        key = "credential-in-network-error"

        def opener(request, timeout):
            raise urllib.error.URLError(f"connection failed for {key}")

        api = runpod_coinrun.RunPodAPI(key, opener=opener)
        with self.assertRaises(runpod_coinrun.APIError) as context:
            api.graphql("query { myself { clientBalance } }")

        self.assertNotIn(key, str(context.exception))
        self.assertIn("<redacted>", str(context.exception))

    def test_log_stream_uses_bearer_token_and_offset(self):
        seen = {}

        def opener(request, timeout):
            seen["url"] = request.full_url
            seen["authorization"] = request.get_header("Authorization")
            return FakeResponse(b"line\n", {"X-Next-Offset": "5"})

        result = runpod_coinrun.fetch_log_chunk(
            "pod-1", 0, "stream-secret", opener=opener
        )

        self.assertEqual(result, (b"line\n", 5))
        self.assertEqual(seen["authorization"], "Bearer stream-secret")
        self.assertTrue(seen["url"].endswith("/log?offset=0"))


class GitAndPreflightGateTests(unittest.TestCase):
    def test_accepts_only_clean_pushed_cl1koi_tip(self):
        state = runpod_coinrun.validate_git_preflight(
            Path("/repo"), runner=git_runner()
        )
        self.assertEqual(state.commit_sha, COMMIT)
        self.assertEqual(state.branch, BRANCH)

    def test_refuses_dirty_branch(self):
        with self.assertRaisesRegex(runpod_coinrun.DeploymentError, "dirty"):
            runpod_coinrun.validate_git_preflight(
                Path("/repo"), runner=git_runner(dirty=True)
            )

    def test_refuses_unpushed_or_diverged_branch(self):
        with self.assertRaisesRegex(
            runpod_coinrun.DeploymentError, "not the pushed tip"
        ):
            runpod_coinrun.validate_git_preflight(
                Path("/repo"), runner=git_runner(remote_sha="b" * 40)
            )

    def test_report_must_exist_pass_and_match_clean_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.json"
            with self.assertRaisesRegex(
                runpod_coinrun.DeploymentError, "absent"
            ):
                runpod_coinrun.validate_preflight_report(missing, COMMIT)

            failed = root / "failed.json"
            passed_report(failed, status="failed")
            with self.assertRaisesRegex(
                runpod_coinrun.DeploymentError, "has not passed"
            ):
                runpod_coinrun.validate_preflight_report(failed, COMMIT)

            mismatched = root / "mismatch.json"
            passed_report(mismatched, commit="b" * 40)
            with self.assertRaisesRegex(
                runpod_coinrun.DeploymentError, "does not match"
            ):
                runpod_coinrun.validate_preflight_report(mismatched, COMMIT)


class DiscoveryAndRemoteBootstrapTests(unittest.TestCase):
    def test_queries_balance_and_all_bounded_gpu_offers(self):
        api = FakeAPI()
        result = runpod_coinrun.discover_runpod(api, "secure")

        self.assertEqual(result.balance, Decimal("100.0"))
        self.assertEqual(set(result.offers), {"H100", "H200", "B200"})
        self.assertTrue(all(offer.available for offer in result.offers.values()))
        query = api.graphql_calls[0]
        for gpu_id in runpod_coinrun.GPU_IDS.values():
            self.assertIn(gpu_id, query)
        self.assertIn("secureCloud: true", query)

    def test_missing_price_is_an_unavailable_offer_not_a_parse_crash(self):
        payload = discovery_payload()
        payload["b200"][0]["lowestPrice"] = None
        result = runpod_coinrun.discover_runpod(
            FakeAPI(discovery=payload), "secure"
        )
        self.assertFalse(result.offers["B200"].available)

    def test_remote_script_pins_github_branch_commit_and_both_bounds(self):
        script = runpod_coinrun.build_remote_script(
            branch=BRANCH,
            commit_sha=COMMIT,
            experiment_command="uv run train.py",
            runtime_seconds=runpod_coinrun.MAX_RUNTIME_SECONDS,
            artifact_dir="/workspace/coinrun-artifacts",
            stream_token="stream-token",
            deadline_epoch=2_000_000_000,
        )

        self.assertIn(
            f"git clone --single-branch --branch {BRANCH} --no-checkout",
            script,
        )
        self.assertIn(runpod_coinrun.GITHUB_REPO_URL, script)
        self.assertIn(f"git checkout --detach {COMMIT}", script)
        self.assertIn("HARD_DEADLINE_EPOCH=2000000000", script)
        self.assertIn(
            '"$watchdog_delay" hard-remote-deadline',
            script,
        )
        self.assertIn(
            'timeout --signal=TERM --kill-after=60 "$experiment_timeout"',
            script,
        )
        self.assertIn("unset RUNPOD_API_KEY COINRUN_STREAM_TOKEN", script)
        self.assertIn("/tmp/coinrun-experiment-failed", script)
        self.assertIn("artifact-manifest.json", runpod_coinrun.REMOTE_WRITE_MANIFEST)
        self.assertNotIn("rsync", script)

    def test_remote_runtime_cannot_exceed_four_hours(self):
        with self.assertRaisesRegex(
            runpod_coinrun.DeploymentError, "Remote runtime"
        ):
            runpod_coinrun.build_remote_script(
                branch=BRANCH,
                commit_sha=COMMIT,
                experiment_command="true",
                runtime_seconds=runpod_coinrun.MAX_RUNTIME_SECONDS + 1,
                artifact_dir="/workspace/artifacts",
                stream_token="token",
                deadline_epoch=2_000_000_000,
            )


class LaunchFlowTests(unittest.TestCase):
    def test_dry_run_discovers_price_but_never_creates_pod_or_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = launch_args(root)
            api = FakeAPI()

            result = runpod_coinrun.run_launch(
                args,
                api_factory=lambda key: api,
                git_runner=git_runner(),
            )

            self.assertEqual(result, 0)
            self.assertEqual(api.create_calls, [])
            self.assertFalse(args.state.exists())

    def test_refuses_balance_below_projected_cost_plus_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = launch_args(root)
            api = FakeAPI(discovery=discovery_payload(balance=10, price=2))

            with self.assertRaisesRegex(
                runpod_coinrun.DeploymentError, "below projected"
            ):
                runpod_coinrun.run_launch(
                    args,
                    api_factory=lambda key: api,
                    git_runner=git_runner(),
                )

            self.assertEqual(api.create_calls, [])

    def test_refuses_active_state_before_any_api_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = launch_args(root)
            runpod_coinrun.write_json_atomic(
                args.state,
                {
                    "phase": "running",
                    "pod_id": "existing",
                    "termination_status": "not_requested",
                },
            )
            api_factory = mock.Mock()

            with self.assertRaisesRegex(
                runpod_coinrun.DeploymentError, "Active RunPod state"
            ):
                runpod_coinrun.run_launch(
                    args,
                    api_factory=api_factory,
                    git_runner=git_runner(),
                )

            api_factory.assert_not_called()

    def test_execute_persists_required_state_streams_and_terminates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = launch_args(root, execute=True)
            api = FakeAPI()
            observed = {}

            def monitor(api_arg, state_path, **kwargs):
                state = runpod_coinrun.read_json(state_path)
                observed.update(state)
                return {
                    "complete": True,
                    "returncode": 0,
                    "commit_sha": state["commit_sha"],
                }

            result = runpod_coinrun.run_launch(
                args,
                api_factory=lambda key: api,
                git_runner=git_runner(),
                monitor=monitor,
                clock=mock.Mock(return_value=100.0),
                now=lambda: datetime(2026, 7, 27, tzinfo=timezone.utc),
            )

            self.assertEqual(result, 0)
            self.assertEqual(api.delete_calls, ["pod-123"])
            state = runpod_coinrun.read_json(args.state)
            self.assertEqual(state["pod_id"], "pod-123")
            self.assertEqual(state["commit_sha"], COMMIT)
            self.assertEqual(state["launch_time_utc"], "2026-07-27T00:00:00+00:00")
            self.assertEqual(state["hourly_price"], 2.0)
            self.assertEqual(state["max_projected_spend"], 8.0)
            self.assertEqual(state["termination_status"], "terminated")
            self.assertEqual(len(state["stream_token"]) > 20, True)
            self.assertEqual(
                api.create_calls[0]["gpuTypeIds"],
                [runpod_coinrun.GPU_IDS["H200"]],
            )
            self.assertNotIn("top-secret", json.dumps(state))
            self.assertEqual(observed["phase"], "running")

    def test_monitor_failure_still_terminates_pod(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = launch_args(root, execute=True)
            api = FakeAPI()

            def monitor(*unused_args, **unused_kwargs):
                raise runpod_coinrun.DeploymentError("stream failed")

            with self.assertRaisesRegex(
                runpod_coinrun.DeploymentError, "stream failed"
            ):
                runpod_coinrun.run_launch(
                    args,
                    api_factory=lambda key: api,
                    git_runner=git_runner(),
                    monitor=monitor,
                )

            self.assertEqual(api.delete_calls, ["pod-123"])
            state = runpod_coinrun.read_json(args.state)
            self.assertEqual(state["termination_status"], "terminated")
            self.assertEqual(
                state["termination_reason"], "launch-or-experiment-failure"
            )

    def test_ambiguous_create_failure_stays_active_for_reconciliation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = launch_args(root, execute=True)
            api = FakeAPI(
                create_error=runpod_coinrun.APIError("network outcome unknown")
            )

            with self.assertRaises(runpod_coinrun.APIError):
                runpod_coinrun.run_launch(
                    args,
                    api_factory=lambda key: api,
                    git_runner=git_runner(),
                )

            state = runpod_coinrun.read_json(args.state)
            self.assertEqual(state["phase"], "launch_unknown")
            self.assertTrue(runpod_coinrun.state_is_active(state))
            self.assertEqual(
                state["termination_status"], "reconciliation_required"
            )


class LifecycleAndWatchdogTests(unittest.TestCase):
    def test_stop_is_idempotent_when_state_is_missing_or_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                state=root / "state.json",
                config=root / "missing.toml",
                execute=True,
                reason="test",
            )
            api_factory = mock.Mock()
            self.assertEqual(
                runpod_coinrun.run_stop(args, api_factory=api_factory), 0
            )
            runpod_coinrun.write_json_atomic(
                args.state,
                {
                    "phase": "terminated",
                    "pod_id": "pod-1",
                    "termination_status": "terminated",
                },
            )
            self.assertEqual(
                runpod_coinrun.run_stop(args, api_factory=api_factory), 0
            )
            api_factory.assert_not_called()

    def test_stop_defaults_to_dry_run_and_execute_deletes_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.toml"
            config.write_text('[default]\napi_key="secret"\n', encoding="utf-8")
            state_path = root / "state.json"
            state = {
                "phase": "running",
                "pod_id": "pod-1",
                "pod_name": "coinrun-reconstruction-test",
                "termination_status": "not_requested",
            }
            runpod_coinrun.write_json_atomic(state_path, state)
            api = FakeAPI(
                pod={
                    "id": "pod-1",
                    "name": state["pod_name"],
                    "desiredStatus": "RUNNING",
                }
            )
            args = SimpleNamespace(
                state=state_path,
                config=config,
                execute=False,
                reason="operator",
            )

            self.assertEqual(
                runpod_coinrun.run_stop(args, api_factory=lambda key: api), 0
            )
            self.assertEqual(api.delete_calls, [])
            args.execute = True
            self.assertEqual(
                runpod_coinrun.run_stop(args, api_factory=lambda key: api), 0
            )
            self.assertEqual(api.delete_calls, ["pod-1"])
            self.assertEqual(
                runpod_coinrun.read_json(state_path)["termination_status"],
                "terminated",
            )
            self.assertEqual(
                runpod_coinrun.run_stop(args, api_factory=lambda key: api), 0
            )
            self.assertEqual(api.delete_calls, ["pod-1"])

    def test_status_automatically_terminates_an_expired_active_pod(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.toml"
            config.write_text('[default]\napi_key="secret"\n', encoding="utf-8")
            state_path = root / "state.json"
            deadline = datetime(2026, 7, 27, tzinfo=timezone.utc)
            state = {
                "phase": "running",
                "pod_id": "pod-1",
                "pod_name": "coinrun-reconstruction-test",
                "hard_deadline_utc": deadline.isoformat(),
                "termination_status": "not_requested",
            }
            runpod_coinrun.write_json_atomic(state_path, state)
            api = FakeAPI(
                pod={
                    "id": "pod-1",
                    "name": state["pod_name"],
                    "desiredStatus": "RUNNING",
                }
            )
            args = SimpleNamespace(state=state_path, config=config)

            result = runpod_coinrun.run_status(
                args,
                api_factory=lambda key: api,
                now=lambda: deadline + timedelta(seconds=1),
            )

            self.assertEqual(result, 0)
            self.assertEqual(api.delete_calls, ["pod-1"])
            self.assertEqual(
                runpod_coinrun.read_json(state_path)["termination_status"],
                "terminated",
            )

    def test_local_monitor_refuses_to_poll_after_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            runpod_coinrun.write_json_atomic(
                state_path,
                {
                    "pod_id": "pod-1",
                    "stream_token": "token",
                    "commit_sha": COMMIT,
                },
            )
            api = mock.Mock()
            clock = mock.Mock(return_value=10.0)

            with self.assertRaises(runpod_coinrun.LocalDeadlineExceeded):
                runpod_coinrun.monitor_pod(
                    api,
                    state_path,
                    deadline=10.0,
                    poll_seconds=1,
                    max_artifact_bytes=1024,
                    clock=clock,
                    sleeper=mock.Mock(),
                )

            api.get_pod.assert_not_called()


if __name__ == "__main__":
    unittest.main()
