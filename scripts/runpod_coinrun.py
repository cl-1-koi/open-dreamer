#!/usr/bin/env python3
"""Bounded, fail-closed RunPod launcher for the CoinRun reconstruction sprint."""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence


MAX_RUNTIME_SECONDS = 4 * 60 * 60
MIN_RUNTIME_SECONDS = 5 * 60
REMOTE_FINALIZATION_SECONDS = 120
PROXY_STARTUP_GRACE_SECONDS = 20 * 60
DEFAULT_CONFIG = Path.home() / ".runpod" / "config.toml"
GITHUB_REPO_URL = "https://github.com/cl-1-koi/open-dreamer.git"
POD_NAME_PREFIX = "coinrun-reconstruction-"
GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_URL = "https://rest.runpod.io/v1"
DEFAULT_IMAGE = (
    "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
)
DEFAULT_STATE = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "runpod_coinrun"
    / "state.json"
)
GPU_IDS = {
    "H100": "NVIDIA H100 80GB HBM3",
    "H200": "NVIDIA H200",
    "B200": "NVIDIA B200",
}
TERMINAL_TERMINATION_STATES = {"pod_absent", "terminated"}
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class DeploymentError(RuntimeError):
    """A fail-closed deployment or lifecycle error."""


class APIError(DeploymentError):
    """A sanitized RunPod API error."""


class LocalDeadlineExceeded(DeploymentError):
    """The local four-hour watchdog expired."""


@dataclass(frozen=True)
class GitState:
    branch: str
    commit_sha: str
    remote_url: str


@dataclass(frozen=True)
class GPUOffer:
    choice: str
    gpu_id: str
    display_name: str
    memory_gb: int
    stock_status: str
    available_gpu_counts: tuple[int, ...]
    hourly_price: Decimal

    @property
    def available(self) -> bool:
        return (
            self.stock_status.lower() != "none"
            and (
                not self.available_gpu_counts
                or 1 in self.available_gpu_counts
            )
            and self.hourly_price > 0
        )


@dataclass(frozen=True)
class Discovery:
    balance: Decimal
    current_spend_per_hour: Decimal
    pods: tuple[dict[str, Any], ...]
    offers: dict[str, GPUOffer]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _decimal(value: Any, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise DeploymentError(f"RunPod returned invalid {field}: {value!r}") from exc
    if not result.is_finite():
        raise DeploymentError(f"RunPod returned nonfinite {field}")
    return result


def _money(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.01')):.2f}"


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DeploymentError(f"State file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DeploymentError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DeploymentError(f"Expected a JSON object in {path}")
    return value


@contextlib.contextmanager
def state_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def update_state(path: Path, **updates: Any) -> dict[str, Any]:
    with state_lock(path):
        state = read_json(path)
        state.update(updates)
        write_json_atomic(path, state)
    return state


def state_is_active(state: dict[str, Any]) -> bool:
    phase = state.get("phase")
    if phase in {"launching", "launch_unknown"}:
        return True
    return bool(
        state.get("pod_id")
        and state.get("termination_status") not in TERMINAL_TERMINATION_STATES
    )


def ensure_no_active_state(path: Path) -> None:
    if not path.exists():
        return
    with state_lock(path):
        state = read_json(path)
        if state_is_active(state):
            pod = state.get("pod_id") or state.get("pod_name") or "unknown"
            raise DeploymentError(
                f"Active RunPod state already exists at {path} ({pod}); "
                "run status or stop before launching"
            )


def load_api_key(path: Path) -> str:
    try:
        with path.expanduser().open("rb") as handle:
            config = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise DeploymentError(f"RunPod config not found: {path.expanduser()}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise DeploymentError(f"RunPod config is invalid TOML: {path.expanduser()}") from exc

    default = config.get("default", {})
    key = default.get("api_key") if isinstance(default, dict) else None
    if not key:
        key = config.get("api_key")
    if not isinstance(key, str) or not key.strip():
        raise DeploymentError(
            f"RunPod API key missing from {path.expanduser()}"
        )
    return key.strip()


class RunPodAPI:
    """Small stdlib-only RunPod client that never includes credentials in errors."""

    def __init__(
        self,
        api_key: str,
        *,
        opener: Callable[..., Any] = urllib.request.urlopen,
        timeout_seconds: float = 30.0,
    ):
        self._api_key = api_key
        self._opener = opener
        self._timeout_seconds = timeout_seconds

    def _request_json(
        self,
        request: urllib.request.Request,
        *,
        allow_not_found: bool = False,
    ) -> Any:
        try:
            with self._opener(request, timeout=self._timeout_seconds) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            if allow_not_found and exc.code == 404:
                return None
            raise APIError(
                f"RunPod API request failed with HTTP {exc.code}"
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = str(getattr(exc, "reason", type(exc).__name__))
            reason = reason.replace(self._api_key, "<redacted>")
            raise APIError(f"RunPod API request failed: {reason}") from None
        if not body:
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise APIError("RunPod API returned invalid JSON") from exc

    def graphql(self, query: str) -> dict[str, Any]:
        request = urllib.request.Request(
            GRAPHQL_URL,
            data=json.dumps({"query": query}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "cl-1-koi-open-dreamer-coinrun/1",
            },
            method="POST",
        )
        response = self._request_json(request)
        if not isinstance(response, dict):
            raise APIError("RunPod GraphQL response was not an object")
        if response.get("errors"):
            messages = [
                str(error.get("message", "unknown GraphQL error"))
                for error in response["errors"]
                if isinstance(error, dict)
            ]
            detail = "; ".join(messages[:3]).replace(
                self._api_key, "<redacted>"
            )
            raise APIError(f"RunPod GraphQL error: {detail}")
        data = response.get("data")
        if not isinstance(data, dict):
            raise APIError("RunPod GraphQL response omitted data")
        return data

    def _rest(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        allow_not_found: bool = False,
    ) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{REST_URL}{path}",
            data=data,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        return self._request_json(request, allow_not_found=allow_not_found)

    def create_pod(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._rest("POST", "/pods", payload)
        if not isinstance(response, dict) or not response.get("id"):
            raise APIError("RunPod create response omitted the pod ID")
        return response

    def get_pod(self, pod_id: str) -> dict[str, Any] | None:
        response = self._rest(
            "GET",
            f"/pods/{urllib.parse.quote(pod_id, safe='')}",
            allow_not_found=True,
        )
        if response is not None and not isinstance(response, dict):
            raise APIError("RunPod pod response was not an object")
        return response

    def delete_pod(self, pod_id: str) -> bool:
        response = self._rest(
            "DELETE",
            f"/pods/{urllib.parse.quote(pod_id, safe='')}",
            allow_not_found=True,
        )
        return response is not None


def build_discovery_query(cloud: str) -> str:
    secure = "true" if cloud == "secure" else "false"
    gpu_fields = """
        id
        displayName
        memoryInGb
        lowestPrice(input: {gpuCount: 1, secureCloud: %s}) {
          stockStatus
          uninterruptablePrice
          availableGpuCounts
        }
    """ % secure
    aliases = []
    for choice, gpu_id in GPU_IDS.items():
        aliases.append(
            f'{choice.lower()}: gpuTypes(input: {{id: {json.dumps(gpu_id)}}}) '
            f"{{ {gpu_fields} }}"
        )
    return """
      query CoinRunRunPodDiscovery {
        myself {
          clientBalance
          currentSpendPerHr
          pods { id name desiredStatus costPerHr }
        }
        %s
      }
    """ % "\n".join(aliases)


def discover_runpod(api: RunPodAPI, cloud: str) -> Discovery:
    data = api.graphql(build_discovery_query(cloud))
    myself = data.get("myself")
    if not isinstance(myself, dict):
        raise DeploymentError("RunPod discovery omitted account information")

    offers: dict[str, GPUOffer] = {}
    for choice, gpu_id in GPU_IDS.items():
        candidates = data.get(choice.lower())
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise DeploymentError(
                f"RunPod discovery did not return exactly one {choice} offer"
            )
        candidate = candidates[0]
        price = candidate.get("lowestPrice")
        if not isinstance(price, dict):
            price = {}
        counts = price.get("availableGpuCounts") or []
        hourly_value = price.get("uninterruptablePrice")
        offers[choice] = GPUOffer(
            choice=choice,
            gpu_id=str(candidate.get("id") or gpu_id),
            display_name=str(candidate.get("displayName") or choice),
            memory_gb=int(candidate.get("memoryInGb") or 0),
            stock_status=str(price.get("stockStatus") or "None"),
            available_gpu_counts=tuple(int(value) for value in counts),
            hourly_price=(
                _decimal(hourly_value, f"{choice} hourly price")
                if hourly_value is not None
                else Decimal(0)
            ),
        )

    pods = myself.get("pods") or []
    if not isinstance(pods, list):
        raise DeploymentError("RunPod discovery returned invalid pod inventory")
    return Discovery(
        balance=_decimal(myself.get("clientBalance"), "account balance"),
        current_spend_per_hour=_decimal(
            myself.get("currentSpendPerHr", 0), "current spend per hour"
        ),
        pods=tuple(pod for pod in pods if isinstance(pod, dict)),
        offers=offers,
    )


def _run_git(
    repo_root: Path,
    args: Sequence[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    completed = runner(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise DeploymentError(
            f"Git preflight failed for {' '.join(args)}: {detail}"
        )
    return completed.stdout.strip()


def _is_cl1koi_remote(url: str) -> bool:
    normalized = url.strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized in {
        "https://github.com/cl-1-koi/open-dreamer",
        "git@github.com:cl-1-koi/open-dreamer",
        "ssh://git@github.com/cl-1-koi/open-dreamer",
    }


def validate_git_preflight(
    repo_root: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> GitState:
    status = _run_git(repo_root, ["status", "--porcelain"], runner=runner)
    if status:
        raise DeploymentError(
            "Refusing RunPod launch because the local Git worktree is dirty"
        )
    branch = _run_git(
        repo_root, ["branch", "--show-current"], runner=runner
    )
    if not branch:
        raise DeploymentError("Refusing RunPod launch from detached HEAD")
    commit_sha = _run_git(repo_root, ["rev-parse", "HEAD"], runner=runner)
    if not SHA_RE.fullmatch(commit_sha):
        raise DeploymentError(f"Git returned invalid commit SHA: {commit_sha!r}")
    remote_url = _run_git(
        repo_root, ["remote", "get-url", "origin"], runner=runner
    )
    if not _is_cl1koi_remote(remote_url):
        raise DeploymentError(
            "origin must be the cl-1-koi/open-dreamer GitHub repository"
        )
    remote_ref = f"refs/heads/{branch}"
    remote_output = _run_git(
        repo_root,
        ["ls-remote", GITHUB_REPO_URL, remote_ref],
        runner=runner,
    )
    remote_lines = [
        line.split()
        for line in remote_output.splitlines()
        if line.strip()
    ]
    if len(remote_lines) != 1 or len(remote_lines[0]) < 2:
        raise DeploymentError(
            f"Branch {branch!r} is absent from cl-1-koi GitHub"
        )
    remote_sha, returned_ref = remote_lines[0][:2]
    if returned_ref != remote_ref or remote_sha != commit_sha:
        raise DeploymentError(
            f"Local commit {commit_sha} is not the pushed tip of "
            f"cl-1-koi/{branch}"
        )
    return GitState(branch=branch, commit_sha=commit_sha, remote_url=remote_url)


def validate_preflight_report(path: Path, commit_sha: str) -> dict[str, Any]:
    if not path.is_file():
        raise DeploymentError(f"CoinRun preflight report is absent: {path}")
    report = read_json(path)
    if report.get("status") != "passed":
        raise DeploymentError(
            f"CoinRun preflight report has not passed: "
            f"status={report.get('status')!r}"
        )
    report_git = report.get("git")
    if not isinstance(report_git, dict):
        raise DeploymentError("CoinRun preflight report omits Git state")
    if report_git.get("commit") != commit_sha:
        raise DeploymentError(
            "CoinRun preflight report commit does not match local HEAD"
        )
    if report_git.get("dirty") is not False:
        raise DeploymentError(
            "CoinRun preflight report was produced from a dirty worktree"
        )
    stages = report.get("stages")
    if not isinstance(stages, list) or not stages:
        raise DeploymentError("CoinRun preflight report contains no stages")
    failed = [
        stage.get("name", "unknown")
        for stage in stages
        if not isinstance(stage, dict) or stage.get("status") != "passed"
    ]
    if failed:
        raise DeploymentError(
            "CoinRun preflight report contains non-passing stages: "
            + ", ".join(str(value) for value in failed)
        )
    return report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def managed_active_pods(pods: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    inactive_statuses = {"EXITED", "STOPPED", "TERMINATED"}
    return [
        pod
        for pod in pods
        if str(pod.get("name", "")).startswith(POD_NAME_PREFIX)
        and str(pod.get("desiredStatus", "")).upper() not in inactive_statuses
    ]


REMOTE_SELF_TERMINATE = r"""
import os
import sys
import time
import urllib.error
import urllib.request

delay = max(0, int(sys.argv[1]))
reason = sys.argv[2]
if len(sys.argv) > 3:
    trigger = sys.argv[3]
    while not os.path.exists(trigger):
        time.sleep(1)
time.sleep(delay)
pod_id = os.environ.get("RUNPOD_POD_ID", "")
api_key = os.environ.get("RUNPOD_API_KEY", "")
if not pod_id or not api_key:
    print(f"remote termination unavailable ({reason})", flush=True)
    raise SystemExit(2)
request = urllib.request.Request(
    f"https://rest.runpod.io/v1/pods/{pod_id}",
    headers={"Authorization": f"Bearer {api_key}"},
    method="DELETE",
)
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()
    print(f"remote termination sent ({reason})", flush=True)
except urllib.error.HTTPError as exc:
    if exc.code != 404:
        print(f"remote termination HTTP {exc.code} ({reason})", flush=True)
        raise SystemExit(3)
except Exception as exc:
    print(f"remote termination failed: {type(exc).__name__} ({reason})", flush=True)
    raise SystemExit(4)
"""


REMOTE_STREAM_SERVER = r"""
import http.server
import hmac
import os
import pathlib
import urllib.parse

root = pathlib.Path(os.environ["COINRUN_STREAM_DIR"]).resolve()

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        supplied = urllib.parse.parse_qs(parsed.query).get("token", [""])[0]
        expected = os.environ["COINRUN_STREAM_TOKEN"]
        if not hmac.compare_digest(supplied, expected):
            self.send_error(403)
            return
        if parsed.path == "/log":
            try:
                offset = max(0, int(urllib.parse.parse_qs(parsed.query).get("offset", ["0"])[0]))
            except ValueError:
                self.send_error(400)
                return
            path = root / "experiment.log"
            data = b""
            if path.is_file():
                with path.open("rb") as handle:
                    handle.seek(min(offset, path.stat().st_size))
                    data = handle.read(1024 * 1024)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Next-Offset", str(offset + len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        allowed = {
            "/artifact-manifest.json": "artifact-manifest.json",
            "/artifacts.tar.gz": "artifacts.tar.gz",
        }
        name = allowed.get(parsed.path)
        if name is None:
            self.send_error(404)
            return
        path = root / name
        if not path.is_file():
            self.send_error(404)
            return
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

http.server.ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
"""


REMOTE_WRITE_MANIFEST = r"""
import hashlib
import json
import os
import pathlib
from datetime import datetime, timezone

root = pathlib.Path(os.environ["COINRUN_ARTIFACT_DIR"]).resolve()
stream = pathlib.Path(os.environ["COINRUN_STREAM_DIR"]).resolve()
files = []
if root.is_dir():
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        })
archive = stream / "artifacts.tar.gz"
archive_digest = hashlib.sha256()
with archive.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        archive_digest.update(chunk)
payload = {
    "schema_version": 1,
    "complete": True,
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "returncode": int(os.environ["COINRUN_EXPERIMENT_RETURNCODE"]),
    "commit_sha": os.environ["COINRUN_COMMIT_SHA"],
    "branch": os.environ["COINRUN_BRANCH"],
    "files": files,
    "archive": {
        "name": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": archive_digest.hexdigest(),
    },
}
temporary = stream / ".artifact-manifest.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, stream / "artifact-manifest.json")
"""


def build_remote_script(
    *,
    branch: str,
    commit_sha: str,
    experiment_command: str,
    runtime_seconds: int,
    artifact_dir: str,
    stream_token: str,
    deadline_epoch: int,
) -> str:
    if not 0 < runtime_seconds <= MAX_RUNTIME_SECONDS:
        raise DeploymentError("Remote runtime must be in (0, 14400] seconds")
    if not SHA_RE.fullmatch(commit_sha):
        raise DeploymentError("Remote checkout requires a full commit SHA")
    if not branch or "\n" in branch:
        raise DeploymentError("Remote checkout requires a valid branch")
    if not experiment_command.strip():
        raise DeploymentError("--experiment-command must not be empty")
    if not artifact_dir.startswith("/workspace/") or "\n" in artifact_dir:
        raise DeploymentError(
            "--remote-artifact-dir must be an absolute path under /workspace"
        )
    if not stream_token or "\n" in stream_token:
        raise DeploymentError("Remote stream token is invalid")
    if deadline_epoch <= 0:
        raise DeploymentError("Remote watchdog deadline is invalid")

    helper_sources = {
        "self_terminate_b64": base64.b64encode(
            REMOTE_SELF_TERMINATE.encode("utf-8")
        ).decode("ascii"),
        "stream_server_b64": base64.b64encode(
            REMOTE_STREAM_SERVER.encode("utf-8")
        ).decode("ascii"),
        "write_manifest_b64": base64.b64encode(
            REMOTE_WRITE_MANIFEST.encode("utf-8")
        ).decode("ascii"),
    }
    values = {
        "repo_url": shlex.quote(GITHUB_REPO_URL),
        "branch": shlex.quote(branch),
        "commit": shlex.quote(commit_sha),
        "command": shlex.quote(experiment_command),
        "runtime": runtime_seconds,
        "artifact_dir": shlex.quote(artifact_dir),
        "stream_token": shlex.quote(stream_token),
        "deadline_epoch": deadline_epoch,
        **helper_sources,
    }
    return textwrap.dedent(
        """\
        #!/usr/bin/env bash
        set -Eeuo pipefail
        export COINRUN_STREAM_DIR=/workspace/coinrun-stream
        export COINRUN_ARTIFACT_DIR=%(artifact_dir)s
        export COINRUN_COMMIT_SHA=%(commit)s
        export COINRUN_BRANCH=%(branch)s
        export COINRUN_STREAM_TOKEN=%(stream_token)s
        HARD_DEADLINE_EPOCH=%(deadline_epoch)d
        watchdog_delay=$((HARD_DEADLINE_EPOCH - $(date +%%s)))
        if [ "$watchdog_delay" -lt 0 ]; then watchdog_delay=0; fi
        experiment_timeout=$((watchdog_delay - %(finalization_seconds)d))
        if [ "$experiment_timeout" -lt 1 ]; then experiment_timeout=1; fi
        REPO_DIR=/workspace/open-dreamer
        mkdir -p "$COINRUN_STREAM_DIR" "$COINRUN_ARTIFACT_DIR"
        : > "$COINRUN_STREAM_DIR/experiment.log"

        printf %%s %(self_terminate_b64)s | base64 -d \
          > /tmp/coinrun_self_terminate.py
        printf %%s %(stream_server_b64)s | base64 -d \
          > /tmp/coinrun_stream_server.py
        printf %%s %(write_manifest_b64)s | base64 -d \
          > /tmp/coinrun_write_manifest.py

        nohup python3 /tmp/coinrun_stream_server.py \
          > "$COINRUN_STREAM_DIR/server.log" 2>&1 &
        nohup python3 /tmp/coinrun_self_terminate.py \
          "$watchdog_delay" hard-remote-deadline \
          > "$COINRUN_STREAM_DIR/remote-watchdog.log" 2>&1 &
        nohup python3 /tmp/coinrun_self_terminate.py \
          %(remote_failure_grace)d experiment-failure \
          /tmp/coinrun-experiment-failed \
          >> "$COINRUN_STREAM_DIR/remote-watchdog.log" 2>&1 &
        unset RUNPOD_API_KEY COINRUN_STREAM_TOKEN

        exec >> "$COINRUN_STREAM_DIR/experiment.log" 2>&1
        echo "CoinRun remote bootstrap started"
        rc=0
        set +e
        (
          set -Eeuo pipefail
          command -v git >/dev/null
          rm -rf "$REPO_DIR"
          git clone --single-branch --branch %(branch)s --no-checkout \
            %(repo_url)s "$REPO_DIR"
          cd "$REPO_DIR"
          git checkout --detach %(commit)s
          test "$(git rev-parse HEAD)" = %(commit)s
          export COINRUN_ARTIFACT_DIR
          timeout --signal=TERM --kill-after=60 "$experiment_timeout" \
            bash -lc %(command)s
        )
        rc=$?
        set -e

        tar -C "$COINRUN_ARTIFACT_DIR" -czf \
          "$COINRUN_STREAM_DIR/artifacts.tar.gz" .
        export COINRUN_EXPERIMENT_RETURNCODE="$rc"
        python3 /tmp/coinrun_write_manifest.py
        echo "CoinRun experiment complete: returncode=$rc"
        if [ "$rc" -ne 0 ]; then
          touch /tmp/coinrun-experiment-failed
        fi
        sleep infinity
        """
        % {
            **values,
            "remote_failure_grace": REMOTE_FINALIZATION_SECONDS,
            "finalization_seconds": REMOTE_FINALIZATION_SECONDS,
        }
    )


def encode_start_command(remote_script: str) -> list[str]:
    encoded = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
    command = (
        f"printf %s {shlex.quote(encoded)} | base64 -d "
        "> /tmp/run_coinrun.sh && exec bash /tmp/run_coinrun.sh"
    )
    return ["bash", "-lc", command]


def make_pod_payload(
    args: argparse.Namespace,
    offer: GPUOffer,
    git_state: GitState,
    pod_name: str,
    stream_token: str,
    deadline_epoch: int,
) -> dict[str, Any]:
    remote_script = build_remote_script(
        branch=git_state.branch,
        commit_sha=git_state.commit_sha,
        experiment_command=args.experiment_command,
        runtime_seconds=args.runtime_seconds,
        artifact_dir=args.remote_artifact_dir,
        stream_token=stream_token,
        deadline_epoch=deadline_epoch,
    )
    return {
        "name": pod_name,
        "imageName": args.image,
        "computeType": "GPU",
        "gpuTypeIds": [offer.gpu_id],
        "gpuTypePriority": "custom",
        "gpuCount": 1,
        "cloudType": args.cloud.upper(),
        "interruptible": False,
        "locked": False,
        "containerDiskInGb": args.container_disk_gb,
        "volumeInGb": 0,
        "ports": ["8000/http"],
        "allowedCudaVersions": ["12.8", "12.9", "13.0"],
        "dockerStartCmd": encode_start_command(remote_script),
    }


def fetch_log_chunk(
    pod_id: str,
    offset: int,
    stream_token: str,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
    timeout: float = 15.0,
) -> tuple[bytes, int] | None:
    query = urllib.parse.urlencode(
        {"offset": offset, "token": stream_token}
    )
    url = f"https://{pod_id}-8000.proxy.runpod.net/log?{query}"
    request = urllib.request.Request(url, method="GET")
    try:
        with opener(request, timeout=timeout) as response:
            data = response.read()
            next_offset = int(response.headers.get("X-Next-Offset", offset + len(data)))
        return data, next_offset
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 404, 502, 503, 504}:
            return None
        raise DeploymentError(
            f"RunPod log stream returned HTTP {exc.code}"
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def fetch_manifest(
    pod_id: str,
    stream_token: str,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
    timeout: float = 15.0,
) -> dict[str, Any] | None:
    url = (
        f"https://{pod_id}-8000.proxy.runpod.net/"
        "artifact-manifest.json?"
        + urllib.parse.urlencode({"token": stream_token})
    )
    request = urllib.request.Request(url, method="GET")
    try:
        with opener(request, timeout=timeout) as response:
            data = response.read(8 * 1024 * 1024)
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 404, 502, 503, 504}:
            return None
        raise DeploymentError(
            f"RunPod artifact manifest returned HTTP {exc.code}"
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    try:
        value = json.loads(data)
    except json.JSONDecodeError as exc:
        raise DeploymentError("Remote artifact manifest is invalid JSON") from exc
    if not isinstance(value, dict):
        raise DeploymentError("Remote artifact manifest is not an object")
    return value


def download_artifact_archive(
    pod_id: str,
    manifest: dict[str, Any],
    stream_token: str,
    destination: Path,
    *,
    deadline: float,
    max_bytes: int,
    opener: Callable[..., Any] = urllib.request.urlopen,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    archive = manifest.get("archive")
    if not isinstance(archive, dict):
        raise DeploymentError("Remote artifact manifest omits archive metadata")
    expected_size = int(archive.get("bytes", -1))
    expected_sha = str(archive.get("sha256", ""))
    if expected_size < 0 or expected_size > max_bytes:
        raise DeploymentError(
            f"Remote artifact archive size {expected_size} exceeds "
            f"the {max_bytes} byte bound"
        )
    if not SHA_RE.fullmatch(expected_sha):
        raise DeploymentError("Remote artifact archive has an invalid SHA-256")

    url = (
        f"https://{pod_id}-8000.proxy.runpod.net/artifacts.tar.gz?"
        + urllib.parse.urlencode({"token": stream_token})
    )
    request = urllib.request.Request(url, method="GET")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    written = 0
    try:
        remaining = deadline - clock()
        if remaining <= 0:
            raise LocalDeadlineExceeded(
                "Local watchdog expired before artifact download"
            )
        with opener(request, timeout=min(30.0, remaining)) as response:
            with temporary.open("wb") as handle:
                while True:
                    if clock() >= deadline:
                        raise LocalDeadlineExceeded(
                            "Local watchdog expired during artifact download"
                        )
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        raise DeploymentError(
                            "Remote artifact archive exceeded its download bound"
                        )
                    digest.update(chunk)
                    handle.write(chunk)
        if written != expected_size:
            raise DeploymentError(
                f"Artifact archive size mismatch: {written} != {expected_size}"
            )
        if digest.hexdigest() != expected_sha:
            raise DeploymentError("Artifact archive SHA-256 mismatch")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def monitor_pod(
    api: RunPodAPI,
    state_path: Path,
    *,
    deadline: float,
    poll_seconds: float,
    max_artifact_bytes: int,
    opener: Callable[..., Any] = urllib.request.urlopen,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    state = read_json(state_path)
    pod_id = str(state["pod_id"])
    stream_token = str(state["stream_token"])
    local_dir = state_path.parent / pod_id
    local_dir.mkdir(parents=True, exist_ok=True)
    log_path = local_dir / "experiment.log"
    manifest_path = local_dir / "artifact-manifest.json"
    archive_path = local_dir / "artifacts.tar.gz"
    log_offset = 0
    proxy_ready = False
    proxy_ready_deadline = min(
        deadline, clock() + PROXY_STARTUP_GRACE_SECONDS
    )

    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            raise LocalDeadlineExceeded("Local four-hour watchdog expired")

        chunk = fetch_log_chunk(
            pod_id,
            log_offset,
            stream_token,
            opener=opener,
            timeout=min(15.0, remaining),
        )
        if chunk is not None:
            proxy_ready = True
            data, log_offset = chunk
            if data:
                with log_path.open("ab") as handle:
                    handle.write(data)
                sys.stdout.write(data.decode("utf-8", errors="replace"))
                sys.stdout.flush()

        manifest = fetch_manifest(
            pod_id,
            stream_token,
            opener=opener,
            timeout=min(15.0, remaining),
        )
        if manifest is not None:
            proxy_ready = True
            write_json_atomic(manifest_path, manifest)
            if manifest.get("complete") is True:
                if manifest.get("commit_sha") != state.get("commit_sha"):
                    raise DeploymentError(
                        "Remote artifact manifest commit does not match launch state"
                    )
                download_artifact_archive(
                    pod_id,
                    manifest,
                    stream_token,
                    archive_path,
                    deadline=deadline,
                    max_bytes=max_artifact_bytes,
                    opener=opener,
                    clock=clock,
                )
                update_state(
                    state_path,
                    phase="experiment_finished",
                    experiment_returncode=manifest.get("returncode"),
                    local_log_path=str(log_path),
                    local_manifest_path=str(manifest_path),
                    local_artifact_archive=str(archive_path),
                    last_observed_utc=isoformat(utc_now()),
                )
                return manifest
        if not proxy_ready and clock() >= proxy_ready_deadline:
            raise DeploymentError(
                "RunPod HTTP proxy did not expose the telemetry service "
                f"within {PROXY_STARTUP_GRACE_SECONDS} seconds"
            )

        pod = api.get_pod(pod_id)
        if pod is None:
            raise DeploymentError(
                "RunPod pod disappeared before a complete artifact manifest arrived"
            )
        update_state(
            state_path,
            phase="monitoring",
            last_pod_status=pod.get("desiredStatus"),
            last_observed_utc=isoformat(utc_now()),
            streamed_log_bytes=log_offset,
        )
        sleeper(min(poll_seconds, max(0.0, deadline - clock())))


def terminate_managed_pod(
    api: RunPodAPI,
    state_path: Path,
    *,
    reason: str,
) -> str:
    state = read_json(state_path)
    pod_id = state.get("pod_id")
    if not pod_id:
        update_state(
            state_path,
            termination_status="pod_absent",
            termination_reason=reason,
            termination_checked_utc=isoformat(utc_now()),
        )
        return "pod_absent"
    if state.get("termination_status") in TERMINAL_TERMINATION_STATES:
        return str(state["termination_status"])

    pod = api.get_pod(str(pod_id))
    if pod is None:
        update_state(
            state_path,
            phase="terminated",
            termination_status="pod_absent",
            termination_reason=reason,
            termination_checked_utc=isoformat(utc_now()),
        )
        return "pod_absent"
    if pod.get("name") != state.get("pod_name"):
        update_state(
            state_path,
            termination_status="termination_refused_name_mismatch",
            termination_reason=reason,
            termination_checked_utc=isoformat(utc_now()),
        )
        raise DeploymentError(
            f"Refusing to terminate {pod_id}: pod name does not match state"
        )

    update_state(
        state_path,
        termination_status="requesting",
        termination_reason=reason,
        termination_requested_utc=isoformat(utc_now()),
    )
    try:
        api.delete_pod(str(pod_id))
    except BaseException:
        update_state(
            state_path,
            termination_status="termination_failed",
            termination_checked_utc=isoformat(utc_now()),
        )
        raise
    update_state(
        state_path,
        phase="terminated",
        termination_status="terminated",
        termination_reason=reason,
        terminated_utc=isoformat(utc_now()),
    )
    return "terminated"


def print_discovery(discovery: Discovery, cloud: str) -> None:
    print(
        f"RunPod balance=${_money(discovery.balance)} "
        f"current_spend=${_money(discovery.current_spend_per_hour)}/hr"
    )
    print(f"{cloud.capitalize()} Cloud on-demand offers:")
    for choice in GPU_IDS:
        offer = discovery.offers[choice]
        counts = ",".join(str(value) for value in offer.available_gpu_counts) or "-"
        print(
            f"  {choice}: {offer.memory_gb}GB "
            f"${_money(offer.hourly_price)}/hr "
            f"stock={offer.stock_status} counts={counts}"
        )


def validate_runtime(value: int) -> None:
    if value < MIN_RUNTIME_SECONDS or value > MAX_RUNTIME_SECONDS:
        raise DeploymentError(
            f"--runtime-seconds must be between {MIN_RUNTIME_SECONDS} "
            f"and {MAX_RUNTIME_SECONDS}"
        )


def run_launch(
    args: argparse.Namespace,
    *,
    api_factory: Callable[[str], RunPodAPI] = RunPodAPI,
    git_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = utc_now,
    monitor: Callable[..., dict[str, Any]] = monitor_pod,
) -> int:
    validate_runtime(args.runtime_seconds)
    state_path = Path(args.state).expanduser().resolve()
    repo_root = Path(args.repo_root).resolve()
    preflight_path = Path(args.preflight_report).expanduser().resolve()
    ensure_no_active_state(state_path)
    git_state = validate_git_preflight(repo_root, runner=git_runner)
    validate_preflight_report(preflight_path, git_state.commit_sha)

    api_key = load_api_key(Path(args.config))
    api = api_factory(api_key)
    discovery = discover_runpod(api, args.cloud)
    print_discovery(discovery, args.cloud)
    active = managed_active_pods(discovery.pods)
    if active:
        names = ", ".join(str(pod.get("name") or pod.get("id")) for pod in active)
        raise DeploymentError(
            f"Managed CoinRun pod already exists in RunPod inventory: {names}"
        )

    offer = discovery.offers[args.gpu]
    if not offer.available:
        raise DeploymentError(
            f"Selected {args.gpu} is not available as one on-demand GPU "
            f"in {args.cloud} cloud"
        )
    runtime_hours = Decimal(args.runtime_seconds) / Decimal(3600)
    projected = offer.hourly_price * runtime_hours
    buffer = _decimal(args.balance_buffer, "balance buffer")
    required_balance = projected + buffer
    if buffer < 0:
        raise DeploymentError("--balance-buffer must be nonnegative")
    if discovery.balance < required_balance:
        raise DeploymentError(
            f"RunPod balance ${_money(discovery.balance)} is below projected "
            f"cost ${_money(projected)} plus buffer ${_money(buffer)}"
        )

    print(
        f"Selected {args.gpu} ({offer.gpu_id}); maximum projected compute "
        f"spend=${_money(projected)} for {args.runtime_seconds}s"
    )
    print(
        f"Remote checkout: cl-1-koi/open-dreamer "
        f"{git_state.branch}@{git_state.commit_sha}"
    )
    if not args.execute:
        print("Dry run only; no pod was started. Pass --execute to launch.")
        return 0

    launch_clock = clock()
    launch_time = now()
    deadline_utc = launch_time + timedelta(seconds=args.runtime_seconds)
    pod_name = (
        f"{POD_NAME_PREFIX}"
        f"{launch_time.strftime('%Y%m%d-%H%M%S')}-{git_state.commit_sha[:8]}"
    )
    stream_token = secrets.token_urlsafe(32)
    ensure_no_active_state(state_path)
    provisional_state = {
        "schema_version": 1,
        "phase": "launching",
        "pod_id": None,
        "pod_name": pod_name,
        "launch_time_utc": isoformat(launch_time),
        "hard_deadline_utc": isoformat(deadline_utc),
        "runtime_seconds": args.runtime_seconds,
        "gpu_choice": args.gpu,
        "gpu_type_id": offer.gpu_id,
        "cloud": args.cloud,
        "hourly_price": float(offer.hourly_price),
        "max_projected_spend": float(projected),
        "balance_at_preflight": float(discovery.balance),
        "balance_buffer": float(buffer),
        "commit_sha": git_state.commit_sha,
        "branch": git_state.branch,
        "github_repo": GITHUB_REPO_URL,
        "preflight_report": str(preflight_path),
        "preflight_report_sha256": sha256_file(preflight_path),
        "termination_status": "not_requested",
        "stream_token": stream_token,
    }
    with state_lock(state_path):
        if state_path.exists() and state_is_active(read_json(state_path)):
            raise DeploymentError("Active RunPod state appeared during launch")
        write_json_atomic(state_path, provisional_state)

    payload = make_pod_payload(
        args,
        offer,
        git_state,
        pod_name,
        stream_token,
        int(deadline_utc.timestamp()),
    )
    local_deadline = launch_clock + args.runtime_seconds
    try:
        pod = api.create_pod(payload)
    except BaseException as exc:
        update_state(
            state_path,
            phase="launch_unknown",
            launch_error=f"{type(exc).__name__}: {exc}",
            termination_status="reconciliation_required",
        )
        raise

    pod_id = str(pod["id"])
    actual_hourly = _decimal(
        pod.get("costPerHr", offer.hourly_price), "launched pod hourly price"
    )
    actual_projected = actual_hourly * runtime_hours
    update_state(
        state_path,
        phase="running",
        pod_id=pod_id,
        hourly_price=float(actual_hourly),
        max_projected_spend=float(actual_projected),
        last_pod_status=pod.get("desiredStatus"),
    )

    failure: BaseException | None = None
    manifest: dict[str, Any] | None = None
    if actual_hourly > offer.hourly_price:
        failure = DeploymentError(
            f"Launched pod price ${_money(actual_hourly)}/hr exceeds "
            f"approved quote ${_money(offer.hourly_price)}/hr"
        )
    else:
        try:
            manifest = monitor(
                api,
                state_path,
                deadline=local_deadline,
                poll_seconds=args.poll_seconds,
                max_artifact_bytes=args.max_artifact_bytes,
            )
            if int(manifest.get("returncode", -1)) != 0:
                failure = DeploymentError(
                    f"Remote experiment failed with return code "
                    f"{manifest.get('returncode')}"
                )
        except BaseException as exc:
            failure = exc

    termination_error: BaseException | None = None
    reason = "experiment-complete" if failure is None else "launch-or-experiment-failure"
    try:
        terminate_managed_pod(api, state_path, reason=reason)
    except BaseException as exc:
        termination_error = exc

    if failure is not None:
        if termination_error is not None:
            raise DeploymentError(
                f"{failure}; automatic termination also failed: {termination_error}"
            ) from failure
        raise failure
    if termination_error is not None:
        raise termination_error
    print(
        f"CoinRun experiment completed and pod {pod_id} was terminated. "
        f"State: {state_path}"
    )
    return 0


def _deadline_has_passed(state: dict[str, Any], now: datetime) -> bool:
    value = state.get("hard_deadline_utc")
    if not isinstance(value, str):
        return False
    try:
        deadline = datetime.fromisoformat(value)
    except ValueError:
        raise DeploymentError("State contains an invalid hard_deadline_utc")
    if deadline.tzinfo is None:
        raise DeploymentError("State hard_deadline_utc is not timezone-aware")
    return now >= deadline


def run_status(
    args: argparse.Namespace,
    *,
    api_factory: Callable[[str], RunPodAPI] = RunPodAPI,
    now: Callable[[], datetime] = utc_now,
) -> int:
    state_path = Path(args.state).expanduser().resolve()
    if not state_path.exists():
        print(f"No RunPod state exists at {state_path}")
        return 0
    state = read_json(state_path)
    if not state_is_active(state):
        print(
            f"Pod {state.get('pod_id') or 'none'} is inactive; "
            f"termination_status={state.get('termination_status')}"
        )
        return 0

    api = api_factory(load_api_key(Path(args.config)))
    if not state.get("pod_id") and state.get("phase") in {
        "launching",
        "launch_unknown",
    }:
        discovery = discover_runpod(api, str(state.get("cloud", "secure")))
        matches = [
            pod for pod in discovery.pods if pod.get("name") == state.get("pod_name")
        ]
        if len(matches) == 1:
            state = update_state(
                state_path,
                pod_id=matches[0].get("id"),
                last_pod_status=matches[0].get("desiredStatus"),
            )
        elif not matches:
            state = update_state(
                state_path,
                phase="launch_failed",
                termination_status="pod_absent",
            )
            print("Recorded launch has no matching active RunPod pod")
            return 0
        else:
            raise DeploymentError(
                "Multiple RunPod pods match the recorded launch name"
            )

    if _deadline_has_passed(state, now()):
        result = terminate_managed_pod(
            api, state_path, reason="status-detected-hard-deadline"
        )
        print(f"Hard deadline passed; termination_status={result}")
        return 0

    pod = api.get_pod(str(state["pod_id"]))
    if pod is None:
        update_state(
            state_path,
            phase="terminated",
            termination_status="pod_absent",
            termination_checked_utc=isoformat(now()),
        )
        print(f"Pod {state['pod_id']} is already absent")
        return 0
    if pod.get("name") != state.get("pod_name"):
        raise DeploymentError("RunPod pod name does not match local state")
    update_state(
        state_path,
        last_pod_status=pod.get("desiredStatus"),
        last_observed_utc=isoformat(now()),
    )
    print(
        f"Pod {state['pod_id']} status={pod.get('desiredStatus')} "
        f"cost=${pod.get('costPerHr', state.get('hourly_price'))}/hr "
        f"deadline={state.get('hard_deadline_utc')}"
    )
    return 0


def run_stop(
    args: argparse.Namespace,
    *,
    api_factory: Callable[[str], RunPodAPI] = RunPodAPI,
) -> int:
    state_path = Path(args.state).expanduser().resolve()
    if not state_path.exists():
        print(f"No RunPod state exists at {state_path}; nothing to stop")
        return 0
    state = read_json(state_path)
    if not state_is_active(state):
        print(
            f"Pod {state.get('pod_id') or 'none'} is already inactive "
            f"({state.get('termination_status')})"
        )
        return 0
    if not args.execute:
        print(
            f"Dry run only; would terminate pod "
            f"{state.get('pod_id') or state.get('pod_name')}. "
            "Pass --execute to stop it."
        )
        return 0
    api = api_factory(load_api_key(Path(args.config)))
    result = terminate_managed_pod(api, state_path, reason=args.reason)
    print(f"termination_status={result}")
    return 0


def add_common_state_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch = subparsers.add_parser(
        "launch", help="Validate and plan a pod; launch only with --execute"
    )
    add_common_state_args(launch)
    launch.add_argument("--repo-root", type=Path, default=repo_root)
    launch.add_argument("--preflight-report", type=Path, required=True)
    launch.add_argument("--gpu", choices=tuple(GPU_IDS), required=True)
    launch.add_argument(
        "--experiment-command",
        required=True,
        help=(
            "Command executed in the exact remote Git checkout; write outputs "
            "under $COINRUN_ARTIFACT_DIR"
        ),
    )
    launch.add_argument(
        "--runtime-seconds", type=int, default=MAX_RUNTIME_SECONDS
    )
    launch.add_argument(
        "--cloud", choices=("secure", "community"), default="secure"
    )
    launch.add_argument("--balance-buffer", type=Decimal, default=Decimal("5"))
    launch.add_argument("--image", default=DEFAULT_IMAGE)
    launch.add_argument("--container-disk-gb", type=int, default=100)
    launch.add_argument(
        "--remote-artifact-dir",
        default="/workspace/coinrun-artifacts",
    )
    launch.add_argument("--poll-seconds", type=float, default=5.0)
    launch.add_argument(
        "--max-artifact-bytes", type=int, default=20 * 1024 * 1024 * 1024
    )
    launch.add_argument(
        "--execute",
        action="store_true",
        help="Perform the paid RunPod mutation after all gates pass",
    )
    launch.set_defaults(handler=run_launch)

    status = subparsers.add_parser("status", help="Reconcile local and remote state")
    add_common_state_args(status)
    status.set_defaults(handler=run_status)

    stop = subparsers.add_parser("stop", help="Idempotently terminate the managed pod")
    add_common_state_args(stop)
    stop.add_argument("--reason", default="operator-request")
    stop.add_argument(
        "--execute",
        action="store_true",
        help="Perform the termination; omitted means dry run",
    )
    stop.set_defaults(handler=run_stop)
    return parser


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.command != "launch":
        return
    if args.container_disk_gb < 50:
        raise DeploymentError("--container-disk-gb must be at least 50")
    if args.poll_seconds <= 0:
        raise DeploymentError("--poll-seconds must be positive")
    if args.max_artifact_bytes <= 0:
        raise DeploymentError("--max-artifact-bytes must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_cli_args(args)
        return int(args.handler(args))
    except KeyboardInterrupt:
        print(
            "Interrupted; executed launches attempt termination before returning.",
            file=sys.stderr,
        )
        return 130
    except DeploymentError as exc:
        print(f"RunPod CoinRun helper refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
