#!/usr/bin/env python3
"""Build the CoinRun runner image and append machine-readable build telemetry.

Writes one JSON object per run to artifacts/coinrun_runner/build_history.jsonl
(append-only) plus the full BuildKit log and a host resource sample per run. The
schema is documented in docs/ops/COINRUN_RUNNER_IMAGE.md; summaries are meant to
be queried with jq, e.g.

    jq -r 'select(.returncode==0) | [.started_utc,.cache_label,.duration_seconds,
           .image_bytes] | @tsv' artifacts/coinrun_runner/build_history.jsonl

This wrapper only observes; it never changes what is built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = REPO_ROOT / "artifacts" / "coinrun_runner"
HISTORY_PATH = HISTORY_DIR / "build_history.jsonl"
SCHEMA = "coinrun-runner-build-history-v1"
# BuildKit plain progress: "#12 DONE 77.9s" / "#12 CACHED"
STEP_DONE_RE = re.compile(r"^#(?P<step>\d+)\s+(?P<state>DONE|CACHED)(?:\s+(?P<seconds>[\d.]+)s)?\s*$")
STEP_NAME_RE = re.compile(r"^#(?P<step>\d+)\s+\[(?P<name>[^\]]+)\]\s*(?P<rest>.*)$")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def capture(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def host_sample() -> dict[str, object]:
    """Coarse host state, so a slow build can be attributed later."""
    usage = shutil.disk_usage(REPO_ROOT)
    sample: dict[str, object] = {
        "cpu_count": os.cpu_count(),
        "load_average_1m": os.getloadavg()[0],
        "disk_free_bytes": usage.free,
        "disk_total_bytes": usage.total,
    }
    try:
        meminfo = dict(
            (parts[0].rstrip(":"), int(parts[1]))
            for line in Path("/proc/meminfo").read_text().splitlines()
            if len(parts := line.split()) >= 2
        )
        sample["mem_total_bytes"] = meminfo.get("MemTotal", 0) * 1024
        sample["mem_available_bytes"] = meminfo.get("MemAvailable", 0) * 1024
    except (OSError, ValueError):
        pass
    return sample


# --------------------------------------------------------------------------
# Periodic resource sampling
#
# Raw /proc counters are read cheaply and turned into per-interval rates, so a
# slow build can be attributed to network, CPU, RAM or disk after the fact.
# Every metric is optional: if a file is unreadable the key is simply absent.
# --------------------------------------------------------------------------

def read_counters() -> dict[str, float]:
    """Cumulative counters from /proc. Missing metrics are omitted."""
    counters: dict[str, float] = {}
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("cpu "):
                values = [float(v) for v in line.split()[1:]]
                counters["cpu_total"] = sum(values)
                # idle + iowait
                counters["cpu_idle"] = values[3] + (values[4] if len(values) > 4 else 0.0)
                break
    except (OSError, ValueError, IndexError):
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            parts = line.split()
            if parts and parts[0] == "MemAvailable:":
                counters["mem_available_bytes"] = float(parts[1]) * 1024
                break
    except (OSError, ValueError):
        pass
    try:
        counters["load_average_1m"] = os.getloadavg()[0]
    except OSError:
        pass
    read_bytes = write_bytes = 0.0
    try:
        for line in Path("/proc/diskstats").read_text().splitlines():
            fields = line.split()
            if len(fields) < 10:
                continue
            name = fields[2]
            # Whole devices only; partitions would double-count.
            if not Path(f"/sys/block/{name}").exists() or name.startswith(("loop", "ram")):
                continue
            read_bytes += float(fields[5]) * 512
            write_bytes += float(fields[9]) * 512
        counters["disk_read_bytes"] = read_bytes
        counters["disk_write_bytes"] = write_bytes
    except (OSError, ValueError):
        pass
    rx = tx = 0.0
    try:
        for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
            name, _, rest = line.partition(":")
            if name.strip() == "lo":
                continue
            fields = rest.split()
            rx += float(fields[0])
            tx += float(fields[8])
        counters["net_rx_bytes"] = rx
        counters["net_tx_bytes"] = tx
    except (OSError, ValueError, IndexError):
        pass
    return counters


def derive_sample(previous: dict[str, float], current: dict[str, float], seconds: float) -> dict[str, object]:
    """Turn two counter snapshots into one rate row."""
    row: dict[str, object] = {"utc": isoformat(utc_now()), "interval_seconds": round(seconds, 3)}
    for key in ("load_average_1m", "mem_available_bytes"):
        if key in current:
            row[key] = current[key]
    total = current.get("cpu_total", 0.0) - previous.get("cpu_total", 0.0)
    idle = current.get("cpu_idle", 0.0) - previous.get("cpu_idle", 0.0)
    if total > 0:
        row["cpu_utilization_percent"] = round(100.0 * (1.0 - idle / total), 2)
    if seconds > 0:
        for key, out in (
            ("disk_read_bytes", "disk_read_bytes_per_s"),
            ("disk_write_bytes", "disk_write_bytes_per_s"),
            ("net_rx_bytes", "net_rx_bytes_per_s"),
            ("net_tx_bytes", "net_tx_bytes_per_s"),
        ):
            if key in current and key in previous:
                row[out] = round((current[key] - previous[key]) / seconds, 1)
    return row


def summarize_resource_samples(rows: list[dict[str, object]]) -> dict[str, object]:
    """Mean/peak fields for the build summary record."""
    summary: dict[str, object] = {"sample_count": len(rows)}
    for field, stats in (
        ("cpu_utilization_percent", ("mean", "peak")),
        ("net_rx_bytes_per_s", ("mean", "peak")),
        ("net_tx_bytes_per_s", ("peak",)),
        ("disk_read_bytes_per_s", ("peak",)),
        ("disk_write_bytes_per_s", ("peak",)),
        ("load_average_1m", ("peak",)),
    ):
        values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
        if not values:
            continue
        if "mean" in stats:
            summary[f"{field}_mean"] = round(sum(values) / len(values), 2)
        if "peak" in stats:
            summary[f"{field}_peak"] = round(max(values), 2)
    lows = [float(r["mem_available_bytes"]) for r in rows
            if isinstance(r.get("mem_available_bytes"), (int, float))]
    if lows:
        summary["mem_available_bytes_min"] = min(lows)
    return summary


class ResourceSampler:
    """Append one rate row every `interval` seconds while the build runs."""

    def __init__(self, path: Path, interval: float = 5.0) -> None:
        self.path = path
        self.interval = interval
        self.rows: list[dict[str, object]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="resource-sampler", daemon=True)

    def _loop(self) -> None:
        previous, marker = read_counters(), time.monotonic()
        with self.path.open("w", encoding="utf-8") as handle:
            while not self._stop.wait(self.interval):
                current, now = read_counters(), time.monotonic()
                row = derive_sample(previous, current, now - marker)
                self.rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                previous, marker = current, now

    def __enter__(self) -> "ResourceSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval + 5)


def parse_stage_durations(log_text: str) -> dict[str, float]:
    """Coarse per-step seconds from BuildKit plain progress, when present."""
    names: dict[str, str] = {}
    durations: dict[str, float] = {}
    for line in log_text.splitlines():
        named = STEP_NAME_RE.match(line.strip())
        if named:
            names.setdefault(named.group("step"), named.group("name"))
            continue
        done = STEP_DONE_RE.match(line.strip())
        if not done:
            continue
        step = done.group("step")
        label = names.get(step, f"step-{step}")
        if done.group("state") == "CACHED":
            durations[label] = 0.0
        elif done.group("seconds"):
            durations[label] = float(done.group("seconds"))
    return durations


def image_facts(tag: str) -> dict[str, object]:
    raw = capture(["sudo", "-n", "docker", "image", "inspect", tag,
                   "--format", "{{.Id}}\t{{.Size}}"])
    if not raw or "\t" not in raw:
        return {}
    image_id, size = raw.split("\t", 1)
    try:
        return {"image_id": image_id, "image_bytes": int(size)}
    except ValueError:
        return {"image_id": image_id}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="coinrun-runner:local")
    parser.add_argument(
        "--cache-label",
        choices=("cold", "warm", "unknown"),
        default="unknown",
        help="Operator's label for whether layer/download caches were primed",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--note", default="")
    parser.add_argument(
        "--sample-seconds", type=float, default=5.0,
        help="Resource sampling cadence during the build",
    )
    args = parser.parse_args(argv)

    lock_sha = hashlib.sha256((REPO_ROOT / "uv.lock").read_bytes()).hexdigest()
    commit = capture(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    log_path = HISTORY_DIR / f"build-{started.strftime('%Y%m%dT%H%M%SZ')}.log"

    command = [
        "sudo", "-n", "docker", "build", "--progress=plain",
        "--build-arg", f"SOURCE_COMMIT={commit}",
        "--build-arg", f"UV_LOCK_SHA256={lock_sha}",
        "-t", args.tag, ".",
    ]
    if args.no_cache:
        command.insert(4, "--no-cache")

    before = host_sample()
    resource_path = log_path.with_name(log_path.stem + "-resources.jsonl")
    monotonic = time.monotonic()
    with ResourceSampler(resource_path, interval=args.sample_seconds) as sampler:
        with log_path.open("w", encoding="utf-8") as handle:
            returncode = subprocess.run(
                command, cwd=str(REPO_ROOT), stdout=handle,
                stderr=subprocess.STDOUT, check=False,
            ).returncode
    duration = time.monotonic() - monotonic
    ended = utc_now()

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    record: dict[str, object] = {
        "schema": SCHEMA,
        "started_utc": isoformat(started),
        "ended_utc": isoformat(ended),
        "duration_seconds": round(duration, 3),
        "git_commit": commit,
        "uv_lock_sha256": lock_sha,
        "image_tag": args.tag,
        "cache_label": args.cache_label,
        "no_cache": bool(args.no_cache),
        "returncode": returncode,
        "log_path": str(log_path.relative_to(REPO_ROOT)),
        "host_before": before,
        "host_after": host_sample(),
        "resource_samples_path": str(resource_path.relative_to(REPO_ROOT)),
        "resource_summary": summarize_resource_samples(sampler.rows),
        "stage_durations_seconds": parse_stage_durations(log_text),
        "source": "wrapper",
        "note": args.note,
    }
    if returncode == 0:
        record.update(image_facts(args.tag))
    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"build rc={returncode} duration={duration:.1f}s log={log_path}")
    if returncode == 0:
        print(f"image={record.get('image_id')} bytes={record.get('image_bytes')}")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
