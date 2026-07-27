# CoinRun runner image — operator guide

Prebuilt OCI image that removes per-pod dependency installation from the CoinRun
sprint. Nothing in this document launches paid compute.

| Item | Value |
|---|---|
| Image | `ghcr.io/cl-1-koi/open-dreamer-coinrun-runner` |
| Base | `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` (CUDA 12.8) |
| Tags | `sha-<40 hex commit>` only. **No `latest`.** |
| Entrypoint | `coinrun-runner` (exec form), default `CMD` is `smoke` |
| Build | `Dockerfile` (two stages); publish workflow staged at `docs/ops/publish-coinrun-runner.yml` |

## What the image does and does not contain

Contains: the exact `uv.lock` environment at `/opt/coinrun/venv`, a warmed uv
cache at `/opt/coinrun/uv-cache` holding the source-built Procgen wheel, git,
uv, GNU timeout, grep, ffmpeg, Procgen's Qt5/cmake prerequisites, and
`scripts/coinrun_runner.py`.

Does **not** contain: credentials, datasets, or an importable copy of the
project. `uv sync` runs with `--no-install-project`, so `dreamer` and `scripts`
are always imported from the runtime Git checkout via `PYTHONPATH`. A pod
therefore cannot silently run image code instead of the pushed commit.

## Build and verify locally

```bash
# Wrapper build: same docker build, plus append-only telemetry.
python scripts/build_coinrun_runner.py --tag coinrun-runner:local --cache-label cold

# Or directly, if you do not want a telemetry record:
LOCK=$(sha256sum uv.lock | cut -d' ' -f1)
sudo docker build \
  --build-arg SOURCE_COMMIT="$(git rev-parse HEAD)" \
  --build-arg UV_LOCK_SHA256="$LOCK" \
  -t coinrun-runner:local .
```

The build fails closed if `UV_LOCK_SHA256` does not match the `uv.lock` it
copied, and if the Procgen prewarm does not produce `libenv.so`.

### Inspect labels, contract env, and digest

```bash
sudo docker inspect coinrun-runner:local --format '{{json .Config.Labels}}' | jq
sudo docker inspect coinrun-runner:local --format '{{range .Config.Env}}{{println .}}{{end}}' | grep COINRUN_
sudo docker inspect coinrun-runner:local --format 'Entrypoint={{json .Config.Entrypoint}} Cmd={{json .Config.Cmd}}'

# After a GHCR push, the digest is what you pass to the launcher:
sudo docker buildx imagetools inspect ghcr.io/cl-1-koi/open-dreamer-coinrun-runner:sha-<commit>
```

`io.coinrun.uv.lock.sha256` and `COINRUN_UV_LOCK_SHA256` must both equal
`sha256sum uv.lock` of the commit the image was built from.

### Local dry runs (no GPU required)

```bash
# CMD is replaceable -- the entrypoint must not swallow it.
sudo docker run --rm coinrun-runner:local experiment --help

# Offline dependency check against a mounted checkout.
sudo docker run --rm --network=none \
  --entrypoint /opt/coinrun/venv/bin/python \
  -e PYTHONPATH=/workspace/open-dreamer \
  -v "$PWD":/workspace/open-dreamer:ro coinrun-runner:local \
  -c "import jax, dreamer.models; print(jax.__version__, dreamer.models.__file__)"

# Offline Procgen generation -- proves the cache is warm, not that the science works.
sudo docker run --rm --network=none -w /workspace/open-dreamer \
  -v "$PWD":/workspace/open-dreamer:ro --entrypoint bash coinrun-runner:local \
  -c 'time uv run --isolated --script dreamer/data/generate_coinrun_dataset.py \
        --collector=scripted --num-episodes-train=2 --num-episodes-val=2 \
        --num-episodes-test=2 --max-episode-length=64 --chunk-size=32 \
        --chunks-per-file=1 --output-dir=/tmp/o --overwrite'
```

On a CPU-only host the full `smoke` mode correctly stops at the GPU gate
(`missing required binaries: nvidia-smi`) and writes
`<artifact-dir>/runner/smoke.json` with `"status": "failed"`. That is the
expected fail-closed result, not a bug.

## Publish

**Supported path today: publish locally.** The finished image is 26.46 GB and
the two-stage build additionally materialises the builder stage and the base, so
peak disk usage is well above what a standard GitHub-hosted runner provides
(~14 GB free on `/`, ~65 GB on `/mnt`). That has not been proven to fit, so the
CI workflow is `workflow_dispatch`-only and fails fast on a disk preflight
rather than exhausting disk mid-push.

```bash
echo "$GITHUB_TOKEN" | sudo docker login ghcr.io -u cl-1-koi --password-stdin
COMMIT=$(git rev-parse HEAD)
IMAGE=ghcr.io/cl-1-koi/open-dreamer-coinrun-runner
sudo docker tag coinrun-runner:slim "$IMAGE:sha-$COMMIT"
sudo docker push "$IMAGE:sha-$COMMIT"

# Record the digest -- this is what the launcher takes.
sudo docker buildx imagetools inspect "$IMAGE:sha-$COMMIT" | head -3
```

The token needs `write:packages`. If the package is new, make it visible to the
RunPod puller once via GitHub → Packages → `open-dreamer-coinrun-runner` →
Package settings.

`docs/ops/publish-coinrun-runner.yml` performs the same build and push.
It is **not installed** at `.github/workflows/` yet: the push credential for
this branch lacks the GitHub `workflow` OAuth scope. Install it with a
credential that has that scope (the file's header has the exact command). It
runs the build and push
with the workflow's own `GITHUB_TOKEN` (`packages: write`); no registry
credential exists in the repository. Run it only after pointing `runner` at a
large or self-hosted runner with at least ~80 GB free. On `ubuntu-latest` the
disk preflight is expected to fail by design.

## RunPod 5-minute smoke

Runs the same gates the experiment will run, at negligible cost.

```bash
python scripts/runpod_coinrun.py launch \
  --gpu H200 \
  --image ghcr.io/cl-1-koi/open-dreamer-coinrun-runner@sha256:<digest> \
  --preflight-report artifacts/coinrun_preflight/telemetry.json \
  --runtime-seconds 300 \
  --experiment-command 'coinrun-runner smoke --artifact-dir=$COINRUN_ARTIFACT_DIR'
# add --execute only when the dry-run output is what you expect
```

The launcher refuses a mutable image reference, refuses an
`--experiment-command` containing `uv sync`, and the remote bootstrap verifies
`COINRUN_UV_LOCK_SHA256` against the checked-out `uv.lock` before running
anything.

### Smoke success gates

All must hold in `<artifact-dir>/runner/smoke.json`:

1. `status` is `passed`.
2. `contract.uv_lock_sha256` equals the local `sha256sum uv.lock`.
3. `contract.commit` equals the pushed commit you launched.
4. `gpu.device_count` is 1 and `gpu.device_kind` is H100/H200/B200.
5. Both `smoke.commands[]` entries have `returncode` 0 and `shard_count` > 0.
6. `smoke.elapsed_seconds` is within the 300s budget.

Any failure exits nonzero and leaves `error` plus the failing stage in the same
file. Do not proceed to the four-hour run on a partial pass.

## Four-hour experiment

```bash
python scripts/runpod_coinrun.py launch \
  --gpu H200 \
  --image ghcr.io/cl-1-koi/open-dreamer-coinrun-runner@sha256:<digest> \
  --preflight-report artifacts/coinrun_preflight/telemetry.json \
  --runtime-seconds 14400 \
  --experiment-command 'coinrun-runner experiment --artifact-dir=$COINRUN_ARTIFACT_DIR' \
  --execute
```

`coinrun-runner experiment` runs the contract gates and then `exec`s
`scripts/run_coinrun_h100_experiment.sh`, so the controller's own 13,500s
deadline, per-phase timeouts, and signal handling apply unchanged. The
launcher's four-hour watchdog, automatic termination, cost gate, manifest
SHA/size checks, and no-secret logging are untouched by this work.

## Build telemetry

`scripts/build_coinrun_runner.py` appends one JSON object per build to
`artifacts/coinrun_runner/build_history.jsonl` (schema
`coinrun-runner-build-history-v1`) and writes the full BuildKit log beside it.

| Field | Meaning |
|---|---|
| `started_utc`, `ended_utc`, `duration_seconds` | Wall clock for the build |
| `git_commit`, `uv_lock_sha256` | Source identity of the build inputs |
| `image_tag`, `image_id`, `image_bytes` | Result (last two only on success) |
| `cache_label` | Operator label: `cold`, `warm`, `unknown` |
| `no_cache`, `returncode`, `log_path` | Build mode, outcome, log location |
| `host_before`, `host_after` | CPU count, load, memory, disk free |
| `resource_samples_path` | JSONL of per-interval samples taken during the build |
| `resource_summary` | Mean/peak CPU, mean/peak net RX, peak net TX, peak disk read/write, peak load, min MemAvailable, sample count |
| `stage_durations_seconds` | Coarse per-step seconds parsed from plain progress; `0.0` means CACHED |
| `source` | Always `wrapper` for generated records |

```bash
# Successful builds, newest last.
jq -r 'select(.returncode==0) | [.started_utc, .cache_label, .duration_seconds, .image_bytes] | @tsv' \
  artifacts/coinrun_runner/build_history.jsonl

# Slowest stages of the most recent build.
jq -s '.[-1].stage_durations_seconds | to_entries | sort_by(-.value) | .[:5]' \
  artifacts/coinrun_runner/build_history.jsonl

# Was the last build network-bound, CPU-bound, or neither?
jq -s '.[-1].resource_summary' artifacts/coinrun_runner/build_history.jsonl
```

Each build also writes `build-<timestamp>-resources.jsonl` beside its log: one
row every `--sample-seconds` (default 5) with UTC time, CPU utilization derived
from `/proc/stat` idle share, 1-minute load, `MemAvailable`, and per-second disk
read/write and network RX/TX rates derived from `/proc/diskstats` and
`/proc/net/dev`. Metrics unavailable on a host are omitted rather than zeroed.
These artifacts are gitignored; only the reconstructed historical CSV is
committed.

Builds observed before this wrapper existed are recorded separately, and marked
as reconstructed rather than measured, in
`docs/coinrun_runner_build_observations.csv`.

## Image size

The first single-stage build was 45.2 GB because a final
`chmod -R a+rX /opt/coinrun` rewrote metadata for every file in the venv and
cache, which overlayfs copies up as a second full 12.8 GB layer. The current
two-stage build removes that recursive chmod and confines the `uv sync`
downloads to a BuildKit cache mount, so `/opt/coinrun` lands as one flat layer.

| Build | Image | Size | `/opt/coinrun` layer |
|---|---|---:|---:|
| single stage, recursive chmod | `e40264b07eb9` | 45.2 GB | 12.1 GB + 0.63 GB + 12.8 GB |
| two stage (current) | `d58ea48ff863` | 26.5 GB | 6.85 GB |

The remaining ~19.6 GB is inherited from the `runpod/pytorch` base and is not
something this Dockerfile can shrink. From `docker history coinrun-runner:slim`,
the base contributes several large layers, the largest single one being the
7.06 GB Torch/CUDA wheel install (`RUN ... TORCH=torch==2.9.1 ...`), plus apt
layers of 5.99 GB, 3.11 GB, 1.15 GB and 1.05 GB. Those figures are the visible
per-layer sizes reported by `docker history`; the base image was pulled as part
of this build and is not tagged locally on its own, so the total inherited size
is stated as `docker image inspect` total (26,461,584,632 B) minus our single
6.85 GB `/opt/coinrun` layer, not as a separately measured base image.

The download cache mount is a build-time artifact only; the pod never depends
on it.
