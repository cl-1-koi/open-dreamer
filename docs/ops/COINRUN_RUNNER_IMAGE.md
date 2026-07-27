# CoinRun runner image — operator guide

Prebuilt OCI image that removes per-pod dependency installation from the CoinRun
sprint. Nothing in this document launches paid compute.

**Default transport is now the direct bundle, not a registry image.** Publishing
26.46 GB is unnecessary when only `/opt/coinrun` (3.00 GB compressed) is absent
from the pinned base. GHCR is optional/legacy; see "Image transport (legacy)".

| Item | Value |
|---|---|
| Bundle manifest | `manifests/coinrun_runner_bundle.json` (tracked) |
| Bundle archive | `artifacts/coinrun_bundle/*.tar.zst` (gitignored, rebuildable) |
| Pinned base | `runpod/pytorch@sha256:60baa36d…` (from the manifest) |
| Image (legacy) | `ghcr.io/cl-1-koi/open-dreamer-coinrun-runner` |
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

## Direct-bundle transport (default)

### 1. Build the bundle from a verified local image

```bash
python scripts/build_coinrun_bundle.py --image coinrun-runner:slim
```

It refuses to package an image whose `COINRUN_UV_LOCK_SHA256` and
`io.coinrun.uv.lock.sha256` disagree, or that was built for a different
`uv.lock` than this checkout. It does **not** care whether the image's commit
equals HEAD — see "Provenance vs. compatibility" below. It writes the archive under
`artifacts/coinrun_bundle/` (never committed), the manifest to
`manifests/coinrun_runner_bundle.json` (committed), and package telemetry to
`artifacts/coinrun_bundle/package_telemetry.jsonl`.

The pinned base must be present locally so its digest can be recorded:

```bash
sudo docker pull runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
```

Measured on this host: 6.91 GB tar → 3.00 GB `.tar.zst` in 188 s at level 9.

### 2. Stage the package and create an ephemeral pod key

The archive is sent directly and is never checked in. On this host,
local-to-RunPod egress was unusably slow, while staging the 3.00 GB package on
`fw-robot1` took 247 seconds:

```bash
ssh fw-robot1 'mkdir -p /root/coinrun-bundles'
rsync --archive --partial --append-verify --compress-level=0 \
  artifacts/coinrun_bundle/coinrun-opt-195069f66bce.tar.zst \
  manifests/coinrun_runner_bundle.json \
  fw-robot1:/root/coinrun-bundles/

# Use a fresh key for this pod. The relay receives it only for the transfer and
# the launcher removes its relay copy in a finally path.
mkdir -p artifacts/runpod_coinrun
ssh-keygen -q -t ed25519 -N '' \
  -f artifacts/runpod_coinrun/pod-transfer-key
```

The launcher validates size and SHA-256 of both staged files against the local
manifest before creating paid compute. The original package remains available
locally and on the operator-owned relay; neither copy is a Git object.

### 3. Dry run (validates everything locally, launches nothing)

The preferred path uses the persistent RunPod network-volume cache. The
volume and pod must be in the same data center:

```bash
python scripts/runpod_coinrun.py launch \
  --transport bundle --gpu H200 \
  --preflight-report artifacts/coinrun_preflight/telemetry.json \
  --runtime-seconds 300 --setup-timeout-seconds 600 \
  --network-volume-id 77iy0y26si \
  --network-volume-data-center-id US-NC-1 \
  --network-volume-mount-path /runpod-volume \
  --bundle-volume-dir /runpod-volume/coinrun-bundles \
  --ssh-key artifacts/runpod_coinrun/pod-transfer-key \
  --ssh-public-key artifacts/runpod_coinrun/pod-transfer-key.pub \
  --experiment-command 'coinrun-runner smoke --artifact-dir=$COINRUN_ARTIFACT_DIR'
```

With `--bundle-volume-dir`, setup verifies and extracts the cached archive but
does not run rsync. The launcher records the volume ID, data center, mount,
verified byte count, and zero network-transfer bytes in its state and transfer
telemetry. `--bundle-volume-dir` and `--bundle-relay-host` are mutually
exclusive.

The relay path remains available as a reconstruction fallback:

```bash
python scripts/runpod_coinrun.py launch \
  --transport bundle --gpu H200 \
  --preflight-report artifacts/coinrun_preflight/telemetry.json \
  --runtime-seconds 300 --setup-timeout-seconds 1200 \
  --bundle-relay-host fw-robot1 \
  --bundle-relay-dir /root/coinrun-bundles \
  --ssh-key artifacts/runpod_coinrun/pod-transfer-key \
  --ssh-public-key artifacts/runpod_coinrun/pod-transfer-key.pub \
  --experiment-command 'coinrun-runner smoke --artifact-dir=$COINRUN_ARTIFACT_DIR'
```

Before any RunPod mutation this checks the manifest schema, that the manifest's
`uv_lock_sha256` matches the local `uv.lock`, that `contract_env` agrees with
it, that the extract target is `/opt/coinrun`, that the archive filename is not
a traversal, and that the archive's size **and** SHA-256 match. Add `--execute`
to launch.

### 4. What the pod does

1. Boots the pinned base by digest with `ports: ["8000/http", "22/tcp"]`,
   `supportPublicIp`, and `PUBLIC_KEY` in env — no account credential is ever
   put in the payload; the remote watchdog uses RunPod's injected pod-scoped
   `RUNPOD_API_KEY`, sourced from `/etc/rp_environment` in SSH sessions.
2. Its start command arms the self-terminate watchdog against the absolute
   paid-work deadline, then `exec /start.sh` so sshd comes up. The pod is
   bounded from boot, so a local crash cannot leave it running.
3. The launcher polls REST `publicIp` + `portMappings["22"]` and waits for
   sshd. With `--bundle-volume-dir`, it reads the archive and manifest directly
   from the attached network volume. Otherwise, it asks the selected relay to
   rsync (`--partial --append-verify`, resumable) both files to
   `/workspace/coinrun-bundle`. The relay receives the pod's ephemeral private
   key only for this operation and removes it and its pod-specific
   `known_hosts` file even if rsync fails. Without either source option, the
   original local-to-pod path remains available.
4. Remotely verifies the size and SHA-256 of *both* files against values
   computed locally, extracts atomically into `/opt/coinrun` (`.incoming` then
   rename), and checks `venv/bin/python`, a `jax/flax/optax` import, and the
   Procgen `libenv.so`.
5. Installs the bounded remote experiment script in one synchronous SSH call
   (verifying its bytes and hash), then starts it in a second call. The
   existing HTTP log/artifact monitor on port 8000 is unchanged.

### 5. Budgets and cost

`--setup-timeout-seconds` (default 1200, max 3600) bounds everything up to the
moment the experiment starts. It is added to `--runtime-seconds` for the
projected-spend and balance check, for the local watchdog, and for the pod's
absolute remote deadline. The experiment itself is bounded by exactly
`--runtime-seconds`, so unused setup budget never extends training.

Any setup failure raises, and the existing automatic-termination path runs.

### 6. Transfer telemetry

Appended to `artifacts/coinrun_bundle/transfer_telemetry.jsonl` and mirrored in
the launcher state under `bundle_transfer`: `launch_to_endpoint_seconds`,
`launch_to_ssh_seconds`, `transfer_seconds`, `transfer_bytes`,
`transfer_bytes_per_second`, `transfer_source`, `bundle_bytes_verified`,
`remote_verify_extract_seconds`,
`experiment_started_utc`, `setup_seconds`. No credentials are recorded.

Small, manually verified historical transfer probes are tracked in
`docs/coinrun_bundle_transfer_observations.csv`. The archive itself remains
untracked; only its manifest and hashes belong in Git.

Observed Hetzner-to-RunPod transfer rates varied from 17.27 MB/s to 78.40 MB/s
for the same 3.00 GB package. Both successful setups nevertheless reached the
experiment in about 275-280 seconds because endpoint publication and transfer
time traded off. Treat provider UI "HTTP service initializing" as the state of
the port-8000 proxy, not proof that the pod or SSH is unavailable.

The runner shim prepends every `site-packages/nvidia/*/lib` directory to
`LD_LIBRARY_PATH`. The locked JAX CUDA wheels contain cuSPARSE and the other
libraries, but JAX 0.10.1 falls back to CPU when the dynamic loader cannot see
those wheel directories.

### 7. Recovery and reconstruction

The archive is deliberately not committed. To recover it from a clean checkout:

```bash
python scripts/build_coinrun_runner.py --tag coinrun-runner:slim --cache-label cold
python scripts/build_coinrun_bundle.py --image coinrun-runner:slim
git diff --stat manifests/coinrun_runner_bundle.json   # expect archive.sha256 to differ
```

Reconstruction is **semantic, not bit-exact**. tar metadata is normalised
(`--sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner`), but the payload
still contains absolute paths, `UV_COMPILE_BYTECODE=1` `.pyc` files whose
headers embed source mtimes, and uv cache entries with build-time-dependent
contents. Expect a different `archive.sha256` with the same behaviour; the
`source.dockerfile_sha256`, `source.uv_lock_sha256` and `base_image.digest`
fields are what make the rebuild verifiable.

### 8. Provenance vs. compatibility (do not chase HEAD)

`source.commit` and `COINRUN_SOURCE_COMMIT` are **provenance**: they say which
commit the environment was built from. They are *not* a runtime-identity check
and must never be required to equal the checkout you launch.

The manifest is itself checked in, so committing an updated manifest always
creates a newer HEAD. Requiring `source.commit == HEAD` is an impossible
self-reference loop — the bundle would be stale the instant you recorded it.

The two concerns are enforced separately:

| Question | Enforced by |
|---|---|
| Is the pod running the exact code I pushed? | `--expect-commit` / the bootstrap's `git rev-parse HEAD` check |
| Is this bundle valid for this checkout? | `uv.lock` SHA, plus the gated build inputs below |

A bundle built at an older commit is **correct and expected** for a later
checkout, as long as the build inputs match.

#### Rebuild triggers

Rebuild the image and the bundle when any of these change:

The bundle is a **dependency environment**. Only inputs that change the
dependencies baked into `/opt/coinrun` may gate a launch. All executable
orchestration runs from the pinned runtime checkout, so it can never stale a
bundle.

| Input | Gated? | Why |
|---|---|---|
| `uv.lock` | **yes, fails closed** | Determines the venv contents |
| PEP 723 block of `dreamer/data/generate_coinrun_dataset.py` | **yes, fails closed** | Pins Procgen, whose built wheel is prewarmed into the bundle's uv cache |
| Base image digest | rebuild manually | A different base changes the ABI the venv was built against |
| Dockerfile apt/`uv sync`/prewarm steps | rebuild manually | They assemble `/opt/coinrun`; recorded via `dockerfile_sha256` but not gated, since comments and layer order leave the payload identical |
| That generator's **body** | no | Runs from the checkout at runtime |
| `scripts/coinrun_runner.py` | **no** | Executed from the checkout via the shim below, never from the archive |
| `scripts/runpod_coinrun.py`, docs | no | Launcher-side only; never enter the bundle |
| `scripts/build_coinrun_bundle.py` | no | Packaging logic; changes the archive, not its semantic content |

So: rebuild for dependency, base, or prewarm changes. Do **not** rebuild for
docs, launcher, or runner changes — and never merely to advance
`source.commit`. Gated hashes live under `build_inputs` and are enforced only
when present, so manifests written before that field remain usable.

#### Where the runner executable comes from

In bundle mode the setup step recreates `/usr/local/bin/coinrun-runner` as:

```sh
exec /opt/coinrun/venv/bin/python \
  "${COINRUN_CHECKOUT_ROOT:-/workspace/open-dreamer}/scripts/coinrun_runner.py" "$@"
```

The interpreter and its packages come from the bundle; the orchestration code
comes from the exact commit the bootstrap cloned and verified, which exports
`COINRUN_CHECKOUT_ROOT` before running the experiment command. The archive's own
`runner/coinrun_runner.py` is inert.

Image mode is unchanged: its baked shim runs the image's copy, which is pinned
by the image's immutable digest or `sha-<commit>` tag.

## Image transport (legacy, optional)

The image transport still works and is unchanged: pass `--transport image`
with an immutable `--image`. It is only needed if you specifically want a
registry-distributed runner; the bundle transport requires no registry at all.

**If you do publish: locally.** The finished image is 26.46 GB and
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

## Layer ordering and commit-only rebuilds

BuildKit folds every in-scope `ARG` into the cache key of each later `RUN`
(visible in `docker history` as `RUN |3 SOURCE_COMMIT=... `). Declaring the
per-commit metadata near the top of a stage therefore rebuilt apt, `uv sync`
and the Procgen prewarm on every source-only commit -- measured at 74.8 s for a
`SOURCE_COMMIT`-only change versus 0.5 s for a fully cached build.

The Dockerfile now keeps `SOURCE_COMMIT`/`SOURCE_REPO` out of the builder
entirely, and in the final stage declares every per-commit `ARG`/`LABEL`/`ENV`
after the last `RUN` and `COPY`. `UV_LOCK_SHA256` is declared in the builder
immediately before the `uv sync` that verifies against it, so it rekeys only
what `uv.lock` already invalidates. Nothing reads the `COINRUN_*` contract
variables at build time -- only at pod runtime -- so declaring them last costs
no correctness. `tests/test_coinrun_runner.py` pins this ordering.
