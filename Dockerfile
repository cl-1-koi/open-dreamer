# Prebuilt CoinRun runner image.
#
# Base: the image scripts/runpod_coinrun.py already provisions (its
# DEFAULT_IMAGE), which satisfies the pod payload's
# allowedCudaVersions=["12.8","12.9","13.0"] gate. Staying on it keeps the CUDA
# 12.8 userspace identical to the paid attempts, so this is a packaging change
# only.
#
# Removed from every paid pod: the `uv sync` of the jax[cuda12] tree, the
# Procgen source build (cmake + Qt5 + C++), and ad-hoc apt installs.
#
# Never baked in: credentials, datasets, or an importable copy of the project.
# The dependency environment is baked; `dreamer`/`scripts` are always imported
# from the runtime Git checkout via PYTHONPATH, so a pod cannot silently run
# image code instead of the pushed commit.
#
# Two stages on purpose. A single stage cannot shrink: `uv cache prune` and
# `chmod -R` in a later layer only shadow bytes, they never remove them, and a
# recursive metadata change makes overlayfs copy up the whole ~12.8 GB venv as a
# duplicate layer. The builder assembles and prunes /opt/coinrun; the final
# stage copies the finished tree once, so the published image contains exactly
# one copy of it.
#
# LAYER-ORDER RULE. BuildKit folds every in-scope ARG into the cache key of each
# subsequent RUN -- visible in `docker history` as `RUN |3 SOURCE_COMMIT=... `.
# Declaring the per-commit metadata ARGs near the top therefore invalidated apt,
# `uv sync` and the Procgen prewarm on every source-only commit (measured: a
# SOURCE_COMMIT-only change forced a 74.8s rebuild instead of the 0.5s fully
# cached path). So: the builder never sees SOURCE_COMMIT/SOURCE_REPO at all, and
# in the final stage every ARG/LABEL/ENV carrying per-commit values is declared
# AFTER the last expensive RUN and COPY. Nothing at build time reads the
# COINRUN_* contract variables -- they are only read at pod runtime -- so
# declaring them last costs no correctness.

ARG BASE_IMAGE=runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.10.4

FROM ${UV_IMAGE} AS uv-src


# --------------------------------------------------------------------------
# Builder: assemble /opt/coinrun. Depends only on uv.lock, pyproject.toml and
# the dataset generator. No LABEL here (labels on a non-final stage are
# discarded) and no source-identity ARGs.
# --------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/coinrun/venv \
    UV_CACHE_DIR=/opt/coinrun/uv-cache \
    UV_PYTHON_INSTALL_DIR=/opt/coinrun/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

# git: exact-commit checkout. coreutils: GNU timeout for phase deadlines.
# grep: controller GPU-model gate. ffmpeg: imageio rollout MP4s.
# cmake/build-essential/qtbase5-dev/libgl1-mesa-dev: Procgen builds libenv.so
# from source (its CMakeLists does find_package(Qt5 COMPONENTS Gui REQUIRED)).
# libqt5gui5/libgl1/libglx0/libx11-6: runtime load of that libenv.so. Build
# tooling is kept so an isolated script can still rebuild if the cache misses.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake coreutils curl ffmpeg git grep \
        gzip libgl1 libgl1-mesa-dev libglx0 libqt5gui5 libx11-6 qtbase5-dev tar \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv-src /uv /uvx /usr/local/bin/

RUN uv python install 3.11
ENV UV_PYTHON_DOWNLOADS=never

WORKDIR /opt/coinrun/build

# --no-install-project is deliberate (see header). --frozen forbids lock drift.
#
# The download cache is a BuildKit cache mount, NOT the image's runtime cache:
# a canceled or retried build reuses already-downloaded wheels instead of
# re-fetching the multi-GB jax[cuda12] set. UV_LINK_MODE=copy (set above) means
# the venv is a real copy, so nothing at pod runtime depends on this mount. It
# also keeps those wheels out of /opt/coinrun/uv-cache entirely, which is why no
# `uv cache prune` is needed afterwards.
COPY pyproject.toml uv.lock README.md ./

# Declared here, immediately before its only use, so apt and `uv python install`
# above are not rekeyed by it. It changes exactly when uv.lock changes, which
# already invalidates the COPY above, so it costs no extra rebuild.
ARG UV_LOCK_SHA256=unknown

RUN --mount=type=cache,id=coinrun-uv-build,target=/root/.cache/uv-build,sharing=locked \
    UV_CACHE_DIR=/root/.cache/uv-build \
    uv sync --frozen --no-install-project \
    && test -x /opt/coinrun/venv/bin/python \
    && test "$(sha256sum uv.lock | cut -d' ' -f1)" = "${UV_LOCK_SHA256}"

# Prewarm Procgen. A real one-episode generation, not --help: Procgen compiles
# libenv.so lazily on first env construction, so only this populates
# UV_CACHE_DIR with the built wheel and proves the toolchain works. The tiny
# output is deleted -- no dataset is baked in.
COPY dreamer/data/generate_coinrun_dataset.py ./generate_coinrun_dataset.py
RUN uv run --isolated --script ./generate_coinrun_dataset.py \
        --num-episodes-train=1 --num-episodes-val=1 --num-episodes-test=1 \
        --max-episode-length=64 --chunk-size=32 --chunks-per-file=1 \
        --output-dir=/tmp/procgen-warmup \
    && test -n "$(find /opt/coinrun/uv-cache -name libenv.so -print -quit)" \
    && rm -rf /tmp/procgen-warmup ./generate_coinrun_dataset.py

# Drop the build scratch dir. Keep the exact uv binary beside the cache it
# created: the RunPod base image may carry an older uv which cannot consume a
# newer cache format. No `uv cache prune` here on purpose: with the sync's
# downloads confined to the cache mount above, /opt/coinrun/uv-cache now holds
# only what the offline Procgen smoke needs -- the source-built Procgen wheel
# plus the isolated script's own wheels (gym3, numpy<2, tyro). Pruning would
# delete the latter and force a network install on the pod.
RUN set -eu; \
    mkdir -p /opt/coinrun/bin /opt/coinrun/runtime-libs; \
    cp /usr/local/bin/uv /opt/coinrun/bin/uv; \
    libenv="$(find /opt/coinrun/uv-cache -name libenv.so -print -quit)"; \
    ldd "$libenv" \
      | awk '$2 == "=>" && $3 ~ "^/" { print $3 }' \
      | while IFS= read -r library; do \
          case "$(basename "$library")" in \
            libc.so.*|libm.so.*|libpthread.so.*|libdl.so.*|librt.so.*|libresolv.so.*|libutil.so.*) \
              continue ;; \
          esac; \
          cp -L "$library" "/opt/coinrun/runtime-libs/$(basename "$library")"; \
        done; \
    rm -rf /opt/coinrun/build; \
    test -x /opt/coinrun/bin/uv; \
    test -s /opt/coinrun/runtime-libs/libQt5Gui.so.5; \
    ! LD_LIBRARY_PATH=/opt/coinrun/runtime-libs ldd "$libenv" | grep -q "not found"


# --------------------------------------------------------------------------
# Final stage: static dependency layers first, per-commit metadata last.
# --------------------------------------------------------------------------
FROM ${BASE_IMAGE}

# Static environment only. The per-commit COINRUN_* contract variables are set
# at the bottom of this file, after the expensive layers.
ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/coinrun/venv \
    UV_CACHE_DIR=/opt/coinrun/uv-cache \
    UV_PYTHON_INSTALL_DIR=/opt/coinrun/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_OFFLINE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake coreutils curl ffmpeg git grep \
        gzip libgl1 libgl1-mesa-dev libglx0 libqt5gui5 libx11-6 qtbase5-dev tar \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv-src /uv /uvx /usr/local/bin/

# One flat copy of the finished, pruned environment. No recursive chmod follows:
# `chmod -R` on this tree would copy up every file again. Root-created files are
# already 644/755, which is what the pod (running as root) needs.
COPY --from=builder /opt/coinrun /opt/coinrun

COPY scripts/coinrun_runner.py /opt/coinrun/runner/coinrun_runner.py
RUN nvidia_libs="$(find /opt/coinrun/venv/lib/python3.11/site-packages/nvidia \
        -type d -name lib -print 2>/dev/null | sort | paste -sd: -)" \
    && runtime_libs="/opt/coinrun/runtime-libs${nvidia_libs:+:$nvidia_libs}" \
    && dataset_python="$(find /opt/coinrun/uv-cache/environments-v2 \
        -mindepth 3 -maxdepth 3 -path '*/bin/python' -type l -print -quit)" \
    && test -n "$dataset_python" \
    && ln -s "$dataset_python" /usr/local/bin/coinrun-dataset-python \
    && printf '#!/bin/sh\nexport LD_LIBRARY_PATH=%s${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\nexec /opt/coinrun/venv/bin/python /opt/coinrun/runner/coinrun_runner.py "$@"\n' \
      "$runtime_libs" \
      > /usr/local/bin/coinrun-runner \
    && chmod 0755 /usr/local/bin/coinrun-runner

WORKDIR /workspace

# --- per-commit metadata: everything below is rekeyed on every commit ---
# Nothing after this point runs a command or copies a payload, so a
# source-only change rewrites metadata layers and nothing else.
ARG SOURCE_COMMIT=unknown
ARG UV_LOCK_SHA256=unknown
ARG SOURCE_REPO=https://github.com/cl-1-koi/open-dreamer

LABEL org.opencontainers.image.title="open-dreamer-coinrun-runner" \
      org.opencontainers.image.source="${SOURCE_REPO}" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.licenses="LicenseRef-All-Rights-Reserved" \
      io.coinrun.uv.lock.sha256="${UV_LOCK_SHA256}"

# The runner contract, read by scripts/coinrun_runner.py and by the remote
# bootstrap in scripts/runpod_coinrun.py. Labels above mirror it for
# `docker inspect`; env is what the pod actually checks. Read only at pod
# runtime, never during the build, which is why it can be declared last.
ENV COINRUN_IMAGE_CONTRACT=1 \
    COINRUN_SOURCE_COMMIT=${SOURCE_COMMIT} \
    COINRUN_UV_LOCK_SHA256=${UV_LOCK_SHA256}

# Exec-form ENTRYPOINT plus a separate CMD: `docker run IMAGE` smokes, and
# `docker run IMAGE experiment` replaces only the CMD. Commit 10a0252 fixed the
# inverse bug (an entrypoint swallowing CMD); tests pin this shape.
ENTRYPOINT ["/usr/local/bin/coinrun-runner"]
CMD ["smoke"]
