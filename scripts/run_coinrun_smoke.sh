#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_seconds="${COINRUN_SMOKE_RUNTIME_SECONDS:-1200}"
outer_grace_seconds="${COINRUN_SMOKE_OUTER_GRACE_SECONDS:-30}"

if ! [[ "${runtime_seconds}" =~ ^[0-9]+$ ]] || (( runtime_seconds <= 0 )); then
  echo "COINRUN_SMOKE_RUNTIME_SECONDS must be a positive integer" >&2
  exit 2
fi
if ! [[ "${outer_grace_seconds}" =~ ^[0-9]+$ ]]; then
  echo "COINRUN_SMOKE_OUTER_GRACE_SECONDS must be a non-negative integer" >&2
  exit 2
fi

cd "${repo_root}"
exec timeout \
  --foreground \
  --signal=TERM \
  --kill-after=15s \
  "$((runtime_seconds + outer_grace_seconds))s" \
  uv run python scripts/coinrun_preflight.py \
  "$@" \
  --runtime-seconds "${runtime_seconds}"
