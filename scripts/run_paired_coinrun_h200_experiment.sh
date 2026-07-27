#!/usr/bin/env bash
# Bounded controller for the exact paired CoinRun pixel experiment. This script
# runs only inside an already-provisioned high-memory GPU host.
set -Eeuo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly MAX_TOTAL_SECONDS="${COINRUN_MAX_TOTAL_SECONDS:-13500}"
readonly TRAINING_SECONDS="${COINRUN_PAIRED_TRAINING_SECONDS:-10800}"
readonly EVAL_SECONDS="${COINRUN_EVAL_SECONDS:-600}"

artifact_dir=""
dataset_dir=""
manifest_sha256=""
seed="${COINRUN_SEED:-20260727}"
preflight_only=0

usage() {
  cat <<'EOF'
Usage: scripts/run_paired_coinrun_h200_experiment.sh \
  --artifact-dir /absolute/output \
  --dataset-dir /absolute/coinrun_paired_v2_open_dreamer \
  --manifest-sha256 SHA256 [--seed INTEGER]

Runs the fixed paired CoinRun tokenizer, held-out latent-stat probe, dynamics
model, and held-out evaluator. It never provisions or terminates a GPU host.
Use --preflight-only to validate the corpus and fully resolve both model
configurations without requiring a GPU or launching training.
EOF
}

die() {
  FINAL_ERROR="$*"
  printf 'paired-coinrun-h200: %s\n' "$*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --artifact-dir)
      (($# >= 2)) || die "--artifact-dir requires a path"
      artifact_dir="$2"
      shift 2
      ;;
    --dataset-dir)
      (($# >= 2)) || die "--dataset-dir requires a path"
      dataset_dir="$2"
      shift 2
      ;;
    --manifest-sha256)
      (($# >= 2)) || die "--manifest-sha256 requires a digest"
      manifest_sha256="$2"
      shift 2
      ;;
    --seed)
      (($# >= 2)) || die "--seed requires an integer"
      seed="$2"
      shift 2
      ;;
    --preflight-only)
      preflight_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$artifact_dir" = /* ]] || die "--artifact-dir must be absolute"
[[ "$dataset_dir" = /* ]] || die "--dataset-dir must be absolute"
[[ "$manifest_sha256" =~ ^[0-9a-f]{64}$ ]] || die "manifest digest must be lowercase SHA-256"
[[ "$seed" =~ ^[0-9]+$ ]] || die "seed must be a non-negative integer"
for seconds in "$MAX_TOTAL_SECONDS" "$TRAINING_SECONDS" "$EVAL_SECONDS"; do
  [[ "$seconds" =~ ^[1-9][0-9]*$ ]] || die "timeouts must be positive integers"
done

ARTIFACT_DIR="${artifact_dir%/}"
DATASET_DIR="${dataset_dir%/}"
RUN_DIR="$ARTIFACT_DIR/paired_run"
PREFLIGHT_DIR="$ARTIFACT_DIR/preflight"
LOG_DIR="$ARTIFACT_DIR/logs"
TELEMETRY_DIR="$ARTIFACT_DIR/telemetry"
GATE_DIR="$ARTIFACT_DIR/gates"
MANIFEST_PATH="$ARTIFACT_DIR/final_manifest.json"
PHASE_TIMINGS="$TELEMETRY_DIR/phase_timings.tsv"
START_EPOCH=0
DEADLINE_EPOCH=0
CURRENT_PHASE="setup"
FINAL_STATUS="failed"
FINAL_ERROR=""

if [[ -d "$ARTIFACT_DIR" ]] && [[ -n "$(find "$ARTIFACT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  die "artifact directory must be empty: $ARTIFACT_DIR"
fi
[[ -f "$DATASET_DIR/manifest.json" ]] || die "paired manifest is missing"
[[ -d "$DATASET_DIR/train" && -d "$DATASET_DIR/val" ]] || die "paired train/val splits are missing"
mkdir -p "$ARTIFACT_DIR" "$LOG_DIR" "$TELEMETRY_DIR" "$GATE_DIR" "$ARTIFACT_DIR/git"

write_manifest() {
  export ARTIFACT_DIR MANIFEST_PATH PHASE_TIMINGS START_EPOCH DEADLINE_EPOCH
  export MAX_TOTAL_SECONDS CURRENT_PHASE FINAL_STATUS FINAL_ERROR DATASET_DIR
  python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["ARTIFACT_DIR"])
timings = []
timing_path = Path(os.environ["PHASE_TIMINGS"])
if timing_path.exists():
    for line in timing_path.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) == 5:
            timings.append({
                "name": fields[0],
                "started_epoch": int(fields[1]),
                "elapsed_seconds": int(fields[2]),
                "returncode": int(fields[3]),
                "deadline_seconds": int(fields[4]),
            })

def text(path):
    path = Path(path)
    return path.read_text(encoding="utf-8").strip() if path.exists() else None

contract_path = root / "paired_run" / "paired_run_contract.json"
contract = None
if contract_path.exists():
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

manifest = {
    "schema_version": "paired-coinrun-h200-controller-v1",
    "status": os.environ["FINAL_STATUS"],
    "failure": os.environ["FINAL_ERROR"] or None,
    "current_phase": os.environ["CURRENT_PHASE"],
    "started_epoch": int(os.environ["START_EPOCH"]),
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "hard_total_deadline_seconds": int(os.environ["MAX_TOTAL_SECONDS"]),
    "deadline_epoch": int(os.environ["DEADLINE_EPOCH"]),
    "dataset_dir": os.environ["DATASET_DIR"],
    "git": {
        "commit": text(root / "git" / "commit.txt"),
        "branch": text(root / "git" / "branch.txt"),
        "status_porcelain": text(root / "git" / "status_porcelain.txt"),
    },
    "phases": timings,
    "paired_contract": contract,
    "required_gates": {
        name: text(root / "gates" / f"{name}.txt")
        for name in (
            "jax_gpu",
            "manifest",
            "tokenizer_checkpoints",
            "dynamics_checkpoints",
            "evaluator",
            "rollout_mp4",
        )
    },
    "gpu_telemetry": sorted(
        str(path.relative_to(root))
        for path in (root / "telemetry").glob("gpu_*.csv")
    ),
}
Path(os.environ["MANIFEST_PATH"]).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

finish() {
  local rc="$1"
  trap - EXIT
  if ((rc != 0)) && [[ -z "$FINAL_ERROR" ]]; then
    FINAL_ERROR="controller exited with status $rc during $CURRENT_PHASE"
  fi
  write_manifest
  exit "$rc"
}
trap 'finish $?' EXIT

remaining_seconds() {
  printf '%s\n' "$((DEADLINE_EPOCH - $(date +%s)))"
}

run_phase() {
  local name="$1"
  local phase_limit="$2"
  shift 2
  local started remaining effective rc monitor_pid=""
  CURRENT_PHASE="$name"
  started="$(date +%s)"
  remaining="$(remaining_seconds)"
  ((remaining > 0)) || { FINAL_ERROR="hard deadline reached before $name"; return 124; }
  effective="$phase_limit"
  ((remaining < effective)) && effective="$remaining"
  printf '%q ' "$@" > "$ARTIFACT_DIR/${name}.command"
  printf '\n' >> "$ARTIFACT_DIR/${name}.command"
  if command -v nvidia-smi >/dev/null 2>&1; then
    (
      while true; do
        date -u +%Y-%m-%dT%H:%M:%SZ
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
          --format=csv,noheader,nounits || true
        sleep 5
      done
    ) >> "$TELEMETRY_DIR/gpu_${name}.csv" 2>&1 &
    monitor_pid="$!"
  fi
  set +e
  timeout --foreground --signal=TERM --kill-after=30s "${effective}s" "$@" >"$LOG_DIR/${name}.log" 2>&1
  rc=$?
  set -e
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$started" "$(( $(date +%s) - started ))" "$rc" "$effective" >> "$PHASE_TIMINGS"
  if ((rc != 0)); then
    FINAL_ERROR="$name failed with status $rc; see $LOG_DIR/${name}.log"
    tail -n 100 "$LOG_DIR/${name}.log" >&2 || true
  fi
  return "$rc"
}

check_checkpoints() {
  local stage="$1"
  local directory="$2"
  local count
  count="$(find "$directory" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | grep -Ec '^[0-9]+$' || true)"
  ((count >= 2)) || {
    FINAL_ERROR="$stage checkpoint gate expected at least two numbered checkpoints; found $count"
    return 1
  }
  printf 'passed: %s checkpoints\n' "$count" > "$GATE_DIR/${stage}_checkpoints.txt"
}

paired_args() {
  local mode="$1"
  local output_dir="$2"
  PAIRED_COMMAND=(
    uv run python "$REPO_ROOT/scripts/paired_coinrun_training.py" "$mode"
    "--repo-root=$REPO_ROOT"
    "--run-dir=$output_dir"
    "--paired-manifest=$DATASET_DIR/manifest.json"
    "--paired-manifest-sha256=$manifest_sha256"
    "--train-array-record-dir=$DATASET_DIR/train"
    "--validation-array-record-dir=$DATASET_DIR/val"
  )
  local override
  for override in \
    "dataset.dataloader_cfg.B=128" \
    "dataset.dataloader_cfg.short_T=32" \
    "dataset.dataloader_cfg.long_T=32" \
    "dataset.dataloader_cfg.num_workers=8" \
    "dataset.dataloader_cfg.prefetch_buffer_size=8" \
    "dataset.dataloader_cfg.device_prefetch_buffer_size=2" \
    "max_steps=12000" \
    "lr_schedule.max_steps=12000" \
    "ckpt.max_steps=12000" \
    "ckpt.max_to_keep=3" \
    "ckpt.save_interval_steps=1000" \
    "logger.log_every=25" \
    "logger.use_wandb=false" \
    "use_wandb=false" \
    "lpips_weight=0" \
    "visualize_every=0" \
    "tokenizer_loss_type=mse" \
    "seed=$seed" \
    "tokenizer.encoder.n_latents=64" \
    "tokenizer.encoder.d_bottleneck=16" \
    "tokenizer.encoder.depth=8" \
    "tokenizer.encoder.d_model=512" \
    "tokenizer.encoder.n_heads=8" \
    "tokenizer.encoder.n_kv_heads=1" \
    "tokenizer.encoder.context_length=32" \
    "tokenizer.decoder.n_latents=64" \
    "tokenizer.decoder.d_bottleneck=16" \
    "tokenizer.decoder.depth=8" \
    "tokenizer.decoder.d_model=512" \
    "tokenizer.decoder.n_heads=8" \
    "tokenizer.decoder.n_kv_heads=1"; do
    PAIRED_COMMAND+=("--tokenizer-override=$override")
  done
  for override in \
    "dataset.dataloader_cfg.B=64" \
    "dataset.dataloader_cfg.short_T=48" \
    "dataset.dataloader_cfg.long_T=48" \
    "dataset.dataloader_cfg.long_ratio=0" \
    "dataset.dataloader_cfg.num_workers=8" \
    "dataset.dataloader_cfg.prefetch_buffer_size=8" \
    "dataset.dataloader_cfg.device_prefetch_buffer_size=2" \
    "max_steps=10000" \
    "lr_schedule.max_steps=10000" \
    "ckpt.max_steps=10000" \
    "ckpt.max_to_keep=3" \
    "ckpt.save_interval_steps=1000" \
    "logger.log_every=25" \
    "logger.use_wandb=false" \
    "use_wandb=false" \
    "seed=$seed" \
    "bootstrap_start=10000" \
    "bootstrap_fraction=0" \
    "image_fraction=0" \
    "ot.enabled=false" \
    "dynamics.k_max=8" \
    "write_video_every=10000" \
    "dynamics.d_bottleneck=16" \
    "dynamics.depth=12" \
    "dynamics.d_model=512" \
    "dynamics.n_heads=8" \
    "dynamics.n_kv_heads=1" \
    "dynamics.packing_factor=2" \
    "dynamics.n_register=8" \
    "dynamics.context_length=32"; do
    PAIRED_COMMAND+=("--dynamics-override=$override")
  done
}

START_EPOCH="$(date +%s)"
DEADLINE_EPOCH="$((START_EPOCH + MAX_TOTAL_SECONDS))"
printf 'name\tstarted_epoch\telapsed_seconds\treturncode\tdeadline_seconds\n' > "$PHASE_TIMINGS"
git -C "$REPO_ROOT" rev-parse HEAD > "$ARTIFACT_DIR/git/commit.txt"
git -C "$REPO_ROOT" branch --show-current > "$ARTIFACT_DIR/git/branch.txt"
git -C "$REPO_ROOT" status --porcelain > "$ARTIFACT_DIR/git/status_porcelain.txt"

command -v uv >/dev/null 2>&1 || die "uv is required"
command -v timeout >/dev/null 2>&1 || die "GNU timeout is required"
actual_manifest_sha="$(sha256sum "$DATASET_DIR/manifest.json" | cut -d' ' -f1)"
[[ "$actual_manifest_sha" == "$manifest_sha256" ]] || die "paired manifest hash mismatch"
printf 'passed: %s\n' "$actual_manifest_sha" > "$GATE_DIR/manifest.txt"

paired_args preflight "$PREFLIGHT_DIR"
run_phase paired_preflight 300 "${PAIRED_COMMAND[@]}" || exit $?

if ((preflight_only)); then
  FINAL_STATUS="preflight_passed"
  CURRENT_PHASE="complete"
  printf 'Paired CoinRun preflight passed: %s\n' "$MANIFEST_PATH"
  exit 0
fi

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"
nvidia-smi --query-gpu=name --format=csv,noheader | grep -Eqi 'H100|H200|B200' \
  || die "an NVIDIA H100, H200, or B200 is required"

run_phase jax_gpu 180 uv run python -c '
import json
import jax
import jax.numpy as jnp
devices = [device for device in jax.devices() if device.platform == "gpu"]
if len(devices) != 1:
    raise SystemExit(f"expected one JAX GPU, got {jax.devices()}")
(jnp.ones((16, 16)) @ jnp.ones((16, 16))).block_until_ready()
print(json.dumps({"device": devices[0].device_kind}))
' || exit $?
printf 'passed\n' > "$GATE_DIR/jax_gpu.txt"

paired_args launch "$RUN_DIR"
run_phase paired_training "$TRAINING_SECONDS" "${PAIRED_COMMAND[@]}" || exit $?
check_checkpoints tokenizer "$RUN_DIR/tokenizer/checkpoints" || exit $?
check_checkpoints dynamics "$RUN_DIR/dynamics/checkpoints" || exit $?

run_phase evaluator "$EVAL_SECONDS" uv run python "$REPO_ROOT/scripts/eval_coinrun.py" \
  "--dynamics-ckpt=$RUN_DIR/dynamics/checkpoints" \
  "--tokenizer-ckpt=$RUN_DIR/tokenizer/checkpoints" \
  "--array-record-path=$DATASET_DIR/val" \
  "--out-dir=$ARTIFACT_DIR/evaluation" \
  --context=8 --horizon=8 --num-windows=32 --batch-size=8 \
  --denoise-steps=4 --one-step-positions=4 --num-videos=4 \
  --p-include-reward=0.25 "--seed=$seed" || exit $?
printf 'passed\n' > "$GATE_DIR/evaluator.txt"

rollout_mp4="$(find "$ARTIFACT_DIR/evaluation" -type f -name '*.mp4' -size +1024c | head -n 1 || true)"
[[ -n "$rollout_mp4" ]] || die "evaluator produced no nonempty rollout MP4"
printf 'passed: %s\n' "$rollout_mp4" > "$GATE_DIR/rollout_mp4.txt"

FINAL_STATUS="passed"
CURRENT_PHASE="complete"
printf 'Paired CoinRun H200 experiment passed: %s\n' "$MANIFEST_PATH"
