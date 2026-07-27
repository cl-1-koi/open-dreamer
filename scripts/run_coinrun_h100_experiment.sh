#!/usr/bin/env bash
# Bounded, local-only CoinRun H100 experiment controller. It deliberately has
# no RunPod API dependency; launch it only from an already-provisioned H100.
set -Eeuo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly MAX_TOTAL_SECONDS=13500 # 3h45m hard maximum, including finalization.
readonly DEFAULT_COLLECTION_SECONDS=1600
readonly DEFAULT_CONFIG_SECONDS=240
readonly DEFAULT_TOKENIZER_SECONDS=4800
readonly DEFAULT_PROBE_SECONDS=600
readonly DEFAULT_DYNAMICS_SECONDS=4800
readonly DEFAULT_EVAL_SECONDS=600

artifact_dir="${COINRUN_ARTIFACT_DIR:-}"
dry_run=0
requested_preset="initial"
seed="${COINRUN_SEED:-20260726}"

usage() {
  cat <<'EOF'
Usage: COINRUN_ARTIFACT_DIR=/absolute/path scripts/run_coinrun_h100_experiment.sh [options]

Runs a bounded, local H100 CoinRun experiment. It never creates or manages a pod.

Options:
  --artifact-dir PATH  Override COINRUN_ARTIFACT_DIR.
  --seed INTEGER       Dataset and training seed (default: 20260726).
  --preset NAME        initial or fallback (default: initial).
  --dry-run            Render commands and manifest without Procgen or GPU access.
  -h, --help           Show this help.
EOF
}

die() {
  FINAL_ERROR="$*"
  printf 'coinrun-h100: %s\n' "$*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --artifact-dir)
      (($# >= 2)) || die "--artifact-dir requires a path"
      artifact_dir="$2"
      shift 2
      ;;
    --seed)
      (($# >= 2)) || die "--seed requires an integer"
      seed="$2"
      shift 2
      ;;
    --preset)
      (($# >= 2)) || die "--preset requires initial or fallback"
      requested_preset="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$artifact_dir" ]] || die "COINRUN_ARTIFACT_DIR or --artifact-dir is required"
[[ "$artifact_dir" = /* ]] || die "artifact directory must be an absolute path"
[[ "$seed" =~ ^[0-9]+$ ]] || die "seed must be a non-negative integer"
[[ "$requested_preset" == "initial" || "$requested_preset" == "fallback" ]] || die "preset must be initial or fallback"

ARTIFACT_DIR="${artifact_dir%/}"
COMMAND_DIR="$ARTIFACT_DIR/commands"
CONFIG_DIR="$ARTIFACT_DIR/resolved_configs"
LOG_DIR="$ARTIFACT_DIR/logs"
TELEMETRY_DIR="$ARTIFACT_DIR/telemetry"
DATASET_DIR="$ARTIFACT_DIR/dataset"
MANIFEST_PATH="$ARTIFACT_DIR/final_manifest.json"
PHASE_TIMINGS="$TELEMETRY_DIR/phase_timings.tsv"
ACTIVE_PRESET="$requested_preset"
FALLBACK_USED=false
FINAL_STATUS="failed"
FINAL_ERROR=""
START_EPOCH=0
DEADLINE_EPOCH=0
CURRENT_PHASE="setup"

if [[ -d "$ARTIFACT_DIR" ]] && [[ -n "$(find "$ARTIFACT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  die "artifact directory must be empty: $ARTIFACT_DIR"
fi
mkdir -p "$COMMAND_DIR" "$CONFIG_DIR" "$LOG_DIR" "$TELEMETRY_DIR"
if [[ -e "$MANIFEST_PATH" ]]; then
  die "refusing to overwrite existing manifest: $MANIFEST_PATH"
fi

write_manifest() {
  local status="$1"
  local error="${2:-}"
  export ARTIFACT_DIR MANIFEST_PATH PHASE_TIMINGS START_EPOCH DEADLINE_EPOCH MAX_TOTAL_SECONDS
  export FINAL_STATUS="$status" FINAL_ERROR="$error" ACTIVE_PRESET FALLBACK_USED CURRENT_PHASE
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
                "name": fields[0], "started_epoch": int(fields[1]),
                "elapsed_seconds": float(fields[2]), "returncode": int(fields[3]),
                "deadline_seconds": int(fields[4]),
            })

def text(path):
    path = Path(path)
    return path.read_text(encoding="utf-8").strip() if path.exists() else None

manifest = {
    "schema_version": 1,
    "status": os.environ["FINAL_STATUS"],
    "failure": os.environ["FINAL_ERROR"] or None,
    "current_phase": os.environ["CURRENT_PHASE"],
    "started_epoch": int(os.environ["START_EPOCH"]),
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "hard_total_deadline_seconds": int(os.environ["MAX_TOTAL_SECONDS"]),
    "deadline_epoch": int(os.environ["DEADLINE_EPOCH"]),
    "active_preset": os.environ["ACTIVE_PRESET"],
    "oom_fallback_used": os.environ["FALLBACK_USED"] == "true",
    "git": {
        "commit": text(root / "git" / "commit.txt"),
        "branch": text(root / "git" / "branch.txt"),
        "status_porcelain": text(root / "git" / "status_porcelain.txt"),
    },
    "phases": timings,
    "commands": sorted(str(p.relative_to(root)) for p in (root / "commands").glob("*.command")),
    "resolved_configs": sorted(str(p.relative_to(root)) for p in (root / "resolved_configs").glob("*.yaml")),
    "gpu_telemetry": sorted(str(p.relative_to(root)) for p in (root / "telemetry").glob("gpu_*.csv")),
    "required_gates": {
        "dataset": text(root / "gates" / "dataset.txt"),
        "tokenizer_checkpoints": text(root / "gates" / "tokenizer_checkpoints.txt"),
        "dynamics_checkpoints": text(root / "gates" / "dynamics_checkpoints.txt"),
        "heldout_probe_collectors": text(root / "gates" / "tokenizer_probe_collectors.json"),
        "rollout_mp4": text(root / "gates" / "rollout_mp4.txt"),
        "evaluator": text(root / "gates" / "evaluator.txt"),
    },
}
Path(os.environ["MANIFEST_PATH"]).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
}

finish() {
  local rc="$1"
  trap - EXIT
  if ((rc == 0)) && [[ "$FINAL_STATUS" == "passed" || "$FINAL_STATUS" == "dry_run" ]]; then
    write_manifest "$FINAL_STATUS" ""
  else
    [[ -n "$FINAL_ERROR" ]] || FINAL_ERROR="controller exited with status $rc during $CURRENT_PHASE"
    write_manifest failed "$FINAL_ERROR"
  fi
  exit "$rc"
}
trap 'finish $?' EXIT

record_command() {
  local name="$1"
  shift
  printf '%q ' "$@" > "$COMMAND_DIR/$name.command"
  printf '\n' >> "$COMMAND_DIR/$name.command"
}

remaining_seconds() {
  local now
  now="$(date +%s)"
  printf '%s\n' "$((DEADLINE_EPOCH - now))"
}

run_phase() {
  local name="$1"
  local phase_limit="$2"
  shift 2
  local started remaining effective rc monitor_pid=""
  CURRENT_PHASE="$name"
  started="$(date +%s)"
  remaining="$(remaining_seconds)"
  if ((remaining <= 0)); then
    FINAL_ERROR="hard total deadline reached before phase $name"
    return 124
  fi
  effective="$phase_limit"
  ((remaining < effective)) && effective="$remaining"
  record_command "$name" "$@"
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
  timeout --foreground --signal=TERM --kill-after=30s "${effective}s" "$@" >"$LOG_DIR/$name.log" 2>&1
  rc=$?
  set -e
  [[ -z "$monitor_pid" ]] || { kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; }
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$started" "$(( $(date +%s) - started ))" "$rc" "$effective" >> "$PHASE_TIMINGS"
  if ((rc != 0)); then
    FINAL_ERROR="phase $name failed with status $rc; see $LOG_DIR/$name.log"
  fi
  return "$rc"
}

select_preset() {
  local preset="$1"
  case "$preset" in
    initial)
      tokenizer_batch=128; tokenizer_length=32; tokenizer_steps="${COINRUN_TOKENIZER_STEPS:-12000}"
      tokenizer_latents=64; tokenizer_bottleneck=16; tokenizer_encoder_depth=8; tokenizer_decoder_depth=8; tokenizer_width=512
      dynamics_batch=64; dynamics_length=48; dynamics_steps="${COINRUN_DYNAMICS_STEPS:-10000}"
      dynamics_depth=12; dynamics_width=512; dynamics_registers=8; dynamics_context=32
      ;;
    fallback)
      tokenizer_batch=64; tokenizer_length=24; tokenizer_steps="${COINRUN_TOKENIZER_STEPS:-12000}"
      tokenizer_latents=48; tokenizer_bottleneck=12; tokenizer_encoder_depth=6; tokenizer_decoder_depth=6; tokenizer_width=384
      dynamics_batch=32; dynamics_length=32; dynamics_steps="${COINRUN_DYNAMICS_STEPS:-10000}"
      dynamics_depth=8; dynamics_width=384; dynamics_registers=6; dynamics_context=24
      ;;
    *) die "internal unknown preset: $preset" ;;
  esac
  [[ "$tokenizer_steps" =~ ^[1-9][0-9]*$ && "$dynamics_steps" =~ ^[1-9][0-9]*$ ]] || die "step overrides must be positive integers"
  ACTIVE_PRESET="$preset"
}

is_oom_failure() {
  local log="$1"
  rg -qi 'out of memory|RESOURCE_EXHAUSTED|CUDA_ERROR_OUT_OF_MEMORY|oom-kill' "$log"
}

capture_config() {
  local name="$1"
  local stage="$2"
  local config_name="$3"
  shift 3
  run_phase "$name" "$DEFAULT_CONFIG_SECONDS" uv run python "$REPO_ROOT/scripts/train_${stage}.py" \
    --config-name="$config_name" --cfg job --resolve "$@" || return $?
  cp "$LOG_DIR/$name.log" "$CONFIG_DIR/$name.yaml"
}

build_tokenizer_overrides() {
  local run_dir="$1"
  tokenizer_overrides=(
    "hydra.run.dir=$run_dir" "run_name=coinrun-h100-tokenizer-$ACTIVE_PRESET"
    "dataset.array_record_path=$DATASET_DIR/train" "dataset.dataloader_cfg.B=$tokenizer_batch"
    "dataset.dataloader_cfg.short_T=$tokenizer_length" "dataset.dataloader_cfg.long_T=$tokenizer_length"
    "dataset.dataloader_cfg.num_workers=8" "dataset.dataloader_cfg.prefetch_buffer_size=8"
    "dataset.dataloader_cfg.device_prefetch_buffer_size=2" "max_steps=$tokenizer_steps"
    "lr_schedule.max_steps=$tokenizer_steps" "ckpt.max_steps=$tokenizer_steps" "ckpt.max_to_keep=3"
    "ckpt.save_interval_steps=1000" "logger.log_every=25" "logger.use_wandb=false" "use_wandb=false"
    "lpips_weight=0" "visualize_every=0" "tokenizer_loss_type=mse" "seed=$seed"
    "tokenizer.encoder.n_latents=$tokenizer_latents" "tokenizer.encoder.d_bottleneck=$tokenizer_bottleneck"
    "tokenizer.encoder.depth=$tokenizer_encoder_depth" "tokenizer.encoder.d_model=$tokenizer_width"
    "tokenizer.encoder.n_heads=8" "tokenizer.encoder.n_kv_heads=1" "tokenizer.encoder.context_length=$tokenizer_length"
    "tokenizer.decoder.n_latents=$tokenizer_latents" "tokenizer.decoder.d_bottleneck=$tokenizer_bottleneck"
    "tokenizer.decoder.depth=$tokenizer_decoder_depth" "tokenizer.decoder.d_model=$tokenizer_width"
    "tokenizer.decoder.n_heads=8" "tokenizer.decoder.n_kv_heads=1"
  )
}

build_dynamics_overrides() {
  local run_dir="$1" tokenizer_ckpt="$2" latent_json="$3"
  local latent_mean latent_std
  latent_mean="$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["latent_mean"], separators=(",", ":")))' "$latent_json")"
  latent_std="$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["latent_std"], separators=(",", ":")))' "$latent_json")"
  dynamics_overrides=(
    "hydra.run.dir=$run_dir" "run_name=coinrun-h100-dynamics-$ACTIVE_PRESET"
    "tokenizer_ckpt=$tokenizer_ckpt" "dataset.array_record_path=$DATASET_DIR/train"
    "dataset.dataloader_cfg.B=$dynamics_batch" "dataset.dataloader_cfg.short_T=$dynamics_length"
    "dataset.dataloader_cfg.long_T=$dynamics_length" "dataset.dataloader_cfg.long_ratio=0"
    "dataset.dataloader_cfg.num_workers=8" "dataset.dataloader_cfg.prefetch_buffer_size=8"
    "dataset.dataloader_cfg.device_prefetch_buffer_size=2" "max_steps=$dynamics_steps"
    "lr_schedule.max_steps=$dynamics_steps" "ckpt.max_steps=$dynamics_steps" "ckpt.max_to_keep=3"
    "ckpt.save_interval_steps=1000" "logger.log_every=25" "logger.use_wandb=false" "use_wandb=false"
    "seed=$seed" "bootstrap_start=$dynamics_steps" "bootstrap_fraction=0" "image_fraction=0"
    "ot.enabled=false" "dynamics.k_max=8" "write_video_every=$dynamics_steps"
    "dynamics.d_bottleneck=$tokenizer_bottleneck" "dynamics.depth=$dynamics_depth"
    "dynamics.d_model=$dynamics_width" "dynamics.n_heads=8" "dynamics.n_kv_heads=1"
    "dynamics.packing_factor=2" "dynamics.n_register=$dynamics_registers"
    "dynamics.context_length=$dynamics_context" "dynamics.latent_mean=$latent_mean" "dynamics.latent_std=$latent_std"
  )
}

check_checkpoint_gate() {
  local stage="$1" directory="$2" count
  count="$(find "$directory" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | rg -c '^[0-9]+$' || true)"
  [[ "$count" =~ ^[0-9]+$ ]] || count=0
  ((count >= 2)) || { FINAL_ERROR="$stage checkpoint gate failed: expected at least two saved checkpoints in $directory"; return 1; }
  mkdir -p "$ARTIFACT_DIR/gates"
  printf 'passed: %s checkpoints\n' "$count" > "$ARTIFACT_DIR/gates/${stage}_checkpoints.txt"
}

merge_mixed_dataset() {
  local source_random="$ARTIFACT_DIR/dataset_sources/random"
  local source_scripted="$ARTIFACT_DIR/dataset_sources/scripted"
  mkdir -p "$DATASET_DIR"
  for split in train val test; do
    mkdir -p "$DATASET_DIR/$split"
    local source shard base
    for source in "$source_random" "$source_scripted"; do
      for shard in "$source/$split"/*.array_record; do
        [[ -f "$shard" ]] || { printf 'missing %s shard in %s\n' "$split" "$source" >&2; return 1; }
        base="$(basename "$shard")"
        cp "$shard" "$DATASET_DIR/$split/$(basename "$source")-$base"
      done
    done
  done
  python3 - "$source_random/metadata.json" "$source_scripted/metadata.json" "$DATASET_DIR/metadata.json" <<'PY'
import json
import sys
from pathlib import Path

random_meta = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
scripted_meta = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if random_meta["num_actions"] != 15 or scripted_meta["num_actions"] != 15:
    raise SystemExit("CoinRun action dimension must be 15 in both sources")
metadata = {
    "schema_version": 1,
    "serialization_format": "pickle",
    "env": "coinrun",
    "collector": "mixed_random_scripted",
    "collector_identity": "isolated_procgen_mixed_random_scripted_v1",
    "num_actions": 15,
    "categorical_noop": 4,
    "action_alignment": "action_applied_after_frame",
    "reward_alignment": "reward_resulting_from_action",
    "sources": {"random": random_meta, "scripted": scripted_meta},
}
Path(sys.argv[3]).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

validate_dataset() {
  uv run python - "$DATASET_DIR" "$REPO_ROOT" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

dataset_dir = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("coinrun_preflight", repo_root / "scripts" / "coinrun_preflight.py")
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)
metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
stats = module.summarize_dataset_records(module.iter_array_records(dataset_dir))
config = module.load_coinrun_dataset_config(repo_root / "configs")
gates = module.validate_dataset_gates(stats, metadata, config, required_sequence_length=24)
if set(stats["collector_frame_histogram"]) != {"random", "scripted"}:
    raise SystemExit("mixed corpus gate failed: both random and scripted records are required")
print(json.dumps({"stats": stats, "gates": gates}, indent=2, sort_keys=True))
PY
}

START_EPOCH="$(date +%s)"
DEADLINE_EPOCH="$((START_EPOCH + MAX_TOTAL_SECONDS))"
printf 'name\tstarted_epoch\telapsed_seconds\treturncode\tdeadline_seconds\n' > "$PHASE_TIMINGS"
mkdir -p "$ARTIFACT_DIR/git" "$ARTIFACT_DIR/gates"
git -C "$REPO_ROOT" rev-parse HEAD > "$ARTIFACT_DIR/git/commit.txt"
git -C "$REPO_ROOT" branch --show-current > "$ARTIFACT_DIR/git/branch.txt"
git -C "$REPO_ROOT" status --porcelain > "$ARTIFACT_DIR/git/status_porcelain.txt"
select_preset "$requested_preset"

if ((dry_run)); then
  build_tokenizer_overrides "$ARTIFACT_DIR/tokenizer_initial"
  record_command tokenizer_train_initial uv run python "$REPO_ROOT/scripts/train_tokenizer.py" --config-name=coinrun_tokenizer "${tokenizer_overrides[@]}"
  printf '%s\n' "uv run --isolated --script $REPO_ROOT/dreamer/data/generate_coinrun_dataset.py --collector=random|scripted --output-dir=$ARTIFACT_DIR/dataset_sources/<collector>" > "$COMMAND_DIR/collection.command"
  dry_probe="$ARTIFACT_DIR/dry_run_latent_stats.json"
  python3 - "$dry_probe" "$tokenizer_bottleneck" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"latent_mean": [0.0] * int(sys.argv[2]), "latent_std": [1.0] * int(sys.argv[2])}) + "\n")
PY
  printf '%s\n' "uv run python $REPO_ROOT/scripts/coinrun_preflight.py _tokenizer_probe --checkpoint-dir=<tokenizer> --dataset-dir=$DATASET_DIR --output=<stats>" > "$COMMAND_DIR/tokenizer_probe.command"
  build_dynamics_overrides "$ARTIFACT_DIR/dynamics_initial" "$ARTIFACT_DIR/tokenizer_initial/checkpoints" "$dry_probe"
  record_command dynamics_train_initial uv run python "$REPO_ROOT/scripts/train_dynamics.py" --config-name=coinrun_dynamics "${dynamics_overrides[@]}"
  printf 'not run in dry-run\n' > "$ARTIFACT_DIR/gates/dataset.txt"
  FINAL_STATUS="dry_run"
  printf 'Dry run manifest: %s\n' "$MANIFEST_PATH"
  exit 0
fi

command -v uv >/dev/null 2>&1 || die "uv is required"
command -v timeout >/dev/null 2>&1 || die "GNU timeout is required"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required for this H100-only pipeline"
nvidia-smi --query-gpu=name --format=csv,noheader | rg -qi 'H100' || die "an NVIDIA H100 is required"

generator="$REPO_ROOT/dreamer/data/generate_coinrun_dataset.py"
episodes_train="${COINRUN_EPISODES_TRAIN_PER_COLLECTOR:-400}"
episodes_val="${COINRUN_EPISODES_VAL_PER_COLLECTOR:-32}"
episodes_test="${COINRUN_EPISODES_TEST_PER_COLLECTOR:-32}"
for count in "$episodes_train" "$episodes_val" "$episodes_test"; do [[ "$count" =~ ^[1-9][0-9]*$ ]] || die "episode counts must be positive integers"; done

run_phase collect_random "$DEFAULT_COLLECTION_SECONDS" uv run --isolated --script "$generator" \
  "--num-episodes-train=$episodes_train" "--num-episodes-val=$episodes_val" "--num-episodes-test=$episodes_test" \
  "--output-dir=$ARTIFACT_DIR/dataset_sources/random" --min-episode-length=64 --max-episode-length=256 \
  --chunk-size=256 --chunks-per-file=32 --collector=random --keep-short-terminated "--seed=$seed" --overwrite || exit $?
run_phase collect_scripted "$DEFAULT_COLLECTION_SECONDS" uv run --isolated --script "$generator" \
  "--num-episodes-train=$episodes_train" "--num-episodes-val=$episodes_val" "--num-episodes-test=$episodes_test" \
  "--output-dir=$ARTIFACT_DIR/dataset_sources/scripted" --min-episode-length=64 --max-episode-length=256 \
  --chunk-size=256 --chunks-per-file=32 --collector=scripted --keep-short-terminated "--seed=$((seed + 1))" --overwrite || exit $?
export DATASET_DIR REPO_ROOT ARTIFACT_DIR
run_phase dataset_merge 120 bash -c "$(declare -f merge_mixed_dataset); merge_mixed_dataset" || exit $?
run_phase dataset_validation 300 bash -c "$(declare -f validate_dataset); validate_dataset" > /dev/null || exit $?
cp "$LOG_DIR/dataset_validation.log" "$ARTIFACT_DIR/dataset_validation.json"
printf 'passed: isolated Procgen random+scripted corpus with disjoint split seed ranges\n' > "$ARTIFACT_DIR/gates/dataset.txt"

run_tokenizer() {
  local preset="$1" run_dir="$ARTIFACT_DIR/tokenizer_$preset"
  select_preset "$preset"
  build_tokenizer_overrides "$run_dir"
  capture_config "tokenizer_config_$preset" tokenizer coinrun_tokenizer "${tokenizer_overrides[@]}" || return $?
  run_phase "tokenizer_train_$preset" "$DEFAULT_TOKENIZER_SECONDS" uv run python "$REPO_ROOT/scripts/train_tokenizer.py" --config-name=coinrun_tokenizer "${tokenizer_overrides[@]}"
}

if ! run_tokenizer "$ACTIVE_PRESET"; then
  failed_preset="$ACTIVE_PRESET"
  if [[ "$failed_preset" == "initial" ]] && is_oom_failure "$LOG_DIR/tokenizer_train_initial.log"; then
    FALLBACK_USED=true
    FINAL_ERROR=""
    run_tokenizer fallback || exit $?
  else
    exit 1
  fi
fi
TOKENIZER_DIR="$ARTIFACT_DIR/tokenizer_$ACTIVE_PRESET"
check_checkpoint_gate tokenizer "$TOKENIZER_DIR/checkpoints" || exit $?

probe_path="$ARTIFACT_DIR/tokenizer_heldout_probe.json"
validate_probe_collectors() {
  local path="$1"
  python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source = payload.get("source")
if not isinstance(source, dict):
    raise SystemExit("held-out probe collector gate: source metadata is missing")
for field in ("collector_records_seen", "collector_records_selected"):
    counts = source.get(field)
    if not isinstance(counts, dict):
        raise SystemExit(f"held-out probe collector gate: {field} is missing")
    missing = [arm for arm in ("random", "scripted") if not isinstance(counts.get(arm), int) or counts[arm] <= 0]
    if missing:
        raise SystemExit(
            f"held-out probe collector gate: {field} lacks positive counts for {', '.join(missing)}"
        )
print(json.dumps({
    "passed": True,
    "collector_records_seen": source["collector_records_seen"],
    "collector_records_selected": source["collector_records_selected"],
}, indent=2, sort_keys=True))
PY
}

run_heldout_probe() {
  run_phase "tokenizer_heldout_probe_$ACTIVE_PRESET" "$DEFAULT_PROBE_SECONDS" uv run python "$REPO_ROOT/scripts/coinrun_preflight.py" _tokenizer_probe \
    "--checkpoint-dir=$TOKENIZER_DIR/checkpoints" "--dataset-dir=$DATASET_DIR" "--output=$probe_path" \
    "--sequence-length=$tokenizer_length" --max-records=16 --std-epsilon=0.000001 || return $?
  [[ -s "$probe_path" ]] || { FINAL_ERROR="held-out tokenizer probe did not write $probe_path"; return 1; }
  if ! validate_probe_collectors "$probe_path" > "$ARTIFACT_DIR/gates/tokenizer_probe_collectors.json"; then
    FINAL_ERROR="held-out probe collector gate failed; see $ARTIFACT_DIR/gates/tokenizer_probe_collectors.json"
    return 1
  fi
}
run_heldout_probe || exit $?

run_dynamics() {
  local preset="$1" run_dir="$ARTIFACT_DIR/dynamics_$preset"
  select_preset "$preset"
  build_dynamics_overrides "$run_dir" "$TOKENIZER_DIR/checkpoints" "$probe_path"
  capture_config "dynamics_config_$preset" dynamics coinrun_dynamics "${dynamics_overrides[@]}" || return $?
  run_phase "dynamics_train_$preset" "$DEFAULT_DYNAMICS_SECONDS" uv run python "$REPO_ROOT/scripts/train_dynamics.py" --config-name=coinrun_dynamics "${dynamics_overrides[@]}"
}

if ! run_dynamics "$ACTIVE_PRESET"; then
  failed_preset="$ACTIVE_PRESET"
  if [[ "$failed_preset" == "initial" ]] && is_oom_failure "$LOG_DIR/dynamics_train_initial.log"; then
    FALLBACK_USED=true
    FINAL_ERROR=""
    run_tokenizer fallback || exit $?
    TOKENIZER_DIR="$ARTIFACT_DIR/tokenizer_fallback"
    check_checkpoint_gate tokenizer "$TOKENIZER_DIR/checkpoints" || exit $?
    run_heldout_probe || exit $?
    run_dynamics fallback || exit $?
  else
    exit 1
  fi
fi
DYNAMICS_DIR="$ARTIFACT_DIR/dynamics_$ACTIVE_PRESET"
check_checkpoint_gate dynamics "$DYNAMICS_DIR/checkpoints" || exit $?

if [[ -f "$REPO_ROOT/scripts/eval_coinrun.py" ]]; then
  eval_dir="$ARTIFACT_DIR/evaluation"
  run_phase evaluator "$DEFAULT_EVAL_SECONDS" uv run python "$REPO_ROOT/scripts/eval_coinrun.py" \
    "--dynamics-ckpt=$DYNAMICS_DIR/checkpoints" "--tokenizer-ckpt=$TOKENIZER_DIR/checkpoints" \
    "--array-record-path=$DATASET_DIR/val" "--out-dir=$eval_dir" \
    --context=8 --horizon=8 --num-windows=32 --batch-size=8 \
    --denoise-steps=4 --one-step-positions=4 --num-videos=4 \
    --p-include-reward=0.25 "--seed=$seed" || exit $?
  printf 'passed: scripts/eval_coinrun.py exited successfully\n' > "$ARTIFACT_DIR/gates/evaluator.txt"
else
  printf 'not required: scripts/eval_coinrun.py is absent; trainer rollout evaluator is gated below\n' > "$ARTIFACT_DIR/gates/evaluator.txt"
fi

rollout_mp4="$(find "$DYNAMICS_DIR" "$ARTIFACT_DIR/evaluation" -type f -name '*.mp4' -size +1024c 2>/dev/null | head -n 1 || true)"
[[ -n "$rollout_mp4" ]] || { FINAL_ERROR="rollout MP4 gate failed: no nonempty MP4 produced by dynamics/evaluator"; exit 1; }
printf 'passed: %s\n' "$rollout_mp4" > "$ARTIFACT_DIR/gates/rollout_mp4.txt"

FINAL_STATUS="passed"
CURRENT_PHASE="complete"
printf 'CoinRun H100 experiment passed. Manifest: %s\n' "$MANIFEST_PATH"
