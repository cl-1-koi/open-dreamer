"""Bounded CoinRun evaluation for tokenizer + dynamics Orbax checkpoints.

Consumes held-out CoinRun RGB/action windows (never training batches) and emits:

  1. tokenizer reconstruction metrics against dataset-mean and copy-previous
     baselines (review G1),
  2. measured latent scale statistics; fails closed when the dynamics
     checkpoint carries no latent_mean/latent_std (review G2),
  3. one-step metrics (predict frame t given ground-truth context up to t-1),
  4. open-loop horizon metrics (autoregressive rollout, predictions fed back),
  5. two action-sensitivity arms against the true actions (review G5):
     - episode-level permutation (future actions from another window),
     - action-index permutation restricted to the 9 behaviorally distinct
       Procgen actions 0-8 (indices 9-14 are no-op aliases),
     each reported with a paired effect size across windows,
  6. reward prevalence (review G6),
  7. non-trivial MP4 rollouts (review G8): the generated region is rejected
     when constant or when it copies the last context frame. The declared
     context/horizon split is encoded in-band (green seam bar = context,
     red seam bar = predicted frames) and in the filename.

CoinRun action semantics (review G3): Procgen CoinRun has 15 discrete actions
and the no-op is index 4 (procgen/env.py action table). The shift applied
here prepends 4, NOT categorical_action_dim // 2.

The evaluator never manufactures a checkpoint and never teacher-forces future
frames: the open-loop path only ever receives context latents, and missing
artifacts abort the run with a clear error before any model is built.

Usage:
    uv run scripts/eval_coinrun.py --dynamics-ckpt logs/dynamics_coinrun/checkpoints \
        --array-record-path datasets/coinrun_episodes/val
    uv run scripts/eval_coinrun.py --dynamics-ckpt <dir> --tokenizer-ckpt <dir> \
        --context 8 --horizon 8 --num-windows 16 --batch-size 4
"""
from __future__ import annotations

import json
import logging
import math
import os
import pickle
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
import tyro

logging.getLogger("absl").setLevel(logging.WARNING)

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import grain  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402

from dreamer.actions import Actions  # noqa: E402
from dreamer.checkpointing import (  # noqa: E402
    DynamicsCheckpointBundle,
    TokenizerCheckpointBundle,
)
from dreamer.configs import DataloaderConfig, DatasetConfig  # noqa: E402
from dreamer.data import build_iterator  # noqa: E402
from dreamer.data.path_utils import discover_array_record_paths  # noqa: E402
from dreamer.generation import DenoiseSchedule, latent_rollout  # noqa: E402
from dreamer.parallel import build_parallel  # noqa: E402
from dreamer.sampler import decode_jit, encode_jit  # noqa: E402

SCHEMA_VERSION = "coinrun-eval/2"

# --- CoinRun action semantics (adversarial review G3, procgen/env.py) ---
# Action table: 0 (LEFT,DOWN) 1 (LEFT) 2 (LEFT,UP) 3 (DOWN) 4 () 5 (UP)
#               6 (RIGHT,DOWN) 7 (RIGHT) 8 (RIGHT,UP) 9-14 special keys.
# CoinRun has no special actions; basic-abstract-game.cpp maps action % 9 and
# forces 4 for action >= 9, so indices 9-14 are no-op aliases. Behaviorally
# distinct actions are 0-8, and the no-op is index 4.
COINRUN_NUM_ACTIONS = 15
COINRUN_NOOP_INDEX = 4
COINRUN_DISTINCT_ACTIONS = 9
# Derangement of the 9 distinct actions: x -> (x + 5) % 9. Every index maps to
# a behaviorally different one (checked against the table above, e.g.
# 4 (no-op) -> 0 (LEFT,DOWN), 8 (RIGHT,UP) -> 4 (no-op), 7 (RIGHT) -> 3 (DOWN)).
ACTION_INDEX_PERMUTATION = tuple((x + 5) % COINRUN_DISTINCT_ACTIONS for x in range(COINRUN_DISTINCT_ACTIONS))

# Video layout constants. The stacked layout must stay divisible by the H.264
# macro block size (16) so imageio/ffmpeg does not resize frames on write.
SEAM_GAP_ROWS = 16
SEAM_GREEN = (0, 200, 0)  # context frames
SEAM_RED = (200, 0, 0)    # predicted (horizon) frames
SEAM_CHANNEL_MARGIN = 30  # tolerated chroma bleed when classifying the seam bar

# MP4 non-triviality thresholds on the uint8 readback (review G8). Lossy H.264
# decodes flat regions with ~2 units of noise but repeats identical source
# frames nearly bit-exact (P-frame residuals ~= 0), so spatial flatness uses a
# generous threshold while temporal-copy thresholds stay tight.
CONSTANT_STD_THRESHOLD = 3.0   # mean per-frame spatial std at/below this is "constant"
COPY_MAD_THRESHOLD = 1.0       # mean |pred[t] - last_context| below this for the
                               # whole horizon is a "copy-last-context" rollout
FROZEN_MAD_THRESHOLD = 1.5     # mean |pred[t] - pred[t-1]| below this across the
                               # horizon is a frozen (copy-own-prediction) rollout

PSNR_CAP_DB = 99.0  # reported when MSE == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class Args:
    """CoinRun evaluation arguments."""

    # Artifacts
    dynamics_ckpt: str = ""  # Orbax checkpoint dir (with or without '/checkpoints' suffix)
    array_record_path: str = "datasets/coinrun_episodes/val"  # held-out split (train is refused)
    tokenizer_ckpt: str | None = None  # optional separate tokenizer checkpoint; default: bundled in dynamics ckpt
    dynamics_model: str = "dynamics_ema"  # "dynamics_ema" | "dynamics"

    # Output
    out_dir: str = "logs/eval_coinrun"

    # Evaluation shape
    context: int = 8          # number of ground-truth context frames
    horizon: int = 8          # number of open-loop predicted frames
    num_windows: int = 16     # total evaluation windows
    batch_size: int = 4       # windows per batch (>= 2 for the episode-permutation arm)
    denoise_steps: int = 4    # tau-ladder steps per predicted frame (must divide k_max)
    one_step_positions: int = 4  # max number of future positions scored one-step
    num_videos: int = 4       # windows exported as MP4 (one file per arm each)
    fps: int = 10
    seed: int = 0

    # Data pipeline
    p_include_reward: float = 0.0  # reward-biased window sampling (0 = uniform)
    num_workers: int = 4
    preflight_scan_records: int = 256  # records decoded during the preflight check


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without JAX)
# ---------------------------------------------------------------------------

def roll_permutation(batch_size: int) -> np.ndarray:
    """Episode-level permutation: window i receives window (i+1) % B's actions.

    Every window gets a foreign action sequence when batch_size >= 2.
    """
    if batch_size < 2:
        raise ValueError(
            f"Action permutation requires batch_size >= 2, got {batch_size}."
        )
    return np.roll(np.arange(batch_size), -1)


def evenly_spaced_positions(horizon: int, max_positions: int) -> list[int]:
    """Up to `max_positions` evenly spaced future positions in [0, horizon).

    Always includes the first (0) and last (horizon - 1) position.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if max_positions < 1:
        raise ValueError(f"max_positions must be >= 1, got {max_positions}")
    if max_positions >= horizon:
        return list(range(horizon))
    positions = np.linspace(0, horizon - 1, max_positions)
    return sorted(set(int(p) for p in np.round(positions)))


def psnr_from_mse(mse: float, peak: float = 255.0) -> float:
    """PSNR in dB from mean squared error; capped when mse == 0."""
    if mse <= 0.0:
        return PSNR_CAP_DB
    return float(min(20.0 * math.log10(peak) - 10.0 * math.log10(mse), PSNR_CAP_DB))


def frame_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Pixel metrics between uint8 frames of identical shape."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"frame shape mismatch: {pred.shape} vs {gt.shape}")
    diff = pred - gt
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    return {"pixel_mse": mse, "pixel_mae": mae, "psnr": psnr_from_mse(mse)}


def latent_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Latent-space MSE between (..., n_latents, d_bottleneck) arrays."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"latent shape mismatch: {pred.shape} vs {gt.shape}")
    diff = pred - gt
    return {"latent_mse": float(np.mean(diff * diff))}


def paired_effect_size(deltas: np.ndarray) -> dict:
    """Paired effect size (Cohen's dz) of per-window metric deltas.

    Positive deltas must mean "consistent with action sensitivity" by the
    caller's sign convention. effect_size_dz is None when the delta variance
    is zero (undefined), with mean_delta still reported.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    n = int(deltas.size)
    if n == 0:
        raise ValueError("paired_effect_size requires at least one window")
    mean = float(deltas.mean())
    std = float(deltas.std(ddof=1)) if n > 1 else 0.0
    dz = None if std == 0.0 else float(mean / std)
    return {
        "mean_delta": mean,
        "std_delta": std,
        "effect_size_dz": dz,
        "frac_positive": float((deltas > 0).mean()),
        "n": n,
    }


def shift_actions_coinrun(actions: Actions) -> Actions:
    """Shift actions right by one, prepending the CoinRun no-op (index 4).

    This intentionally does NOT use dreamer.actions.shift_actions, which
    prepends categorical_action_dim // 2 (the Minecraft camera-grid center) —
    index 8 = (RIGHT, UP) under the Procgen action table (review G3).
    """
    def _shift(x):
        if x is None:
            return None
        start = jnp.full_like(x[:, 0:1], COINRUN_NOOP_INDEX)
        return jnp.concatenate([start, x[:, :-1]], axis=1)

    return jax.tree.map(_shift, actions)


def permute_action_indices(
    actions: Actions,
    permutation: tuple[int, ...] = ACTION_INDEX_PERMUTATION,
    num_distinct: int = COINRUN_DISTINCT_ACTIONS,
    num_actions: int = COINRUN_NUM_ACTIONS,
) -> Actions:
    """Remap categorical actions among behaviorally distinct indices 0-8.

    Indices >= num_distinct (no-op aliases 9-14) are left unchanged (review
    G5): permuting them would map no-ops onto no-ops and weaken the gate.
    """
    lut = np.arange(num_actions, dtype=np.int32)
    lut[:num_distinct] = np.asarray(permutation, dtype=np.int32)

    def _remap(x):
        if x is None:
            return None
        return jnp.asarray(lut)[x]

    return Actions(
        binary=None if actions.binary is None else _remap(actions.binary),
        categorical=None if actions.categorical is None else _remap(actions.categorical),
        continuous=actions.continuous,  # not used by CoinRun
    )


def build_eval_video(
    gt_frames: np.ndarray,
    pred_frames: np.ndarray,
    context: int,
    gap_rows: int = SEAM_GAP_ROWS,
) -> np.ndarray:
    """Stack GT row over prediction row with an in-band context/horizon seam bar.

    Args:
        gt_frames: (T, H, W, C) uint8 ground-truth frames.
        pred_frames: (T, H, W, C) uint8 predicted frames (context region may be
            tokenizer reconstructions; horizon region is model rollout output).
        context: number of leading frames that are context. Frames [0, context)
            get a green seam bar, frames [context, T) a red one.

    Returns:
        (T, 2*H + gap_rows, W, C) uint8 video.
    """
    gt_frames = np.asarray(gt_frames)
    pred_frames = np.asarray(pred_frames)
    if gt_frames.shape != pred_frames.shape:
        raise ValueError(
            f"gt/pred shape mismatch: {gt_frames.shape} vs {pred_frames.shape}"
        )
    T, H, W, C = gt_frames.shape
    if not 1 <= context < T:
        raise ValueError(f"context must be in [1, T), got context={context}, T={T}")
    if W % 2 != 0 or (2 * H + gap_rows) % 16 != 0:
        raise ValueError(
            f"video layout {W}x{2 * H + gap_rows} must keep W even and total "
            "height divisible by 16 (H.264 macro blocks)"
        )

    canvas = np.zeros((T, 2 * H + gap_rows, W, C), dtype=np.uint8)
    canvas[:, :H] = gt_frames
    canvas[:, H + gap_rows:] = pred_frames
    for t in range(T):
        canvas[t, H:H + gap_rows] = SEAM_GREEN if t < context else SEAM_RED
    return canvas


def classify_seam(color: np.ndarray) -> str:
    """Classify a mean RGB color as 'green', 'red', or 'unknown'."""
    r, g, b = (float(color[0]), float(color[1]), float(color[2]))
    if g > r + SEAM_CHANNEL_MARGIN and g > b + SEAM_CHANNEL_MARGIN:
        return "green"
    if r > g + SEAM_CHANNEL_MARGIN and r > b + SEAM_CHANNEL_MARGIN:
        return "red"
    return "unknown"


def verify_mp4(
    path: str | Path,
    expected_frames: int,
    context: int,
    gap_rows: int = SEAM_GAP_ROWS,
) -> dict:
    """Read an exported MP4 back and verify structure and non-triviality.

    Hard-fails (AssertionError) when the file is unreadable, has the wrong
    frame count, is constant overall, the seam bar does not decode at the
    declared context/horizon boundary, or the generated (horizon) region is
    constant / a copy of the last context frame (review G8).

    Returns:
        Stats dict recorded in results.json.
    """
    path = Path(path)
    frames = iio.imread(str(path))  # (T, H', W', C)
    if frames.ndim != 4:
        raise AssertionError(f"{path}: expected 4D video, got shape {frames.shape}")
    n_frames, height = int(frames.shape[0]), int(frames.shape[1])
    if n_frames != expected_frames:
        raise AssertionError(
            f"{path}: expected {expected_frames} frames "
            f"(context + horizon), found {n_frames}"
        )

    frame_std = float(frames.std())
    if frame_std <= 1.0:
        raise AssertionError(
            f"{path}: video is blank/constant (std={frame_std:.4f})"
        )

    # The GT row occupies the top half of the stacked layout; derive H.
    gt_row_h = (height - gap_rows) // 2
    seam_labels = []
    for t in range(n_frames):
        strip = frames[t, gt_row_h + 2: gt_row_h + gap_rows - 2].reshape(-1, 3)
        seam_labels.append(classify_seam(strip.mean(axis=0)))
    expected_labels = ["green"] * context + ["red"] * (n_frames - context)
    seam_marker_ok = seam_labels == expected_labels
    if not seam_marker_ok:
        raise AssertionError(
            f"{path}: seam bar does not match the declared context/horizon split "
            f"(expected {expected_labels}, decoded {seam_labels})"
        )

    # Non-triviality of the generated region (bottom row, horizon frames).
    pred_row = frames[:, gt_row_h + gap_rows:].astype(np.float64)
    horizon_region = pred_row[context:]
    horizon_std = float(horizon_region.std(axis=(1, 2, 3)).mean())
    if horizon_std <= CONSTANT_STD_THRESHOLD:
        raise AssertionError(
            f"{path}: generated rollout is constant "
            f"(mean per-frame spatial std={horizon_std:.4f} <= "
            f"{CONSTANT_STD_THRESHOLD}); the model produced a blank video "
            "(review G8)."
        )
    last_context = pred_row[context - 1]
    copy_mad = np.abs(horizon_region - last_context).mean(axis=(1, 2, 3))
    copy_mad_max = float(copy_mad.max())
    if copy_mad_max < COPY_MAD_THRESHOLD:
        raise AssertionError(
            f"{path}: generated rollout copies the last context frame "
            f"(max mean-abs-diff {copy_mad_max:.4f} < {COPY_MAD_THRESHOLD}); "
            "the model repeated its context instead of predicting (review G8)."
        )
    frozen_mad_mean = None
    if horizon_region.shape[0] >= 2:
        frozen_mad_mean = float(
            np.abs(horizon_region[1:] - horizon_region[:-1]).mean()
        )
        if frozen_mad_mean < FROZEN_MAD_THRESHOLD:
            raise AssertionError(
                f"{path}: generated rollout is frozen across the horizon "
                f"(mean frame-to-frame diff {frozen_mad_mean:.4f} < "
                f"{FROZEN_MAD_THRESHOLD}); the model repeated its own "
                "prediction (review G8)."
            )

    return {
        "frames_readback": n_frames,
        "frame_std": frame_std,
        "seam_marker_ok": seam_marker_ok,
        "horizon_region_std": horizon_std,
        "copy_last_context_mad_max": copy_mad_max,
        "frozen_mad_mean": frozen_mad_mean,
        "nontrivial": True,
    }


def resolve_checkpoint_dir(path: str | None, *, label: str) -> str:
    """Resolve a checkpoint directory, accepting an optional '/checkpoints' suffix.

    Never creates anything: raises FileNotFoundError when the path is missing.
    """
    if not path:
        raise FileNotFoundError(
            f"{label} checkpoint path is empty. A trained Orbax checkpoint is "
            "required; the evaluator does not create or fall back to random weights."
        )
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{label} checkpoint not found: {p}. Train a model first; the "
            "evaluator does not create or fall back to random weights."
        )
    if p.name != "checkpoints":
        candidate = p / "checkpoints"
        if candidate.is_dir():
            return str(candidate)
    return str(p)


def validate_args(args: Args) -> None:
    """Validate argument combinations before any heavy work happens."""
    if args.context < 1:
        raise ValueError(f"context must be >= 1, got {args.context}")
    if args.horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {args.horizon}")
    if args.batch_size < 2:
        raise ValueError(
            f"batch_size must be >= 2 (the episode-level permutation arm "
            f"compares windows within a batch), got {args.batch_size}"
        )
    if args.num_windows < 1:
        raise ValueError(f"num_windows must be >= 1, got {args.num_windows}")
    if args.denoise_steps < 1:
        raise ValueError(f"denoise_steps must be >= 1, got {args.denoise_steps}")
    if args.one_step_positions < 1:
        raise ValueError(
            f"one_step_positions must be >= 1, got {args.one_step_positions}"
        )
    if args.num_videos < 0:
        raise ValueError(f"num_videos must be >= 0, got {args.num_videos}")
    if args.fps < 1:
        raise ValueError(f"fps must be >= 1, got {args.fps}")
    if not 0.0 <= args.p_include_reward <= 1.0:
        raise ValueError(
            f"p_include_reward must be in [0, 1], got {args.p_include_reward}"
        )
    if args.dynamics_model not in ("dynamics", "dynamics_ema"):
        raise ValueError(
            f"dynamics_model must be 'dynamics' or 'dynamics_ema', got "
            f"{args.dynamics_model!r}"
        )
    split_name = Path(args.array_record_path).name.lower()
    if "train" in split_name:
        raise ValueError(
            f"Refusing to evaluate on {args.array_record_path!r}: the path "
            "looks like a training split. Held-out evaluation must draw from "
            "val/test records (review G4); rename the split directory if this "
            "is genuinely held-out data."
        )


def preflight_dataset(
    array_record_path: str,
    seq_len: int,
    batch_size: int,
    max_scan: int,
    p_include_reward: float = 0.0,
) -> dict:
    """Bounded dataset preflight: decode up to `max_scan` records and verify.

    Checks that records exist, look like pickle-serialized CoinRun chunks
    (raw_video / sequence_length / actions / rewards), use legal CoinRun
    action indices, and that at least `batch_size` of the scanned records are
    long enough. Reports reward prevalence (review G6) and whether level seeds
    are recorded (review G4). Fails clearly instead of hanging the Grain
    pipeline on an all-filtered dataset.

    Returns:
        Info dict recorded in results.json (plus first-record details used
        for cross-checks once model configs are known).
    """
    paths = discover_array_record_paths(array_record_path)
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise FileNotFoundError(
            f"No .array_record files found at {array_record_path!r}. Generate "
            "the CoinRun dataset first (see dreamer/data/generate_coinrun_dataset.py)."
        )

    source = grain.sources.ArrayRecordDataSource(paths)
    num_records = len(source)
    scan = min(num_records, max(max_scan, batch_size))

    required_keys = ("raw_video", "sequence_length", "actions", "rewards")
    usable = 0
    nonzero_reward_frames = 0
    scanned_frames = 0
    action_min, action_max = None, None
    level_seed_key = None
    level_seeds: set = set()
    first_info: dict = {}
    for i in range(scan):
        raw = source[i]
        try:
            record = pickle.loads(raw)
        except Exception as exc:
            raise ValueError(
                f"Record {i} in {paths[0]} is not a pickle-serialized CoinRun "
                f"record (pickle.loads failed: {exc!r}). This evaluator expects "
                "the format written for the CoinRun dataset pipeline."
            ) from exc
        missing = [k for k in required_keys if k not in record]
        if missing:
            raise ValueError(
                f"Record {i} is missing required CoinRun keys {missing}; has "
                f"{sorted(record.keys())}."
            )
        record_len = int(record["sequence_length"])
        if record_len >= seq_len:
            usable += 1

        actions = np.asarray(record["actions"])
        lo, hi = int(actions.min()), int(actions.max())
        action_min = lo if action_min is None else min(action_min, lo)
        action_max = hi if action_max is None else max(action_max, hi)

        rewards = np.asarray(record["rewards"])
        nonzero_reward_frames += int((rewards > 0).sum())
        scanned_frames += int(rewards.size)

        for key in ("level_seed", "seed"):
            if key in record:
                level_seed_key = key
                level_seeds.add(int(record[key]))
                break

        if i == 0:
            first_info = {
                "first_record_sequence_length": record_len,
                "first_record_raw_video_nbytes": len(record["raw_video"]),
            }

    if action_min is not None and (action_min < 0 or action_max >= COINRUN_NUM_ACTIONS):
        raise ValueError(
            f"Scanned records contain action indices [{action_min}, {action_max}], "
            f"outside CoinRun's 15 legal actions [0, {COINRUN_NUM_ACTIONS}) "
            "(review G3)."
        )
    if usable == 0:
        raise ValueError(
            f"None of the {scan} scanned records at {array_record_path!r} has "
            f"sequence_length >= {seq_len}; cannot evaluate."
        )
    if usable < batch_size:
        raise ValueError(
            f"Only {usable} of {scan} scanned records have sequence_length >= "
            f"{seq_len}, but batch_size={batch_size} windows are needed per "
            "batch. Provide a dataset with more long records or lower "
            "batch_size/context+horizon."
        )
    if p_include_reward > 0.0 and nonzero_reward_frames == 0:
        raise ValueError(
            f"p_include_reward={p_include_reward} was requested, but none of "
            f"the {scanned_frames} scanned frames carries nonzero reward. The "
            "reward-biased arm would be vacuous (review G6); generate a "
            "dataset that retains coin/death episodes or set p_include_reward=0."
        )

    return {
        "num_files": len(paths),
        "num_records": num_records,
        "scanned_records": scan,
        "usable_records_in_scan": usable,
        "action_min": action_min,
        "action_max": action_max,
        "nonzero_reward_frames_in_scan": nonzero_reward_frames,
        "scanned_frames": scanned_frames,
        "level_seed_key": level_seed_key,
        "unique_level_seeds_in_scan": len(level_seeds) if level_seed_key else 0,
        **first_info,
    }


def git_commit(repo_root: Path) -> str | None:
    """Best-effort git commit of the repository containing this script."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Model loading and config validation (JAX)
# ---------------------------------------------------------------------------

def _latest_step(ckpt_dir: str) -> int:
    with ocp.CheckpointManager(ckpt_dir) as manager:
        step = manager.latest_step()
    if step is None:
        raise FileNotFoundError(f"No checkpoint steps found in {ckpt_dir}")
    return int(step)


def load_models(args: Args, mesh_rules):
    """Load tokenizer + dynamics from Orbax checkpoints. Never manufactures weights."""
    dynamics_ckpt = resolve_checkpoint_dir(args.dynamics_ckpt, label="dynamics")
    dynamics_step = _latest_step(dynamics_ckpt)

    bundle = DynamicsCheckpointBundle.from_pretrained(
        dynamics_ckpt,
        mesh_rules=mesh_rules,
        model_names={args.dynamics_model, "tokenizer"},
    )
    dynamics = getattr(bundle, args.dynamics_model)

    if args.tokenizer_ckpt:
        tokenizer_ckpt = resolve_checkpoint_dir(args.tokenizer_ckpt, label="tokenizer")
        tokenizer_step = _latest_step(tokenizer_ckpt)
        tokenizer = TokenizerCheckpointBundle.from_pretrained(
            tokenizer_ckpt, mesh_rules=mesh_rules, model_names={"tokenizer"}
        ).tokenizer
        tokenizer_source = "separate"
    else:
        tokenizer_ckpt = dynamics_ckpt
        tokenizer_step = dynamics_step
        tokenizer = bundle.tokenizer
        tokenizer_source = "dynamics_bundle"

    info = {
        "dynamics_ckpt": dynamics_ckpt,
        "dynamics_step": dynamics_step,
        "dynamics_model": args.dynamics_model,
        "tokenizer_ckpt": tokenizer_ckpt,
        "tokenizer_step": tokenizer_step,
        "tokenizer_source": tokenizer_source,
    }
    return tokenizer, dynamics, info


def validate_model_configs(dyn_cfg, tok_cfg, dataset_info: dict) -> None:
    """Cross-check checkpoint configs against CoinRun semantics and the dataset.

    Fail-closed gates (adversarial review):
      - G2: latent_mean/latent_std must be set on the dynamics checkpoint.
      - G3: the action space must be CoinRun's 15 discrete actions.
    """
    cat_dim = int(dyn_cfg.categorical_action_dim)
    if cat_dim != COINRUN_NUM_ACTIONS:
        raise ValueError(
            f"The dynamics checkpoint declares categorical_action_dim={cat_dim}, "
            f"but Procgen CoinRun has exactly {COINRUN_NUM_ACTIONS} discrete "
            "actions (review G3). A 16-wide table indicates the old, "
            "mismatched convention (and the wrong no-op index); refusing to "
            "evaluate against it."
        )
    if dyn_cfg.latent_mean is None or dyn_cfg.latent_std is None:
        raise ValueError(
            "The dynamics checkpoint has latent_mean/latent_std=None, so "
            "latent normalization would be silently disabled (review G2). "
            "Fail-closed: retrain or re-save the checkpoint with explicit "
            "latent scale stats before evaluating."
        )
    d_bottleneck = int(dyn_cfg.d_bottleneck)
    if len(dyn_cfg.latent_mean) != d_bottleneck or len(dyn_cfg.latent_std) != d_bottleneck:
        raise ValueError(
            f"latent_mean/latent_std length must match d_bottleneck={d_bottleneck}, "
            f"got {len(dyn_cfg.latent_mean)}/{len(dyn_cfg.latent_std)}."
        )
    frame_h, frame_w = int(tok_cfg.decoder.H), int(tok_cfg.decoder.W)
    expected_nbytes = dataset_info["first_record_sequence_length"] * frame_h * frame_w * 3
    if dataset_info["first_record_raw_video_nbytes"] != expected_nbytes:
        raise ValueError(
            f"Dataset record holds {dataset_info['first_record_raw_video_nbytes']} "
            f"video bytes for {dataset_info['first_record_sequence_length']} frames, "
            f"but the tokenizer checkpoint expects {frame_h}x{frame_w}x3 frames "
            f"({expected_nbytes} bytes). Dataset/config mismatch."
        )
    if dataset_info.get("action_max") is not None and dataset_info["action_max"] >= cat_dim:
        raise ValueError(
            f"Dataset action index {dataset_info['action_max']} is outside the "
            f"dynamics checkpoint's action space [0, {cat_dim}). Dataset/config mismatch."
        )


# ---------------------------------------------------------------------------
# Rollout helpers (JAX)
# ---------------------------------------------------------------------------

def _to_uint8(frames) -> np.ndarray:
    return np.asarray(jax.device_get(jnp.clip(frames, 0, 255).astype(jnp.uint8)))


def _open_loop_rollout(dynamics, schedule, z_ctx, act_ctx, act_future, horizon, rng):
    """Autoregressive rollout. Only context latents are provided; predicted
    latents are fed back through the KV cache, so future ground truth cannot
    leak into the rollout (no teacher forcing)."""
    result = latent_rollout(
        dynamics,
        actions_future=act_future,
        schedule=schedule,
        latents_ctx=z_ctx,
        actions_ctx=act_ctx,
        num_steps=horizon,
        rng=rng,
    )
    return result["latents"]  # (B, context + horizon, n_latents, d), unnormalized


def _one_step_prediction(dynamics, schedule, z, actions, position, context, rng):
    """Predict the latent at context+position given ground truth up to it."""
    end = context + position  # predict index `end` from latents [0, end)
    result = latent_rollout(
        dynamics,
        actions_future=actions[:, end:end + 1],
        schedule=schedule,
        latents_ctx=z[:, :end],
        actions_ctx=actions[:, :end],
        num_steps=1,
        rng=rng,
    )
    return result["latents"][:, -1]  # (B, n_latents, d)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

ROLLOUT_ARMS = ("true", "episode_permuted", "action_permuted")


def run(args: Args) -> dict:
    """Run the bounded evaluation and write results.json + MP4s."""
    t_start = time.perf_counter()
    validate_args(args)

    seq_len = args.context + args.horizon
    out_dir = Path(args.out_dir)
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = preflight_dataset(
        args.array_record_path, seq_len, args.batch_size, args.preflight_scan_records,
        p_include_reward=args.p_include_reward,
    )

    mesh, _data_sharding, mesh_rules = build_parallel("data")

    with jax.set_mesh(mesh):
        t_load = time.perf_counter()
        tokenizer, dynamics, artifact_info = load_models(args, mesh_rules)
        load_sec = time.perf_counter() - t_load

        validate_model_configs(dynamics.cfg, tokenizer.cfg, dataset_info)

        k_max = int(dynamics.cfg.k_max)
        if args.denoise_steps > k_max or k_max % args.denoise_steps != 0:
            raise ValueError(
                f"denoise_steps={args.denoise_steps} must divide the dynamics "
                f"checkpoint's k_max={k_max}."
            )
        categorical_action_dim = int(dynamics.cfg.categorical_action_dim)

        schedule = DenoiseSchedule.init(args.denoise_steps, k_max)
        positions = evenly_spaced_positions(args.horizon, args.one_step_positions)
        frame_h, frame_w = int(tokenizer.cfg.decoder.H), int(tokenizer.cfg.decoder.W)

        dataset_cfg = DatasetConfig(
            name="coinrun",
            dataloader_cfg=DataloaderConfig(
                B=args.batch_size,
                num_workers=args.num_workers,
                prefetch_buffer_size=2,
                device_prefetch_buffer_size=1,
                short_T=seq_len,
                long_T=seq_len,
                long_ratio=0.0,
                dtype="float32",
            ),
            H=frame_h,
            W=frame_w,
            C=3,
            patch_size=int(tokenizer.cfg.encoder.patch_size),
            categorical_action_dim=categorical_action_dim,
            array_record_path=args.array_record_path,
            p_include_reward=args.p_include_reward,
        )
        dataloader = build_iterator(dataset_cfg, seed=args.seed, print_filter_warnings=False)

        print(
            f"Evaluating {args.dynamics_model} (step {artifact_info['dynamics_step']}) "
            f"+ tokenizer (step {artifact_info['tokenizer_step']}, "
            f"{artifact_info['tokenizer_source']}) on {args.num_windows} held-out windows "
            f"(context={args.context}, horizon={args.horizon}, "
            f"denoise_steps={args.denoise_steps}, one-step positions {positions})."
        )

        H = args.horizon
        # Per-window metric series, retained for paired effect sizes:
        # {arm: {"psnr": [rows of H], "latent_mse": [rows of H]}}
        series = {arm: {"psnr": [], "latent_mse": []} for arm in ROLLOUT_ARMS}
        pixel_sums = {arm: {"pixel_mse": np.zeros(H), "pixel_mae": np.zeros(H)} for arm in ROLLOUT_ARMS}
        one_step_sums = {
            pos: {"pixel_mse": 0.0, "pixel_mae": 0.0, "psnr": 0.0, "latent_mse": 0.0}
            for pos in positions
        }
        recon_sums = {"pixel_mse": 0.0, "pixel_mae": 0.0, "psnr": 0.0}
        baseline_sums = {
            "dataset_mean": {"pixel_mse": 0.0, "psnr": 0.0},
            "copy_previous": {"pixel_mse": 0.0, "psnr": 0.0},
        }
        baseline_h_sums = {
            name: {"pixel_mse": np.zeros(H), "psnr": np.zeros(H)}
            for name in ("dataset_mean", "copy_previous")
        }
        # Latent scale accumulators (per bottleneck dimension).
        d_bottleneck = int(dynamics.cfg.d_bottleneck)
        z_sum = np.zeros(d_bottleneck, dtype=np.float64)
        z_sumsq = np.zeros(d_bottleneck, dtype=np.float64)
        z_count = 0
        reward_frame_count = 0
        reward_nonzero_count = 0

        windows: list[dict] = []
        videos_meta: list[dict] = []
        batch_times: list[float] = []

        # Dataset-mean baseline frame from the checkpoint's own dataset stats.
        dataset_mean_pixel = np.asarray(
            tokenizer.cfg.encoder.dataset_mean, dtype=np.float64
        ) * 255.0  # (C,)

        rng = jax.random.PRNGKey(args.seed)
        n_batches = max(1, math.ceil(args.num_windows / args.batch_size))
        windows_done = 0
        video_budget = args.num_videos

        for batch_idx in range(n_batches):
            if windows_done >= args.num_windows:
                break
            t_batch = time.perf_counter()
            batch = next(dataloader)
            videos_np = np.asarray(jax.device_get(batch["videos"]))  # (B, T, H, W, C) uint8
            B, T = videos_np.shape[:2]
            if T != seq_len:
                raise RuntimeError(
                    f"Dataloader returned sequence length {T}, expected "
                    f"{seq_len} (= context + horizon)."
                )
            rewards_np = np.asarray(jax.device_get(batch["rewards"]))
            actions = shift_actions_coinrun(batch["actions"])

            rng, batch_rng = jax.random.split(rng)
            rollout_rng, one_step_rng = jax.random.split(batch_rng)

            # Tokenizer reconstruction (G1) and latent scale measurement (G2).
            z = encode_jit(tokenizer, batch["videos"])  # (B, T, n, d)
            gt_decoded = _to_uint8(decode_jit(tokenizer, z))

            z_np = np.asarray(jax.device_get(z), dtype=np.float64)
            z_sum += z_np.sum(axis=(0, 1, 2))
            z_sumsq += (z_np ** 2).sum(axis=(0, 1, 2))
            z_count += int(np.prod(z_np.shape[:3]))

            z_ctx, act_ctx = z[:, :args.context], actions[:, :args.context]
            act_future = actions[:, args.context:]

            # Paired rollouts: identical context latents and sampling RNG;
            # only the future action sequence differs between arms.
            act_future_ep_perm = jax.tree.map(lambda x: x[roll_permutation(B)], act_future)
            act_future_ac_perm = permute_action_indices(act_future)
            rollouts = {
                arm: _open_loop_rollout(
                    dynamics, schedule, z_ctx, act_ctx, arm_actions, H, rollout_rng
                )
                for arm, arm_actions in (
                    ("true", act_future),
                    ("episode_permuted", act_future_ep_perm),
                    ("action_permuted", act_future_ac_perm),
                )
            }
            preds = {arm: _to_uint8(decode_jit(tokenizer, lat)) for arm, lat in rollouts.items()}
            lat_np = {arm: np.asarray(jax.device_get(lat), dtype=np.float64) for arm, lat in rollouts.items()}

            # One-step predictions at the declared positions.
            one_step_lat: dict[int, np.ndarray] = {}
            one_step_frame: dict[int, np.ndarray] = {}
            for pos in positions:
                one_step_rng, pos_rng = jax.random.split(one_step_rng)
                lat_1 = _one_step_prediction(
                    dynamics, schedule, z, actions, pos, args.context, pos_rng
                )
                one_step_lat[pos] = np.asarray(jax.device_get(lat_1))
                one_step_frame[pos] = _to_uint8(
                    decode_jit(tokenizer, lat_1[:, None])
                )[:, 0]

            for i in range(B):
                rec = frame_metrics(gt_decoded[i], videos_np[i])
                for name in recon_sums:
                    recon_sums[name] += rec[name]

                # Baselines (G1): dataset-mean over all frames; copy-previous
                # over frames [1, T) (frame t predicted by frame t-1).
                bm = frame_metrics(
                    np.broadcast_to(dataset_mean_pixel, videos_np[i].shape), videos_np[i]
                )
                baseline_sums["dataset_mean"]["pixel_mse"] += bm["pixel_mse"]
                baseline_sums["dataset_mean"]["psnr"] += bm["psnr"]
                cp = frame_metrics(videos_np[i, :-1], videos_np[i, 1:])
                baseline_sums["copy_previous"]["pixel_mse"] += cp["pixel_mse"]
                baseline_sums["copy_previous"]["psnr"] += cp["psnr"]
                for h in range(H):
                    t = args.context + h
                    for name, pred in (
                        ("dataset_mean", np.broadcast_to(dataset_mean_pixel, videos_np[i, t].shape)),
                        ("copy_previous", videos_np[i, t - 1]),
                    ):
                        fm = frame_metrics(pred, videos_np[i, t])
                        baseline_h_sums[name]["pixel_mse"][h] += fm["pixel_mse"]
                        baseline_h_sums[name]["psnr"][h] += fm["psnr"]

                reward_frame_count += int(rewards_np[i].size)
                reward_nonzero_count += int((rewards_np[i] > 0).sum())

                per_window = {
                    "index": windows_done + i,
                    "reward_mean": float(rewards_np[i].mean()),
                    "recon_psnr": rec["psnr"],
                }
                for arm in ROLLOUT_ARMS:
                    psnrs = []
                    for h in range(H):
                        fm = frame_metrics(preds[arm][i, args.context + h], videos_np[i, args.context + h])
                        lm = latent_metrics(lat_np[arm][i, args.context + h], z_np[i, args.context + h])
                        pixel_sums[arm]["pixel_mse"][h] += fm["pixel_mse"]
                        pixel_sums[arm]["pixel_mae"][h] += fm["pixel_mae"]
                        series[arm]["psnr"].append(fm["psnr"])
                        series[arm]["latent_mse"].append(lm["latent_mse"])
                        psnrs.append(fm["psnr"])
                    per_window[f"open_loop_{arm}_psnr_mean"] = float(np.mean(psnrs))
                for pos in positions:
                    fm = frame_metrics(one_step_frame[pos][i], videos_np[i, args.context + pos])
                    lm = latent_metrics(one_step_lat[pos][i], z_np[i, args.context + pos])
                    for name in one_step_sums[pos]:
                        one_step_sums[pos][name] += fm.get(name, lm.get(name, 0.0))
                windows.append(per_window)

                # MP4 export (GT row over model row, seam bar declares the split).
                if video_budget > 0 and windows_done + i < args.num_windows:
                    for arm in ROLLOUT_ARMS:
                        pred_row = np.concatenate(
                            [gt_decoded[i, :args.context], preds[arm][i, args.context:]],
                            axis=0,
                        )
                        video = build_eval_video(videos_np[i], pred_row, args.context)
                        fname = f"window{windows_done + i:04d}_ctx{args.context}_hor{H}_{arm}.mp4"
                        fpath = video_dir / fname
                        iio.imwrite(str(fpath), video, fps=args.fps, codec="libx264")
                        # Hard-fails (review G8) on constant or copy-last-context output.
                        stats = verify_mp4(fpath, seq_len, args.context)
                        videos_meta.append({
                            "path": str(fpath),
                            "arm": arm,
                            "window": windows_done + i,
                            "context": args.context,
                            "horizon": H,
                            **stats,
                        })
                    video_budget -= 1

            windows_done += B
            batch_times.append(time.perf_counter() - t_batch)
            print(
                f"batch {batch_idx + 1}/{n_batches}: {windows_done} windows "
                f"({batch_times[-1]:.1f}s)"
            )

    # Aggregate over windows (the last batch may overshoot num_windows; the
    # overshoot windows are still counted, which is reported below).
    n = float(windows_done)
    windows = windows[: args.num_windows]

    arm_arrays = {
        arm: {name: np.asarray(rows, dtype=np.float64).reshape(-1, H) for name, rows in series[arm].items()}
        for arm in ROLLOUT_ARMS
    }

    def _arm_block(arm: str) -> dict:
        return {
            "horizon": list(range(H)),
            "psnr": arm_arrays[arm]["psnr"].mean(axis=0).tolist(),
            "latent_mse": arm_arrays[arm]["latent_mse"].mean(axis=0).tolist(),
            "pixel_mse": (pixel_sums[arm]["pixel_mse"] / n).tolist(),
            "pixel_mae": (pixel_sums[arm]["pixel_mae"] / n).tolist(),
        }

    def _sensitivity_block(arm: str) -> dict:
        """Paired effect sizes of an action-permutation arm vs true actions.

        Sign convention: positive delta = consistent with action sensitivity
        (PSNR drops / latent MSE rises when future actions are corrupted).
        """
        psnr_drop = arm_arrays["true"]["psnr"] - arm_arrays[arm]["psnr"]
        mse_increase = arm_arrays[arm]["latent_mse"] - arm_arrays["true"]["latent_mse"]
        return {
            "psnr_drop": {
                "per_horizon": [paired_effect_size(psnr_drop[:, h]) for h in range(H)],
                "overall": paired_effect_size(psnr_drop.mean(axis=1)),
            },
            "latent_mse_increase": {
                "per_horizon": [paired_effect_size(mse_increase[:, h]) for h in range(H)],
                "overall": paired_effect_size(mse_increase.mean(axis=1)),
            },
        }

    z_mean = z_sum / z_count
    z_std = np.sqrt(np.maximum(z_sumsq / z_count - z_mean ** 2, 0.0))

    metrics = {
        "recon": {name: recon_sums[name] / n for name in recon_sums},
        "baselines": {
            "dataset_mean": {
                "frames": "all T frames of each window",
                **{name: baseline_sums["dataset_mean"][name] / n for name in ("pixel_mse", "psnr")},
            },
            "copy_previous": {
                "frames": "frames 1..T-1 (frame t predicted by frame t-1)",
                **{name: baseline_sums["copy_previous"][name] / n for name in ("pixel_mse", "psnr")},
            },
            "dataset_mean_horizon": {
                "horizon": list(range(H)),
                **{name: (baseline_h_sums["dataset_mean"][name] / n).tolist() for name in ("pixel_mse", "psnr")},
            },
            "copy_previous_horizon": {
                "horizon": list(range(H)),
                **{name: (baseline_h_sums["copy_previous"][name] / n).tolist() for name in ("pixel_mse", "psnr")},
            },
        },
        "latent_scale": {
            "per_dim_mean": z_mean.tolist(),
            "per_dim_std": z_std.tolist(),
            "max_abs_mean": float(np.abs(z_mean).max()),
            "max_abs_std_minus_1": float(np.abs(z_std - 1.0).max()),
        },
        "reward": {
            "frame_fraction_nonzero": reward_nonzero_count / max(reward_frame_count, 1),
            "frames_evaluated": reward_frame_count,
        },
        "one_step": {
            "positions": positions,
            **{name: [one_step_sums[pos][name] / n for pos in positions]
               for name in ("pixel_mse", "pixel_mae", "psnr", "latent_mse")},
        },
        "open_loop_true": _arm_block("true"),
        "open_loop_episode_permuted": _arm_block("episode_permuted"),
        "open_loop_action_permuted": _arm_block("action_permuted"),
        "sensitivity": {
            "episode_permutation": _sensitivity_block("episode_permuted"),
            "action_permutation": _sensitivity_block("action_permuted"),
            "action_permutation_mapping": list(ACTION_INDEX_PERMUTATION),
            "action_permutation_note": (
                "indices 0-8 remapped by x -> (x + 5) % 9 (all behaviorally "
                "distinct); no-op aliases 9-14 unchanged; no-op index 4 used "
                "for the start-of-sequence shift"
            ),
        },
    }
    metrics["summary"] = {
        "recon_psnr": metrics["recon"]["psnr"],
        "recon_minus_dataset_mean_psnr_db": metrics["recon"]["psnr"] - metrics["baselines"]["dataset_mean"]["psnr"],
        "recon_minus_copy_previous_psnr_db": metrics["recon"]["psnr"] - metrics["baselines"]["copy_previous"]["psnr"],
        "one_step_psnr_mean": float(np.mean(metrics["one_step"]["psnr"])),
        "open_loop_true_psnr_mean": float(np.mean(metrics["open_loop_true"]["psnr"])),
        "open_loop_true_psnr_last_horizon": float(metrics["open_loop_true"]["psnr"][-1]),
        "copy_previous_horizon_psnr_mean": float(np.mean(metrics["baselines"]["copy_previous_horizon"]["psnr"])),
        "open_loop_true_minus_copy_previous_db": float(
            np.mean(metrics["open_loop_true"]["psnr"]) - np.mean(metrics["baselines"]["copy_previous_horizon"]["psnr"])
        ),
        "episode_permutation_psnr_drop": metrics["sensitivity"]["episode_permutation"]["psnr_drop"]["overall"]["mean_delta"],
        "episode_permutation_effect_size_dz": metrics["sensitivity"]["episode_permutation"]["psnr_drop"]["overall"]["effect_size_dz"],
        "action_permutation_psnr_drop": metrics["sensitivity"]["action_permutation"]["psnr_drop"]["overall"]["mean_delta"],
        "action_permutation_effect_size_dz": metrics["sensitivity"]["action_permutation"]["psnr_drop"]["overall"]["effect_size_dz"],
        "latent_scale_max_abs_mean": metrics["latent_scale"]["max_abs_mean"],
        "latent_scale_max_abs_std_minus_1": metrics["latent_scale"]["max_abs_std_minus_1"],
        "reward_frame_fraction_nonzero": metrics["reward"]["frame_fraction_nonzero"],
    }

    results = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "git_commit": git_commit(Path(__file__).resolve().parent.parent),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": asdict(args),
        "artifacts": {**artifact_info, "dataset": dataset_info,
                      "array_record_path": args.array_record_path},
        "model": {
            "dynamics_params": int(dynamics.num_scaling_params()),
            "tokenizer_params": int(tokenizer.num_scaling_params()),
            "k_max": k_max,
            "categorical_action_dim": categorical_action_dim,
            "coinrun_noop_index": COINRUN_NOOP_INDEX,
        },
        "setup": {
            "context": args.context,
            "horizon": H,
            "seq_len": seq_len,
            "batch_size": args.batch_size,
            "num_windows_requested": args.num_windows,
            "num_windows_evaluated": int(windows_done),
            "denoise_steps": args.denoise_steps,
            "one_step_positions": positions,
            "p_include_reward": args.p_include_reward,
            "seed": args.seed,
            "rollout_arms": list(ROLLOUT_ARMS),
            "teacher_forcing": "none (open-loop rollouts feed back predictions; "
                               "one-step metrics use ground-truth context only)",
        },
        "timing_sec": {
            "load_models": load_sec,
            "per_batch": batch_times,
            "total": time.perf_counter() - t_start,
        },
        "metrics": metrics,
        "windows": windows,
        "videos": videos_meta,
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    s = metrics["summary"]

    def _fmt_dz(dz):
        return "n/a" if dz is None else f"{dz:+.2f}"

    print("\n" + "=" * 68)
    print(f"CoinRun eval: {windows_done} held-out windows, context={args.context}, horizon={H}")
    print("=" * 68)
    print(f"  tokenizer recon PSNR                 : {s['recon_psnr']:.2f} dB "
          f"(dataset-mean {metrics['baselines']['dataset_mean']['psnr']:.2f}, "
          f"copy-previous {metrics['baselines']['copy_previous']['psnr']:.2f})")
    print(f"  latent scale |mean| / |std-1| (max)  : "
          f"{s['latent_scale_max_abs_mean']:.4f} / {s['latent_scale_max_abs_std_minus_1']:.4f}")
    print(f"  one-step PSNR (mean over positions)  : {s['one_step_psnr_mean']:.2f} dB")
    print(f"  open-loop PSNR true (mean/last)      : "
          f"{s['open_loop_true_psnr_mean']:.2f} / {s['open_loop_true_psnr_last_horizon']:.2f} dB "
          f"(copy-previous baseline {s['copy_previous_horizon_psnr_mean']:.2f})")
    print(f"  sensitivity episode-perm PSNR drop   : "
          f"{s['episode_permutation_psnr_drop']:+.3f} dB "
          f"(dz={_fmt_dz(s['episode_permutation_effect_size_dz'])})")
    print(f"  sensitivity action-perm PSNR drop    : "
          f"{s['action_permutation_psnr_drop']:+.3f} dB "
          f"(dz={_fmt_dz(s['action_permutation_effect_size_dz'])})")
    print(f"  reward frame fraction nonzero        : {s['reward_frame_fraction_nonzero']:.4f}")
    print(f"  results: {results_path}")
    print(f"  videos : {video_dir} ({len(videos_meta)} files)")
    print("=" * 68)

    return results


def main() -> None:
    run(tyro.cli(Args, description=__doc__))


if __name__ == "__main__":
    main()
