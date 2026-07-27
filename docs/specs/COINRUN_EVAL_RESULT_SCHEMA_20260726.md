# CoinRun Evaluation Result Schema

Date: 2026-07-26

Producer: `scripts/eval_coinrun.py` (`schema_version: "coinrun-eval/2"`)

This document defines the `results.json` written by the bounded CoinRun
evaluator, the sign conventions, and the fail-closed gates it implements from
`docs/reviews/COINRUN_RECONSTRUCTION_ADVERSARIAL_REVIEW_20260726.md` (G1–G9)
on top of `docs/specs/COINRUN_RECONSTRUCTION_SPRINT_20260726.md`.

## Scope and invariants

- Evaluation windows come from a held-out ArrayRecord split
  (`--array-record-path`, default `datasets/coinrun_episodes/val`). A path
  whose final component contains `train` is refused before any work (G4).
  The evaluator never touches the training batch and never calls the
  trainer's `input_tensor[:4]` path.
- Checkpoints are loaded read-only via Orbax (`from_pretrained`). Missing
  checkpoint directories, checkpoint dirs without steps, missing dataset
  files, non-pickle records, or record/model shape mismatches abort the run
  with a clear error before evaluation. The evaluator never creates or
  falls back to randomly initialized weights.
- Open-loop rollouts are structurally incapable of teacher-forcing future
  frames: `latent_rollout` receives context latents only and feeds its own
  predictions back through the KV cache. One-step metrics are the only
  ground-truth-conditioned predictions and are labelled as such.
- CoinRun action semantics (G3): 15 discrete actions, no-op index 4
  (Procgen action table). The start-of-sequence shift prepends 4, not
  `categorical_action_dim // 2`. The dynamics checkpoint must declare
  `categorical_action_dim == 15` or evaluation fails closed.
- Latent scale (G2): if the dynamics checkpoint has
  `latent_mean`/`latent_std = None`, evaluation fails closed (latent
  normalization would otherwise be silently disabled).

## Top-level fields

| field | meaning |
|---|---|
| `schema_version` | `"coinrun-eval/2"` |
| `status` | `"completed"` (runs that hit a gate raise and write no file) |
| `git_commit` | commit of the evaluation repo, best effort |
| `timestamp_utc` | run start, UTC |
| `args` | resolved CLI arguments |
| `artifacts` | checkpoint paths/steps actually loaded, `dynamics_model` (`dynamics` or `dynamics_ema`), `tokenizer_source` (`dynamics_bundle` = the online tokenizer the dynamics model was trained against, or `separate`), dataset preflight info |
| `model` | parameter counts, `k_max`, `categorical_action_dim` (15), `coinrun_noop_index` (4) |
| `setup` | declared `context`, `horizon`, `seq_len` (= context + horizon), batch/window counts, `denoise_steps`, `one_step_positions`, `p_include_reward`, `seed`, `rollout_arms`, teacher-forcing statement |
| `timing_sec` | model load, per-batch, total wall time |
| `metrics` | see below |
| `windows` | per-window rows: `reward_mean`, `recon_psnr`, per-arm mean PSNR |
| `videos` | per-MP4 verification stats (see below) |

## `metrics`

PSNR is in dB on uint8 frames, capped at 99. All per-horizon arrays are
indexed by `h = 0 .. horizon-1`, i.e. prediction of frame `context + h`.

| block | meaning |
|---|---|
| `recon` | tokenizer `decode(encode(x))` pixel MSE/MAE/PSNR over all frames of each window (G1) |
| `baselines.dataset_mean` | MSE/PSNR of a constant per-channel dataset-mean frame over all frames |
| `baselines.copy_previous` | MSE/PSNR of frame `t-1` predicting frame `t`, over frames `1..T-1` |
| `baselines.*_horizon` | the same two baselines restricted to the horizon frames, per `h` — the reference for judging open-loop curves |
| `latent_scale` | per-bottleneck-dimension mean/std of tokenizer latents measured on the evaluated windows, plus `max_abs_mean` and `max_abs_std_minus_1` (G2 measurement) |
| `reward` | `frame_fraction_nonzero` over evaluated windows (G6) |
| `one_step` | per-position pixel/latent metrics; prediction of frame `context + pos` given ground truth up to `context + pos - 1` |
| `open_loop_true` | open-loop rollout with true future actions: per-`h` PSNR, latent MSE, pixel MSE/MAE |
| `open_loop_episode_permuted` | same rollout with another window's future action sequence (episode-level permutation, paired RNG) |
| `open_loop_action_permuted` | same rollout with future action indices remapped by `x -> (x+5) % 9` over the behaviorally distinct actions 0–8; no-op aliases 9–14 unchanged (G5) |
| `sensitivity.episode_permutation` / `sensitivity.action_permutation` | paired effect sizes vs the true arm |
| `summary` | headline scalars (see stdout summary for the same numbers) |

### Sensitivity sign conventions

For each window and horizon step, with `arm ∈ {episode_permuted, action_permuted}`:

- `psnr_drop = psnr_true − psnr_arm` — positive is consistent with action sensitivity.
- `latent_mse_increase = latent_mse_arm − latent_mse_true` — positive is consistent with action sensitivity.

Each is reported per horizon step and overall (window means) as
`{mean_delta, std_delta, effect_size_dz, frac_positive, n}` where
`effect_size_dz` is the paired Cohen's dz (`mean/std`, `ddof=1`) and is
`null` when the delta variance is zero. A clean negative (deltas ≈ 0 or
negative, dz ≈ 0) is a valid, reportable outcome.

## `videos` entries

One entry per exported MP4 (`num_videos` windows × 3 arms). Filename:
`windowNNNN_ctxC_horH_<arm>.mp4`. Layout: top row ground truth, bottom row
model output (context region = tokenizer reconstruction, horizon region =
rollout), separated by a 16-row seam bar that is green for context frames
and red for predicted frames — the declared context/horizon split is
therefore verifiable from the video itself.

| field | meaning |
|---|---|
| `frames_readback` | decoded frame count; must equal `context + horizon` |
| `frame_std` | whole-video std; must exceed 1.0 |
| `seam_marker_ok` | seam bar decoded green→red exactly at `context` |
| `horizon_region_std` | mean per-frame spatial std of the generated horizon region |
| `copy_last_context_mad_max` | max over horizon frames of mean-abs-diff vs the last context frame |
| `frozen_mad_mean` | mean frame-to-frame abs-diff across the horizon (`null` when horizon = 1) |
| `nontrivial` | `true` when all G8 checks passed |

G8 is fail-closed: the run raises when any exported video has a constant
generated region (`horizon_region_std ≤ 3.0`), copies the last context frame
(`copy_last_context_mad_max < 1.0`), or is frozen across the horizon
(`frozen_mad_mean < 1.5`). Thresholds are on the lossy MP4 readback.

## Dataset preflight (`artifacts.dataset`)

`num_files`, `num_records`, `scanned_records`, `usable_records_in_scan`,
`action_min`/`action_max` (must stay within `[0, 15)`),
`nonzero_reward_frames_in_scan` / `scanned_frames` (reward prevalence, G6;
`p_include_reward > 0` with zero nonzero-reward frames is refused),
`level_seed_key` + `unique_level_seeds_in_scan` (whether level seeds are
recorded, G4), and first-record shape facts used for model/dataset
cross-checks.

## Known conventions and limitations

- `rewards[t]` records the reward for the transition *into* frame `t`
  (caused by `actions[t-1]`), matching the generator's loop order; the
  evaluator reports reward prevalence only and does not re-align.
- Episode-level permutation uses distinct sampled windows as episode
  proxies; records do not carry episode ids, so two windows in a batch can
  in principle come from the same source episode.
- Action-sensitive-state restriction from G5 ("grounded frames") is not
  implementable from stored records (no env access at eval time); the
  action-index permutation is applied on all horizon steps instead.
- `dynamics_ema` is the default evaluated model; at very small training
  budgets the EMA weights remain close to initialization (review §4.5), so
  `--dynamics-model dynamics` is available and the choice is recorded.
- Multi-device sharding is out of scope; evaluation runs on a single
  process/device mesh (`parallel_strategy: data`).
