# CoinRun Reconstruction Sprint — Adversarial Review

Date: 2026-07-26. Reviewer: Claude (Opus 5), independent read-only audit.
No files edited, no training launched.

Audited: `docs/specs/COINRUN_RECONSTRUCTION_SPRINT_20260726.md` against the
official OpenDreamer source at pinned commit `797e41f`, plus the pinned Procgen
checkout at `/home/ubuntu/reference/procgen`.

**Method note.** The working tree was being edited during this audit
(`scripts/train_dynamics.py` gained 78 lines and two untracked `configs/coinrun_*.yaml`
appeared mid-review). Every finding below was therefore re-verified against
`git show 797e41f:<path>`; all cited files except `scripts/train_dynamics.py`
are byte-identical between the worktree and the pinned commit (md5 checked).
Line numbers for `train_dynamics.py` refer to the **pinned** blob.

## Verdict

**NO-GO for the sprint as written. GO for a re-scoped four-hour "plumbing
certificate" sprint** with the gates in §6.

Two separate reasons:

1. **The released CoinRun data path has never been executed at this commit.**
   It fails with a `TypeError` on the first episode, and has two further
   independent defects behind it (§1). Deliverable 1 is a from-scratch build,
   not a fix — and the sprint budgets no time for that.
2. **Four hours cannot produce an interpretable pixel CoinRun result**, only a
   plumbing certificate. The spec's "Scientific Readout" (§Scientific Readout,
   five comparisons) is not reachable at the proposed 100–500-step scale (§5).
   The spec conflates these two deliverables; they must be named separately.

A clean negative is available and cheap: the sprint can honestly conclude
"the released CoinRun path is non-functional and the reconstruction is a new
implementation" within the first hour.

## 1. Blocking: the released CoinRun data generator cannot run

Three independent defects in `dreamer/data/generate_coinrun_dataset.py`
(pinned, unmodified). All verified by executing the constructor and a
write/read round-trip in the repo's own `.venv` — no training involved.

**1.1 `TypeError` on the first episode.** Lines 110–114 call
`ShardWriter(output_dir_split, records_per_shard=..., serialization_format="pickle")`.
`ShardWriter.__init__` (`dreamer/data/shard_writer.py:25-28`) has signature
`(output_dir, records_per_shard=1000)`. Verified:

```
ShardWriter.__init__ signature: (self, output_dir: Path | str, records_per_shard: int = 1000)
ShardWriter(serialization_format=...) -> TypeError: unexpected keyword argument 'serialization_format'
```

**1.2 Writer/reader format mismatch behind it.** `ShardWriter.write`
(`shard_writer.py:57`) calls `serialize_msgpack_record`. The CoinRun read path
uses `pickle.loads` in both `EpisodeLengthFilter.filter`
(`dreamer/data/transforms.py:85`) and `ProcessEpisodeAndSlice.random_map`
(`transforms.py:172`). Verified round-trip:

```
bytes head: b'\x84\xa9raw_video\xc4'          # msgpack
coinrun reader pickle.loads -> ValueError: unregistered extension code 2002875049
```

The dead `serialization_format="pickle"` argument is evidence the writer once
supported pickle and the support was removed or never landed.

**1.3 `AttributeError` at the end of every split.** Line 193 reads
`writer.num_shards`. `ShardWriter` defines no `num_shards` or `total_records`
property — both appear only in its docstring (`shard_writer.py:17-20`).
Verified: `hasattr(ShardWriter, "num_shards") == False`.

**Implication.** The sprint's assumption that a CoinRun generator exists to be
adapted is false. Budget a full rewrite of record writing plus a real
serialization contract, and add a round-trip test (the spec's deliverable 6
already asks for one — it is now load-bearing, not nice-to-have).

## 2. Blocking: silent action-semantics corruption

**2.1 The "no-op" start action is *run-right + jump*.**
`create_noop_action_like` (`dreamer/actions.py:40-51`) fills the categorical
start action with `categorical_action_dim // 2`, with the comment "verified that
this is equal to `mouse_movement_to_categorical(dx=0, dy=0)`". That is correct
**only for Minecraft**, whose categorical action is the 11×11 mu-law camera grid
(121 // 2 = 60 = centre bin).

For CoinRun the categorical action is the Procgen discrete action.
`configs/dataset/coinrun.yaml` declares `categorical_action_dim: 16`, so the
"no-op" becomes index **8**. Procgen's action table
(`procgen/env.py:155-172`) is:

```
0 (LEFT,DOWN) 1 (LEFT) 2 (LEFT,UP) 3 (DOWN) 4 () 5 (UP)
6 (RIGHT,DOWN) 7 (RIGHT) 8 (RIGHT,UP) 9 (D) 10 (A) 11 (W) 12 (S) 13 (Q) 14 (E)
```

Index 8 is `("RIGHT","UP")` — run right and jump. The true no-op is **index 4**.
`shift_actions` is applied on the critical path in both
`scripts/train_dynamics.py:314` (pinned) and `scripts/eval_fvd.py:126`, so every
training sequence and every rollout is prefixed with a jump-right action labelled
"no-op". Nothing asserts.

This is precisely the "silently remapped" failure the spec's deliverable 2 warns
about — but the risk is silent corruption, not rejection (see §4.1).

**2.2 The declared action dimension is wrong.** Procgen CoinRun has **15**
actions (`num_actions = len(self.combos)`, `env.py:115`), not 16.
`validate_dynamics_config` (`train_dynamics.py:176-198`) only checks that
`dataset.categorical_action_dim == dynamics.categorical_action_dim`; it never
checks either against the environment. The released config's 16 therefore
passes, leaving one dead embedding row and the wrong no-op index. With the
correct 15, `15 // 2 = 7 = ("RIGHT")` — still wrong.

**2.3 Nearly half of uniformly-random actions are behaviourally identical.**
`BasicAbstractGame::game_step` (`basic-abstract-game.cpp:687-694`) computes
`move_action = action % 9`, and for `action >= 9` forces `move_action = 4`.
CoinRun has no special actions, so **actions {4, 9, 10, 11, 12, 13, 14} all mean
"stand still"** — 7 of 15, i.e. **46.7% of uniform random actions are no-ops**.

Two consequences for the sprint:

- the random collector wastes ~47% of its steps, worsening an already severe
  coverage problem;
- PASS gate 4 ("dynamics output changes under a legal action permutation") is
  weakened toward a **false negative**: a uniform permutation over 15 indices
  frequently maps one no-op onto another. The permutation must be restricted to
  behaviourally distinct actions (a subset of 0–8).

## 3. Blocking for interpretability: nothing is held out

**3.1 The "validation" batch is the training batch.** Pinned
`scripts/train_dynamics.py:279-291`:

```python
do_eval = (cfg.write_video_every>0 and ...) or step == cfg.max_steps - 1
if do_eval:
    val_data = input_tensor[:4]
    val_actions = actions[:4]
    run_evaluation(..., val_data=val_data, val_actions=val_actions, ...)
```

There is **no validation split anywhere in the released dynamics trainer**. The
generated MP4 that PASS gate 5 accepts is rendered from the current training
batch. This is the single most likely route to a convincing fake pass: at
100–500 steps on a small corpus, reproducing four training clips is close to
memorisation.

**3.2 Level splits are not disjoint and level identity is not recorded.**
`generate_coinrun_dataset.py:117` draws `seed = np.random.randint(0, 10000)` for
train, val **and** test from a single `np.random.seed(args.seed)` stream with no
exclusion (lines 204-208), and the record schema
(lines 49-62: `raw_video`, `sequence_length`, `actions`, `rewards`) does **not**
store the level seed. So val/test levels overlap train almost surely, and the
overlap cannot even be measured after the fact. No generalization claim is
available from the released generator. The spec's deliverable 1 correctly
requires seeds to be recorded — note that this is an addition, not a fix.

## 4. High-risk silent traps

**4.1 Latent normalization silently disabled.** `latent_mean` / `latent_std`
default to `None` (`dreamer/configs.py:60-61`) and `configs/dataset/coinrun.yaml`
sets neither. `normalize_latents` (`dreamer/utils.py:254-255`) **returns the
input unchanged when either is `None`** — no warning. It is called on the
critical path at `dreamer/training.py:344` inside `shortcut_forcing_step`, and
at `generation.py:302,366`.

So a CoinRun run (`data_type: video`, no latent stats) trains flow matching on
**unnormalized** tokenizer latents. The tokenizer bottleneck is a plain
`nnx.Linear` with no output normalization (`dreamer/models.py:755`), so unit
scale is not guaranteed. Flow matching interpolates against `N(0,1)` noise; a
latent scale far from 1 mis-conditions the objective and produces exactly the
"loss falls, generation is garbage" outcome the spec says it wants to detect.
This is the highest-value gate to add (G2, §6).

**4.2 Dataset degeneracy at released defaults — reward is identically zero.**
`Args` defaults are `min_episode_length = max_episode_length = 1000`
(lines 82-83), and an episode is kept only if `step_t + 1 >= min_episode_length`
(line 149). Procgen's `timeout` is **1000** (`game.cpp:26`). Therefore only
episodes that survive to timeout are retained — episodes with **no coin and no
death**. Under a random policy, coin/death episodes terminate early and are
discarded ("Episode too short, resampling...").

Consequences:

- the retained corpus has **reward ≡ 0**, so `p_include_reward: 0.5` is a no-op
  (`transforms.py:197-198`: `reward_ts.size == 0` falls through to uniform), and
  the spec's readout #2 (uniform vs reward-biased windows) is **vacuous**;
- most sampled episodes are discarded, so generation is several times slower
  than the episode count suggests;
- the sprint *must* lower `min_episode_length` to keep coin/death episodes —
  which activates 4.3.

**4.3 Cross-episode contamination once episode lengths are lowered.** The loop
appends the observation *before* testing the boundary
(lines 134-136 then 144-145), so the first frame of the **next** level is
appended to the current episode, together with a discontinuous action and
reward. Harmless at the released defaults (kept episodes exit via range
exhaustion), but it becomes real corruption the moment the sprint shortens
episodes — which §4.2 forces it to do.

**4.4 Reward is shifted one step relative to its causal action.** At index `t`
the loop records `obs_t` from `env.observe()` (pre-action), `action_t` applied to
`obs_t`, and `rew_t` — which gym3 returns for the transition *into* `obs_t`,
i.e. caused by `action_{t-1}`. `ProcessEpisodeAndSlice` slices videos, actions
and rewards with the same indices (`transforms.py:208, 223, 226`), so the
off-by-one propagates. This convention must be declared and tested; anything
built on `p_include_reward` or a reward readout depends on it.

**4.5 EMA is ~60% random init at the proposed scale.** `ema_decay: 0.999`
(`configs/tokenizer.yaml:203`, `configs/dynamics.yaml`) means that after 500
steps `0.999^500 = 0.61` of the EMA weights are still the random
initialization. The released generation path treats EMA as the deployment
model. `run_evaluation` (`training.py:520-526`) helpfully renders online and EMA
columns side by side, so this is visible — but PASS gate 5 does not say which
weights must produce the nonblank MP4. Either scale `ema_decay` to the step
budget or exclude EMA readouts from the gates at this scale, explicitly.

**4.6 Dynamics uses the *online*, not EMA, tokenizer.**
`train_dynamics.py` takes `tokenizer_bundle.tokenizer`, while
`TokenizerCheckpointBundle._model_registry` (`checkpointing.py:246-249`) holds
both `tokenizer` and `tokenizer_ema`. Whether intended upstream or not, record
it: the latent distribution the dynamics model trains on is the online
tokenizer's, and any EMA-tokenizer decode path would see a different one.

**4.7 Checkpoint cadence contradicts the rental gate.**
`build_checkpoint_manager` sets `save_on_steps=[ckpt.max_steps - 1]`
(`checkpointing.py:48`) and `common.yaml` wires `ckpt.max_steps: ${max_steps}`,
so a short run *does* save one final checkpoint — good. But
`save_interval_steps` is 1000 (tokenizer) / 5000 (dynamics), so a four-hour paid
run would checkpoint **only at the very end**, directly violating the rental
gate's "checkpoint at least every 15 minutes". Must be set from a measured step
rate, and a restore must be exercised locally first.

## 5. Missing release components, and what is better than feared

**Confirmed missing at `797e41f`:**

- No CoinRun tokenizer or dynamics training config. `configs/dataset/coinrun.yaml`
  is the only CoinRun config; `tokenizer.yaml` defaults to `dataset: minecraft_vpt`
  and `dynamics.yaml` to `dataset: minecraft_vpt_latent`.
- `dreamer/data/README.md` contains **zero** occurrences of "coinrun".
- No RL collector and no BC policy (README roadmap unchecked; blog: "The
  repository we released does not include the behaviour-cloning or RL code").
- No scripted collector of any kind — the released generator is random-action only.
- No CoinRun dataset, manifest, checkpoint, or reported metric. The blog's
  CoinRun claim is qualitative ("run and jump to the right to get the coin").
- `procgen` is **not** a declared dependency in `pyproject.toml` and is not
  installed in `.venv` (verified). PASS gate 1 fails today. procgen ships no
  cp311 wheel while `pyproject.toml` requires Python 3.11, so it needs a source
  build (cmake + C++) — a material fraction of a four-hour budget. A prebuilt
  checkout exists at `/home/ubuntu/reference/procgen`, but for a different
  interpreter.
- `LICENSE` is All Rights Reserved — no reuse or redistribution right.

**Better than feared — do not budget time for these:**

- **The offline tokenization stage is not on the critical path.**
  `train_dynamics.py` sets `use_latent_data = cfg.dataset.data_type == "latent"`;
  for raw video it encodes inline with the frozen tokenizer
  (`train_step`, pinned lines 96-101). So `scripts/tokenize_minecraft_dataset.py`
  can be skipped entirely for CoinRun, and rewards remain available to the
  dynamics path (the latent path drops them — `ProcessLatentAndSlice` returns
  only latents and actions).
- **Action validation is generic, not Minecraft-specific.**
  `validate_dynamics_config` and `validate_action_batch`
  (`train_dynamics.py:176-240`) check dimensions and shapes without any
  VPT assumption. Deliverable 2's stated fear ("rejected by Minecraft/VPT-specific
  assertions") is largely unfounded; the real defect is the *silent* no-op index
  (§2.1), which these validators cannot catch.
- **FVD needs no download.** `dreamer/fvd/i3d_pretrained_400.npz` (50 MB) is
  bundled in the repo.
- **`scripts/train_dynamics.py` at the pinned commit is complete** — `train_step`,
  `ema_update_step`, checkpointing and `hydra.main` are all present and wired.

## 6. Minimum interpretable scale, and the ladder

**Scale.** The released tokenizer budget is 10,000 steps at `B=8, T=16` =
**1.28M frames** (`configs/tokenizer.yaml:185`, `:24-33`). The sprint's
100–500-step pilot is **1–5% of that** (12.8k–64k frames). A tokenizer at that
budget will not reconstruct CoinRun adequately, and a bad tokenizer makes every
downstream dynamics number uninterpretable — garbage in.

Rough estimates (speculative, stated as such): CoinRun at 64×64 with patch 8 is
64 patches/frame and a far easier reconstruction target than Minecraft, so a
small tokenizer plausibly reaches usable reconstruction in **~2,000–5,000 steps**
(0.25–0.64M frames); dynamics needs **~5,000–20,000 steps** before open-loop
structure is meaningfully assessable. Data-side, an interpretable result needs
at least a few thousand episodes over **disjoint** level ranges plus a scripted
collector — with random actions alone, 47% no-ops (§2.3) and reward ≡ 0 (§4.2)
guarantee the coverage failure the blog itself reported.

**Conclusion:** four hours buys a genuine end-to-end **plumbing certificate**.
It does not buy any of the five comparisons in the spec's Scientific Readout.
Rename the deliverable and move the readout to a follow-on with a stated budget.

**Ladder.** A10 → H100/H200 is sound *in kind*: same code path, single device,
`parallel_strategy: data` (and `common.yaml` notes that `train_tokenizer.py`
forces data parallelism in code regardless). At the proposed smoke profile
(depth 2–6, width 128–384, T=16–32, B=2–8) the local A10 (one device, 23 GB —
verified via `nvidia-smi`) is not the constraint; JAX/XLA compile time and
CPU-side Procgen generation are. But the H100-vs-H200 choice currently has **no
measured basis** — the profile that would size it does not exist yet. Make the
rental decision a function of a measured A10 activation-memory number at the
target `B`/`T`, not a preference.

## 7. PASS gates to add before any paid compute

The existing six gates are necessary but insufficient — gates 3, 4 and 5 are all
satisfiable by a pipeline that has learned nothing. Add:

| # | Gate | Rationale |
|---|---|---|
| **G1** | **Tokenizer reconstruction on held-out levels.** Report PSNR/MSE of `decode(encode(x))` and require it to beat both a per-frame dataset-mean baseline and a copy-previous-frame baseline by a declared margin. | Without this, every dynamics number is garbage-in (§5). |
| **G2** | **Latent-scale gate.** Measure per-dimension mean/std of tokenizer latents on a held-out batch; require \|mean\| and \|std − 1\| inside a declared band, or set `latent_mean`/`latent_std` explicitly. **Fail closed if either is `None` while `data_type: video`.** | §4.1 — silent, and the exact failure mode the sprint says it wants to catch. |
| **G3** | **Action-semantics gate.** Assert `env.ac_space.eltype.n == categorical_action_dim`; assert the configured no-op index equals Procgen's **4**; unit-test that `shift_actions` prepends 4, not 8. | §2.1, §2.2 — silent and on the critical path. |
| **G4** | **Held-out split gate.** Level seed recorded per record; train/val/test level sets provably disjoint; evaluation and MP4 generation drawn from the val split — never `input_tensor[:4]`. | §3.1, §3.2 — the main fake-pass route. |
| **G5** | **Action-sensitivity gate, restricted.** Permute only among behaviourally distinct actions (subset of 0–8), restricted to action-sensitive states (grounded frames), with a declared effect-size threshold — not merely "output changes". | §2.3 — current gate 4 risks a false negative and accepts a trivial pass. |
| **G6** | **Reward-prevalence gate.** Report the fraction of retained frames with nonzero reward and the coin/death/timeout episode split. Refuse to run the reward-biased arm if nonzero-reward frames are zero. | §4.2 — otherwise readout #2 is vacuous and will be reported as "no difference". |
| **G7** | **Checkpoint-cadence gate.** Set `save_interval_steps` from the measured step rate so a checkpoint lands ≥ every 15 min, and exercise a real restore locally before the pod starts. | §4.7 — contradicts the rental gate today. |
| **G8** | **Non-triviality gate on the MP4.** Assert the generated rollout differs from both a constant frame **and** from a copy of the last context frame. | "Nonblank" alone passes for a model that repeats context. |
| **G9** | **Data round-trip gate.** Byte-exact recovery of frames, actions, rewards, seed and frame count through the writer/reader pair, with the serialization format asserted. | §1 — the released pair is mutually incompatible. |

## 8. Recommended re-scope

1. **Hour 0–1.** Install `procgen` (declare it in `pyproject.toml`); rewrite the
   record writer/reader with G9; record level seeds and collector identity; fix
   the episode-length and boundary logic (§4.2, §4.3); declare the reward
   alignment (§4.4). If procgen cannot be built inside the hour, **stop and
   report that as the negative** — it is a real and publishable finding.
2. **Hour 1–2.** CoinRun tokenizer and dynamics configs as new files; G3 as
   unit tests; disjoint level splits; a real val iterator replacing
   `input_tensor[:4]`.
3. **Hour 2–3.** Tokenizer pilot to whatever step count fits; G1 and G2
   measured and reported.
4. **Hour 3–4.** Dynamics pilot; G5 and G8; write the telemetry the spec asks
   for. Report a plumbing certificate, explicitly **not** a scientific result.
5. **Rental only after G1–G9 pass**, and with the H100/H200 choice justified by
   a measured A10 memory profile.

Claims to keep off the record throughout: any reproduction claim (the CoinRun
config, dataset, checkpoint, collector and metrics are unpublished, and the
LICENSE forbids reuse); any generalization claim until G4 holds; and any
"it works" claim from an MP4 rendered off the training batch.
