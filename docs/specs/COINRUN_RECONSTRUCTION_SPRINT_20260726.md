# OpenDreamer CoinRun Reconstruction Sprint

Date: 2026-07-26

Upstream: `next-state/open-dreamer`

Pinned upstream commit: `797e41f052b5996740938fd2fe8161f1866de3a2`

Experiment branch: `experiment/coinrun-reconstruction-20260726`

## Objective

Within four hours, establish whether the released OpenDreamer implementation
can train an action-conditioned pixel CoinRun tokenizer and dynamics model
end-to-end. Produce enough telemetry and generated output to distinguish a
working reconstruction from a pipeline that merely lowers training loss.

This is not an exact reproduction claim. The original CoinRun model config,
checkpoint, dataset manifest, RL collector, and behavior-cloning code were not
released.

## Required Deliverables

1. A deterministic, bounded CoinRun dataset generator that records RGB frames,
   actions, rewards, episode boundaries, seeds, and collector identity.
2. Generic action handling in the dynamics trainer. CoinRun action dimensions
   must not be rejected by Minecraft/VPT-specific assertions or silently
   remapped.
3. Dedicated small CoinRun tokenizer and dynamics configs. Do not mutate the
   Minecraft defaults to make the smoke pass.
4. A bounded preflight command that runs:
   - dataset generation and decoding,
   - tokenizer initialization and at least one optimizer step,
   - checkpoint save/restore,
   - dynamics initialization and at least one optimizer step,
   - a short action-conditioned rollout.
5. Telemetry containing:
   - git commit and resolved config,
   - parameter counts,
   - dataset records, frames, reward prevalence, and action histogram,
   - JAX compile time,
   - steady-state step time and examples/frames per second,
   - peak and steady GPU memory,
   - losses and gradient norm,
   - checkpoint and generated MP4 paths.
6. Tests for action shape/shift semantics, CoinRun config composition, data
   round-trip, and bounded smoke behavior.

## First Local Profiles

These are engineering profiles, not scientific endpoints.

### Tokenizer smoke

- CoinRun RGB: 64x64x3
- sequence length: 16
- batch: 2-8, selected by measured A10 memory
- encoder depth: 2-4
- encoder width: 128-256
- decoder depth: 2-4
- decoder width: 128-256
- latent count: 32-64
- bottleneck width: 8-16
- steps: 10 preflight, then 100-500 bounded pilot

### Dynamics smoke

- sequence length: 16-32
- batch: 2-8, selected by measured A10 memory
- depth: 2-6
- width: 128-384
- context: full smoke sequence
- register tokens: 4-16
- shortcut/bootstrap disabled for the first 100 steps
- steps: 10 preflight, then 100-500 bounded pilot

## Dataset Arms

The smoke may use random actions to validate plumbing. The first interpretable
comparison must contain two declared collectors:

- `random`: uniform legal Procgen actions.
- `scripted`: a deterministic or seeded run-right/jump policy that reaches
  later-level and reward-adjacent states substantially more often.

Window sampling must report both uniform sampling and the released
`p_include_reward=0.5` behavior. Collector identity and reward bias may not be
confounded without reporting the joint distribution.

## PASS Gates

The local preflight passes only if all of the following are true:

1. A fresh environment can install dependencies without manually editing the
   environment after failure.
2. CoinRun records round-trip with exact action, reward, seed, and frame count.
3. Tokenizer and dynamics each complete real forward, backward, optimizer, and
   checkpoint-restore steps on the A10.
4. Dynamics output changes under a legal action permutation on action-sensitive
   states. Shape compatibility alone is not evidence of conditioning.
5. A generated rollout MP4 is nonblank and has the declared context/horizon.
6. Telemetry is written before any run longer than 20 minutes is launched.

## Rental Gate

Do not start a paid pod until the local preflight passes. Once it passes:

- prefer one H200 141 GB for memory headroom, or one H100 SXM 80 GB when the
  measured profile fits comfortably;
- use a four-hour hard termination guard;
- checkpoint at least every 15 minutes;
- stream logs and telemetry back to the local machine;
- never leave a paid pod running without an active process and watchdog.

## Scientific Readout

At minimum compare:

1. random versus scripted collector;
2. uniform versus reward-biased windows;
3. one-step validation loss versus open-loop rollout degradation;
4. true actions versus episode-level action permutation;
5. pixel dynamics versus the existing transparent-state model at matched
   trajectory splits and horizons where possible.

A clean negative is a valid deliverable. Lower loss without action sensitivity
or usable open-loop rollouts is a failure, not a partial success.

## Non-goals

- Reproducing the unpublished CoinRun checkpoint exactly.
- Training the released 1.6B Minecraft configuration.
- Adding a Dreamer actor-critic or imagination-training loop in this sprint.
- Claiming pixel-policy performance from qualitative rollouts.
