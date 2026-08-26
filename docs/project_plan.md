# Project Plan: Sim-to-Real World Models vs. Diffusion Policy on SO-101

Last updated: 2026-08-26

## Research Question

Given equal sim-pretraining and a small real-world data budget, does a
world-model-based planner (TD-MPC2-style) transfer to the real SO-101 more
sample-efficiently and more robustly than a diffusion policy — and does the
world model's "imagine-then-act" replanning make it more resilient to
disturbances than the diffusion policy's reflex-style execution?

## Task

Pick up a single cube and place it in a marked zone. Deliberately kept to one
object / one task so the comparison between the two approaches stays clean.

## Hardware / Environment

- Real robot: SO-101 leader + follower pair (physical, available to the user)
- Simulator: NVIDIA Isaac Lab (GPU-parallelized; chosen over Genesis /
  ManiSkill3 / MuJoCo MJX for domain-randomization tooling and fit with
  available NVIDIA GPU)
- Compute: Ubuntu machine (keerthan@100.71.12.16, via Tailscale), NVIDIA GPU,
  robots physically connected there
- Data format: LeRobot dataset format, for compatibility with both the
  diffusion policy trainer and the world-model pipeline

## The Two Models Under Comparison

1. **World model + planner** — TD-MPC2-style latent dynamics model trained on
   sim rollouts; action selection via online planning (CEM/MPPI); fine-tuned
   on a small real-world dataset. Model-based: explicitly predicts
   consequences of candidate actions before committing.
2. **Diffusion policy** — LeRobot's existing implementation; trained the same
   way (sim rollouts + small real fine-tune); no explicit dynamics model,
   direct observation → action-chunk generation via denoising.

## Scope

Focused: one task, one object, no fixed calendar timeline. Not attempting a
multi-task benchmark study. A diffusion-based world model (DIAMOND/Genie-style)
is a possible stretch goal, not part of the core comparison (see
decisions.md for reasoning).

## Phases

1. **Sim environment** — Import/build SO-101 asset in Isaac Lab, construct the
   pick-and-place scene, add domain randomization (lighting, cube
   texture/color, cube start pose, friction/mass variation).
2. **Real data collection** — Small teleop dataset (~20-50 episodes) via the
   leader/follower pair, logged in LeRobot dataset format.
3. **Train both models** — On sim rollouts, then fine-tune both on the same
   small real dataset (equal budget for fair comparison).
4. **Evaluation**:
   - Zero-shot sim→real success rate (both models)
   - Success rate vs. number of real fine-tune episodes (sample-efficiency
     curve — the headline result)
   - Behavior under mid-episode disturbance (nudge the cube after the policy
     commits) — tests replanning (world model) vs. fixed-chunk execution
     (diffusion policy)
5. **Write-up** — Results + mechanistic explanation, with rollout videos from
   sim and real.

## Immediate Next Step

Get an SO-101 asset into Isaac Lab and a minimal scripted pick-and-place scene
running — this unblocks everything downstream (data generation, both models).

## Status

Not yet started on implementation. Planning phase complete as of 2026-08-26.
