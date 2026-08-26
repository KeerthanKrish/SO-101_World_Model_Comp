# SO-101 World Model Comparison

Research project comparing a world-model-based planner against a diffusion
policy for sim-to-real manipulation on the SO-101 robot arm.

## Research Question

Given equal sim-pretraining and a small real-world data budget, does a
world-model-based planner (TD-MPC2-style) transfer to the real SO-101 more
sample-efficiently and more robustly than a diffusion policy — and does the
world model's "imagine-then-act" replanning make it more resilient to
disturbances than the diffusion policy's reflex-style execution?

Task: pick up a single cube and place it in a marked zone, trained in
simulation (NVIDIA Isaac Lab) and evaluated on the real SO-101 arm.

## Repo Structure

- `docs/` — project plan, decisions, progress log, and preferences
- `IsaacLab/`, `lerobot/` — external dependencies (not tracked here, see
  `.gitignore`); cloned locally during setup

## Status

See [docs/progress.md](docs/progress.md) for the current state and
[docs/project_plan.md](docs/project_plan.md) for the full plan.
