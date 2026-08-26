# Progress Log

### 2026-08-26

- Brainstormed project direction: world models + SO-101 arm.
- Landed on research question: sim-to-real transfer comparison between a
  TD-MPC2-style world model (with online planning) and a diffusion policy
  (LeRobot's implementation), on a single pick-and-place task.
- Confirmed simulator choice: NVIDIA Isaac Lab.
- Confirmed scope: focused (one task/object), no fixed timeline.
- Set up project structure and documentation:
  - `docs/project_plan.md`
  - `docs/preferences.md`
  - `docs/questions_log.md`
  - `docs/decisions.md`
  - `docs/progress.md` (this file)
- Set up remote project root at `~/SO-101-WM` on the Ubuntu machine
  (`keerthan@100.71.12.16`), confirmed passwordless SSH access working.
- Local mirror established at `c:\Keerthan\Projects\SO-101-WM`.

**Status**: Planning complete. Next step is building the SO-101 Isaac Lab
scene (asset import + minimal scripted pick-and-place task).

### 2026-08-26 (continued) — Environment setup

- Surveyed Ubuntu machine: Ubuntu 24.04.4, no NVIDIA driver installed
  despite GPU being present (RTX 5060 Ti, 16GB VRAM — meets Isaac Sim's
  recommended spec). No conda, no docker, no ttyUSB/ACM devices (arms not
  yet connected).
- Installed NVIDIA driver `nvidia-driver-595-open` (>= 580.65.06 required by
  Isaac Sim), build tools (cmake, build-essential, gcc-11/g++-11 for Isaac
  Lab's build step), added user to `dialout` group for future serial access
  to the arms.
- Installed Miniforge (conda) since nothing managed Python environments
  before.
- Rebooted; confirmed driver working (`nvidia-smi` shows RTX 5060 Ti,
  driver 595.84, CUDA 13.2) and Tailscale auto-reconnected as expected.
- Created two conda envs: `env_isaaclab` (Python 3.11, for Isaac
  Sim/Lab) and `lerobot` (Python 3.11, for LeRobot/real-arm control) — kept
  separate to avoid dependency clashes between Isaac Sim's pinned
  torch/numpy and LeRobot's requirements.
- Started `pip install "isaacsim[all,extscache]==5.1.0"` in `env_isaaclab`
  (large download, running in background).
- Set up GitHub repo (`KeerthanKrish/SO-101_World_Model_Comp`): generated
  SSH key on the Ubuntu machine, added it to GitHub, initialized git in
  `~/SO-101-WM`, pushed initial commit with `docs/` and `.gitignore`
  (IsaacLab/ and lerobot/ clones excluded from version control — external
  dependencies, not part of this project's own history).

**Status**: Isaac Sim install in progress. Next: verify Isaac Sim launches,
install Isaac Lab from source, install LeRobot in its own env, then set up
physical arm connectivity once arms are plugged in.
