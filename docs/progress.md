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

### 2026-08-26 (continued) — Isaac Sim, Isaac Lab, LeRobot installed

- Isaac Sim 5.1.0 installed via pip in `env_isaaclab`. Bundled torch (cu126)
  didn't support this GPU's compute capability (RTX 5060 Ti is Blackwell,
  sm_120) — reinstalled torch 2.7.0 with the cu128 build; verified with an
  actual GPU matmul (no more sm_120 compatibility warning).
- Cloned Isaac Lab (`~/SO-101-WM/IsaacLab`) and ran `./isaaclab.sh --install`
  using gcc-11/g++-11 (Ubuntu 24.04 ships gcc 13 by default, which Isaac
  Lab's build doesn't yet support). Isaac Sim's first run needed
  `OMNI_KIT_ACCEPT_EULA=YES` since the interactive EULA prompt fails over a
  non-interactive SSH session — noting explicitly that this accepts the
  Omniverse EULA.
- LeRobot: initial `pip install -e ".[feetech]"` failed — LeRobot now
  requires Python >=3.12 (docs/earlier research said 3.11, which was
  outdated). Recreated the `lerobot` conda env with Python 3.12; installed
  cleanly, `lerobot-find-port` / `lerobot-calibrate` CLIs confirmed present.
- Found the official SO-101 URDF source: `TheRobotStudio/SO-ARM100` repo,
  `Simulation/SO101/` folder (URDF + meshes + MuJoCo XML). Cloned into
  `~/SO-101-WM/assets/SO-ARM100`. Confirmed joint structure in the URDF:
  `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`,
  `gripper` (6 revolute joints, matches the real SO-101's 6 servos).

**Status**: Verifying Isaac Lab runs headless (`create_empty.py` tutorial).
Next: convert the SO-101 URDF to USD via Isaac Lab's `convert_urdf.py`,
write an articulation config (joint stiffness/damping/limits, modeled on
Isaac Lab's existing manipulator configs), then build a minimal scene
(ground plane, table, cube, SO-101) and a scripted (non-learned) reach/grasp
motion to sanity-check the physics before any learning goes in.

### 2026-08-26 (continued) — Isaac Lab verified working; SO-101 spawns and simulates

- Confirmed Isaac Lab actually runs (not just installs): tutorial scripts
  in this project loop forever by design (meant for GUI viewing), so
  "hangs" during verification were expected behavior plus two of my own
  mistakes (stdout block-buffering hid log output until I set
  `PYTHONUNBUFFERED=1`; `timeout` failed to kill nested Kit subprocesses,
  leaving orphaned GPU-using processes that needed `pkill -9 -f <script>`
  cleanup). Worth remembering for any future bounded test run.
- Converted the SO-101 URDF to USD via
  `IsaacLab/scripts/tools/convert_urdf.py` (`--fix-base`,
  `--joint-target-type position`). Output at
  `~/SO-101-WM/assets/usd/so101/so101.usd`. Only warnings in the log (about
  `gripper_frame_link` having no visual mesh of its own — harmless), no
  errors.
- Wrote `sim/robots/so101.py`: an Isaac Lab `ArticulationCfg` for the
  SO-101, modeled on Isaac Lab's Franka config. Joint torque/velocity limits
  are estimated from STS3215 datasheet figures (~1.9 N*m peak torque, ~4.7
  rad/s no-load speed) since the raw URDF's placeholder limits
  (effort=10, velocity=10 for every joint) don't reflect the real servo —
  flagged in the code as estimates to validate against the real arm later.
  See docs/so101_asset_notes.md for the full reasoning.
- Wrote `sim/scripts/test_so101_spawn.py`, a bounded (300-step, not
  infinite) sanity-check script. Result: robot spawns correctly, joint
  names match the expected 6 DOF, physics is stable (max joint velocity
  0.17 rad/s while holding default pose against gravity, no NaNs).

**Status**: SO-101 confirmed working end-to-end in Isaac Lab (spawn +
stable physics). Next: build the actual pick-and-place scene (table, cube,
camera) and a scripted (non-learned) reach/grasp motion, before any domain
randomization or learning.

### 2026-08-27 — Pick-and-place scene built and rendered; driver downgrade required

- Wrote `sim/scenes/pickplace_scene.py` (`InteractiveSceneCfg`): ground
  plane, dome light, static table (top surface at z=0), SO-101 robot, a
  0.05kg cube 0.2m in front of the robot base (approximate reach guess, not
  yet validated), and a fixed-viewpoint camera for offscreen rendering.
- Wrote `sim/scripts/capture_scene_image.py`: builds the scene, settles
  physics for 60 steps, aims the camera via `set_world_poses_from_view`,
  and saves an RGB frame -- since there's no practical GPU-accelerated way
  to view Isaac Sim's live 3D GUI over X11 forwarding to MobaXterm (see
  docs/preferences.md), offscreen rendering is the way we'll visually
  check scenes going forward.
- Hit a real blocker: Isaac Sim 5.1.0 segfaulted on first camera-enabled
  run (`librtx.scenedb.plugin.so` crash in the RTX/Hydra render path).
  Root cause: a known incompatibility between Isaac Sim 5.1.0's RTX
  renderer and the NVIDIA 595.x driver branch on Blackwell GPUs (confirmed
  via multiple NVIDIA forum posts and isaac-sim/IsaacLab GitHub issues
  with the same crash signature). Isaac Sim's officially validated driver
  is the 580 branch. Downgraded: purged `nvidia-driver-595-open`,
  installed `nvidia-driver-580-open` (580.173.02), rebooted. Camera
  rendering worked immediately afterward -- confirms this was the actual
  root cause, not a scene/config bug.
- Also hit (again) the Kit shutdown-hang issue: `capture_scene_image.py`
  finished and saved its output but the process didn't exit on its own;
  had to kill it by exact PID (using `pkill -f <script>` risks matching
  its own invocation's command-line text over SSH and killing the SSH
  session itself -- happened twice this session; exact-PID `kill` is
  safer).
- Result: successfully rendered and pulled back an image confirming the
  SO-101 spawns correctly on the table next to the cube, correct
  gripper/link geometry, no clipping or unstable physics.

### 2026-08-27 (continued) — WebRTC live streaming attempted and abandoned

- Tried to set up live remote viewing of Isaac Sim via its WebRTC
  Streaming Client (user connected from the Windows laptop, server on the
  Ubuntu machine at `100.71.12.16:8011` over Tailscale). Three attempts,
  three different failure modes -- see docs/decisions.md for details.
  Matches known, widely-reported unreliability of Isaac Sim's WebRTC
  streaming outside NVIDIA's own cloud infrastructure.
- Decision: stick with offscreen image capture (already working, see the
  2026-08-27 entry above) for visual checks. NoMachine remote desktop was
  identified as the reliable alternative if live interactive viewing
  becomes important later, but declined for now.
- Wrote `sim/scripts/stream_scene.py` during this attempt (loads the actual
  pick-and-place scene with streaming flags, rather than an empty
  instance) -- kept in the repo since it's still a valid script if
  streaming is revisited later, just unused for now.

**Status**: Pick-and-place scene confirmed visually correct. Next: scripted
(non-learned) reach/grasp/place motion -- likely via Isaac Lab's
differential IK controller (`scripts/tutorials/05_controllers/run_diff_ik.py`)
to command Cartesian targets rather than guessing joint angles -- as the
sanity check before any domain randomization or learning goes in. Also
worth validating the cube's placement distance against the real arm's
actual reach once the physical arm is connected.
