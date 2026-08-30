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

### 2026-08-27 (continued) — First scripted grasp attempt (didn't grasp yet)

- Wrote `sim/scripts/run_pickplace_demo.py`: scripted (non-learned)
  reach/grasp/lift/place/release motion using Isaac Lab's
  `DifferentialIKController` (position-only command mode) to command
  Cartesian end-effector targets for the 5 arm joints, with the gripper
  joint commanded directly and separately. Waypoint positions and
  gripper open/closed joint values are initial guesses, explicitly flagged
  in the code as unverified.
- Wrote `sim/scripts/stitch_video.sh` (ffmpeg wrapper) to turn saved frame
  sequences into mp4s.
- Ran the demo: completed all 8 waypoints, 128 frames saved, stitched to
  video and pulled to the local laptop mirror at
  `sim/output/pickplace_demo.mp4`. Result: robot moved through the full
  sequence quickly (expected -- no motion smoothing/speed limiting
  implemented yet), but the cube's final position was unchanged from its
  start -- the grasp did not succeed. Likely candidates: wrong
  grasp-height waypoint, wrong gripper open/closed joint direction, or
  both -- needs iterating against the video.
- Realized (prompted by a good question from the user) that the current
  scene/camera/motor setup was built for a first physics sanity check, not
  for sim-to-real consistency. Created docs/sim_to_real_checklist.md to
  track what still needs deliberate matching (camera extrinsics/intrinsics,
  control interface/frequency, torque/PD gains) versus what's left to
  domain randomization (the plan's actual strategy for the sim-to-real
  gap, not full parameter matching).

**Status**: Pick-and-place scene confirmed visually correct. Scripted grasp
motion runs end-to-end but doesn't yet successfully grasp the cube -- next
step is tuning waypoints/gripper direction against the video. Also tracking
sim-to-real consistency gaps for later (see checklist doc), not blocking
current sim-only iteration. Worth validating the cube's placement distance
against the real arm's actual reach once the physical arm is connected.

### 2026-08-27/28 -- Extended grasp-tuning session: real progress, still not grasping

Long iterative session narrowing down why the scripted grasp wasn't working.
Summary of what was tried, in order:

1. **Jaw-offset bug found and fixed**: the tracked IK frame
   (`gripper_frame_link`) is not at the fingertips -- it's ~8cm away from
   where the jaws actually meet (confirmed both analytically, from the
   URDF's fixed joint transforms, and empirically via `calibrate_grasp.py`
   -- the two measurements agreed to within a few mm). Fixed by computing
   a local-frame offset correction (`_JAW_OFFSET_LOCAL` in
   `run_pickplace_demo.py`) and aiming for the corrected target instead of
   the raw frame position. Real, working fix.
2. **First version of the fix was unstable**: recomputing the offset
   correction *every step* using the still-changing current orientation
   created a feedback loop -- during a big motion (large orientation
   swings), the estimated correction swung wildly, observed as the arm
   lurching toward/through the floor. Fixed with a two-phase approach per
   waypoint: move roughly into place first (orientation settles), *then*
   compute the correction once from the settled pose and aim at that fixed
   target for the rest of the waypoint's steps. This resolved the
   floor-smashing entirely.
3. **Wrist camera**: abandoned per user request after ~8 failed mount
   configurations (see the "WebRTC...abandoned" entry above for the
   general pattern -- same story, different feature). Scene now has two
   additional *fixed* cameras instead (`side_camera`, `top_camera` in
   `pickplace_scene.py`) for multi-angle diagnosis without relying on an
   eye-in-hand view.
4. **Robot color**: tried a uniform blue `visual_material` override for
   debug visibility -- this replaced ALL materials including the
   originally-black motor housings, losing that contrast (user preferred
   the original look). Reverted entirely; getting "blue links, black
   motors" specifically would need per-mesh material assignment (targeting
   only link meshes, preserving motor sub-mesh materials), which needs
   real USD-scripting effort not yet justified. Robot is back to its
   original converted colors (yellow-ish links, black motors).
5. **Tried rotating the gripper 90 deg** (locking `wrist_roll` to a fixed
   absolute value -- learned along the way that this must be an absolute
   override, not an incremental add-on-top-of-IK's-output-every-step,
   since the latter compounds indefinitely). Visually confirmed the
   rotation took effect (gripper's opening plane visibly changed), but
   didn't fix the grasp -- the cube got shoved to a completely different
   location (near the elbow/wrist), revealing a *different* problem: during
   the uncorrected "phase 1" of each waypoint, the arm's body/wrist can
   sweep low across the table at an uncontrolled angle and clip the cube
   before the gripper even gets there, since position-only IK never
   constrains orientation.
6. **Tried full pose control** (position + a fixed "gripper points
   straight down" target orientation, derived analytically as the
   shortest-arc rotation sending the local jaw direction onto world -Z) to
   fix the uncontrolled-approach-angle problem properly instead of
   patching wrist_roll. This made things *worse*: the target orientation
   turned out to be unreachable for this 5-DOF arm at this position, and
   the damped-least-squares solver converged to a completely collapsed
   configuration near the robot's base instead of reaching at all.
   **Reverted** to the position-only two-phase approach (item 2 above) per
   user decision -- it's the best-working version so far, even though the
   grasp still isn't succeeding.
7. Found and killed an orphaned process from `probe_gripper_geometry.py`
   that had been silently running (and holding ~866MB of GPU memory) for
   over 11 hours, undetected until an explicit `nvidia-smi` check --
   process-name-specific checks (`ps aux | grep <script>`) can miss
   orphans from earlier, differently-named scripts. Worth periodically
   checking `nvidia-smi` broadly, not just for the specific process just
   launched.

**Current state**: position-only IK with the two-phase jaw-offset
correction is the working baseline. The arm reliably reaches the correct
neighborhood of the cube (no more floor-smashing, no more wild misses) and
nudges the cube by a few cm on contact, but does not yet close around it
and lift it. The remaining gap is almost certainly the uncontrolled
approach angle during the coarse (phase 1) part of each waypoint --
solving this without breaking anything else (as full pose control did) is
the open problem.

**Status**: Grasp still not succeeding. Position-only + two-phase offset
correction is the stable baseline to keep iterating from. Wrist camera and
per-mesh robot recoloring are explicitly parked, not abandoned.

### 2026-08-29 -- Real arm connected, teleop working

Machine now has a monitor, keyboard, and the leader arm physically
attached (follower arm not yet connected). Full details in
docs/real_arm_setup.md; summary here.

- Reconnected on a new LAN IP (`10.0.0.240`, DHCP-assigned, changes on
  reboot -- machine previously briefly power-cycled by accident, see
  below).
- Confirmed a real GUI window (not just headless offscreen rendering) can
  be opened on the physical monitor from an SSH session, via
  `DISPLAY=:1 XAUTHORITY=/run/user/1001/gdm/Xauthority`. Verified with a
  screenshot (`ffmpeg -f x11grab`) before asking the user to confirm on
  their own screen.
- Leader arm calibrated (`lerobot-calibrate`, id `leader1`) -- needed
  external power (not just USB) to be detected at all; calibration itself
  is interactive and needs a real TTY, doesn't work through a
  non-interactive SSH command.
- **Discovered LeRobot and Isaac Sim can't share a Python process**:
  LeRobot's source uses Python 3.12+ syntax, Isaac Sim requires 3.11.
  Solved with a two-process file-based bridge: `leader_reader.py` (in the
  `lerobot` env) writes live leader-arm state to a shared JSON file;
  `teleop_bridge.py` (in `env_isaaclab`) polls it and drives the simulated
  robot's joints. **Confirmed working** by the user -- moving the real
  leader arm visibly drives the simulated arm in real time.
- User-reported issues from the first teleop session, both fixed:
  - Gripper visually clipping into the cube instead of colliding -- root
    cause was `solver_velocity_iteration_count=0`, which every prior run's
    log had actually warned about (PhysX explicitly recommending 1-2), just
    never addressed. Bumped to 2 for both the robot and the cube.
  - Noticeable lag, including in native mouse-driven viewport navigation
    (not just leader-arm responsiveness) -- root cause was 3-4 `CameraCfg`
    sensors rendering every frame regardless of whether their output was
    read. Split the scene into `PickPlaceSceneBaseCfg` (physical scene
    only, no cameras -- now used for teleop) and `PickPlaceSceneCfg`
    (adds cameras back, for headless scripted/recording runs).
- Accidentally shut down the user's own Windows laptop instead of just the
  Ubuntu machine earlier in this session (misread "shutdown my computer"
  as referring to the laptop) -- caught and aborted (`shutdown /a`) before
  it triggered, per the user's immediate correction.

**Status**: Teleoperation confirmed working end-to-end (real arm -> live
sim). Fixes for clipping/lag just applied, not yet re-confirmed by the
user. Next: have the user teleoperate an actual successful pick-and-place,
recorded via teleop_bridge.py's JSON output, then use that recording to
(a) extract real grasp parameters for the scripted demo and (b) serve as
demonstration data for the model-training side of the project.

### 2026-08-29 (continued) -- Researched grasping physics issues, applied real fixes

After the lag fix, user reported the gripper still clipping into the cube,
the cube "escaping" (bouncing out) when the gripper closed fully on it,
and general slipperiness -- and asked me to specifically research how
others handle this for the SO-101 / robot arms in general before just
guessing more fixes.

**Research findings** (Isaac Lab GitHub discussions/forums, community
SO-ARM100/101 Isaac Lab projects):
- An IsaacLab GitHub discussion with the literal title "Why my gripper can
  not close when I want to grasp the cube?" diagnosed and fixed the exact
  same symptom: the gripper's USD collision used the default `convex_hull`
  approximation, which **cannot represent concave geometry at all** -- a
  convex hull "fills in" any concave notch, so the gripper's pincer mouth
  had no actual gap in its collision shape. Confirmed fix: switch to
  `convex_decomposition` for the gripper meshes.
- Community tips for grasping-task stability: higher solver position
  iteration counts (10 vs. a default of 4) reduce penetration; explicit
  higher friction on both contacting surfaces addresses slipperiness;
  contacts are sensitive to timestep size.
- Found (but didn't need) additional options for later: SDF collision
  approximation (even more accurate than convex decomposition, higher
  cost), and PhysX "compliant contact" materials (a spring-based soft
  contact model) for finer control if convex decomposition alone isn't
  enough.

**Fixes applied**, in order of impact:
1. **Re-converted the URDF to USD with `collider_type="convex_decomposition"`**
   instead of the default `convex_hull` (`sim/scripts/reconvert_urdf_convex_decomp.py`
   -- convert_urdf.py's CLI doesn't expose this option, had to call
   `UrdfConverter`/`UrdfConverterCfg` directly). This is the fix that
   actually addresses the root cause per the research above.
2. Bumped `solver_position_iteration_count` from 8 to 12 (robot and cube).
3. Cube `physics_material`: friction 0.5->1.2 (static and dynamic) with
   `friction_combine_mode="max"` -- ensures effective contact friction is
   reliably high regardless of the gripper's own (default) material,
   since PhysX uses the higher-priority combine mode's value. (Tried
   adding a matching physics_material to the robot too, but hit a real
   API gap: `physics_material` isn't a field on the base `UsdFileCfg`,
   only on `UsdFileWithCompliantContactCfg`, which needs an explicit
   per-prim path rather than a whole-asset override -- reverted that part,
   not worth the extra complexity given the cube-side fix alone should be
   sufficient.)
4. Softer gripper actuator gains (stiffness 50->15, damping 2->1, gripper
   joint only) -- a stiff PD gain fighting to reach "fully closed" against
   an object it physically can't close past was building up the
   interpenetration that caused the pop-out; softer gains let it yield
   more like a torque-limited real servo.
5. Lowered `max_depenetration_velocity` 5.0->1.0 (robot and cube) -- caps
   how violently PhysX's depenetration correction can eject an object,
   converting a "launch" into a gentler separation.

**Result** (user-confirmed): successfully picked up the cube for the first
time. Still some residual clipping and occasional escaping, but
noticeably less slippery. Real, measurable progress, not fully solved.

**Also**: moved the cube from 0.2m to 0.28m in front of the robot base --
user found 0.2m uncomfortably close (awkward reach angle) once actually
teleoperating the real arm. This is the first cube-placement number
informed by actual real-arm feedback rather than a guess.

**Status**: Grasping via teleop now works but isn't fully solid --
residual clipping/escaping remain, improved not eliminated. Cube starting
position corrected based on real teleop feedback. Next: keep iterating on
remaining contact issues if the user wants to continue, or move on to
using recorded teleop trajectories for parameter extraction / demo data.
