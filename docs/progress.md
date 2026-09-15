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

### 2026-08-29 (continued) -- Further perf tuning, first successful pick, and a real data-loss bug

- Added `sim.step(render=(step % 2 == 0))` to teleop_bridge.py -- renders
  every other physics step instead of every step (physics still runs at
  full 100Hz), roughly halving GPU render cost for a further
  responsiveness improvement the user asked for.
- User's cube got trapped in a small gap near the fixed jaw housing (not
  between the pincer tips at all) after switching to convex decomposition
  -- a known tradeoff: decomposing a mesh into multiple convex pieces to
  fix the "can't represent concave shapes" problem can introduce tiny
  gaps *between* those pieces that a small object gets caught in. No
  targeted fix yet; a session restart (which respawns the cube) clears it
  immediately when it happens.
- **First successful teleop pick-and-place**, confirmed by checking the
  recording mid-session (cube height reached ~0.19m, well above the
  ~0.015m resting height). Did several more reps afterward (some
  successful, some "jiggling" without a real lift, per the user).
- **Real bug: recording data loss.** Stopped the session with SIGINT to
  trigger a clean final save via the script's `finally` block. The
  recording file (a single JSON array, rewritten in full every 2 seconds)
  ended up truncated/corrupted -- the interrupt landed mid-`json.dump()`
  write, likely during a periodic autosave, and left a half-written file.
  Recovered what could be salvaged by finding the last complete JSON
  object and closing the array there -- recovered 16,162 of what had been
  a longer recording, but the recovered portion turned out to be only the
  early, mostly-idle part of the session (max cube height ~0.025m
  throughout) -- **the actual successful pick-and-place reps were in the
  lost tail and are gone.** Saved the partial recovery as
  `teleop_recording_partial_2026-08-29.json` for reference, not because
  it's useful demonstration data.
- **Root cause fixed**: added `atomic_save_json()` to teleop_bridge.py --
  writes to a `.tmp` file and `os.replace()`s it into place, so a
  mid-write interrupt can now only ever leave the *previous* complete
  save intact, never a truncated one. Used for both the periodic autosave
  and the final `finally`-block save.
- Wrote `sim/scripts/segment_teleop_episodes.py`: segments a continuous
  teleop recording into individual pick-and-place episodes by detecting
  spans where the cube's height rises above a threshold and back down,
  discarding spans that never rose high enough to count as a genuine
  lift (vs. jiggling the cube on the table without lifting it). Plain
  Python, no Isaac Lab/torch dependency. Ran on the recovered (jiggle-only)
  data as a smoke test -- correctly found 0 genuine episodes, matching
  the data's actual content.

**Status**: Teleop pick-and-place demonstrated as achievable (once), but
no usable recorded demonstration data survived from today's session due to
the now-fixed save-corruption bug. Next: redo a batch of demonstrations
with the fixed (atomic-write) teleop_bridge.py, then actually run
segment_teleop_episodes.py on real successful data.

### 2026-08-29 (continued) -- First real demonstration batch collected

Redid the teleop session with the atomic-write fix in place. Stopped
cleanly via SIGINT -- recording saved correctly this time (29,748 steps,
valid JSON, no corruption). Ran `segment_teleop_episodes.py` on it:

- 10 candidate lifted spans detected
- 2 correctly discarded as jiggling (peak height 0.068m / 0.076m, below
  the 0.08m real-pick threshold) -- matches the user's own description of
  some messy reps that didn't actually lift the cube
- **8 genuine pick episodes** kept (peak heights 0.082m-0.149m), saved to
  `sim/output/teleop_episodes/episode_000.json` through `episode_007.json`

First real, usable batch of demonstration data for this project -- not
huge, but a genuine starting point for both scripted-parameter extraction
and (eventually) model training data, per the plan from earlier.

**Status**: 8 genuine demonstration episodes on disk (gitignored,
`sim/output/`, not committed to git -- data artifacts, not code). Did not
start a new Isaac Sim session per user's explicit request this time. Next:
either collect more episodes in a future session to grow the batch, or
start using these 8 for parameter extraction / training data prep.

### 2026-08-30 (continued) -- Wrist camera round-3 correction

User sent additional reference photos and pointed out the round-2 "working"
wrist camera didn't actually match: it aimed at the moving jaw's pivot
point (a hinge near the base) instead of down the fingers toward the grasp
point. Also clarified the camera does NOT move with the gripper's
open/close motion -- it's solidly mounted on the static housing
("gripper_link"), confirming that attachment point was right, just the aim
direction was wrong.

Fix: re-aimed at "gripper_frame_link" (Isaac Lab's own IK end-effector
frame, via "gripper_frame_joint"'s URDF origin) instead of the "gripper"
joint's origin -- that's almost exactly local -Z from "gripper_link",
i.e. actually down the fingers. Found the remaining "which lateral axis is
up" offset empirically (test_wrist_camera_angles5.py,
test_wrist_camera_angles6.py) -- "-Y" at a 7cm standoff reproduced the
reference photo's framing (fingertips converging at the bottom of frame,
tabletop filling the rest). Updated "pickplace_scene.py"'s "wrist_camera"
and docs/real_camera_setup.md with full details and the corrected lesson
(check what a URDF child frame actually represents, not just that one
exists).

**Status**: Wrist camera now matches the reference photo's framing.
Rendered confirmation at
`sim/output/wrist_cam_test6/negy_7cm.png` (same pos/rot now baked into
`pickplace_scene.py`).

### 2026-08-30 (continued) -- Camera FOV, real hardware queried directly, top-down camera fixed

User connected both real cameras (wrist "Web Camera" + a Logitech C922
for top-down) to the Ubuntu box, live. Queried both directly via
`v4l2-ctl`/`ffmpeg` instead of guessing specs:

- Wrist camera: native 1280x720. Top-down (C922): native up to 1920x1080.
  Both 16:9 -- sim cameras were mismatched (wrist 480x480 square,
  `top_camera` 960x720 4:3). Corrected both to 16:9.
- Wrist `focal_length` widened per user feedback after reviewing the
  first working render; user compared 6/8/10mm side by side and picked
  10.0 as the closer match to the real camera's actual FOV.
- `top_camera`'s position was wrong in a way only a live real-camera
  capture exposed: it assumed the robot sits at the *near edge* of the
  visible table (matching the user's real cardboard sheet, which only
  extends forward from the arm), but the sim table is centered ON the
  robot -- only 0-0.3m ahead of the base is usable. The camera was aimed
  mostly past that edge, showing empty ground for most of the frame.
  Recentered and re-tuned; now closely matches the real camera's
  proportions (tabletop fills most of the frame, arm near the top).
- Real wrist camera's raw output has a visible vignette + warm color
  cast, absent from sim. Decision: leave uncorrected -- domain
  randomization already varies sim color/lighting broadly, and the
  planned real-data fine-tuning phase trains directly on this camera's
  actual output anyway, which is a more direct fix than synthetically
  reproducing a lens artifact. See docs/sim_to_real_checklist.md.

Also noticed (unrelated, caught while investigating): Ubuntu's crash
reporter (`apport`) had generated a crash report for
`xdg-desktop-portal-gnome`, timestamped right when a script was
accidentally launched in GUI mode with no display attached (forgot
`--headless`) -- explains why the user was seeing intermittent "Ubuntu
has experienced an internal error" popups on the physical monitor mid-
session. Not a hardware/OS health issue, just fallout from that one
mistake; fixed by always passing `--headless` going forward.

**Status**: Both cameras now aspect-matched and position/FOV-tuned
against live real hardware, not just static reference photos. See
docs/real_camera_setup.md and docs/sim_to_real_checklist.md for full
details.

### 2026-08-30 (continued) -- Reward function implemented and validated

Implemented the reward function for the TD-MPC2 world-model side of the
comparison (the diffusion policy never needs one -- see
docs/reward_function.md for the full writeup, condensed here).

- Extracted `JAW_OFFSET_LOCAL` (the calibrated gripper-frame-to-fingertip
  offset) out of `run_pickplace_demo.py` into a new dependency-free
  module, `sim/robots/grasp_geometry.py` -- the demo script boots Isaac
  Sim as an import side effect, so the constant couldn't be safely
  imported from anywhere else before this.
- Wrote `sim/envs/pickplace_reward.py`: two-phase potential-based shaping
  (approach -> transport, switching on the grasp event, to avoid a naive
  height-reward fighting against ever placing the cube), tanh-bounded
  distance terms, a proximity+height+joint-angle grasp heuristic, a
  sparse success bonus, and a small action-energy penalty. Fully
  self-contained (stdlib only) so it's importable/testable without
  booting Isaac Sim. Reuses the scripted demo's existing place-zone
  location `(-0.15, 0.15, ...)` rather than inventing a second one.
- Validated two ways: (1) a synthetic self-test with hand-constructed
  states, all passing; (2) replaying real captured teleop demonstrations
  (`sim/scripts/validate_reward_function.py`) through the actual reward
  function using live simulated state + real forward-kinematics-derived
  gripper position. This caught two real bugs neither the self-test nor
  code review found: the validation script was resetting the cube/robot
  to scene defaults instead of the episode's own recorded starting state
  (these episodes are segments of one continuous session, so they don't
  start from a clean reset), and once fixed, a false-positive grasp
  detection (cube barely above rest height + coincidentally-closed
  gripper joint far away registering as "grasped") -- fixed by adding a
  `grasp_proximity_threshold` requirement, with a regression test added.
- Confirmed after fixes: approach-phase reward rises/falls sensibly
  across ~930 combined real simulated steps (two episodes), no false
  positives, no NaNs/crashes. Confirmed limitation, not swept under the
  rug: neither replayed episode ever reproduced an actual lift in
  open-loop replay, even the one with the strongest original recorded
  peak (0.149m) -- matches `replay_episode.py`'s own documented caveat
  that open-loop replay is sensitive to small differences and doesn't
  reliably reproduce a contact-dependent outcome even from matched
  starting conditions. So the grasp/transport/success-bonus half of the
  reward is validated only synthetically, not against real data --
  documented explicitly as an open item.

**Status**: Reward function implemented, self-tested, and empirically
validated against real data as far as open-loop replay allows. Not yet
wired into an actual Gym-style training environment (`reset()`/`step()`)
-- that's the next piece of infrastructure needed before RL training can
start. See docs/reward_function.md for the complete design rationale and
validation details.

### 2026-08-30 (continued) -- Top-down camera still not matching; evaluation plan written

User re-checked all four camera views after the FOV/aspect-ratio fixes
and felt the top-down camera still doesn't match the real C922's view
well. Diagnosis: the remaining gap isn't a camera-angle problem anymore,
it's a **table size mismatch** -- the sim table is a fixed 0.6m square
centered on the robot, while the user's real cardboard workspace is much
larger and only extends forward from the arm. No camera position can
fully compensate for that scale difference.

Recommended two options rather than physically repositioning the real
camera to chase the current (fairly arbitrary) sim table size: (1)
enlarge the sim table to better match the real workspace's proportions
-- the more durable, root-cause fix -- or (2) lean on camera-pose domain
randomization during training so the trained policy tolerates imperfect
real-camera alignment rather than needing pixel-perfect matching,
consistent with the project's existing "not aiming for a perfect
match" philosophy (sim_to_real_checklist.md). **Not yet implemented --
awaiting user decision on whether to resize the table.**

User then asked for a detailed look ahead at how the world model and
diffusion policy will actually be compared once both are trained -- this
being "the important part of the project." Wrote
docs/evaluation_plan.md: eight concrete comparison tests (sample
efficiency, novel start-position generalization, mid-episode perturbation
recovery, visual domain shift, distractor objects, zero-shot sim-to-real
transfer, inference latency, multi-modal demonstration handling), each
with why it differentiates model-based planning from reactive imitation,
plus a "data collection pre-planning checklist" section -- several of
these tests only work if specific decisions (train/held-out position
splits, reserved real-world eval positions, deliberately varied teleop
demonstration style) are made *before* bulk data collection, not after,
since some of these can only be fixed by recollecting data from scratch
if gotten wrong.

**Status**: Evaluation plan documented in detail, ahead of any actual
training work (none has started yet). Top-down camera fix pending user
decision (resize table vs. rely on randomization). Next: resolve the
table-size question, then proceed toward wiring the reward function into
an actual training env once ready.

### 2026-08-30 (continued) -- Table enlarged, top-down camera re-tuned

User chose to enlarge the sim surface rather than reposition the real
camera. Doubled the table from 0.6m to 1.2m in `pickplace_scene.py`
(`_TABLE_SIZE`), keeping it centered on the robot (simpler than moving
the robot to one edge -- everything else, cube/target positions, stays
valid without further changes, and the extra table behind the robot is
harmless since every camera only looks forward anyway).

Updated `pickplace_reward.py`'s `table_half_extent` (0.3 -> 0.6) to match,
and fixed one self-test assertion that would have silently become wrong
under the new bounds (a fallen-cube test position that was beyond the
old table's edge but not the new, larger one).

Re-tuned `top_camera`: midpoint of the new usable forward depth moved
from x=0.15 to x=0.3, and the standoff height doubled (0.85 -> 1.7) to
match -- a pinhole camera's visible extent scales linearly with height
for fixed focal length, so doubling both the table depth and camera
height should preserve the same framing that worked before. Verified by
re-rendering (not just assumed): the doubled config reproduced the same
good composition confirmed working previously (tabletop filling the full
width and most of the height, arm near the top).

**Status**: Table enlarged and top-down camera re-matched; all four
camera views re-rendered and confirmed working together on the new
table size.

### 2026-08-31 -- Correctness audit, TD-MPC2 wired in and trained, diffusion policy data strategy planned

Before wiring in either algorithm, went back through pickplace_reward.py,
grasp_geometry.py, and pickplace_env.py line by line rather than trusting
earlier validation was sufficient. Found and fixed a real bug: is_grasped()'s
height check also drove the reward's approach/transport phase switch, but
the place target's resting height (0.015m) is below the lift-detection
threshold (0.02m) by construction -- so lowering an already-held cube onto
the target silently fell back to approach-phase shaping right when
transport-phase precision mattered most. Fixed with a new, deliberately
stateful is_holding() function (the one explicit exception to the module's
otherwise stateless design) -- see docs/reward_function.md's correctness-
audit section for the full writeup, including a second flawed self-test
assertion caught in the process.

Integrated the official TD-MPC2 reference implementation (cloned to
tdmpc2/, never modified in place) against PickPlaceEnv. Found and fixed
five more real, previously-unexercised bugs along the way -- three in
TD-MPC2's own reference code (never actually run with a genuine multi-
modal observation by any of its shipped examples), one in pickplace_env.py
(extras dict never populated, so no caller could distinguish success from
failure), and a namespace collision between our envs/ package and theirs.
Full writeup in docs/tdmpc2_integration.md. Smoke test (300 steps, shrunk
config) confirmed the whole pipeline mechanically: env -> two-camera
fusion adapter -> agent.act (real MPC planning) -> replay buffer ->
agent.update, with real gradient updates producing finite, sensible losses.

Ran the first real training run: 30,000 steps, full 500-step episodes,
default batch size, num_envs=1 (required -- TD-MPC2's reference trainer
expects a single non-batched env; sample efficiency from one env stream
is the whole point of the algorithm, not a limitation worked around).
Took 3935s (~66 min). 60 episodes collected, 27,501 real gradient updates,
no crash. Added periodic evaluation interleaved with training in the same
env instance (a second SimulationContext isn't possible in this process) --
every 5000 steps, one deterministic (eval_mode=True) episode, not added to
the buffer, with scene_camera frames captured into an mp4 for a visual
checkpoint. Eval rewards across the run: +14.6, +90.3, +12.6, +54.1,
+97.9 -- noisy (each is a single episode at a random cube position) but
trending upward overall. No successful placement yet in any eval episode
(success=False throughout) -- 30k steps with num_envs=1 shows real
learning signal but isn't enough to fully solve the task yet, consistent
with TD-MPC2's benchmark tasks typically needing substantially more
environment steps even for simpler continuous control.

While training ran, worked through what else needed settling before
diffusion policy's data-collection strategy. Found and fixed one real gap:
teleop_bridge.py never randomized the cube's position at all -- it sat at
one fixed spawn point for an entire session, position across "reps" only
ever determined by wherever a human happened to leave it. Fixed to
randomize within the same train region TD-MPC2 uses, re-randomizable
between reps via a simple trigger file. Researched how others solve the
"need bulk demonstration data but real demos are expensive" problem --
found MimicGen (NVIDIA, CoRL 2023), which generates large datasets from a
handful of human demonstrations via per-subtask SE(3) trajectory
transformation. User decided to implement a simplified version of that
core idea directly (not the full published framework, given how much
simpler our single-object task is) once they return to it -- full
discussion and decision in docs/diffusion_policy_data_strategy.md.

**Status**: Reward function and env now audited and trusted. TD-MPC2
pipeline proven correct end to end with one real training run completed
-- promising reward trend, task not yet solved, more training time would
be the natural next experiment. Diffusion policy blocked only on the
demonstration-data decision, execution deferred by the user; the strategy
itself (small human-demo seed + custom SE(3) augmentation) is decided and
documented.

### 2026-08-31 (continued) -- The "promising" run1 trend was reward hacking

Before starting a longer run on the strength of run1's rising eval trend,
the user asked a pointed question after actually watching the highest-
scoring (+97.9) episode's video: the arm never touches the cube, and
nearly clips into itself. This was the right catch -- I had called the
trend "promising" from the printed numbers alone, without watching the
video myself first, and the real explanation was reward hacking: the old
dense shaping term (`reach_weight * (1 - tanh(d / reach_scale))`) paid
its ABSOLUTE value every step based on current distance alone, so simply
hovering somewhere plausible-looking -- without tracking that episode's
actual cube -- could accumulate ~+0.2/step x 500 steps ~= +97.9 with zero
real engagement. Separately, `enabled_self_collisions` had been left at
its inherited `False` default in `so101.py`, letting the arm pass through
itself -- fixed to `True`.

First attempted reward fix: narrowed `reach_scale` 0.15 -> 0.08 (make
vague proximity worth less) and added direct checkpoint saving to
`train_tdmpc2_pickplace.py` (run1's policy couldn't be reloaded once
concerns surfaced). Launched a second, shorter (15,000-step) run
specifically to check the fix quickly before committing to a longer run.
**Result: worse, not better** -- eval rewards were consistently negative
and non-improving (-3.8, -18.7, -11.2, -33.2), and extracted video frames
(no video-viewer tool available, so frames pulled via ffmpeg and
inspected directly -- an approach used repeatedly from here on) showed
genuinely undirected motion, never engaging the cube in either sampled
episode. Diagnosis: narrowing the scale reduced the reward for vague
proximity but didn't fix the actual mechanism (a policy could still
profit from occupying any fixed position, just a smaller one), while also
removing most of the usable gradient for a policy that starts far from
the cube.

**The actual fix**: potential-based reward shaping (Ng, Harada & Russell,
ICML 1999) -- reward the CHANGE in a potential function between steps,
never its absolute value. Provably policy-invariant (never changes the
optimal policy, for any choice of potential), and practically: a policy
that holds still anywhere now earns exactly zero shaping reward, closing
the hacking mechanism at its root rather than by narrowing a scale.
`reach_scale` reverted to 0.15 (broad is fine again once the exploit that
motivated narrowing it no longer exists). Added a `touch_bonus`
(one-time, genuine contact range) and fixed a second, separately-found
incentive bug: the old `grasp_bonus` was a flat per-step reward while
holding, which -- since placing ends the episode -- made holding the cube
forever a better strategy than finishing. Made it one-time too. Full
technical writeup in docs/reward_function.md's "Reward-hacking finding
and potential-based-shaping redesign" section.

Launched a third training run (15,000 steps) under the new reward.
Reward numbers looked sane throughout (no exploit-scale spikes), and one
checkpoint's video showed the gripper genuinely reaching toward and
briefly hovering at the cube -- real progress, though only one data
point and possibly partly lucky given the cube's position was still
randomized.

**Status**: Reward redesigned around a proven-correct mechanism
(potential-based shaping) rather than a re-tuned scale. Self-collision
bug fixed. Next: confirm this generalizes with a longer run.

### 2026-09-01 -- Two more 50k-step runs, both static, then a mechanistic diagnosis

Ran a fourth (50,000-step) training run under the same potential-based
reward, per the user's own suggestion to see if more time alone resolved
run3's inconclusive result. It didn't -- **direct frame inspection**
across three checkpoints (sampled every 25 frames this time, after
under-sampling missed detail in run3's review) showed the arm holding
the *exact same* folded resting pose in every single frame, regardless
of checkpoint or the cube's randomized position. The small variance in
eval reward turned out to be almost entirely explained by the brief
settling transient at episode start, not real cube-tracking.

Diagnosed the likely mechanism: potential-based shaping correctly pays
*zero* net reward for holding still (that's the whole point of the fix),
which also means there's no reward pressure at all pushing an untrained
policy to move, unless its experience already contains a genuine
touch/grasp trajectory for the value function to learn from -- and
TD-MPC2's default exploration budget (5 random episodes before its own
policy takes over) apparently never produced one against a small,
randomly-positioned target. Raised the seed-episode budget 6x (5 -> 30
episodes) and ran a fifth 50,000-step run to test this directly.

**Result: identical failure mode.** Run5 reproduced run4's exact
fixed-idle-pose behavior. This ruled out "just needs more random
exploration" as the fix, and motivated actually tracing TD-MPC2's own
planning code (`_plan()` in tdmpc2.py) instead of continuing to scale
that one knob. Found a more precise mechanism: TD-MPC2's CEM planner
samples 512 candidate action sequences and narrows to 64 elites over 6
iterations every single env step, but CEM is known to over-confidently
narrow its own sampling std even when the value estimates it's ranking
by are pure noise (no learned signal yet) -- and the actual
training-time exploration noise (`a = a + std * randn(...)`, added only
when not in eval mode) uses exactly that potentially-falsely-converged
std. Also lowered `action_penalty_weight` 5x (0.01 -> 0.002): any
movement, even useful movement, was being penalized on top of zero
reward for standing still, making "don't move" a doubly-safe local
optimum. Added exact `touched`/`holding` logging to eval output
(`extras["touched"]`/`["holding"]` in pickplace_env.py), replacing
error-prone frame-by-frame visual guessing with a precise, objective
signal.

**Status**: Two consecutive 50k-step runs under the (correctly
non-hackable) potential-based reward showed zero directed behavior --
established this is a bootstrapping/exploration problem, not a reward-
correctness problem, and pinpointed the mechanism precisely enough to
design two targeted fixes (see next entry) rather than guessing further.

### 2026-09-02/03 -- Migrated to a new Mac; two curriculum levers produce the first reliable cube contact

User set up a new MacBook Pro and transferred the conversation over.
Local development now happens here instead of the Windows laptop --
practically, this meant re-establishing SSH access from scratch (the
Ubuntu box's LAN IP had changed again, per the DHCP-changes-on-reboot
note in docs/real_arm_setup.md; refound it via `~/.ssh/known_hosts` host-
key matching against previously-seen `10.0.0.x` addresses, confirmed by
matching SSH banner + host key fingerprint before trusting it) and
generating a new passwordless SSH key pair for this machine. **The
sync workflow itself also changed for the better**: code now flows
Ubuntu -> `git push` -> GitHub -> `git pull` on the Mac, replacing the
old scp-based mirroring for code entirely. scp is still used, but now
only for the things `.gitignore` deliberately excludes and always will
(`sim/output/` -- eval videos, checkpoints, recorded teleop episodes;
and `assets/`, the robot USD files) -- these get synced to the Mac
periodically, not on every change. See docs/preferences.md for the
updated arrangement.

With the CEM-collapse mechanism diagnosed, implemented and launched a
sixth training run with two targeted levers: `--cube-pos 0.15 0.0`
(fixing the cube to one point -- the train region's geometric center --
instead of randomizing it every episode, isolating "can this learn to
reach and grasp at all" from the harder "can it generalize across
positions" question) and `--min-std 0.5` (raising the floor under CEM's
training-time exploration noise from its default 0.05, so the planner
can't collapse into false confidence about a flat value landscape quite
so easily). Both were real experiments, not proven fixes, changed
together rather than one at a time given how much time two consecutive
identical-failure runs had already cost.

**Result: the first real behavioral change since the reward redesign.**
`touched=True` fired on every eval checkpoint from step 20,459 onward (6
in a row, through the end of the run) -- reliable, repeated cube contact,
not a static idle pose and not a single lucky-looking frame. `held`
never went true and no episode succeeded, but this was a genuine,
qualitative break from every prior run. Per standing instruction ("start
a 70k run once this one's done, don't wait for me to confirm"), launched
a seventh run automatically at 70,000 steps with identical settings the
moment run6 finished.

**A second false positive, caught the same way as the first.** Partway
through run7, an eval episode logged `held=True` -- watched the video
before trusting it (this project's standing practice, now paying off a
second time), and the gripper had closed fully BESIDE the cube, never
around it, cube untouched the whole episode. Root cause: `is_grasped()`/
`is_holding()` only ever checked a spherical distance from the jaw pivot
to the cube -- identical whether the cube was in front of the closing
jaws or off to one side of them. The user, watching the same run6/7
videos independently, separately flagged two things worth fixing before
another run: the cube felt positioned too close to the arm's base for a
clean grasp angle, and any gripper-closing reward needed to specifically
require the cube be *between* the jaws as they close, not just "closing
somewhere near the cube."

Fixed both, carefully:
- Added `is_between_jaws()` (`sim/envs/pickplace_reward.py`), decomposing
  the cube's position relative to the jaw pivot along the gripper's live
  reach direction (a new `jaw_approach_axis_world()` helper in
  `sim/robots/grasp_geometry.py`, reusing the already-calibrated
  `JAW_OFFSET_LOCAL` offset as a direction rather than needing new
  hardware calibration) into an axial (within-reach) and lateral
  (centered) component. A live measurement first ruled out the naive
  alternative -- the two jaw bodies' origins stay a constant ~3.6cm apart
  regardless of joint angle, since the joint rotates the moving jaw about
  a pivot rather than translating it, so raw origin-to-origin distance
  carries no information about openness at all. `is_grasped()` now
  requires this check to fire at all; `is_holding()` requires it only to
  *establish* holding, not to persist it (deliberately, to avoid a new
  failure mode where minor sway while genuinely carrying the cube could
  flicker a real hold back out of these intentionally tight thresholds).
- Added a `grasp_close_weight` potential-based shaping term for the
  specific act of closing the gripper, gated on `is_between_jaws()` --
  previously nothing rewarded closing at all, only gripper-to-cube
  distance. Gating on the geometric check rather than mere proximity was
  a specific, explicit user requirement: a policy that just snaps the
  gripper shut near-but-not-around the cube must earn nothing, or the fix
  would reproduce the exact run7 incentive one level up.
- Moved the training cube position from `(0.15, 0.0)` to `(0.25, 0.0)` --
  the far edge of the same train region -- based on the user's direct
  observation. Checked against `sim/output/reachability_sweep.json`
  rather than picked freely: `(0.25, 0.0)` has the best empirically-
  measured IK convergence of any point checked in the whole region, so
  this isn't a reachability tradeoff.

Verified via 6 new self-test cases (including a direct regression test
reproducing run7's exact false positive and confirming it's now
correctly rejected), a full local + remote self-test pass, and pushed.
An eighth run, using both new levers with the far-edge cube position, is
queued to launch once run7 finishes.

**Status**: TD-MPC2 has gone from "no directed behavior at all" (runs
3-5) to "reliably reaches and touches the cube" (run6) in the space of
one mechanistic diagnosis and two targeted levers. The remaining gap is
finishing the grasp itself, not finding the cube -- run8 tests whether
the newly-added geometric grasp-closing signal closes that gap. Two
real, video-confirmed false positives in the detection logic have now
been caught and fixed (is_holding()'s height-based flip in the
correctness audit, and now is_between_jaws()) -- both times by watching
video rather than trusting a logged flag, which remains this project's
single most reliable debugging tool. Diffusion policy side is still
fully dormant, deferred since 2026-08-31 pending the user's return to it.

### 2026-09-03 (continued) -- Run7 completes (second false positive), run8's fix holds up, two new reward terms and warm-start capability added

Run7 finished: 70,000 steps, no crash, ~3h31m. Both `held=True`
checkpoints from mid-run turned out to be false positives once watched --
step 20459's gripper closed beside the cube (already root-caused, see
above), step 30439's closed near the cube's base/pivot rather than around
its body, a related but distinct misalignment. Nothing in run7's extra
20,000 steps over run6 produced genuinely new behavior -- the
touched-but-not-held plateau held exactly as before, which is what
actually justified moving to a geometric fix rather than a fourth
uniform-length repeat: "just run longer" had now been tried and had
stopped teaching us anything new.

Ran the full verification suite for the `is_between_jaws()`/
`grasp_close_weight` fix (self-test, both env smoke tests, real-data
replay) before committing, then launched run8: 70,000 steps, cube moved
to `(0.25, 0.0)` -- the train region's far edge, checked against the
empirical reachability sweep data rather than picked freely (turned out
to have the single best measured IK convergence of any point in the
region). Result: real, visible progress -- `touched=True` on 10 of 14
checkpoints, consistent purposeful reaching -- and, just as important,
**zero** `between_jaws=True`/`held=True` false positives across the
entire run, confirming the geometric fix isn't just strict, it's
correctly strict. Direct video review of the best checkpoint (step
55389) showed exactly why the milestone still hadn't fired: the arm
approaches from directly above and pokes the cube with a single
fingertip, repeatably and with clear intent, but never straddles it with
both open jaws.

The user, watching the same video, separately flagged a second pattern:
the gripper closes almost immediately on approach, well before anywhere
near correctly positioned -- and asked directly whether the two fixes
already planned (see below) would address it. They wouldn't have, on
their own -- lateral-alignment shaping only concerns *where* the arm is,
not the gripper's joint angle, and warm-starting just continues training
under whatever incentive already exists. Working through why surfaced a
concrete, testable hypothesis: `min_std` (raised uniformly for run6
onward to fight CEM's own exploration collapse) applies identically
across *every* action dimension, including the gripper -- injecting
persistent noise into gripper actuation regardless of position, with
nothing in the reward pushing back against it, since closing early was
previously just neutral (no reward, no cost).

Implemented three things together, carefully, taking the time the user
explicitly asked for rather than rushing it:

1. `lateral_align_weight` -- a genuine potential-based shaping term over
   the raw lateral offset from `is_between_jaws()`'s own decomposition
   (factored into a shared `_jaw_offsets()` helper so both use identical
   geometry), fixing the binary-gate-with-no-gradient problem directly.
2. `premature_close_weight` -- a small, deliberately ABSOLUTE (not
   potential-based) penalty for closing while not correctly positioned,
   directly targeting the behavior the user flagged. Explicitly confirmed
   this doesn't reintroduce the original reward-hacking mechanism: that
   was an absolute-value *reward* farmable by dwelling in a state; a pure
   *penalty* is minimized by avoiding one instead, so there's nothing to
   exploit. Mutually exclusive with `grasp_close_weight` by construction.
3. `--resume-from` -- warm-start support for `train_tdmpc2_pickplace.py`
   (`TDMPC2.load()` already existed, just unused), so run8's genuinely-
   learned reach/touch skill isn't thrown away while training against the
   two new terms above.

Caught and fixed two real bugs while wiring this up, both in test
plumbing rather than the reward design: the self-test's own
`compute_reward()` calls passed `cfg` as a bare positional argument
immediately after `prev_joint_pos` across all ~27 call sites -- correct
under the signature at the time, but silently reinterpreted as the newly
inserted `prev_lateral` parameter the moment it was added before `cfg`,
caught by an actual test failure and fixed by passing `cfg=cfg`
explicitly everywhere; and a stale test fixture (`gripper_open_joint`,
`1.0`) that was only ever "open enough" for the old binary closed check,
not the gripper's true physical open limit, which the new continuous
closedness potential read as partially closed, incorrectly tripping the
new penalty in unrelated tests.

Verified via 13 new self-test cases, a full local + remote self-test
pass, both env smoke tests, and real-data replay validation -- all clean.
Committed, pushed, and launched a ninth run (40,000 steps -- shorter than
the from-scratch runs, since this is refinement of an existing skill, not
acquisition of a new one -- warm-started from run8's final checkpoint,
minimal 1,000-step seed-exploration phase given the resumed policy
already knows how to act). Confirmed the resumed weights actually loaded
(`[INFO] Resumed agent weights from ...` in the log) before trusting the
run. In progress at time of writing.

**Status**: three real, video-confirmed false positives in the detection
logic have now been caught and fixed over the life of this project
(is_holding()'s height-based flip, is_between_jaws() for the beside-the-
cube case, and implicitly the base/pivot case it also covers) -- all
three caught by watching actual video, none by the printed numbers alone,
which remains the project's single most reliable debugging discipline.
The reward has evolved from "rewards absolute proximity" (exploitable) to
"rewards genuine progress plus milestones" (run1's fix) to "rewards
genuine progress toward a *correctly positioned* grasp, not merely a
nearby one" (this entry) -- each step motivated by a specific, directly-
observed failure mode, not by speculation. Whether the two newest terms
actually close the remaining gap is run9's open question. Diffusion
policy side remains fully dormant, deferred since 2026-08-31.

### 2026-09-04 -- Migrated to a new Mac (again); run9 results reviewed, an
axial-alignment fix proposed, initially misdiagnosed, then corrected

Confirmed context/state was fully preserved across the Mac migration
(docs/ and code sync via `git pull`, `sim/output/` untracked and synced
only on request via `scp` -- see docs/preferences.md). Set up Tailscale/
LAN connectivity and passwordless SSH from the new machine.

Run9 finished: 40,000 steps, warm-started from run8. The user reviewed
`eval_step_036427.mp4` and `eval_step_032435.mp4` directly and reported
real, specific progress -- the arm touching the cube with the stationary
part of the gripper and pushing it along, close enough that closing the
gripper at the right moment "would've basically picked it up." Asked to
look into making that happen.

First pass at this was wrong. Reading the extracted frames, I concluded
the gripper was CLOSED during these pushes and began implementing
gripper-aperture pre-shaping (a `pre_shape_weight` config field) before
the user corrected me directly: "no hang on, the gripper is open when
its pushing it. i just want to make sure you understood that." Fully
reverted the incomplete `pre_shape_weight` edit and re-examined the same
frames with that correction in mind -- the gripper genuinely was open
(a clear V-shape, visible gap between both prongs), but the cube sat
buried near the jaws' pivot/hinge, where that gap is narrowest, rather
than out near the fingertips, where it's widest. An axial reach-distance
problem, not an aperture one. Plausible mechanism: `reach_weight` rewards
driving raw gripper-to-cube distance toward zero, with no notion of a
correct standoff distance, so it keeps paying out for pulling the pivot
itself past the point where the fingers could actually catch the cube.

Implemented `axial_align_weight` (mirroring `lateral_align_weight`
exactly, over the axial component of `_jaw_offsets()` instead of the
lateral one) after confirming the diagnosis with the user, who asked
directly whether this would also address the gripper-closing timing --
answered honestly that it wouldn't on its own (a separate, existing
mechanism, `grasp_close_weight`, already exists for that), and that
warm-starting again from run9 with a longer step budget (~60,000 vs.
run9's 40,000) made sense specifically because this fix needs to
counteract an already-reinforced "drive distance to zero" habit, not
build on neutral ground. Launched run10: 60,000 steps, warm-started from
run9.

**Note for later readers**: this session also discovered and fixed the
eval-video/checkpoint filename-collision data-loss bug documented above,
and reorganized all surviving run output into per-run folders -- see the
data-loss entry above and docs/decisions.md for the `--run-name` fix.

### 2026-09-05 -- Run10 completed but never actually closes; reward
redesigned around the closing incentive itself, not positioning

Run10 finished: 100,000 steps (the longest single run yet), no crash.
Frame-by-frame review of three full eval episodes (steps 85329, 90319,
95309) told a different story than hoped: the arm reaches and makes
contact within 1-2 seconds of a 25-second episode, then holds a
completely static pose for the rest of it, regardless of what happens
next. In two episodes the cube stayed wedged near the pivot the whole
time (the exact problem `axial_align_weight` was meant to fix); in the
third, the initial contact pushed the cube completely out of reach
within the first few seconds, and the arm just kept reaching at empty
space for the remaining ~20 seconds, never re-engaging. Across all three
episodes, traced frame-by-frame, **the gripper never closed even once**.

Since `axial_align_weight` only concerns position, not the actual close
decision, and the real remaining gap was clearly "never attempts to
close at all," kept the axial term implemented (self-tested, working as
designed) but reverted it via `git revert` rather than build further on
top of a fix that was not the actual bottleneck -- preserves the
implementation and its tests in history without carrying dead weight
forward.

Redesigned around the closing mechanism directly: checked
`is_between_jaws()`'s actual trigger rate against where contact was
really happening (almost never true at the policy's real contact point,
given the cube's tendency to land right at/beyond the old
`grasp_reach_min` boundary) and found the likely cause of "never closes"
-- `grasp_close_weight` (rewards closing) almost never got to fire, while
`premature_close_weight` (penalizes closing) fired on nearly every step
the gripper had any closedness at all, since "not between jaws" was the
overwhelmingly common case at the real contact point. Three narrow
changes: widened `grasp_reach_min` (-0.01 -> -0.04) so the window
actually covers where contact happens, raised `grasp_close_weight`
(1.0 -> 2.5) so it dominates once reachable, lowered
`premature_close_weight` (0.3 -> 0.1) so it no longer drowns that signal
out. See docs/reward_function.md and docs/decisions.md for the full
mechanism.

Verified via an updated self-test (one fixture needed adjusting once the
window widened -- a test asserting "behind the pivot must fail" using an
offset that cleared the OLD boundary but not the new one, caught before
it could silently pass for the wrong reason) and a production smoke
test. Launched run11: 60,000 steps, warm-started from **run9's**
checkpoint specifically, not run10's -- run10's extra 100,000 steps were
spent reinforcing the counterproductive "freeze" habit under the old
incentive structure, and building on top of a more deeply entrenched bad
habit seemed like a worse starting point than the less-contaminated
run9 checkpoint.

### 2026-09-05/06 -- Run11 results: a video-read corrected by objective
logging, premature_close_weight zeroed, run12 launched

Run11 finished: 60,000 steps, no crash. Frame-by-frame review of the
early checkpoint (step 5489) showed a much more dynamic, exploratory
policy than run10's frozen reflex -- sweeping through many different
poses and re-approaching repeatedly rather than diving once and
freezing, consistent with the network still adapting to the changed
reward landscape. Later checkpoints (045409, 050399, 055389) appeared,
from extracted still frames, to show the gripper aperture visibly
narrowing during the approach -- read (wrongly, as it turned out) as
genuine closing.

The user watched the actual videos and disputed this directly: "i dont
think i ever saw the gripper close in either of those." Rather than
re-argue from more still frames (the same fragile approach that had
already gone wrong once this project, with the aperture/axial
misdiagnosis above), pulled the objective, ground-truth
`touched`/`between_jaws`/`held` booleans `run_eval_episode()` already
logs once per eval episode, straight from the training log. Verdict:
`held` was `False` in every single one of run11's 11 eval checkpoints.
`between_jaws` HAD flipped `True` in 4 of them (steps 20459, 35429,
50399, 55389) -- something that essentially never happened in runs 9/10
-- so the widened geometric window was genuinely helping positioning,
but closing specifically still was not happening. What I had read as
"narrowing aperture" in the stills was almost certainly a foreshortening
illusion from the fixed external `scene_camera` combined with the arm's
own rotation, exactly the kind of thing full video motion catches and
sparse stills do not.

The user also asked directly whether the eval video was what the policy
itself sees. Checked the actual code to answer precisely rather than
guess: `scene_camera` is a fixed, third-person diagnostic viewpoint,
explicitly documented in `run_eval_episode()`'s own docstring as "a
visual check on what the policy actually does, not [what the policy
sees]." The real policy inputs are `wrist_camera` (mounted on the
gripper) and `top_camera` (fixed top-down) -- neither of us had ever been
looking at what the policy actually observes, only a human-reference
view.

To find out why `between_jaws` firing was not converting into a genuine
close, added `--eval-only` to `train_tdmpc2_pickplace.py` (see
docs/decisions.md) and replayed run11's best checkpoint (step 55389)
with new per-step logging (gripper joint angle, commanded gripper
action, axial/lateral offset, `between_jaws`). This ruled out "brief
pass-through, no time to close" outright: `between_jaws` fired dozens of
times across the 500-step episode, including a 15-consecutive-step
window, plenty of time. The gripper barely moved even during that long
window (under 0.01 radians of drift), and the one place it dipped
meaningfully (steps 368-371, ~0.06 radians toward closed, out of the
~1.44 radians needed to reach `gripper_closed_threshold`) reversed within
a few steps, climbing steadily back to ~fully-open rather than
continuing or settling. Root cause: `premature_close_weight` is gated on
`not between_jaws`, and `between_jaws` itself is unstable (true only
~15% of the episode, mostly short bursts) -- so a policy partway through
a slow, multi-step close is one small drift away from the penalty
resuming on its still-partly-closed gripper. Reopening immediately is
the locally safe strategy; committing to a full close across a window
that might not hold is not.

Set `premature_close_weight` to 0.0 -- the original reason it existed
(run8 closing carelessly far from the cube, regardless of position) is
now separately handled by `grasp_close_weight`'s own `is_between_jaws()`
gate, which did not exist yet when this penalty was first added.
Considered a grace period or `between_jaws` hysteresis instead, but
deliberately chose the cleanest single-variable test first. Verified via
self-test (the mechanism itself stays covered through a local
nonzero-weight config even at a 0.0 production default) and a production
smoke test. Launched run12: 60,000 steps, warm-started from run11's
final checkpoint.

### 2026-09-06 -- Run12 produces a real held=True, verified to be a
marginal graze rather than a genuine lift, three fixes, run13 launched

Run12 finished: 60,000 steps, no crash. **`held=True` fired for the
first time in this project's history**, at step 40419's eval episode.
Pulled all 11 eval videos and reviewed them; asked the user to look at
`eval_step_040419.mp4` specifically, reporting (again, from still-frame
reading) that the aperture appeared to close and stay closed for the
rest of the episode.

The user again disputed this from the actual video: "it never held it
in that video? im not sure where you got that it is working from?" --
and separately asked directly whether the camera being watched was the
policy's own view (answered from the code in the entry above: no, it
never has been). This time, rather than defend a video read at all,
extended `--eval-only`'s CSV with `cube_height` and `gripper_cube_dist`
specifically so every one of `is_holding()`'s four establishing
conditions (height above `lift_threshold`, joint closed enough,
proximity, `between_jaws`) could be checked independently against the
exact checkpoint that produced the result, rather than trusting either
the video or the logged flag at face value.

Finding: the `held=True` result was real by every one of `is_holding()`'s
own checks -- not a geometric false positive like the run7 bug -- but
the actual lift was only ~9mm on a 3cm cube (`lift_threshold` required
just 5mm above resting height), the cube visibly settling back toward
resting height over the following several steps rather than being
carried. It recurred 3 separate times in the one episode (steps ~95-106,
~116, ~140-141), each re-firing the full `grasp_bonus`, because that
bonus was gated on "holding now but wasn't the previous step"
(one-time per continuous streak), not "first time this episode" -- a
policy could cheaply farm the milestone bonus by grazing a razor-thin
threshold repeatedly, collecting the reward spike (+~2.0 each time, by
far the largest reward events in the episode) without ever achieving a
deliberate, sustained pick-up.

The same per-step data also showed `is_between_jaws()` flickering False
for single steps in the middle of otherwise-sustained close attempts --
independent of `premature_close_weight` (already 0.0) being a factor,
this still weakened `grasp_close_weight`'s own gradient during a real
attempt, since the reward zeroed out on those flicker steps even while
genuine progress toward closed was being made.

Implemented three changes together, all surfacing from this one replay:
`between_jaws_grace_steps` (default 2, via a new
`_between_jaws_effective()` helper) forgives up to 2 consecutive
flicker-outs for `grasp_close_weight`/`premature_close_penalty`'s gate
specifically -- deliberately NOT touching `is_between_jaws()` itself or
`is_grasped()`/`is_holding()`'s own establishment logic, which needed to
stay exactly as strict as before given the very next fix depends on it.
`lift_threshold` raised 0.02 -> 0.04 (roughly 2.5x the observed
accidental jostle). `grasp_bonus` regated on a new, genuinely
episode-sticky `was_ever_held` parameter, mirroring `touch_bonus`'s own
`was_touched` pattern, which never had this bug. Took real care wiring
this up given `was_ever_held` had to be a required (no-default)
parameter on `compute_reward()` -- a silent wrong default would let a
real call site forget to thread it through with no error at all, so
every one of the ~46 self-test call sites plus both real call sites
(`pickplace_env.py`, `validate_reward_function.py`) needed updating and
were verified programmatically, not just by eye, to confirm none were
missed.

Added 12 new self-test cases: direct unit tests of
`_between_jaws_effective()`, an end-to-end hysteresis sequence through
`compute_reward()` confirming the shaping genuinely keeps paying out
through a forgiven flicker and genuinely stops once the flicker outlasts
the grace window, and real regression tests for both the lift-threshold
graze and the grasp_bonus re-firing bug, built directly from the actual
observed numbers rather than hypothetical ones. Verified via a full
local + remote self-test pass and a production smoke test, all clean.
Launched run13: 60,000 steps, warm-started from run12's final
checkpoint -- a relatively cheap verification that these three specific
fixes work as intended before committing to the much larger investment
of a full, from-scratch clean retrain against the finalized reward
design (discussed with the user and agreed as the next step once run13
confirms the fixes hold up).

### 2026-09-06/08 -- Run13 inconclusive, a real tool-limitation catch,
### positioning itself found to be regressing, a clean retrain that
### disproved its own premise, and the pivot to demonstration seeding

Run13 finished: 60,000 steps, no crash. `between_jaws=True` on 3 of 12
checkpoints -- down from run11/12's 4 of 11 each, not the improvement
hoped for. Tried to verify the fixes directly by replaying two
checkpoints via `--eval-only`, and caught a real methodology gap instead
of a clean answer: the first replay came back as a completely different,
far less-engaged episode than what training-time eval had actually
logged for that same checkpoint. Root cause: TD-MPC2's CEM planner
samples internally at every planning step, and nothing in this pipeline
seeds that RNG -- `eval_mode=True` only suppresses the final action's
exploration noise, not the planner's own internal search. Every
`--eval-only` replay is a fresh, independent sample from the policy, not
a reproduction of one specific past episode -- documented as a real tool
limitation (see docs/decisions.md) rather than left as an unstated
assumption the next diagnostic might trip over.

Pulled full videos for both run13 and the replays and asked the user to
look them over. Their read, watching the actual footage rather than any
numbers: the arm didn't even seem to be reliably getting positioned
between the jaws anymore, something that had at least been visible
earlier in the project, and asked whether recovering that first -- then
layering the closing-leniency mechanics on top -- was the right
direction. It was: the raw `between_jaws` numbers back up the same read
(4/11 -> 4/11 -> 3/11 across runs 11, 12, 13), and `reach_weight`/
`lateral_align_weight` -- the terms actually responsible for
positioning -- hadn't been touched since run8/9, meaning five
generations of continuous warm-starting through several different,
sometimes-conflicting closing-focused reward regimes since run9 was the
likely culprit, a patchwork value function rather than a clean optimum
for positioning specifically.

Since there's no way to surgically recover just the positioning-relevant
learning from inside an already-warm-started network, launched run14: a
full, clean, from-scratch retrain (no warm start, the real 15,000-step
seed phase, 100,000 steps) against the current, fully-refined reward
design, reasoning that positioning would redevelop cleanly under the
already-good `reach_weight`/`lateral_align_weight` terms without the
inherited drift. **The result flatly contradicted the theory**: `touched`
came back reliably (74% of checkpoints, matching run8's own from-scratch
benchmark), but `between_jaws` fired on only 1 of 19 checkpoints --
worse than every warm-started run, not better. Launched run15 as a
direct test of the fallback theory ("just needs more time on top of
solid touching," deliberately mirroring the run8-to-run9 pattern that is
the only time this project has actually seen `between_jaws` take off):
80,000 more steps warm-started from run14's own checkpoint. Result: 0 of
15 checkpoints, the entire run.

Reassessed rather than defended the original theory: `lateral_align_weight`
has only ever been shown to work refining an ALREADY-touching-reliably
policy (run8 into run9) -- never discovering touching and precise
lateral straddling simultaneously from a random or under-trained one.
The warm-start chain likely wasn't degrading positioning after all; it
may have been the only reason positioning ever worked in the first
place. Seven consecutive RL runs (9 through 15) against this reward
design have now never produced a genuine sustained hold, only ever
momentary, sub-centimeter grazes -- a real, if humbling, result after
this much iteration on reward shaping alone.

Also fixed a real infrastructure gap in passing: Tailscale (installed
since early in the project, `100.71.12.16`) had been assumed flaky based
on an old "stuck starting" note, but turned out to already be fully
connected and running (`tailscaled` active for 5+ days, `BackendState:
Running`) -- the note was simply stale. Confirmed and switched to using
the Tailscale IP for all Ubuntu connections going forward, per the
user's request, rather than continuing to rely on the LAN IP.

Rather than keep iterating on reward shaping or attempting yet another
from-scratch run, the user raised an idea floated earlier in the
project: using the recorded real teleop demonstrations
(`sim/output/teleop_episodes/episode_000.json` through `episode_007.json`)
to directly seed TD-MPC2's replay buffer, rather than relying solely on
the agent's own exploration to ever stumble into a genuine grasp. Before
committing to this direction, checked the actual demonstration data
rather than assuming it would help -- an earlier docs/reward_function.md
note had claimed no validated episode ever reached a genuine
lift-and-carry state, but that reflected only the one or two episodes
spot-checked early in the project. Re-checked all 8 directly: every
single one reaches a cube height of 8.7cm to 14.9cm above the table,
dramatically higher than the ~2.75cm the best RL rollout ever
accidentally achieved -- strong evidence these are genuine, deliberate
lifts, not touches or jostles. Corrected the stale note in
docs/reward_function.md. At the user's explicit request, took a full
documentation checkpoint (this entry, plus the corresponding
docs/decisions.md and docs/tdmpc2_integration.md entries) before
starting this new direction, given how different a departure
demonstration-seeded training is from every run so far. Prototyping in
progress -- see docs/decisions.md for the plan.

While prototyping the demo-to-buffer replay script, found and fixed
three separate, independent bugs that had been silently preventing
every recorded teleop episode from ever registering as a real grasp:
episode segmentation was cutting off almost the entire approach/grasp
phase (fixed by re-segmenting with far more pre-padding against the
still-extant raw recording); the reach/proximity/touch thresholds were
calibrated around an unmeasured guess at roughly half the gripper's true
physical reach (corrected after measuring the real value two independent
ways -- STL mesh parsing and a live replay measurement); and
`jaw_approach_axis_world()` had been pointing the wrong physical
direction the whole time (pivot toward the wrist instead of pivot toward
the fingertips), so the reward function's own grasp-detection geometry
had a real, previously-undetected sign bug. Full details, evidence, and
verification for each in docs/decisions.md.

After all three fixes, replaying the 8 re-segmented episodes through the
real production reward function gets `ever_holding=True` for 5 of 8 --
the first time this project has ever detected a genuine sustained grasp,
from any source. This also raises a real open question worth flagging:
this same axis-direction bug has been present in `is_between_jaws()`
since it was added (2026-09-03) and would have affected every RL run's
own grasp evaluation too (runs 9 through 15), not just this replay
investigation -- not yet acted on beyond fixing it, since the current
focus is specifically the demo-seeding pipeline, but worth keeping in
mind if pure-RL training is revisited later.

Remaining before demo-seeded training can actually run: episode_004
still starts with the gripper closed even at generous re-padding and
needs further investigation; episodes 001 and 007 diverge from their
own recordings under the existing joint-limit clipping workaround
(Option 1), which may need revisiting; and the validated replay logic
still needs to be wired into `train_tdmpc2_pickplace.py` itself as an
actual, usable feature -- `replay_demo_to_buffer.py` remains a
standalone diagnostic script for now.

Wired demo-seeding into real training (`--seed-demos`) and ran four full
training runs (16-19) to find out what it actually takes to turn this
into a genuine hold, not just a working pipeline. Full details, evidence,
and reasoning for each step in docs/decisions.md; summary here:

- run16 (randomized cube position): no `between_jaws`, no `holding`.
- run17 (cube fixed to episode_002's own position, plus periodic
  re-injection of the seed episodes to stop their sampling share from
  fading as real episodes accumulate -- discovered via reading torchrl's
  own sampler source that it picks episodes uniformly by count, not
  weighted by length): `between_jaws=True` for the first time ever, 2 of
  7 checkpoints. A per-step diagnostic trace then showed exactly why
  `holding` still didn't fire: the gripper was genuinely, steadily
  closing, just too slowly relative to the arm's own lateral drift in
  and out of position -- a timing race, not a policy that never tries.
- Raised `between_jaws_grace_steps` (2->5) and `grasp_close_weight`
  (2.5->4.0) in response, then ran run18 (120k steps) WITHOUT
  `--min-std` to isolate the reward changes -- this backfired badly: the
  policy collapsed into reaching toward the cube then freezing in a
  fixed tucked pose for the rest of every episode, the same CEM
  planner-std-collapse failure mode this project fixed once before
  (runs 4/5) but had left out of every seed-demos run to isolate
  variables.
- Also hit and fixed a real machine-level bug on the Ubuntu box mid-way
  through this: its WiFi interface had no IPv4 default gateway route at
  all (only IPv6), so any IPv4-only external fetch failed outright --
  traced to IsaacLab's own default ground-plane asset needing a cloud
  fetch on one particular run. Fixed by manually adding the missing
  routes without touching the live WiFi connection, so the existing
  SSH/Tailscale session was never at risk.
- run19 (55k steps, ~2.8 hours, min-std restored, same reward tuning as
  run18): the best result yet. `touched=True` at all 10 checkpoints
  (no gaps -- run18's freeze pattern is gone), `between_jaws=True` at 6
  of 10 (more and earlier than run17). Video review confirmed real,
  sustained engagement with the cube for the majority of an episode, not
  just a brief pass-by. `holding` still never fired, but the remaining
  gap is now well-characterized: closing proceeds too slowly relative to
  the window available, not a positioning or exploration problem anymore.

This checkpoint (code, docs, and the run19 model checkpoint on the
Ubuntu box) is tagged `run19-close-timing` specifically so it can be
returned to regardless of what the next experiment (making the gripper
close faster/more decisively) does to it.

Added `close_speed_bonus` -- a one-time reward, riding `grasp_bonus`'s
own anti-farming gate, crediting how fast the gripper was closing at
the instant a hold is first established. Full reasoning (including two
rejected continuous-reward designs) in docs/decisions.md.

- run20 (warm-started from run19, `--seed-episodes 2`): the first
  genuine, detected hold this project has ever produced, at the very
  first eval checkpoint. `between_jaws=True` 8 of 10 checkpoints (up
  from run19's 6), `held=True` 1 of 10. A fresh per-step diagnostic
  trace on a non-holding checkpoint found the real remaining issue:
  the gripper now closes genuinely fast, but sometimes stops around
  40% closed and reverses back open right as the arm's own lateral
  position drifts -- confirmed via a smooth (non-spiking)
  gripper-cube distance that this is a policy decision to abandon, not
  a physical bounce. Best explanation: very few genuine "finishing
  pays off" examples exist in training data, worsened by evaluation
  episodes (where recent successes show up) never being added back to
  the buffer.
- run21 (warm-started from run20, 75k steps, `--seed-demos-min-fraction`
  raised 0.15 -> 0.20 to weight the demonstrations' complete closures
  more heavily): a mixed result. Overall `between_jaws=True` only 4 of
  14 checkpoints (worse than run20's 8/10), `held=True` never fired.
  But the last 3 of 4 checkpoints all showed `between_jaws=True` with
  sharply climbing reward, ending on the two best-reward episodes this
  project has produced (+2.57, then +7.65) right as the run ended.
  Not yet clear whether the demo-weight change helped, or the
  improvement is just from more training time/variance -- the middle
  of the run was worse than run20 on positioning for reasons not yet
  understood.

Next: a fresh diagnostic trace against run21's own final checkpoint,
to see directly whether the abandons-partway pattern has actually
improved, before deciding whether to keep warm-starting on the
assumption the late-run trend continues, or intervene more directly
(e.g. discouraging abandoning a close once started).

That diagnostic turned up two real findings. First: run21's own final
checkpoint (step 75001) is actually a regression -- 4 of 4 fresh
`--eval-only` draws failed to even touch the cube, while the same run's
earlier checkpoint at step 70359 immediately produced a genuine
`between_jaws=True` and the best reward yet (+11.06). Training isn't
monotonic; the last checkpoint saved isn't necessarily the best one a
run produced, so any future warm start from run21 should resume from
step 70359, not the final checkpoint.

Second, and more important: that +11.06 episode's high reward was
misleading. The full per-step trace showed the gripper cycling through
the same approach-dip-retreat pattern about 15 times across the
episode, each dip stalling around 35-45% closed before reversing --
`cube_height` never rose above resting height once, the entire episode.
The reward was earned by repeatedly re-collecting ordinary approach
shaping on each fresh attempt, not by getting closer to an actual
grasp. Confirmed independently on video before any fix was written.

Root cause: `grasp_close_weight`'s shaping has no memory of prior
attempts -- redoing an identical shallow dip pays exactly what it paid
the first time, so nothing makes pushing deeper than before worth more
than safely repeating a known-survivable dip. Added
`deepest_close_weight` (docs/decisions.md has the full derivation): a
shaping term measured against the deepest point reached so far THIS
episode, not just the previous step, so only genuinely exceeding a
prior attempt ever pays out again -- verified directly in the self-test
that this can't be gamed by oscillating, not just argued. Self-test and
a full smoke test both pass clean.

Next: a real training run warm-started from
`run21_more_demo_weight/agent_step_070359.pt` (the good checkpoint, not
the regressed final one) with `deepest_close_weight` now active.

Launched run22 (48,000 steps, `run22_deepest_close`, warm-started from
`run21_more_demo_weight/agent_step_070359.pt`, `deepest_close_weight`
active) after tagging `pre-deepest-close-run`. Confirmed healthy at
step ~5545 (GPU active, buffer growing, updates progressing). Left
running in the background per instruction; not polled further until
asked.

While run22 trains, started the diffusion-policy side of the project
(no teleop access currently, so this work doesn't depend on the robot).
Reviewed docs/diffusion_policy_data_strategy.md's already-decided plan
(custom MimicGen-lite: segment a source demo at the grasp event,
SE(3)-retarget the reach-and-grasp segment to a new cube position,
splice into the unchanged transport-and-place segment) and picked the
SE(3) transform-and-replay mechanism as the first validation target,
since it's the part most likely to reveal a hard blocker.

Found that Isaac Lab's own `isaaclab_mimic` module (NVIDIA's MimicGen
port) provides the exact pose-math needed
(`isaaclab.utils.math.pose_in_A_to_pose_in_B`/`make_pose`/`unmake_pose`/
`pose_inv` -- plain, dependency-free 4x4 tensor ops) and that
`isaaclab.controllers.DifferentialIKController` (generic Jacobian
pseudo-inverse IK, `dq = J+dx`, confirmed via
`IsaacLab/scripts/tutorials/05_controllers/run_diff_ik.py`) is
robot-agnostic and needs no SO-101-specific IK derivation. The full
`isaaclab_mimic` generation framework requires `ManagerBasedRLMimicEnv`,
a different env-authoring paradigm than this project's
`PickPlaceEnv(DirectRLEnv)` -- decided to reuse only the standalone math
and IK controller directly in a custom script, not adopt the framework.

Built and validated the first piece: `extract_grasp_segment.py`,
adapted from `replay_demo_to_buffer.py`'s proven state-override replay
pattern. Replays a source episode through the real `PickPlaceEnv`,
records the gripper's own live world pose (position + quaternion, read
back each step -- this IS the forward-kinematics step, no analytic FK
computed by hand) alongside the recorded gripper joint value, and uses
the now-fixed `is_holding()` to find the exact grasp-event boundary
splitting "reach-and-grasp" (needs retargeting) from
"transport-and-place" (fixed target, replayed unchanged).

Ran against `episode_002` (one of the 5/8 episodes already confirmed
`ever_holding=True`, and the same episode run17 fixed the cube position
to). Result: grasp boundary at downsampled step 194 of 545 (of a
645-step raw episode at 100Hz), with holding sustained for 222 of the
306 traced steps after the boundary -- a genuine, non-flickering hold,
not a one-step blip. The extracted 195-step reach-and-grasp segment
travels only ~2.5cm end-to-end (the recorded demo's final approach was
already a short final reach), gripper joint closing from 0.588 to 0.156
over the segment. No crashes, no bugs found on the first real run
(after fixing two Isaac-Sim launch-flag mismatches, not logic errors --
see docs/decisions.md).

Next: use this segment plus a new target cube position to validate the
SE(3) retargeting math itself (`pose_in_A_to_pose_in_B`), then drive the
transformed poses through `DifferentialIKController` in closed loop, and
finally splice the retargeted segment into the source episode's own
unchanged transport-and-place tail and confirm `is_holding()` still
fires at the new cube position.

Meanwhile, run22 finished and it's a real regression: `held=True` never
fired (0/9 eval checkpoints), `between_jaws=True` only once (1/9),
ending on its two weakest checkpoints -- worse than the checkpoint it
was warm-started from. Dug into why with per-step `--eval-only` traces
at three points (the original run21 checkpoint, run22's one
`between_jaws` hit mid-run, and run22's final checkpoint) instead of
guessing. Found a clean, monotonic pattern: the gripper's oscillation
amplitude and how often it dips into the tight lateral window
`is_between_jaws()` needs both shrink together the longer training
continues -- from full-range swings hitting the window 30% of the time,
to modest swings hitting it 4.5% of the time, to barely-there
adjustments that get within 0.1mm of the window but never cross it, 0%
of the time. The approach phase itself is untouched (still a smooth,
purposeful reach every time) -- this is specifically a collapse in the
close-range commitment behavior.

Root cause: `deepest_close_weight` is correctly farm-proof (verified
again here), but farm-proofing it this way means the reward bar rises
every time it's cleared, so genuine reward gets sparser as training goes
on. The existing (small, previously harmless) `action_penalty_weight`
had been swamped by the old exploit's reward before; once that exploit
is closed off, this same small constant cost of attempting starts to
outweigh the shrinking expected payoff, and training correctly grinds
the policy toward smaller, safer motions that increasingly miss the
precision window. Also found the deeper reason run21's oscillation loop
paid out in the first place: gating a potential-based term on
`between_jaws_for_closing` breaks the exact telescoping guarantee
potential-based shaping normally provides (it requires evaluating the
potential every step, not just while a gate happens to be open) --
`deepest_close_weight` avoids that specific hole by persisting its
record independent of the gate, which is also exactly what makes the
reward landscape sparser over time. Full derivation, the data table, and
the proposed fix (phase-gating `action_penalty_weight` down during the
pre-hold approach/grasp phase, not touching `deepest_close_weight`
itself) are in docs/decisions.md. Not yet implemented -- checking with
the user on direction before another reward change and another
multi-hour run.

Implemented the fix: `action_penalty_approach_scale` (0.2 default) cuts
`action_penalty_weight`'s effect by 80% while not holding, full strength
once holding. Verified with 4 new self-test cases (local + Ubuntu) and a
full training-pipeline smoke test (no crash, real call site) -- full
details and the one loose end (a separate, older validation script hung
in Isaac Sim's own scene loading, unrelated to this change, not yet
investigated) in docs/decisions.md. Ready for a real training run to see
whether it actually recovers the oscillation/exploration behavior and
`between_jaws`/`held` rates.

Ran run23 (48k steps, warm-started from the same checkpoint as run22,
with the fix active). Single-draw eval results looked like a partial,
ambiguous improvement over run22 (4/9 between_jaws vs 1/9, still no
held), matching this project's whole recent pattern of "fix something,
get an unclear result, need another fix." Rather than write fix #4,
stepped back and asked why every fix keeps landing here.

Two things this project's methodology had never actually checked: (1)
TD-MPC2's planner is genuinely non-deterministic even in eval mode --
confirmed by reading tdmpc2.py's _plan() directly, the executed action
is always a real stochastic sample over the elite trajectory set, every
single planning call -- so every between_jaws/held number this project
has ever reported for a checkpoint was exactly one noisy draw, never
verified against repeats. (2) Laid the whole warm-start chain side by
side (19->20->21->22/23) and found only the very first hop ever improved
on its starting point -- every hop since has cost something, regardless
of which specific reward term changed alongside it, pointing at the
shared mechanism (weights-only warm-starting, buffer always reset) more
than any one reward term.

Built `--eval-only-repeats N` (loops eval episodes in one Kit boot,
aggregate hit-rate summary) and used it to properly re-test three
checkpoints at 6 draws each, plus a min_std=0.2 vs 0.5 comparison. Real
results, not single draws: the ancestor checkpoint genuinely gets
between_jaws 100% of the time (12/12 across two settings); run22's
collapse to 0% was real, not noise; the action_penalty fix genuinely,
substantially recovered it to 67% (not fully back to 100%, a real
remaining gap); lowering min_std changed nothing about between_jaws or
held rates, ruling out planner noise as what's capping precision.

The actual headline finding: `held=True` came up 0 times in all 18
draws across every checkpoint tested, including the best one available.
Every reward-shaping fix this project has made recently has been aimed
at a target (reliably finishing a hold) that this specific lineage of
checkpoints has never actually demonstrated even once under honest,
repeated measurement -- run20's one recorded hold, this project's only
ever example, has never been reproduced by any descendant checkpoint.
Full numbers and reasoning in docs/decisions.md. Conclusion: keep the
action_penalty fix (it genuinely worked), adopt multi-draw testing as
the standard going forward, and stop adding reactive reward terms --
the next real experiment is more UNINTERRUPTED training time on the
current best checkpoint, properly measured, before considering anything
structural (buffer persistence, demo re-weighting).

Ran that experiment: run24, 96k steps (double the usual budget), resumed
from run23's actually-verified checkpoint, nothing else changed. Tested
properly with `--eval-only-repeats 6` on its two most relevant
checkpoints rather than trusting single-draw training logs. Result: the
best checkpoint plateaued at 0.67 between_jaws -- statistically the same
as run23 already had, no further recovery from 48k more steps of pure
training time. The final checkpoint was a genuine, severe regression
(0/6 touched at all) -- the second independent confirmation (after
run21) that this codebase's last-saved checkpoint should never be
assumed to be its best.

Running total: `held=True` in 0 of 36 independent, honestly-drawn
episodes across 4 checkpoints spanning 3 training runs. That's a solid
null result now, not a small sample. "Just train longer" was the
hypothesis this run tested, and it didn't hold up -- more of the same
isn't the path past this. Two concrete structural options are on the
table (persist the replay buffer across warm starts instead of
weights-only; re-weight demo sampling toward the hold-and-carry portion
specifically) -- full reasoning in docs/decisions.md, deciding with the
user which to try next rather than picking unilaterally.

Decided on demo re-weighting: the ancestor checkpoint (100% between_jaws
across 12 draws, never destabilized by a warm start) still never held,
which points at a persistent data/signal gap rather than transient
instability -- the thing buffer persistence would target.

Caught and corrected my own first-pass reasoning before implementing
anything: initially thought the hold phase was under-represented because
it's a small FRACTION of each demo episode -- checked this against real
data (extract_grasp_segment.py's boundary data across all 5 seed
episodes) and it doesn't hold up, the hold phase is actually LARGER than
the approach phase by step count. The real mechanism: only 5 demo
episodes exist and --seed-demos-min-fraction only floors their EPISODE
count, not how many sampled transitions are genuinely post-grasp -- the
other ~80% of the buffer is self-generated experience that (since the
policy rarely succeeds) contributes close to zero hold-phase transitions
of its own, so hold signal stays thin and non-growing no matter how long
training runs. Confirmed this actually fits how the buffer's sampler
works by reading torchrl's SliceSampler source directly: episodes are
selected uniformly by count, so a short, hold-concentrated pseudo-episode
gets exactly the same selection odds as a full-length one.

Implemented `--seed-demos-hold-segments`: extracts each seed episode's
segment from 50 steps (checked against all 5 episodes' real boundaries,
194-200, before picking this) before its first is_holding()=True through
its end, and injects it as an additional pseudo-episode on the same
re-injection cadence as the whole episodes -- no new replay cost
(reuses replay_episode_to_tds()'s existing single pass), no changes
needed to the existing re-injection machinery. Verified three ways
before calling it done: real extraction against all 5 actual episodes
matched hand-computed expectations exactly; the never-held edge case
(via --smoke-test's short timeout) correctly produces no segment; a
regression check confirmed --seed-demos without the new flag is
byte-for-byte unchanged. All three passed on the first attempt. Full
design reasoning in docs/decisions.md. Ready for a real training run.

Ran run25 (72k steps, warm-started from the ancestor checkpoint,
--seed-demos-hold-segments active). Verified its best-looking checkpoint
(step 45409, +5.08 reward -- the highest single-draw reward this project
has ever recorded) properly with 6 draws: 100% touched, 100%
between_jaws, 0% held, mean reward +3.13. This is the first descendant
checkpoint since the ancestor itself to fully recover its 100%
between_jaws rate -- run22 collapsed to 0%, run23 and run24 both
plateaued at 67%. Not provable that hold-segments specifically caused
this from one run, but 6/6 is real signal, not noise, and it's the best
verified checkpoint this project has produced since the original
ancestor. held is still 0/6 though -- the core goal isn't solved yet.

The final checkpoint (072001) was, again, a regression (0% on
everything) -- the third confirmed instance of this codebase's final
checkpoint being worse than an earlier one from the same run (after
run21 and run24). Treating this as a standing fact going forward rather
than re-diagnosing it each time: never warm-start from a run's own final
checkpoint without checking earlier ones first.

Videos pulled to sim/output/tdmpc2_eval_videos/run25_hold_segments/ on
the Mac (matching the Ubuntu-side path, not a temp directory) for the
user to review -- eval_step_045409.mp4 is the one to watch.

Implemented the flagged "next lever": --seed-demos-hold-segments-fraction
gives hold segments their own independent target buffer-episode
fraction (default 0.20) and their own re-injection schedule, fully
decoupled from --seed-demos-min-fraction -- whole episodes no longer
get quietly diluted to make room for hold segments the way run25's
first version did. Verified with a deliberately extreme test (fraction
0.9) that made the two groups' schedules clearly different: hold
segments re-injected every single real episode while whole episodes
correctly stayed silent on their own, much longer schedule -- confirmed
independent, not just theoretically decoupled. Also confirmed the
actual practice values (both at 0.20) compute identically as expected.
No crashes. Ready for a real run, warm-started from run25's verified
step45409 checkpoint -- the best-verified starting point now, not the
original ancestor.

While run26 trained, started the sim-to-real side: the user connected
the physical FOLLOWER arm for the first time (never before -- only the
leader had ever been connected, purely to read teleop input into sim).
Researched what actually exists before writing anything: no code in
this repo has ever sent a command to a physical robot. Read LeRobot's
SOFollower class directly to understand its real API (interactive,
torque-disabled calibration flow; a read-only get_observation(); a
send_action() with a built-in max_relative_target safety clamp; real
STS3215 PID defaults, a genuine new data point for the sim_to_real
checklist's open PD-gain item). Built follower_reader.py -- connects,
calibrates if needed, then only ever reads and prints joint positions,
never calls send_action() anywhere. User needs to run it themselves,
interactively, since calibration means physically moving the arm by
hand in sync with prompts. This is the deliberately cautious first step
before anything in this project ever writes a motor command to real
hardware. Full details in docs/decisions.md.

User connected the follower, calibrated it, and reported each joint's
sign by hand. Cross-checked all 6 against sim's own convention --
mostly via rendered images (extending calibrate_grasp.py's method to
every joint, not just the gripper), `shoulder_pan` additionally
cross-checked against a real recorded episode's cube position (since a
top-down rotation is hard to judge from an oblique camera), and
`wrist_roll` needed a third attempt (a rendered image was too ambiguous,
then a position-tracking approach came back internally inconsistent)
before landing on directly extracting the true rotation axis/angle from
gripper_link's own measured orientation change -- verified via a sanity
check before trusting it. Result: `elbow_flex` and `wrist_roll` need a
sign flip between a policy's action output and a real follower command;
`shoulder_pan`, `shoulder_lift`, `wrist_flex`, and `gripper` transfer
directly. Full comparison table and reasoning in docs/decisions.md;
recorded in docs/sim_to_real_checklist.md too, as a new checked-off
"joint sign convention" item. Not yet applied anywhere -- no code exists
yet that actually sends a command to the follower.

Meanwhile, run26 (72k steps, warm-started from run25's 100%-verified
checkpoint, independent hold-segment weighting active) finished and its
best checkpoint verified at only 33% between_jaws -- a regression from
run25, not an improvement. Can't tell from one run whether the new
weighting caused this or it's another instance of the same warm-start
variance seen throughout this investigation, but it didn't help. Final
checkpoint was again a total collapse (0% everything) -- the fourth
confirmed instance of that pattern, no longer worth re-diagnosing each
time. run25's step45409 remains the best verified checkpoint this
project has produced since the original ancestor; future work should
build from there, not from run26. Full numbers in docs/decisions.md.

First real command sent to physical hardware: a no-op position readback
(follower_writer_test.py) confirmed the write path works with zero
unexpected motion, then episode_002's reach segment was replayed for
real (follower_replay_reach.py, sign flips applied). Success -- correct
direction, gripper closed appropriately, settled cleanly. Motion was
visibly "stop and go" (expected -- replayed at 10Hz against a recording
made at 50Hz) and stopped ~2-3 inches above the table instead of at
contact -- checked against sim's own data and found sim itself has the
cube ~1 inch up at this exact cutoff (the segment ends where
`is_holding()` first turns true, already mid-lift), accounting for part
of the gap but not all of it -- a real, if modest, sim-to-real height
offset makes up the rest, not yet isolated further. Full details in
docs/decisions.md.

Back to the world-model side: launched run27 (option 1 -- more time
from run25's own 100%-verified checkpoint, otherwise unchanged) after
catching and fixing a real config bug first (a first launch attempt was
accidentally injecting demo material at 2x run25's original rate, not
matching it -- see docs/decisions.md for the exact fix). While that
runs, prepared options 2 (half run25's demo weight, testing run26's
"more weight hurt" finding directly) and 3 (longer CEM planning horizon,
3->5, targeting the hypothesis that the planner can't directly see far
enough ahead to value a genuine multi-step hold) -- both fully verified
and ready to launch immediately if run27 doesn't pan out. Option 3
required adding a new --horizon CLI override and confirming, both by
reading the code and by an actual smoke test, that it's safe to combine
with warm-starting from an existing checkpoint trained at a different
horizon. Exact commands for both in docs/decisions.md.

Ran option 1 (run27) and option 3 (run29, launched in parallel at the
user's request once GPU headroom was confirmed) to completion. run27
(more time, run25's exact settings) did NOT reproduce run25 -- only
1/19 single-draw checkpoints hit between_jaws, and even that one at a
negative reward. run29 (horizon 3->5, otherwise identical to run27) told
a genuinely different story: verified its best checkpoint (step75349) at
100% between_jaws (6/6 draws) -- matching run25's own best result
exactly, via a completely different mechanism and a different training
run, independent confirmation this is a reachable, repeatable outcome
rather than a one-off. This is the clearest real evidence yet that
longer planning horizon helps. held remains 0% throughout, and the final
checkpoint regressed again (17%) -- the fifth confirmed instance of that
pattern. Full numbers in docs/decisions.md.

Diagnosed why run29's 100%-between_jaws checkpoint still never achieves
`held`: per-step CSV analysis across all 6 verified draws showed
`lateral` sitting at roughly double the strict `grasp_lateral_threshold`
while touching the cube, so the closing-shaping reward terms almost
never fired -- a genuine stable "hover, don't close" local optimum, not
a failed attempt. Added `grasp_close_lateral_threshold`, a wider
tolerance used only to gate the closing-shaping terms, leaving the
strict success criteria (`is_between_jaws`/`is_grasped`/`is_holding`)
completely untouched to avoid reopening the run7 false-positive failure
mode. Launched two runs from run29's verified checkpoint to test it:
run30 (standard demo weight) and run31 (half demo weight, auto-queued to
launch after run30 via a log-marker-polling script rather than waiting
on process exit, since every run in this project hangs at shutdown and
never exits on its own).

run30 verified: step020459 (20k of 42k steps) hit a genuine 100%
between_jaws (6/6 draws) -- the fix works exactly as designed for
positioning consistency, matching run25's and run29's own best results
via a third distinct mechanism. `held` is still 0/6, though -- the fix
alone hasn't solved the actual closing problem. step040419 (double the
training) collapsed back to 0/6 between_jaws -- the sixth confirmed
instance of this project's more-training-regresses-the-checkpoint
pattern. run31's equivalent checkpoint is being verified now. Full
numbers and the CSV-based diagnosis in docs/decisions.md.
