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
