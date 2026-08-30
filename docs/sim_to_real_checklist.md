# Sim-to-Real Consistency Checklist

Tracks what needs to be deliberately matched between simulation and the
real SO-101 setup, versus what's left to domain randomization. Created
2026-08-27 after realizing the initial scene/camera/motor setup was built
for a first physics sanity check, not sim-to-real consistency -- worth
tracking explicitly so this doesn't slip.

## Philosophy

We are *not* aiming for a perfectly matched simulation -- that's brittle
and hard to achieve. The plan (see project_plan.md) already uses domain
randomization (lighting, texture, object pose, friction) so a policy
trained in sim is robust across a range rather than dependent on one exact
match. But a few things genuinely need deliberate matching, not
arbitrary/placeholder values, because randomization doesn't fix them:

## Checklist

- [x] **Camera extrinsics (wrist/eye-in-hand camera)** -- Done 2026-08-30.
  User mounted a real webcam on the wrist/gripper housing; sim's
  `wrist_camera` (`sim/scenes/pickplace_scene.py`) now attaches to the
  same body (`gripper_link`) with a position/rotation computed to match
  the real mount, informed by reference photos -- see
  docs/real_camera_setup.md. A real top-down camera (Logitech C922) was
  also connected and matched the same way -- `top_camera` is no longer
  an arbitrary debugging viewpoint, it has a real counterpart now.
  `scene_camera`/`side_camera` remain sim-only debugging viewpoints.
- [x] **Camera resolution/aspect ratio** -- Done 2026-08-30. Both real
  cameras are now physically connected to the Ubuntu box, so this was
  queried directly instead of guessed: wrist camera (generic "Web Camera"
  USB module) native 1280x720, top-down (Logitech C922 Pro Stream Webcam)
  native up to 1920x1080 -- both 16:9. Sim cameras corrected from
  mismatched aspect ratios (wrist was 480x480 square, top_camera was
  960x720 4:3) to 480x270 and 960x540.
- [ ] **Camera FOV/focal length** -- Not measured precisely (no physical
  calibration rig), but tuned by eye: rendered several `focal_length`
  candidates side by side against live real-camera captures and picked
  the closest match (wrist_camera: 10.0). Good enough for now; would need
  a proper checkerboard calibration only if precise pixel-level sim/real
  alignment becomes necessary later.
- [ ] **Real camera color/vignette** -- The real wrist camera's raw output
  has a visible vignette (dark rounded corners) and a warm color cast,
  neither present in sim renders. Decision: leave uncorrected rather than
  post-process it away or try to replicate it in sim. Domain randomization
  already varies sim lighting/color broadly, and the plan's real-data
  fine-tuning phase (see project_plan.md) trains directly on this
  camera's actual output anyway -- that's a more direct fix for this gap
  than synthetically reproducing a lens artifact. Revisit only if a
  future need calls for zero-shot sim-to-real without fine-tuning.
- [ ] **Control interface / frequency** -- Sim currently commands joints
  via Isaac Lab's `ImplicitActuatorCfg` position targets in the physics
  loop (~100Hz, `dt=0.01`). Needs to match LeRobot's actual real-servo
  control loop frequency and units once we're recording/replaying
  policies on hardware, so a trained policy's action outputs transfer
  directly without needing translation.
- [x] **Joint calibration zero-point** -- Aligned. Used the URDF's
  "new_calib" variant (zero = middle of each joint's range), matching
  LeRobot's own calibration convention for the real servos. Should be
  consistent once real calibration is run, but not yet cross-checked
  against an actual calibrated real arm.
- [ ] **Torque/velocity limits** -- Currently estimates from the STS3215
  public datasheet (see docs/so101_asset_notes.md), not measurements from
  the actual servos. Don't need to be exact, but should be validated to be
  in the right ballpark so sim doesn't allow motions the real servos can't
  physically execute (this affects whether a trained policy is even
  executable on hardware).
- [ ] **Joint stiffness/damping (PD gains)** -- Currently generic
  placeholders (50/2) copied from Isaac Lab's Franka template config, not
  tuned to how the real STS3215 servos actually respond.

## When to revisit

Items above become blocking once we (a) connect the real arm and pick an
actual camera mounting position, and (b) get to the point of running a
sim-trained policy on real hardware or comparing sim vs. real rollouts
directly. Not blocking for continued sim-only scene/task development.
