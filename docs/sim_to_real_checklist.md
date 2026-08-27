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

- [ ] **Camera extrinsics (pose relative to robot base)** -- Not yet done.
  Current sim camera (`sim/scenes/pickplace_scene.py`, pos `(0.6, -0.6,
  0.5)` looking at `(0.15, 0, 0.05)`) was placed arbitrarily for visual
  debugging, not matched to any real mounting position. Once a real camera
  mounting position is chosen, either match it exactly or make it the
  center of the randomized camera-pose range (not fully arbitrary).
- [ ] **Camera intrinsics** (FOV/focal length, resolution) -- Not yet
  matched to whatever real camera hardware gets used.
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
