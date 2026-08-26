# SO-101 Asset Notes

Technical notes on the SO-101 robot description used for simulation — source,
structure, and things to watch out for when bringing it into Isaac Lab.

## Source

URDF (and a separate MJCF, unused) pulled from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100),
`Simulation/SO101/` folder. Both files are generated from the same Onshape
CAD model via `onshape-to-robot` — they are independent exports, not a
MuJoCo→Isaac conversion. Only the URDF (`so101_new_calib.urdf`) is used;
the MJCF file is not touched. Cloned locally at
`~/SO-101-WM/assets/SO-ARM100`.

Two calibration variants exist: `so101_new_calib` (zero = middle of each
joint's range — the default) and `so101_old_calib` (zero = fully extended
horizontal). Using the new-calibration variant to match LeRobot's default
calibration convention, since that's what the real-arm calibration will also
produce.

## Joint / Link Structure

6 revolute joints (5 arm DOF + gripper), matching the real SO-101's 6
Feetech STS3215 servos:

| Joint | Link |
|---|---|
| `shoulder_pan` | shoulder_link |
| `shoulder_lift` | upper_arm_link |
| `elbow_flex` | lower_arm_link |
| `wrist_flex` | wrist_link |
| `wrist_roll` | gripper_link |
| `gripper` | moving_jaw_so101_v1_link |

(`gripper_frame_joint` is a fixed joint, not an actuated DOF.)

## Known Risk: URDF Physics Parameters Are Not Simulator-Tuned

CAD-exported URDFs typically carry rough/default mass, inertia, and
joint stiffness/damping values — not tuned for any particular physics
engine. Before trusting sim rollouts, the Isaac Lab articulation config
needs:

- Mass/inertia sanity-checked against real-world estimates (not left as
  CAD placeholder defaults)
- Joint torque/velocity limits set to match the real STS3215 servo specs,
  not whatever raw values are in the URDF — this matters specifically for
  the sim-to-real comparison: if simulated joints can move faster/harder
  than the real servos can, a policy trained in sim will learn behavior the
  real arm can't physically execute.

STS3215 servo specs (gearing): follower arm uses 1/345 gearing; leader arm
uses three differently-geared motors (lower gearing) so it can be moved by
hand without much resistance while still supporting its own weight. See
[project_plan.md](project_plan.md) for how this feeds into the broader plan.

## Conversion Pipeline (planned)

URDF → USD via Isaac Lab's `convert_urdf.py` utility, then a Python
articulation config (modeled on Isaac Lab's existing manipulator configs,
e.g. Franka) referencing the converted USD with corrected joint properties.
