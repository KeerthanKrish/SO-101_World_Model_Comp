# SPDX-License-Identifier: BSD-3-Clause
"""Empirically determine what SIM's positive/negative direction physically
means for each of the 6 SO-101 joints -- extends calibrate_grasp.py's own
"render both extremes, look at the image, don't guess from geometry" method
(which only ever covered the gripper joint) to all 6.

Motivation: this project has been directly burned before by trusting
derived/assumed geometry over rendered ground truth -- see
sim/robots/grasp_geometry.py's own history, where `jaw_approach_axis_world()`
pointed the wrong physical direction (pivot->wrist instead of pivot->
fingertips) for the entire project until an STL-mesh measurement caught it.
The URDF's raw axis="0 0 1" + origin rpy=... math for each joint COULD be
worked out by hand to determine what a positive joint angle means in world
space, but that's exactly the kind of derivation that's easy to get wrong
without rendering it and looking -- so this renders it instead, the same
discipline already applied to the gripper.

This is needed now specifically to cross-check against the REAL follower
arm's just-measured joint sign convention (2026-09-13, read off via
sim/scripts/follower_reader.py with torque disabled) -- to determine
whether a policy's action output (trained against SIM's joint-value sign
convention) needs any per-joint negation before being sent to the real
follower, or transfers as-is. Pure sim -- no physical hardware risk from
running this.

For each of the 5 arm joints, holds the other 4 at the "new_calib" zero
pose (URDF's own middle-of-range convention, same as PickPlaceEnvCfg's
default reset pose) and sweeps that one joint to its lower limit, then its
upper limit, rendering an image at each extreme before returning it to 0.
The gripper is included too (redundant with calibrate_grasp.py, which
already established open=upper/closed=lower, but costs nothing to
reconfirm here in the same unified, one-shot report).

Usage:
    ./isaaclab.sh -p /path/to/calibrate_joint_signs.py --headless --enable_cameras
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Empirically check SO-101 per-joint positive-direction convention.")
parser.add_argument(
    "--output-dir", type=str, default="/home/keerthan/SO-101-WM/sim/output/joint_sign_calibration", help="Output dir."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Force line-buffered stdout -- see calibrate_wrist_roll_sign.py's own
# comment on this same fix for the full reasoning (this box's known Kit
# shutdown-hang can strand unflushed RESULT prints in a full-buffered
# stdout indefinitely).
sys.stdout.reconfigure(line_buffering=True)

import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from scenes.pickplace_scene import PickPlaceSceneCfg  # isort:skip

# (joint_name, lower_limit, upper_limit) -- exact values from
# so101_new_calib.urdf, confirmed directly against the URDF source
# (assets/SO-ARM100/Simulation/SO101/so101_new_calib.urdf), not re-derived
# or assumed from any other file.
_JOINTS = [
    ("shoulder_pan", -1.91986, 1.91986),
    ("shoulder_lift", -1.74533, 1.74533),
    ("elbow_flex", -1.69, 1.69),
    ("wrist_flex", -1.65806, 1.65806),
    ("wrist_roll", -2.74385, 2.84121),
    ("gripper", -0.174533, 1.74533),
]


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = PickPlaceSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    camera = scene["scene_camera"]
    # Pulled back and centered on the arm's base (not the grasp point, unlike
    # calibrate_grasp.py's close-up framing) -- needs to keep the WHOLE arm
    # in frame across its full range of motion for all 6 joints, including
    # shoulder_pan's +-110 degree base swing, not just a fixed grasp pose.
    eye = torch.tensor([[0.55, -0.55, 0.45]], device=sim.device)
    look_at = torch.tensor([[0.15, 0.0, 0.1]], device=sim.device)
    camera.set_world_poses_from_view(eye, look_at)

    joint_names = robot.data.joint_names
    sim_dt = sim.get_physics_dt()

    def hold_pose(joint_pos_dict, steps=90):
        target = torch.zeros((1, len(joint_names)), device=sim.device)
        for name, val in joint_pos_dict.items():
            target[0, joint_names.index(name)] = val
        for _ in range(steps):
            robot.set_joint_position_target(target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

    def capture(label):
        sim.render()
        rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        out_path = os.path.join(args_cli.output_dir, f"{label}.png")
        Image.fromarray(rgb).save(out_path)
        print(f"[RESULT] Saved {out_path}")

    # Establish the neutral ("new_calib" zero, all-joints-0) reference pose
    # first, and capture it once -- every per-joint sweep below returns to
    # this exact pose before testing the next joint, so each extreme image
    # is a clean, single-joint-changed comparison against this same baseline,
    # not a compounding pose from the previous joint's test.
    hold_pose({}, steps=120)
    capture("00_neutral_all_zero")

    for name, lower, upper in _JOINTS:
        for label, value in [("lower", lower), ("upper", upper)]:
            hold_pose({name: value})
            capture(f"{name}_{label}")
        # Return this joint to 0 before moving on to the next one.
        hold_pose({})

    print("[RESULT] No crash -- all 6 joints swept to both limits and captured.")


if __name__ == "__main__":
    main()
    simulation_app.close()
