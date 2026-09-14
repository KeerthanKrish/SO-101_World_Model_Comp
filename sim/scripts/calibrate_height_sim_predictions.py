# SPDX-License-Identifier: BSD-3-Clause
"""Height-calibration step 1 of 2: for a small set of KNOWN joint
configurations (the same ones follower_replay_height_check.py will
command on the real follower), compute what SIM predicts the gripper's
grasp point height (above the table, which sits at z=0 -- see
pickplace_scene.py's own comment) should be at each one.

This is the sim-side half of isolating the ~2-3 inch gap found during
the first real reach-trajectory replay: that test only compared a
single point (the segment's own cutoff) against a proxy (the recorded
cube's height, not the gripper's own), and only informally ("about 2-3
inches" by eye, not measured). This uses 4 poses spanning a real range
of the workspace, and the actual jaw-pivot grasp point
(grasp_point_world(), the same reference this whole project already
uses everywhere else -- reward shaping, MimicGen work -- not an
arbitrary body frame), so the comparison against real measurements is
apples-to-apples with everything else in this project.

Poses: the "new_calib" neutral (all joints 0), plus 3 real, already-
validated joint configurations from episode_002's own reach segment
(steps 0, 97, 194 of the downsampled recording -- start, middle, and the
exact cutoff the first real replay stopped at and the user measured
"2-3 inches above the table" for by eye).

Usage:
    ./isaaclab.sh -p /path/to/calibrate_height_sim_predictions.py --headless --enable_cameras
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir", type=str, default="/home/keerthan/SO-101-WM/sim/output/joint_sign_calibration", help="Output dir for reference images."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import sys  # isort:skip

sys.stdout.reconfigure(line_buffering=True)

import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from robots.grasp_geometry import grasp_point_world  # isort:skip
from scenes.pickplace_scene import PickPlaceSceneCfg  # isort:skip

_JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Same 4 poses follower_replay_height_check.py commands on the real
# follower -- "neutral" plus 3 real, already-validated frames from
# episode_002's own reach segment (steps 0/97/194 of the downsampled
# recording), pulled directly from that file, not re-typed by hand.
_POSES = {
    "neutral": {j: 0.0 for j in _JOINT_ORDER},
    "reach_start": {
        "shoulder_pan": -1.2965, "shoulder_lift": 0.2562, "elbow_flex": 0.0215,
        "wrist_flex": 1.1316, "wrist_roll": -1.015, "gripper": 0.5875,
    },
    "reach_mid": {
        "shoulder_pan": -1.2935, "shoulder_lift": 0.2532, "elbow_flex": 0.0368,
        "wrist_flex": 1.0779, "wrist_roll": -1.0671, "gripper": 0.586,
    },
    "reach_end": {
        "shoulder_pan": -1.3165, "shoulder_lift": 0.0875, "elbow_flex": 0.0338,
        "wrist_flex": 1.2543, "wrist_roll": -1.1254, "gripper": 0.1563,
    },
}


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = PickPlaceSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    camera = scene["scene_camera"]
    joint_names = robot.data.joint_names
    body_names = robot.data.body_names
    ee_body_idx = body_names.index("gripper_frame_link")
    sim_dt = sim.get_physics_dt()

    eye = torch.tensor([[0.55, -0.55, 0.45]], device=sim.device)
    look_at = torch.tensor([[0.15, 0.0, 0.1]], device=sim.device)
    camera.set_world_poses_from_view(eye, look_at)

    def hold_pose(joint_pos_dict, steps=120):
        target = torch.zeros((1, len(joint_names)), device=sim.device)
        for name, val in joint_pos_dict.items():
            target[0, joint_names.index(name)] = val
        for _ in range(steps):
            robot.set_joint_position_target(target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

    print("[RESULT] SIM-predicted grasp-point height (z, meters above the table, table at z=0):")
    for label, pose in _POSES.items():
        hold_pose(pose)
        ee_pos_w = robot.data.body_pos_w[0, ee_body_idx].cpu().tolist()
        ee_quat_w = robot.data.body_quat_w[0, ee_body_idx].cpu().tolist()
        grasp_pos = grasp_point_world(ee_pos_w, ee_quat_w)
        print(f"  {label}: grasp_point = ({grasp_pos[0]:+.4f}, {grasp_pos[1]:+.4f}, {grasp_pos[2]:+.4f}) "
              f"-> height={grasp_pos[2]:+.4f} m")

        sim.render()
        rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        out_path = os.path.join(args_cli.output_dir, f"height_check_{label}.png")
        Image.fromarray(rgb).save(out_path)
        print(f"  [RESULT] Saved {out_path}")

    print("[RESULT] No crash -- all 4 poses captured.")


if __name__ == "__main__":
    main()
    simulation_app.close()
