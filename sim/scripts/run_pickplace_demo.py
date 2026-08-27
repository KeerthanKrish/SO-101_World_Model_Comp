# SPDX-License-Identifier: BSD-3-Clause
"""Scripted (non-learned) reach/grasp/place demo for the SO-101, using
Isaac Lab's differential IK controller to command Cartesian end-effector
targets instead of guessing joint angles directly.

This is a sanity check, not a finished policy: the waypoint positions and
gripper open/closed joint values below are initial guesses (see inline
comments) and will likely need tuning once we can see the rendered result.

Saves one RGB frame every --frame-interval sim steps to --frames-dir; use
stitch_video.py afterward to turn those into an mp4.

Usage:
    ./isaaclab.sh -p /path/to/run_pickplace_demo.py --headless --enable_cameras
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted SO-101 pick-and-place demo.")
parser.add_argument("--frame-interval", type=int, default=4, help="Save a frame every N sim steps.")
parser.add_argument(
    "--frames-dir", type=str, default="/home/keerthan/SO-101-WM/sim/output/pickplace_frames", help="Frame output dir."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import subtract_frame_transforms

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from scenes.pickplace_scene import PickPlaceSceneCfg  # isort:skip

# Joint value guesses (URDF range is -0.174533 to 1.74533) -- direction of
# open vs. closed is NOT yet verified against real behavior. Check rendered
# frames and flip if backwards.
_GRIPPER_OPEN = 1.5
_GRIPPER_CLOSED = -0.1

# Waypoints in the robot root frame: (target_ee_pos_xyz, gripper_cmd, hold_steps).
# Heights/positions are initial guesses -- table top is z=0, cube center at
# (0.2, 0.0, 0.015), place target arbitrarily chosen at (-0.15, 0.15, *).
_WAYPOINTS = [
    ((0.2, 0.0, 0.20), _GRIPPER_OPEN, 60),  # approach above cube
    ((0.2, 0.0, 0.06), _GRIPPER_OPEN, 60),  # descend to grasp height
    ((0.2, 0.0, 0.06), _GRIPPER_CLOSED, 60),  # close gripper
    ((0.2, 0.0, 0.20), _GRIPPER_CLOSED, 60),  # lift
    ((-0.15, 0.15, 0.20), _GRIPPER_CLOSED, 90),  # move to place zone
    ((-0.15, 0.15, 0.06), _GRIPPER_CLOSED, 60),  # descend
    ((-0.15, 0.15, 0.06), _GRIPPER_OPEN, 60),  # release
    ((-0.15, 0.15, 0.20), _GRIPPER_OPEN, 60),  # retreat
]


def main():
    os.makedirs(args_cli.frames_dir, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = PickPlaceSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete.")

    robot = scene["robot"]
    camera = scene["scene_camera"]
    eye = torch.tensor([[0.6, -0.6, 0.5]], device=sim.device)
    target = torch.tensor([[0.15, 0.0, 0.05]], device=sim.device)
    camera.set_world_poses_from_view(eye, target)

    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=sim.device)

    robot_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        body_names=["gripper_frame_link"],
    )
    robot_entity_cfg.resolve(scene)
    gripper_joint_id, _ = robot.find_joints("gripper")

    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]

    sim_dt = sim.get_physics_dt()
    step_count = 0
    frame_count = 0

    def save_frame():
        nonlocal frame_count
        rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        Image.fromarray(rgb).save(os.path.join(args_cli.frames_dir, f"frame_{frame_count:05d}.png"))
        frame_count += 1

    for waypoint_pos, gripper_cmd, hold_steps in _WAYPOINTS:
        # set the new IK target using the current end-effector orientation (position-only command)
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        root_pose_w = robot.data.root_pose_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        target_pos = torch.tensor([waypoint_pos], device=sim.device)
        ik_controller.reset()
        ik_controller.set_command(target_pos, ee_quat=ee_quat_b)

        for _ in range(hold_steps):
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
            robot.set_joint_position_target(
                torch.tensor([[gripper_cmd]], device=sim.device), joint_ids=gripper_joint_id
            )
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            if step_count % args_cli.frame_interval == 0:
                sim.render()
                save_frame()
            step_count += 1

    print(f"[RESULT] Demo complete. Saved {frame_count} frames to {args_cli.frames_dir}")
    cube_pos = scene["cube"].data.root_pos_w[0].cpu().numpy()
    print(f"[RESULT] Final cube position: {cube_pos}")


if __name__ == "__main__":
    main()
    simulation_app.close()
