# SPDX-License-Identifier: BSD-3-Clause
"""Renders all four cameras defined in pickplace_scene.py in one pass:
scene_camera and side_camera (aimed at runtime via set_world_poses_from_view,
same as capture_scene_image.py), plus the final wrist_camera and top_camera
(fixed pose baked into the scene config).

Usage:
    ./isaaclab.sh -p /path/to/capture_all_cameras.py --headless --enable_cameras
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir", type=str, default="/home/keerthan/SO-101-WM/sim/output/all_cameras", help="Output dir."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

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


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = PickPlaceSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    scene["scene_camera"].set_world_poses_from_view(
        torch.tensor([[0.6, -0.6, 0.5]], device=sim.device), torch.tensor([[0.15, 0.0, 0.05]], device=sim.device)
    )
    scene["side_camera"].set_world_poses_from_view(
        torch.tensor([[0.2, -0.5, 0.1]], device=sim.device), torch.tensor([[0.2, 0.0, 0.03]], device=sim.device)
    )

    robot = scene["robot"]
    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=sim.device)

    robot_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        body_names=["gripper_frame_link"],
    )
    robot_entity_cfg.resolve(scene)
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]
    sim_dt = sim.get_physics_dt()

    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    root_pose_w = robot.data.root_pose_w
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    target_pos = torch.tensor([[0.2, 0.0, 0.05]], device=sim.device)
    ik_controller.set_command(target_pos, ee_quat=ee_quat_b)

    gripper_joint_id = robot.data.joint_names.index("gripper")

    for _ in range(200):
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        robot.set_joint_position_target(torch.tensor([[0.7]], device=sim.device), joint_ids=[gripper_joint_id])
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    sim.render()
    for name in ["scene_camera", "side_camera", "top_camera", "wrist_camera"]:
        cam = scene[name]
        rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
        out_path = os.path.join(args_cli.output_dir, f"{name}.png")
        Image.fromarray(rgb).save(out_path)
        print(f"[RESULT] Saved {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
