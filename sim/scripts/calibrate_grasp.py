# SPDX-License-Identifier: BSD-3-Clause
"""Position the arm near the cube via IK, then render one image with the
gripper joint at its lower limit and one at its upper limit, to visually
determine which direction actually closes the jaws (and whether the
approach height looks right) -- an empirical check instead of guessing.

Usage:
    ./isaaclab.sh -p /path/to/calibrate_grasp.py --headless --enable_cameras
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Empirically check SO-101 gripper open/close direction.")
parser.add_argument("--ee-z", type=float, default=0.06, help="Target EE height above table for the check.")
parser.add_argument(
    "--output-dir", type=str, default="/home/keerthan/SO-101-WM/sim/output/grasp_calibration", help="Output dir."
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
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from scenes.pickplace_scene import PickPlaceSceneCfg  # isort:skip

# See run_pickplace_demo.py for where this offset comes from.
_JAW_OFFSET_LOCAL = (-0.0281, 0.019018, -0.0747274)


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = PickPlaceSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    camera = scene["scene_camera"]
    wrist_camera = scene["wrist_camera"]
    eye = torch.tensor([[0.35, -0.35, 0.35]], device=sim.device)
    look_at = torch.tensor([[0.2, 0.0, 0.03]], device=sim.device)
    camera.set_world_poses_from_view(eye, look_at)

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

    # move to the candidate grasp position first, gripper open-ish (mid-range, neutral)
    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    root_pose_w = robot.data.root_pose_w
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    target_pos = torch.tensor([[0.2, 0.0, args_cli.ee_z]], device=sim.device)
    ik_controller.set_command(target_pos, ee_quat=ee_quat_b)

    for step in range(200):
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    print("[INFO]: Reached candidate grasp position. EE pos (world):", ee_pose_w[0, 0:3].cpu().numpy())

    body_names = robot.data.body_names
    jaw_idx = body_names.index("moving_jaw_so101_v1_link")
    frame_idx = body_names.index("gripper_frame_link")
    frame_pos = robot.data.body_pos_w[0, frame_idx].cpu().numpy()
    jaw_pos = robot.data.body_pos_w[0, jaw_idx].cpu().numpy()
    print(f"[RESULT] At this pose: gripper_frame_link pos={frame_pos}, jaw_pivot pos={jaw_pos}")
    print(f"[RESULT] Offset (jaw_pivot - gripper_frame_link) = {jaw_pos - frame_pos}")

    for label, value in [("lower", -0.174533), ("upper", 1.74533)]:
        for _ in range(90):
            robot.set_joint_position_target(
                torch.tensor([[value]], device=sim.device), joint_ids=gripper_joint_id
            )
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
        sim.render()
        rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        out_path = os.path.join(args_cli.output_dir, f"gripper_{label}.png")
        Image.fromarray(rgb).save(out_path)
        print(f"[RESULT] Saved {out_path}")

        sim.render()
        wrist_rgb = wrist_camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        wrist_out_path = os.path.join(args_cli.output_dir, f"wrist_{label}.png")
        Image.fromarray(wrist_rgb).save(wrist_out_path)
        print(f"[RESULT] Saved {wrist_out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
