# SPDX-License-Identifier: BSD-3-Clause
"""Second attempt at the wrist camera, this time informed by a real
reference photo of the user's physical camera mount (see
docs/real_camera_setup.md): mounted on top of gripper_link (the
wrist/servo housing, just before the jaw pivot), angled forward and down
over the fingers -- not attached to gripper_frame_link like the first
(failed) attempt.

Tests several forward-and-down tilt candidates at once, same technique as
the first attempt's test_wrist_camera_angles.py.

Usage:
    ./isaaclab.sh -p /path/to/test_wrist_camera_angles2.py --headless --enable_cameras
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir", type=str, default="/home/keerthan/SO-101-WM/sim/output/wrist_cam_test2", help="Output dir."
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
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import subtract_frame_transforms

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from scenes.pickplace_scene import PickPlaceSceneCfg  # isort:skip

# Candidate (pos, rot) mounts attached to gripper_link, all "ros" convention
# (forward = local +Z). rot is (w, x, y, z). Varying how far forward/down
# the tilt is, small offset up and slightly forward from gripper_link's
# origin (roughly where the real webcam bracket sits on the housing).
# Round 3: rounds 1-2 were blind rotation guesses and both failed (too
# close / looking at the wrong thing entirely). This time the rotation is
# computed analytically -- same shortest-arc-rotation technique that
# worked for _JAW_OFFSET_LOCAL and the grasp-target orientation earlier --
# from a URDF measurement we already have: the "gripper" joint's <origin>
# (0.0202, 0.0188, -0.0234) IS the jaw pivot's position directly in
# gripper_link's own local frame (URDF joint origins are relative to the
# parent link, and gripper_link is the parent of the gripper joint) -- no
# guessing needed. Rotation sends local +Z (ROS "forward") onto that
# normalized direction; position backs off along the opposite direction by
# a few candidate standoff distances (matching the visible real bracket
# height in the reference photo).
_CANDIDATES = {
    "standoff_5cm": ((-0.02792, -0.02598, 0.03234), (0.42027, -0.61815, 0.66412, 0.0)),
    "standoff_7cm": ((-0.03908, -0.03637, 0.04527), (0.42027, -0.61815, 0.66412, 0.0)),
    "standoff_9cm": ((-0.05025, -0.04677, 0.05821), (0.42027, -0.61815, 0.66412, 0.0)),
}


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = PickPlaceSceneCfg(num_envs=1, env_spacing=2.0)
    for name, (pos, rot) in _CANDIDATES.items():
        setattr(
            scene_cfg,
            f"cam_{name}",
            CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/gripper_link/Cam_{name}",
                offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="ros"),
                spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.01, 2.0)),
                width=320,
                height=320,
                data_types=["rgb"],
            ),
        )

    scene = InteractiveScene(scene_cfg)
    sim.reset()

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
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    sim.render()
    for name in _CANDIDATES:
        cam = scene[f"cam_{name}"]
        rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
        out_path = os.path.join(args_cli.output_dir, f"{name}.png")
        Image.fromarray(rgb).save(out_path)
        print(f"[RESULT] Saved {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
