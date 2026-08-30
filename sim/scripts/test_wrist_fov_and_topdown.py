# SPDX-License-Identifier: BSD-3-Clause
"""Tests two things at once, both requested after reviewing the round-6
wrist camera render:

1. Wider wrist camera FOV -- user confirmed the negy_7cm framing/direction
   is right, just wants to see more of the scene. Sweeps a few shorter
   focal lengths (smaller focal length = wider FOV for a fixed sensor
   size) against the exact pos/rot already baked into pickplace_scene.py.
2. A fixed, straight-down top-down camera matching the user's real
   top-down reference photo (arm mounted at one edge, looking straight
   down over the workspace, arm entering from the top of frame). Computed
   analytically: camera looks along world -Z with image "up" (toward
   frame top, where the arm is) aligned to world -X (the direction back
   toward the robot base from the workspace center) -- this is a fixed
   180-degree rotation about the world (1,1,0) axis, quaternion
   (0, 0.70710678, 0.70710678, 0). Position kept at the existing
   top_camera's (0.15, 0.0, 0.55) since that x is already between the
   robot base and the cube.

Usage:
    ./isaaclab.sh -p /path/to/test_wrist_fov_and_topdown.py --headless --enable_cameras
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir", type=str, default="/home/keerthan/SO-101-WM/sim/output/wrist_fov_topdown_test", help="Output dir."
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

# Same pos/rot as the final wrist_camera config -- only focal_length varies.
_WRIST_POS = (-0.00080248, -0.07002216, -0.00996772)
_WRIST_ROT = (0.03478796, 0.02019336, -0.98402225, -0.17344234)
_WRIST_FOCAL_LENGTHS = {"fov_fl10": 10.0, "fov_fl8": 8.0, "fov_fl6": 6.0}

_TOPDOWN_POS = (0.15, 0.0, 0.85)
_TOPDOWN_ROT = (0.0, 0.70710678, 0.70710678, 0.0)


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = PickPlaceSceneCfg(num_envs=1, env_spacing=2.0)
    for name, fl in _WRIST_FOCAL_LENGTHS.items():
        setattr(
            scene_cfg,
            f"cam_{name}",
            CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/gripper_link/Cam_{name}",
                offset=CameraCfg.OffsetCfg(pos=_WRIST_POS, rot=_WRIST_ROT, convention="ros"),
                spawn=sim_utils.PinholeCameraCfg(focal_length=fl, clipping_range=(0.005, 2.0)),
                width=320,
                height=320,
                data_types=["rgb"],
            ),
        )
    for fl_name, fl in {"td_fl8": 8.0, "td_fl10": 10.0, "td_fl12": 12.0, "td_fl16": 16.0}.items():
        setattr(
            scene_cfg,
            f"cam_{fl_name}",
            CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Cam_{fl_name}",
                offset=CameraCfg.OffsetCfg(pos=_TOPDOWN_POS, rot=_TOPDOWN_ROT, convention="ros"),
                spawn=sim_utils.PinholeCameraCfg(focal_length=fl, clipping_range=(0.05, 5.0)),
                width=960,
                height=540,
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
    for name in list(_WRIST_FOCAL_LENGTHS.keys()) + ["td_fl8", "td_fl10", "td_fl12", "td_fl16"]:
        cam = scene[f"cam_{name}"]
        rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
        out_path = os.path.join(args_cli.output_dir, f"{name}.png")
        Image.fromarray(rgb).save(out_path)
        print(f"[RESULT] Saved {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
