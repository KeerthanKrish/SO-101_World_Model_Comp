# SPDX-License-Identifier: BSD-3-Clause
"""Spawn the SO-101 near the cube, then test several candidate wrist-camera
mount rotations at once (rendering one image per candidate) so the correct
one can be picked by inspection instead of guessing one at a time across
full simulation runs.

Usage:
    ./isaaclab.sh -p /path/to/test_wrist_camera_angles.py --headless --enable_cameras
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test candidate wrist camera mount rotations.")
parser.add_argument(
    "--output-dir", type=str, default="/home/keerthan/SO-101-WM/sim/output/wrist_cam_test", help="Output dir."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import subtract_frame_transforms

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from scenes.pickplace_scene import PickPlaceSceneCfg  # isort:skip

# Candidate (pos, rot) mounts to try, all attached to gripper_frame_link.
# rot is (w, x, y, z). Positions back off along different local axes since
# we don't yet know which one points toward the jaw.
_CANDIDATES = {
    "negz_rotx180": ((0.0, 0.0, 0.12), (0.0, 1.0, 0.0, 0.0)),
    "posz_identity": ((0.0, 0.0, -0.12), (1.0, 0.0, 0.0, 0.0)),
    "negx_roty90": ((0.12, 0.0, 0.0), (0.7071, 0.0, 0.7071, 0.0)),
    "posx_rotyneg90": ((-0.12, 0.0, 0.0), (0.7071, 0.0, -0.7071, 0.0)),
    "negy_rotxneg90": ((0.0, 0.12, 0.0), (0.7071, -0.7071, 0.0, 0.0)),
    "posy_rotx90": ((0.0, -0.12, 0.0), (0.7071, 0.7071, 0.0, 0.0)),
    # Analytically computed: quaternion rotating +Z onto the actual
    # (normalized) jaw direction from gripper_frame_link, via the standard
    # shortest-arc-rotation formula (q.xyz = a x b, q.w = 1 + a.b, then
    # normalize, for a=(0,0,1), b=jaw direction). Position backs off 12cm
    # along the opposite (negative) direction so the camera sits behind the
    # frame looking toward the jaw.
    "analytic": ((0.041, -0.0278, 0.109), (0.213, -0.548, -0.809, 0.0)),
}


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = PickPlaceSceneCfg(num_envs=1, env_spacing=2.0)
    # add all candidate cameras to the scene cfg dynamically
    for name, (pos, rot) in _CANDIDATES.items():
        setattr(
            scene_cfg,
            f"cam_{name}",
            CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/gripper_frame_link/Cam_{name}",
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
