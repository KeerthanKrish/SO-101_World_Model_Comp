# SPDX-License-Identifier: BSD-3-Clause
"""Fourth attempt at the wrist/eye-in-hand camera.

Correction from the user after round 3: the camera is mounted on the
gripper ASSEMBLY (gripper_link, the static wrist/servo housing) after
all -- it does NOT move with the moving jaw's open/close motion, it's
solidly mounted. So round 2's attachment point (gripper_link) and
analytically-derived rotation (aiming at the "gripper" joint's URDF
origin, which IS the jaw pivot direction in gripper_link's own local
frame -- no guessing needed, same as round 2) were correct. What's
suspected wrong is the STANDOFF DISTANCE: round 2 used a 9cm standoff and
got a render showing the whole gripper mechanism + the far tabletop/cube,
but the user's actual reference photos show a much tighter framing --
camera very close to the pivot, looking almost straight down into the gap
between the two fingers, with just the two fingertips converging into
frame from the bottom-left/bottom-right and the table filling the rest.

This script keeps round 2's exact rotation and direction, and only sweeps
much shorter standoff distances to find the one that reproduces that tight
framing.

Usage:
    ./isaaclab.sh -p /path/to/test_wrist_camera_angles4.py --headless --enable_cameras
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir", type=str, default="/home/keerthan/SO-101-WM/sim/output/wrist_cam_test4", help="Output dir."
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

# Same rotation as round 2 (aims camera's local +Z at the jaw pivot
# direction, computed from the "gripper" joint's URDF origin in
# gripper_link's own local frame). Unit direction = round 2's 9cm offset
# divided by 0.09 -- (-0.5583, -0.5197, 0.6468). Sweeping much shorter
# standoffs this time (1-5cm) instead of 9cm.
_ROT = (0.42027, -0.61815, 0.66412, 0.0)
_UNIT_DIR = (-0.5583, -0.5197, 0.6468)

_CANDIDATES = {}
for cm in [1, 2, 3, 4, 5]:
    d = cm / 100.0
    pos = tuple(round(c * d, 6) for c in _UNIT_DIR)
    _CANDIDATES[f"standoff_{cm}cm"] = (pos, _ROT)


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
                spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.005, 2.0)),
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
    for name in _CANDIDATES:
        cam = scene[f"cam_{name}"]
        rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy()
        out_path = os.path.join(args_cli.output_dir, f"{name}.png")
        Image.fromarray(rgb).save(out_path)
        print(f"[RESULT] Saved {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
