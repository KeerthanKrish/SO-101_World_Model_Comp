# SPDX-License-Identifier: BSD-3-Clause
"""Fifth attempt at the wrist/eye-in-hand camera.

Rounds 2-4 aimed the camera at the "gripper" joint's URDF origin -- the
position of the MOVING JAW'S PIVOT in gripper_link's local frame. That
was the wrong target: it points at a hinge point near the base, not down
the length of the fingers toward the grasp point, which is why round 2's
9cm render showed a foreshortened view of the whole mechanism (see
wrist_cam_test2/standoff_9cm.png) instead of the reference photos' tight
view of two fingers converging with the table filling the rest.

Better direction, still free (no guessing): gripper_link already has a
second fixed child frame, "gripper_frame_link", attached via
gripper_frame_joint with URDF origin (-0.0079, -0.000218121, -0.0981274)
-- this is Isaac Lab's actual IK end-effector frame, i.e. the grasp
point out at the fingertips. That origin is overwhelmingly along local
-Z (normalized approx (-0.08, -0.002, -1.0)) -- so "down the fingers,
toward the grasp point" is essentially local -Z of gripper_link, not the
direction used in rounds 2-4.

This time: aim the camera's forward axis at that fingertip direction, and
sweep candidate "up" axes/standoffs/tilts for the perpendicular mounting
offset (the axis the real bracket clips onto), same trial approach as
round 3 since we don't have an analytical shortcut for that part.

Usage:
    ./isaaclab.sh -p /path/to/test_wrist_camera_angles5.py --headless --enable_cameras
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir", type=str, default="/home/keerthan/SO-101-WM/sim/output/wrist_cam_test5", help="Output dir."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
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


def _shortest_arc_quat(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    xyz = np.cross(a, b)
    w = 1.0 + float(np.dot(a, b))
    q = np.array([w, xyz[0], xyz[1], xyz[2]])
    return q / np.linalg.norm(q)


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _axis_angle_quat(axis, angle_rad):
    axis = axis / np.linalg.norm(axis)
    s = math.sin(angle_rad / 2.0)
    return np.array([math.cos(angle_rad / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


# Direction from gripper_link's origin to gripper_frame_link (the fingertip
# / grasp point) -- from gripper_frame_joint's URDF <origin>.
_FWD = np.array([-0.0079, -0.000218121, -0.0981274])
_FWD = _FWD / np.linalg.norm(_FWD)
_BASE_ROT = _shortest_arc_quat(np.array([0.0, 0.0, 1.0]), _FWD)

_CANDIDATES = {}
for up_name, up_axis in [
    ("posx", np.array([1.0, 0.0, 0.0])),
    ("negx", np.array([-1.0, 0.0, 0.0])),
    ("posy", np.array([0.0, 1.0, 0.0])),
    ("negy", np.array([0.0, -1.0, 0.0])),
]:
    for tilt_deg in [0, 20, 35]:
        tilt_axis = np.cross(_FWD, up_axis)
        if np.linalg.norm(tilt_axis) < 1e-6:
            continue
        # tilt forward AWAY from up_axis (i.e. toward "down") by tilt_deg
        tilt_q = _axis_angle_quat(tilt_axis, -math.radians(tilt_deg))
        rot = _quat_mul(tilt_q, _BASE_ROT)
        pos = 0.02 * up_axis + _FWD * 0.01
        name = f"{up_name}_tilt{tilt_deg}"
        _CANDIDATES[name] = (tuple(pos.tolist()), tuple(rot.tolist()))


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
