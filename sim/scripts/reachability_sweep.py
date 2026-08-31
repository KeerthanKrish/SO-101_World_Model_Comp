# SPDX-License-Identifier: BSD-3-Clause
"""Sweeps a grid of (x, y) candidate cube positions at a fixed grasp-
approach height and checks whether the arm's IK controller actually
converges to each one -- used to empirically define the arm's real
reachable workspace on the (now 1.2m) table, instead of guessing
train/held-out position bounds for the world-model-vs-diffusion-policy
evaluation plan (see docs/evaluation_plan.md, checklist item 1).

For each grid point: resets the robot to its default pose, runs the same
DifferentialIKController used throughout this project for a bounded
number of steps toward (x, y, z), and records the final position error.
A point is "reachable" if the arm actually converges close to it, not
just if it's geometrically within the arm's maximum extent -- this
matters because some geometrically-close points are still awkward/
unreachable due to joint limits or self-collision-like configurations.

Usage:
    ./isaaclab.sh -p /path/to/reachability_sweep.py --headless
"""

import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--grasp-height", type=float, default=0.03, help="Fixed z for every candidate point.")
parser.add_argument("--x-min", type=float, default=-0.15)
parser.add_argument("--x-max", type=float, default=0.5)
parser.add_argument("--y-min", type=float, default=-0.4)
parser.add_argument("--y-max", type=float, default=0.4)
parser.add_argument("--step", type=float, default=0.05)
parser.add_argument("--ik-steps", type=int, default=150, help="IK steps allowed to converge per point.")
parser.add_argument("--converged-tol", type=float, default=0.01, help="Meters; below this counts as reachable.")
parser.add_argument(
    "--output", type=str, default="/home/keerthan/SO-101-WM/sim/output/reachability_sweep.json"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import subtract_frame_transforms

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from scenes.pickplace_scene import PickPlaceSceneBaseCfg  # isort:skip


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = PickPlaceSceneBaseCfg(num_envs=1, env_spacing=2.0)
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

    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()

    xs = []
    x = args_cli.x_min
    while x <= args_cli.x_max + 1e-9:
        xs.append(round(x, 3))
        x += args_cli.step
    ys = []
    y = args_cli.y_min
    while y <= args_cli.y_max + 1e-9:
        ys.append(round(y, 3))
        y += args_cli.step

    results = []
    total = len(xs) * len(ys)
    done = 0

    for x in xs:
        for y in ys:
            done += 1
            # reset to a clean default pose before every attempt so results
            # are comparable and don't depend on the previous target.
            robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            robot.reset()
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim_dt)

            target_pos = torch.tensor([[x, y, args_cli.grasp_height]], device=sim.device)
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            root_pose_w = robot.data.root_pose_w
            _, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            ik_controller.set_command(target_pos, ee_quat=ee_quat_b)

            for _ in range(args_cli.ik_steps):
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
                sim.step(render=False)
                scene.update(sim_dt)

            final_ee_pos_w = robot.data.body_pose_w[0, robot_entity_cfg.body_ids[0], 0:3].cpu().tolist()
            error = (
                (final_ee_pos_w[0] - x) ** 2 + (final_ee_pos_w[1] - y) ** 2 + (final_ee_pos_w[2] - args_cli.grasp_height) ** 2
            ) ** 0.5
            reachable = error < args_cli.converged_tol
            results.append({"x": x, "y": y, "error": error, "reachable": reachable})
            print(f"[{done}/{total}] ({x:+.2f}, {y:+.2f}) error={error:.4f} reachable={reachable}")

    os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)
    with open(args_cli.output, "w") as f:
        json.dump(
            {
                "grasp_height": args_cli.grasp_height,
                "converged_tol": args_cli.converged_tol,
                "results": results,
            },
            f,
            indent=2,
        )

    reachable_pts = [r for r in results if r["reachable"]]
    print(f"\n[RESULT] {len(reachable_pts)}/{len(results)} grid points reachable")
    if reachable_pts:
        xs_r = [r["x"] for r in reachable_pts]
        ys_r = [r["y"] for r in reachable_pts]
        print(f"[RESULT] Reachable x range: [{min(xs_r):.2f}, {max(xs_r):.2f}]")
        print(f"[RESULT] Reachable y range: [{min(ys_r):.2f}, {max(ys_r):.2f}]")
    print(f"[RESULT] Saved full grid to {args_cli.output}")


if __name__ == "__main__":
    main()
    simulation_app.close()
