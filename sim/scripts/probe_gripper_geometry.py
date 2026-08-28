# SPDX-License-Identifier: BSD-3-Clause
"""Probe the SO-101's actual gripper geometry instead of guessing it.

Commands the gripper joint to both ends of its range, lets it settle, and
reports the world position of the moving jaw link relative to the tracked
IK frame ("gripper_frame_link") at each extreme. This tells us:
  1. Which joint value actually closes the jaws (vs. opens them).
  2. The real offset between gripper_frame_link and where the jaws meet,
     so grasp-height waypoints can be computed instead of guessed.

Usage:
    ./isaaclab.sh -p /path/to/probe_gripper_geometry.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe SO-101 gripper geometry.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import sys

import torch
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from robots.so101 import SO101_CFG  # isort:skip

GRIPPER_LOWER = -0.174533
GRIPPER_UPPER = 1.74533


def settle(robot, sim, sim_dt, joint_target, steps=120):
    for _ in range(steps):
        robot.set_joint_position_target(joint_target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)


def world_bbox(stage, prim_path):
    """World-space axis-aligned bounding box of a prim (and its children)."""
    prim = stage.GetPrimAtPath(prim_path)
    bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_], useExtentsHint=True)
    bbox = bbox_cache.ComputeWorldBound(prim)
    box_range = bbox.ComputeAlignedRange()
    return Gf.Vec3d(box_range.GetMin()), Gf.Vec3d(box_range.GetMax())


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=3000.0))

    robot_cfg = SO101_CFG.copy()
    robot_cfg.prim_path = "/World/SO101"
    robot = Articulation(cfg=robot_cfg)

    sim.reset()
    sim_dt = sim.get_physics_dt()

    body_names = robot.data.body_names
    print("[INFO]: Body names:", body_names)
    jaw_idx = body_names.index("moving_jaw_so101_v1_link")
    frame_idx = body_names.index("gripper_frame_link")

    joint_names = robot.data.joint_names
    gripper_joint_idx = joint_names.index("gripper")

    default_pos = robot.data.default_joint_pos.clone()

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim_paths = {
        "gripper_frame_link": "/World/SO101/gripper_frame_link",
        "gripper_link": "/World/SO101/gripper_link",
        "moving_jaw": "/World/SO101/moving_jaw_so101_v1_link",
    }

    for label, value in [("LOWER", GRIPPER_LOWER), ("UPPER", GRIPPER_UPPER)]:
        target = default_pos.clone()
        target[:, gripper_joint_idx] = value
        settle(robot, sim, sim_dt, target)

        frame_pos = robot.data.body_pos_w[0, frame_idx].cpu().numpy()
        jaw_pos = robot.data.body_pos_w[0, jaw_idx].cpu().numpy()
        print(f"[RESULT] gripper={label} ({value:.4f}): frame_origin={frame_pos}, jaw_origin={jaw_pos}")

        for name, path in prim_paths.items():
            bmin, bmax = world_bbox(stage, path)
            center = [(bmin[i] + bmax[i]) / 2 for i in range(3)]
            print(f"[RESULT]   {name} bbox: min={list(bmin)}, max={list(bmax)}, center={center}")

    print("[RESULT] Probe complete.")


if __name__ == "__main__":
    main()
    simulation_app.close()
