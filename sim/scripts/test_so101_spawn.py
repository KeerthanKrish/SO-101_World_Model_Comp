# SPDX-License-Identifier: BSD-3-Clause
"""Sanity-check script: spawn the SO-101 in an empty scene and step physics.

Runs a bounded number of steps (not an infinite loop) and reports whether
joint states stay finite/bounded, as a basic check that the URDF->USD
conversion and articulation config produce stable physics before building
the actual task scene.

Usage:
    ./isaaclab.sh -p /path/to/test_so101_spawn.py --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sanity-check SO-101 spawn and physics stability.")
parser.add_argument("--num-steps", type=int, default=300, help="Number of physics steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from robots.so101 import SO101_CFG  # isort:skip


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 1.5, 1.5], [0.0, 0.0, 0.2])

    # Ground plane + light
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot_cfg = SO101_CFG.copy()
    robot_cfg.prim_path = "/World/SO101"
    robot = Articulation(cfg=robot_cfg)

    sim.reset()
    print("[INFO]: Setup complete. Joint names:", robot.data.joint_names)

    sim_dt = sim.get_physics_dt()
    default_joint_pos = robot.data.default_joint_pos.clone()

    max_abs_vel = 0.0
    saw_nan = False

    for step in range(args_cli.num_steps):
        robot.set_joint_position_target(default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel
        if torch.isnan(joint_pos).any() or torch.isnan(joint_vel).any():
            saw_nan = True
            print(f"[ERROR] NaN detected at step {step}")
            break
        max_abs_vel = max(max_abs_vel, joint_vel.abs().max().item())

    print(f"[RESULT] Ran {step + 1} steps. Max |joint velocity| observed: {max_abs_vel:.4f} rad/s. NaN: {saw_nan}")
    print(f"[RESULT] Final joint positions: {robot.data.joint_pos}")


if __name__ == "__main__":
    main()
    simulation_app.close()
