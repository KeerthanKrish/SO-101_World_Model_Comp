# SPDX-License-Identifier: BSD-3-Clause
"""Build the SO-101 pick-and-place scene and hold it running with live
WebRTC streaming enabled, so it can be viewed remotely via the Isaac Sim
WebRTC Streaming Client.

This is intentionally a long-running process (not bounded like the sanity
-check scripts) -- it's meant to be watched live. Stop it with Ctrl+C or by
killing the process by PID when done viewing.

Usage:
    ./isaaclab.sh -p /path/to/stream_scene.py --livestream 2 --enable_cameras
"""

import sys

from isaaclab.app import AppLauncher

import argparse

parser = argparse.ArgumentParser(description="Stream the SO-101 pick-and-place scene live.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from scenes.pickplace_scene import PickPlaceSceneCfg  # isort:skip


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = PickPlaceSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete. Streaming scene -- connect a viewer to see it.")

    camera = scene["scene_camera"]
    import torch

    eye = torch.tensor([[0.6, -0.6, 0.5]], device=sim.device)
    target = torch.tensor([[0.15, 0.0, 0.05]], device=sim.device)
    camera.set_world_poses_from_view(eye, target)

    sim_dt = sim.get_physics_dt()
    robot = scene["robot"]
    default_joint_pos = robot.data.default_joint_pos.clone()

    while simulation_app.is_running():
        robot.set_joint_position_target(default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
