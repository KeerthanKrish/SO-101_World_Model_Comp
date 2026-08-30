# SPDX-License-Identifier: BSD-3-Clause
"""Replay a captured teleop episode (from segment_teleop_episodes.py)
open-loop in the sim -- no live leader arm or human input needed. Lets us
autonomously reproduce a demonstrated pick-and-place, given the exact same
starting conditions the demo was recorded under (fixed cube position).

This is NOT the same as generalizing the demo to new object positions --
it replays the exact joint trajectory verbatim, so it only reproduces the
same result under the same starting conditions. Generalizing to varied
cube positions would need closed-loop re-targeting (e.g. IK), a further
step beyond this.

Usage:
    ./isaaclab.sh -p /path/to/replay_episode.py --episode /path/to/episode_000.json [--repeat 3] [--enable_cameras]
"""

import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a captured teleop episode open-loop.")
parser.add_argument("--episode", type=str, required=True, help="Path to an episode_*.json file.")
parser.add_argument("--repeat", type=int, default=1, help="Number of times to replay the episode.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from scenes.pickplace_scene import PickPlaceSceneBaseCfg  # isort:skip

_JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    with open(args_cli.episode) as f:
        episode = json.load(f)
    print(f"[INFO]: Loaded episode with {len(episode)} steps from {args_cli.episode}")

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = PickPlaceSceneBaseCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    joint_names = robot.data.joint_names
    joint_indices = [joint_names.index(j) for j in _JOINT_ORDER]
    sim_dt = sim.get_physics_dt()

    default_root_state = scene["cube"].data.default_root_state.clone()

    for rep in range(args_cli.repeat):
        # reset cube and robot to their default starting states before each replay
        scene["cube"].write_root_pose_to_sim(default_root_state[:, :7])
        scene["cube"].write_root_velocity_to_sim(default_root_state[:, 7:])
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.reset()
        scene.reset()

        for step in episode:
            target = torch.tensor([[step["joint_pos"][j] for j in _JOINT_ORDER]], device=sim.device)
            robot.set_joint_position_target(target, joint_ids=joint_indices)
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim_dt)

        final_cube_pos = scene["cube"].data.root_pos_w[0].cpu().tolist()
        print(f"[RESULT] Replay {rep + 1}/{args_cli.repeat}: final cube position = {final_cube_pos}")


if __name__ == "__main__":
    main()
    simulation_app.close()
