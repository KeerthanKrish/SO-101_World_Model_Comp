# SPDX-License-Identifier: BSD-3-Clause
"""Smoke test for the new PickPlaceEnv (sim/envs/pickplace_env.py) --
constructs it, resets it, steps it with random actions for a bounded
number of steps, and prints observation shapes/reward/done values at
every step. Not a policy -- just confirms the env's reset()/step() loop
runs correctly end to end before anything tries to train through it.

Usage:
    ./isaaclab.sh -p /path/to/test_pickplace_env.py --headless [--use_cameras] [--num_envs N] [--steps N]
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--use_cameras", action="store_true", help="Include the wrist camera in observations.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from envs.pickplace_env import PickPlaceEnv, PickPlaceEnvCfg  # isort:skip


def main():
    # use_cameras/num_envs must be passed at construction time, not set as
    # attributes afterward -- PickPlaceEnvCfg.__post_init__ builds the
    # scene cfg from them and only runs once, at __init__.
    cfg = PickPlaceEnvCfg(use_cameras=args_cli.use_cameras, num_envs=args_cli.num_envs)
    cfg.sim.device = args_cli.device

    env = PickPlaceEnv(cfg)
    print(f"[INFO] num_envs={env.num_envs} action_space={env.action_space} observation_space={env.observation_space}")

    obs, extras = env.reset()
    print("[INFO] Reset OK.")
    print(f"[INFO] policy proprio shape: {obs['policy']['proprio'].shape}")
    if args_cli.use_cameras:
        print(f"[INFO] policy wrist_rgb shape: {obs['policy']['wrist_rgb'].shape}")
        print(f"[INFO] policy top_rgb shape: {obs['policy']['top_rgb'].shape}")
    print(f"[INFO] critic privileged shape: {obs['critic']['privileged'].shape}")

    total_reward = torch.zeros(env.num_envs, device=env.device)
    num_resets = 0

    for step in range(args_cli.steps):
        actions = torch.rand((env.num_envs, 6), device=env.device) * 2.0 - 1.0
        obs, reward, terminated, truncated, extras = env.step(actions)
        total_reward += reward
        num_resets += int((terminated | truncated).sum().item())

        if step % 20 == 0 or step == args_cli.steps - 1:
            print(
                f"[step {step:4d}] reward_mean={reward.mean().item():+.3f} "
                f"reward_min={reward.min().item():+.3f} reward_max={reward.max().item():+.3f} "
                f"terminated={int(terminated.sum().item())} truncated={int(truncated.sum().item())} "
                f"privileged[0]={obs['critic']['privileged'][0].tolist()}"
            )

    print("\n[RESULT] === Smoke test summary ===")
    print(f"[RESULT] Ran {args_cli.steps} steps across {env.num_envs} envs with random actions.")
    print(f"[RESULT] Total resets triggered (terminated or truncated): {num_resets}")
    print(f"[RESULT] Mean cumulative reward per env: {total_reward.mean().item():+.3f}")
    print("[RESULT] No crash -- env reset()/step() loop runs end to end.")


if __name__ == "__main__":
    main()
    simulation_app.close()
