# SPDX-License-Identifier: BSD-3-Clause
"""Measures PickPlaceEnv step throughput at a given num_envs, to find how
many parallel environments this GPU can actually run.

Run once with --use_cameras to find the REAL number that matters (this is
what any actual training run will use -- the trained policy needs both
cameras, see docs/real_camera_setup.md and pickplace_env.py's module
docstring). Optionally run once without it as a diagnostic upper bound,
to see how much camera rendering alone costs.

Reports steps/second (both per-env-step and total env-instances/second,
i.e. num_envs * steps/second -- the number that actually predicts training
wall-clock time) and peak GPU memory used, so a --num_envs value can be
picked before committing to a real training run.

Usage:
    ./isaaclab.sh -p /path/to/num_envs_scaling_test.py --headless [--enable_cameras --use_cameras] --num_envs 64 --steps 100
"""

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--use_cameras", action="store_true")
parser.add_argument("--num_envs", type=int, required=True)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--warmup_steps", type=int, default=10, help="Steps to run before timing starts.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from envs.pickplace_env import PickPlaceEnv, PickPlaceEnvCfg  # isort:skip


def main():
    cfg = PickPlaceEnvCfg(use_cameras=args_cli.use_cameras, num_envs=args_cli.num_envs)
    cfg.sim.device = args_cli.device

    t0 = time.perf_counter()
    try:
        env = PickPlaceEnv(cfg)
    except torch.cuda.OutOfMemoryError as e:
        print(f"[RESULT] num_envs={args_cli.num_envs} use_cameras={args_cli.use_cameras}: OOM during construction: {e}")
        return
    print(f"[PROGRESS] env constructed at t={time.perf_counter() - t0:.1f}s")

    obs, extras = env.reset()
    print(f"[PROGRESS] reset done at t={time.perf_counter() - t0:.1f}s")

    try:
        for i in range(args_cli.warmup_steps):
            actions = torch.rand((env.num_envs, 6), device=env.device) * 2.0 - 1.0
            env.step(actions)
            print(f"[PROGRESS] warmup step {i} done at t={time.perf_counter() - t0:.1f}s")

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()

        for i in range(args_cli.steps):
            actions = torch.rand((env.num_envs, 6), device=env.device) * 2.0 - 1.0
            env.step(actions)
            if i % 10 == 0:
                print(f"[PROGRESS] timed step {i}/{args_cli.steps} done at t={time.perf_counter() - t0:.1f}s")

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    except torch.cuda.OutOfMemoryError as e:
        print(f"[RESULT] num_envs={args_cli.num_envs} use_cameras={args_cli.use_cameras}: OOM during stepping: {e}")
        return

    env_steps_per_sec = (args_cli.steps * args_cli.num_envs) / elapsed
    steps_per_sec = args_cli.steps / elapsed
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

    print(
        f"[RESULT] num_envs={args_cli.num_envs} use_cameras={args_cli.use_cameras} "
        f"steps_per_sec={steps_per_sec:.2f} env_steps_per_sec={env_steps_per_sec:.1f} "
        f"peak_gpu_mem_gb={peak_mem_gb:.2f}"
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
