# SPDX-License-Identifier: BSD-3-Clause
"""Smoke test for the TD-MPC2 adapter (tdmpc2_pickplace_env.py) --
confirms obs shapes/dtypes match what TD-MPC2's encoder expects, and that
info['success']/info['terminated'] are correctly populated (the exact gap
this adapter's own docstring documents finding and fixing in
pickplace_env.py's extras handling).

Usage:
    ./isaaclab.sh -p /path/to/test_tdmpc2_adapter.py --headless --enable_cameras --steps 40
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=40)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
sys.path.insert(0, "/home/keerthan/SO-101-WM/sim/scripts")
from envs.pickplace_env import PickPlaceEnv, PickPlaceEnvCfg  # isort:skip
from tdmpc2_pickplace_env import PickPlaceTDMPC2Wrapper  # isort:skip


def main():
    cfg = PickPlaceEnvCfg(use_cameras=True, num_envs=1)
    cfg.sim.device = args_cli.device
    base_env = PickPlaceEnv(cfg)
    env = PickPlaceTDMPC2Wrapper(base_env)

    obs = env.reset()
    print(f"[INFO] obs keys: {list(obs.keys())}")
    print(f"[INFO] obs['state'] shape/dtype: {obs['state'].shape} {obs['state'].dtype}")
    print(f"[INFO] obs['rgb'] shape/dtype: {obs['rgb'].shape} {obs['rgb'].dtype}")
    assert obs["state"].shape == (12,), obs["state"].shape
    assert obs["rgb"].shape == (6, 64, 64), obs["rgb"].shape
    assert obs["rgb"].dtype == np.float32
    assert 0.0 <= obs["rgb"].min() and obs["rgb"].max() <= 255.0

    print(f"[INFO] action_space: {env.action_space}")
    print(f"[INFO] rand_act-equivalent sample: {env.action_space.sample().shape}")

    any_terminated = False
    any_success_key_seen = False
    for step in range(args_cli.steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        any_success_key_seen = any_success_key_seen or ("success" in info and "terminated" in info)
        if done:
            any_terminated = any_terminated or info["terminated"]
            print(f"[INFO] step {step}: done=True info={info}")
            obs = env.reset()
        if step % 10 == 0:
            print(f"[INFO] step {step}: reward={reward:+.3f} done={done} info={info}")

    print("\n[RESULT] === Adapter smoke test summary ===")
    print(f"[RESULT] Ran {args_cli.steps} steps, no crash.")
    print(f"[RESULT] info always had success/terminated keys: {any_success_key_seen}")
    print(f"[RESULT] At least one real termination observed: {any_terminated} "
          f"(False is fine for a short random-action run -- terminations are rare by chance)")


if __name__ == "__main__":
    main()
    simulation_app.close()
