# SPDX-License-Identifier: BSD-3-Clause
"""One-off headless check for teleop_bridge.py's randomize_cube_position()
-- confirms it doesn't crash and lands within the intended train region,
without needing the physical leader arm connected.

Usage:
    ./isaaclab.sh -p /path/to/test_cube_randomize.py --headless
"""

import sys

from isaaclab.app import AppLauncher

import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from scenes.pickplace_scene import PickPlaceSceneBaseCfg  # isort:skip

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim/scripts")
from teleop_bridge import randomize_cube_position, _CUBE_X_RANGE, _CUBE_Y_RANGE  # isort:skip


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    scene = InteractiveScene(PickPlaceSceneBaseCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    for i in range(10):
        randomize_cube_position(scene)
        pos = scene["cube"].data.root_pos_w[0].cpu().tolist()
        in_range = _CUBE_X_RANGE[0] <= pos[0] <= _CUBE_X_RANGE[1] and _CUBE_Y_RANGE[0] <= pos[1] <= _CUBE_Y_RANGE[1]
        print(f"[INFO] sample {i}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) in_range={in_range}")
        assert in_range, pos

    print("[RESULT] All 10 samples landed within the train region -- randomize_cube_position() works correctly.")


if __name__ == "__main__":
    main()
    simulation_app.close()
