# SPDX-License-Identifier: BSD-3-Clause
"""List mesh prims and their bound materials under the SO-101 robot, to see
if motor-housing meshes are separately identifiable from link body meshes
(for a possible targeted color override).

Usage:
    ./isaaclab.sh -p /path/to/inspect_so101_meshes.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from pxr import Usd, UsdGeom, UsdShade
import sys

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from robots.so101 import SO101_CFG  # isort:skip


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    robot_cfg = SO101_CFG.copy()
    robot_cfg.prim_path = "/World/SO101"
    Articulation(cfg=robot_cfg)
    sim.reset()

    stage = sim_utils.get_current_stage() if hasattr(sim_utils, "get_current_stage") else None
    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()

    prim = stage.GetPrimAtPath("/World/SO101")
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdGeom.Mesh):
            binding_api = UsdShade.MaterialBindingAPI(p)
            mat = binding_api.ComputeBoundMaterial()[0]
            mat_path = mat.GetPath().pathString if mat else "None"
            print(f"[RESULT] {p.GetPath()} -> material={mat_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
