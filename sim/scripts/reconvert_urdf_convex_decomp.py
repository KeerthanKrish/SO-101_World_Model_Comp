# SPDX-License-Identifier: BSD-3-Clause
"""Re-convert the SO-101 URDF to USD with `convex_decomposition` collision
approximation instead of the default `convex_hull`.

Why: a convex hull can't represent concave geometry -- the gripper's
pincer notch is concave, so the hull "fills it in", creating a collision
shape that has no actual gap for an object to sit inside. This is a
documented, common cause of exactly the symptoms we hit during teleop
(gripper clipping through the cube instead of contacting it properly,
objects popping out under grip force) -- see an IsaacLab GitHub discussion
("Why my gripper can not close when I want to grasp the cube?") that
diagnosed and fixed the identical problem the same way. convert_urdf.py's
CLI doesn't expose `collider_type`, so this mirrors its internals directly
with that field set.

Usage:
    ./isaaclab.sh -p /path/to/reconvert_urdf_convex_decomp.py --headless
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    type=str,
    default="/home/keerthan/SO-101-WM/assets/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
)
parser.add_argument("--output", type=str, default="/home/keerthan/SO-101-WM/assets/usd/so101/so101.usd")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


def main():
    urdf_path = os.path.abspath(args_cli.input)
    dest_path = os.path.abspath(args_cli.output)

    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        fix_base=True,
        merge_fixed_joints=False,
        force_usd_conversion=True,
        collider_type="convex_decomposition",
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=1.0),
            target_type="position",
        ),
    )
    urdf_converter = UrdfConverter(urdf_converter_cfg)
    print(f"[RESULT] Generated USD file: {urdf_converter.usd_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
