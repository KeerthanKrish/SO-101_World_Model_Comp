# SPDX-License-Identifier: BSD-3-Clause
"""Build the SO-101 pick-and-place scene, step physics briefly to settle, and
save a single RGB image from the scene camera.

Since we're running headless on the remote GPU box with no practical way to
view Isaac Sim's 3D GUI over X11 forwarding (see docs/preferences.md),
offscreen image capture is the way to visually check scene layout.

Usage:
    ./isaaclab.sh -p /path/to/capture_scene_image.py --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Capture an image of the SO-101 pick-and-place scene.")
parser.add_argument("--num-settle-steps", type=int, default=60, help="Physics steps before capturing.")
parser.add_argument(
    "--output", type=str, default="/home/keerthan/SO-101-WM/sim/output/scene_check.png", help="Output image path."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils import convert_dict_to_backend

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from scenes.pickplace_scene import PickPlaceSceneCfg  # isort:skip


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = PickPlaceSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete. Scene entities:", list(scene.keys()))

    camera = scene["scene_camera"]
    eye = torch.tensor([[0.6, -0.6, 0.5]], device=sim.device)
    target = torch.tensor([[0.15, 0.0, 0.05]], device=sim.device)
    camera.set_world_poses_from_view(eye, target)

    sim_dt = sim.get_physics_dt()
    robot = scene["robot"]
    default_joint_pos = robot.data.default_joint_pos.clone()

    for _ in range(args_cli.num_settle_steps):
        robot.set_joint_position_target(default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    # one more render-only step to make sure the camera buffer is fresh
    sim.step(render=True)
    scene.update(sim_dt)

    rgb = camera.data.output["rgb"]
    print("[INFO]: Captured RGB tensor shape:", rgb.shape)

    os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)
    from PIL import Image

    img = rgb[0, ..., :3].cpu().numpy()
    Image.fromarray(img).save(args_cli.output)
    print(f"[RESULT] Saved image to {args_cli.output}")


if __name__ == "__main__":
    main()
    simulation_app.close()
