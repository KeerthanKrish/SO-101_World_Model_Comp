# SPDX-License-Identifier: BSD-3-Clause
"""Records a short video of PickPlaceEnv actually running -- random
actions (no trained policy exists yet), so this shows the reset/step
CYCLE mechanics working (episodes resetting the cube to a fresh sampled
position, timing out, resetting again), not intelligent behavior.

Saves one frame per env step from a third-person debug camera
(scene_camera, aimed once at the start) plus the wrist camera, side by
side, then stitches them into an mp4 with ffmpeg (same approach as the
existing sim/scripts/stitch_video.sh).

Usage:
    ./isaaclab.sh -p /path/to/record_pickplace_env_video.py --headless --enable_cameras --steps 300
"""

import argparse
import os
import subprocess
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument(
    "--output-dir", type=str, default="/home/keerthan/SO-101-WM/sim/output/pickplace_env_rollout"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from envs.pickplace_env import PickPlaceEnv, PickPlaceEnvCfg  # isort:skip


def main():
    frames_dir = os.path.join(args_cli.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cfg = PickPlaceEnvCfg(use_cameras=True, num_envs=1)
    cfg.sim.device = args_cli.device
    env = PickPlaceEnv(cfg)

    obs, extras = env.reset()

    # aim the third-person debug camera once, matching capture_scene_image.py's convention.
    scene_camera = env.scene["scene_camera"]
    scene_camera.set_world_poses_from_view(
        torch.tensor([[0.6, -0.6, 0.5]], device=env.device), torch.tensor([[0.15, 0.0, 0.05]], device=env.device)
    )
    env.sim.render()

    print(f"[INFO] Recording {args_cli.steps} steps of random-action rollout...")
    for step in range(args_cli.steps):
        actions = torch.rand((env.num_envs, 6), device=env.device) * 2.0 - 1.0
        obs, reward, terminated, truncated, extras = env.step(actions)

        scene_rgb = scene_camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        wrist_rgb = obs["policy"]["wrist_rgb"][0].cpu().numpy()

        # side by side: scene view (960x720ish) + wrist view (480x270),
        # resized to match scene_rgb's height so np.hstack works cleanly.
        h = scene_rgb.shape[0]
        wrist_img = Image.fromarray(wrist_rgb).resize(
            (int(wrist_rgb.shape[1] * h / wrist_rgb.shape[0]), h)
        )
        combined = np.hstack([scene_rgb, np.array(wrist_img)])

        Image.fromarray(combined).save(os.path.join(frames_dir, f"frame_{step:05d}.png"))

        if step % 50 == 0:
            print(f"[INFO] step {step}/{args_cli.steps}")

    print(f"[INFO] Saved {args_cli.steps} frames to {frames_dir}")

    output_mp4 = os.path.join(args_cli.output_dir, "rollout.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", str(args_cli.fps),
            "-i", os.path.join(frames_dir, "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", output_mp4,
        ],
        check=True,
    )
    print(f"[RESULT] Saved video to {output_mp4}")


if __name__ == "__main__":
    main()
    simulation_app.close()
