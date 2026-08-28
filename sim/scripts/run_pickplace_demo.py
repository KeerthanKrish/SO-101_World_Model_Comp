# SPDX-License-Identifier: BSD-3-Clause
"""Scripted (non-learned) reach/grasp/place demo for the SO-101, using
Isaac Lab's differential IK controller (position-only command mode) to
command Cartesian end-effector targets instead of guessing joint angles
directly.

Tried full pose control (position + a fixed "point straight down" target
orientation) in an earlier version -- that made things worse, not better:
the target orientation turned out to be unreachable for this 5-DOF arm at
this position, and the solver converged to a collapsed configuration near
the base instead of actually reaching. Reverted to position-only IK, which
at least gets physically close to the cube, and kept the two-phase
targeting fix (see run_ik_steps / the waypoint loop) that resolved an
earlier bug where recomputing the jaw-offset correction every step (using
the still-changing orientation during a big motion) caused the arm to
lurch toward/through the floor.

This is a sanity check, not a finished policy: the waypoint positions and
gripper open/closed joint values below are initial guesses (see inline
comments) and will likely need tuning once we can see the rendered result.

Saves one RGB frame every --frame-interval sim steps to --frames-dir; use
stitch_video.py afterward to turn those into an mp4.

Usage:
    ./isaaclab.sh -p /path/to/run_pickplace_demo.py --headless --enable_cameras
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted SO-101 pick-and-place demo.")
parser.add_argument("--frame-interval", type=int, default=4, help="Save a frame every N sim steps.")
parser.add_argument(
    "--frames-dir", type=str, default="/home/keerthan/SO-101-WM/sim/output/pickplace_frames", help="Frame output dir."
)
parser.add_argument(
    "--side-frames-dir",
    type=str,
    default="/home/keerthan/SO-101-WM/sim/output/pickplace_side_frames",
    help="Side camera frame output dir.",
)
parser.add_argument(
    "--top-frames-dir",
    type=str,
    default="/home/keerthan/SO-101-WM/sim/output/pickplace_top_frames",
    help="Top camera frame output dir.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

# Fixed local-frame offset from gripper_frame_link's origin to the gripper
# joint's pivot (where the jaws actually meet), derived from the URDF's
# fixed joint transforms (gripper_frame_joint and the gripper joint's
# origin, both relative to gripper_link) -- see docs/so101_asset_notes.md.
# This is constant regardless of arm pose (both frames are rigidly attached
# to gripper_link), unlike a world-frame offset which rotates with the
# wrist. Verified: its magnitude (~8.2cm) matches an empirical world-frame
# measurement taken at one specific arm pose via calibrate_grasp.py.
_JAW_OFFSET_LOCAL = (-0.0281, 0.019018, -0.0747274)

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from scenes.pickplace_scene import PickPlaceSceneCfg  # isort:skip

# Confirmed empirically via calibrate_grasp.py + visual inspection of the
# two comparison images: LOWER limit closes the jaw (narrow pincer near the
# target), UPPER limit opens it wide. Direction was correct in the first
# attempt -- the actual bug was grasp height (see below).
_GRIPPER_OPEN = 1.5
_GRIPPER_CLOSED = -0.174533

# Waypoints: (desired JAW position in world/root frame, gripper_cmd, hold_steps).
# NOTE: these are jaw-pivot targets now, not gripper_frame_link targets --
# the per-step loop below converts to the correct frame_link command using
# _JAW_OFFSET_LOCAL every step. Table top is z=0, cube center at
# (0.2, 0.0, 0.015), place target arbitrarily chosen at (-0.15, 0.15, *).
_WAYPOINTS = [
    ((0.2, 0.0, 0.20), _GRIPPER_OPEN, 60),  # approach above cube
    ((0.2, 0.0, 0.02), _GRIPPER_OPEN, 60),  # descend to grasp height (jaw at cube mid-height)
    ((0.2, 0.0, 0.02), _GRIPPER_CLOSED, 60),  # close gripper
    ((0.2, 0.0, 0.20), _GRIPPER_CLOSED, 60),  # lift
    ((-0.15, 0.15, 0.20), _GRIPPER_CLOSED, 90),  # move to place zone
    ((-0.15, 0.15, 0.02), _GRIPPER_CLOSED, 60),  # descend
    ((-0.15, 0.15, 0.02), _GRIPPER_OPEN, 60),  # release
    ((-0.15, 0.15, 0.20), _GRIPPER_OPEN, 60),  # retreat
]


def main():
    os.makedirs(args_cli.frames_dir, exist_ok=True)
    os.makedirs(args_cli.side_frames_dir, exist_ok=True)
    os.makedirs(args_cli.top_frames_dir, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = PickPlaceSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete.")

    robot = scene["robot"]
    camera = scene["scene_camera"]
    side_camera = scene["side_camera"]
    top_camera = scene["top_camera"]
    camera.set_world_poses_from_view(
        torch.tensor([[0.6, -0.6, 0.5]], device=sim.device), torch.tensor([[0.15, 0.0, 0.05]], device=sim.device)
    )
    side_camera.set_world_poses_from_view(
        torch.tensor([[0.2, -0.5, 0.1]], device=sim.device), torch.tensor([[0.0, 0.0, 0.05]], device=sim.device)
    )
    top_camera.set_world_poses_from_view(
        torch.tensor([[0.15, 0.0, 0.55]], device=sim.device), torch.tensor([[0.15, 0.0, 0.0]], device=sim.device)
    )

    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=sim.device)

    robot_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        body_names=["gripper_frame_link"],
    )
    robot_entity_cfg.resolve(scene)
    gripper_joint_id, _ = robot.find_joints("gripper")

    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]

    sim_dt = sim.get_physics_dt()
    step_count = 0
    frame_count = 0

    def save_frame():
        nonlocal frame_count
        rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        Image.fromarray(rgb).save(os.path.join(args_cli.frames_dir, f"frame_{frame_count:05d}.png"))
        side_rgb = side_camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        Image.fromarray(side_rgb).save(os.path.join(args_cli.side_frames_dir, f"frame_{frame_count:05d}.png"))
        top_rgb = top_camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        Image.fromarray(top_rgb).save(os.path.join(args_cli.top_frames_dir, f"frame_{frame_count:05d}.png"))
        frame_count += 1

    local_offset = torch.tensor([_JAW_OFFSET_LOCAL], device=sim.device)
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device)

    def run_ik_steps(frame_target_w, gripper_cmd, num_steps):
        """Drive the IK toward a fixed (world-frame) target for gripper_frame_link.

        Unlike a per-step recomputed target, this target is fixed for the
        whole call -- avoids the feedback instability where recomputing the
        jaw-offset correction every step (using the still-changing current
        orientation during a big motion) could yield wildly wrong
        corrections, observed as the arm lurching toward/through the floor.
        """
        nonlocal step_count
        root_pose_w = robot.data.root_pose_w
        frame_target_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], frame_target_w, root_pose_w[:, 3:7]
        )
        for _ in range(num_steps):
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            ik_controller.set_command(frame_target_b, ee_quat=ee_quat_b)
            joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
            robot.set_joint_position_target(
                torch.tensor([[gripper_cmd]], device=sim.device), joint_ids=gripper_joint_id
            )
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            if step_count % args_cli.frame_interval == 0:
                sim.render()
                save_frame()
            step_count += 1

    for waypoint_pos, gripper_cmd, hold_steps in _WAYPOINTS:
        desired_jaw_pos_w = torch.tensor([waypoint_pos], device=sim.device)
        ik_controller.reset()

        # Phase 1 (~60% of hold_steps): aim gripper_frame_link directly at the
        # desired jaw position, uncorrected. This is "wrong" by the jaw
        # offset, but gets the arm into the right neighborhood and lets its
        # orientation settle into whatever natural pose it adopts there.
        phase1_steps = max(1, int(hold_steps * 0.6))
        run_ik_steps(desired_jaw_pos_w, gripper_cmd, phase1_steps)

        # Phase 2 (remaining steps): now that orientation is settled, compute
        # the jaw offset correction ONCE using the current (stable)
        # orientation, and aim at that single corrected, fixed target.
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        jaw_pos_w, _ = combine_frame_transforms(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7], local_offset, identity_quat)
        jaw_error = desired_jaw_pos_w - jaw_pos_w
        corrected_frame_target_w = ee_pose_w[:, 0:3] + jaw_error
        phase2_steps = hold_steps - phase1_steps
        run_ik_steps(corrected_frame_target_w, gripper_cmd, phase2_steps)

    print(f"[RESULT] Demo complete. Saved {frame_count} frames to {args_cli.frames_dir}")
    cube_pos = scene["cube"].data.root_pos_w[0].cpu().numpy()
    print(f"[RESULT] Final cube position: {cube_pos}")


if __name__ == "__main__":
    main()
    simulation_app.close()
