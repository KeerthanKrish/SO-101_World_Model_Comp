# SPDX-License-Identifier: BSD-3-Clause
"""Validates pickplace_reward.py against a REAL captured teleop
demonstration, not just hand-constructed test states.

Replays a segmented episode (from segment_teleop_episodes.py) open-loop in
sim -- same mechanism as replay_episode.py -- and at every step computes
the actual reward the training env would produce, using:
  - the LIVE simulated cube pose/velocity (not the recorded JSON's cube_pos --
    replaying the same joint trajectory should reproduce closely-matching
    physics, and using the live state is what a real training rollout
    would actually see)
  - the gripper's grasp point, computed via forward kinematics from
    gripper_frame_link's live body pose plus the calibrated JAW_OFFSET_LOCAL
    correction (grasp_point_world) -- NOT gripper_frame_link's own origin
  - the live simulated joint velocities (for the action penalty)

This can only validate the "approach" and "grasp/lift" halves of the
reward -- these demonstrations pick the cube up and set it back down near
its start, they don't carry it to pickplace_reward.py's (-0.15, 0.15, ...)
place target. The "transport"/"place" phase and the sparse success bonus
are validated only by the hand-constructed cases in
pickplace_reward.py's own self-test, not against real data -- documented
explicitly, not glossed over (see docs/reward_function.md).

Usage:
    ./isaaclab.sh -p /path/to/validate_reward_function.py --episode /path/to/episode_003.json
"""

import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate the pick-place reward function against a real demonstration.")
parser.add_argument("--episode", type=str, required=True, help="Path to an episode_*.json file.")
parser.add_argument("--log-every", type=int, default=10, help="Print reward every N steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from envs.pickplace_reward import PickPlaceRewardConfig, compute_reward  # isort:skip
from robots.grasp_geometry import grasp_point_world  # isort:skip
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

    ee_cfg = SceneEntityCfg("robot", body_names=["gripper_frame_link"])
    ee_cfg.resolve(scene)
    ee_body_id = ee_cfg.body_ids[0]

    # Reset the cube to THIS EPISODE'S OWN recorded starting position, not
    # the scene's unrelated default spawn point. These episodes were cut
    # out of one continuous teleop session (segment_teleop_episodes.py),
    # so each one starts with the cube wherever it physically drifted to
    # after the previous rep -- not at a clean reset state. Replaying the
    # recorded joint trajectory against the wrong cube start position
    # means the gripper is aimed at where the cube used to be, not where
    # it actually is -- silently reproducing nothing like the original
    # pick (this was a real bug in an earlier version of this script,
    # caught by reward output showing cube_h pinned at rest height the
    # entire episode with is_grasped() never firing -- see
    # docs/reward_function.md).
    default_root_state = scene["cube"].data.default_root_state.clone()
    episode_cube_pos = episode[0]["cube_pos"]
    root_pose = default_root_state[:, :7].clone()
    root_pose[0, 0:3] = torch.tensor(episode_cube_pos, device=sim.device)
    scene["cube"].write_root_pose_to_sim(root_pose)
    scene["cube"].write_root_velocity_to_sim(torch.zeros_like(default_root_state[:, 7:]))

    # Same reasoning as the cube: initialize the robot to THIS EPISODE'S
    # own first recorded joint_pos, not the arm's rest pose. Resetting to
    # the rest pose and then commanding episode[0]'s (likely very
    # different, mid-motion) joint target as a PD position target created
    # a large, physically meaningless teleport/settling transient right at
    # step 0 -- exactly where the grasp-proximity bug above was caught.
    initial_joint_pos = robot.data.default_joint_pos.clone()
    for i, joint_name in enumerate(joint_names):
        if joint_name in episode[0]["joint_pos"]:
            initial_joint_pos[0, i] = episode[0]["joint_pos"][joint_name]
    robot.write_joint_state_to_sim(initial_joint_pos, robot.data.default_joint_vel)
    robot.reset()
    scene.reset()

    cfg = PickPlaceRewardConfig()

    rewards = []
    grasped_at = None
    touched_at = None
    between_jaws_at = None
    min_reach_dist = float("inf")
    max_cube_height = 0.0
    first_grasp_gripper_cube_dist = None
    # was_holding/was_touched/prev_dist/prev_joint_pos -- state
    # pickplace_reward.compute_reward() needs across steps. See that
    # module's docstring (was_holding/was_touched/prev_joint_pos) and
    # pickplace_env.py's module docstring (prev_dist, and why it must be
    # reset -- None here plays the same role as NaN there) for why each is
    # needed. All start "empty" at episode start, updated from each step's info.
    was_holding = False
    was_touched = False
    prev_dist = None
    prev_joint_pos = None

    for step_idx, step in enumerate(episode):
        target = torch.tensor([[step["joint_pos"][j] for j in _JOINT_ORDER]], device=sim.device)
        robot.set_joint_position_target(target, joint_ids=joint_indices)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim_dt)

        ee_pos_w = robot.data.body_pos_w[0, ee_body_id].cpu().tolist()
        ee_quat_w = robot.data.body_quat_w[0, ee_body_id].cpu().tolist()
        gripper_pos = grasp_point_world(ee_pos_w, ee_quat_w)

        cube_pos = scene["cube"].data.root_pos_w[0].cpu().tolist()
        cube_lin_vel = scene["cube"].data.root_lin_vel_w[0].cpu().tolist()
        gripper_joint_pos = step["joint_pos"]["gripper"]
        joint_vel = robot.data.joint_vel[0, joint_indices].cpu().tolist()

        reward, info = compute_reward(
            gripper_pos, ee_quat_w, cube_pos, cube_lin_vel, gripper_joint_pos, joint_vel,
            was_holding, was_touched, prev_dist, prev_joint_pos, cfg,
        )
        was_holding = info["holding"]
        was_touched = info["touched"]
        prev_dist = info["dist"]
        prev_joint_pos = info["joint_pos"]
        rewards.append(reward)

        max_cube_height = max(max_cube_height, cube_pos[2])
        if info["phase"] == "approach":
            min_reach_dist = min(min_reach_dist, info["dist"])
        if info["grasped"] and grasped_at is None:
            grasped_at = step_idx
            first_grasp_gripper_cube_dist = (
                sum((gripper_pos[i] - cube_pos[i]) ** 2 for i in range(3)) ** 0.5
            )
        if info["touched"] and touched_at is None:
            touched_at = step_idx
        if info["between_jaws"] and between_jaws_at is None:
            between_jaws_at = step_idx

        if step_idx % args_cli.log_every == 0 or step_idx == len(episode) - 1:
            print(
                f"[step {step_idx:4d}] reward={reward:+.3f} phase={info['phase']:9s} "
                f"holding={info['holding']} grasped={info['grasped']} cube_h={cube_pos[2]:.3f} "
                f"gripper_pos=({gripper_pos[0]:.3f},{gripper_pos[1]:.3f},{gripper_pos[2]:.3f})"
            )

    print("\n[RESULT] === Validation summary ===")
    print(f"[RESULT] Steps: {len(episode)}")
    print(f"[RESULT] Max cube height reached: {max_cube_height:.4f} m")
    # Under the 2026-08-31 potential-based-shaping redesign, "dense" is a
    # per-step DELTA, not an absolute value -- "approaches 1.0" is no longer
    # a meaningful expectation (see pickplace_reward.py's module docstring).
    # The physically meaningful, formulation-independent diagnostic is how
    # close the gripper actually got to the cube during the approach phase.
    print(f"[RESULT] Min gripper-to-cube distance during approach phase (should approach 0): "
          f"{min_reach_dist:.4f} m")
    if touched_at is not None:
        print(f"[RESULT] First touched (is_touching) at step {touched_at}")
    else:
        print("[RESULT] Never reached touch_threshold according to is_touching().")
    if between_jaws_at is not None:
        print(f"[RESULT] First positioned between the jaws (is_between_jaws, added 2026-09-03) at step {between_jaws_at}")
    else:
        print("[RESULT] Never positioned between the jaws according to is_between_jaws().")
    if grasped_at is not None:
        print(f"[RESULT] First grasped at step {grasped_at} "
              f"(gripper-cube distance at that moment: {first_grasp_gripper_cube_dist:.4f} m, "
              f"should be small)")
        print(f"[RESULT] Reward jump at grasp: "
              f"{rewards[grasped_at - 1]:+.3f} -> {rewards[grasped_at]:+.3f}")
    else:
        print("[RESULT] Never grasped according to the reward function's is_grasped() -- "
              "check lift_threshold/gripper_closed_threshold against this episode's actual data.")
    print(f"[RESULT] Reward range across episode: [{min(rewards):+.3f}, {max(rewards):+.3f}]")


if __name__ == "__main__":
    main()
    simulation_app.close()
