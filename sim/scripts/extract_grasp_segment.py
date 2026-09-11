# SPDX-License-Identifier: BSD-3-Clause
"""First step of the MimicGen-lite pipeline for the diffusion-policy side
of this project (see docs/diffusion_policy_data_strategy.md): replays a
recorded teleop episode through the REAL production environment (same
proven pattern as replay_demo_to_buffer.py) and extracts exactly the
data the next stage (SE(3) retargeting via
isaaclab.utils.math.pose_in_A_to_pose_in_B, then closed-loop replay via
isaaclab.controllers.DifferentialIKController) needs:

  - The gripper's own world-frame Cartesian pose (position + quaternion)
    at every step of the "reach-and-grasp" segment -- the segment that
    actually needs retargeting to a new cube position. Recorded via
    forward kinematics (i.e. just reading back the live simulated pose
    each step), not computed analytically -- the simulator itself is
    the most reliable source of truth here, exactly the same reasoning
    already applied throughout this project's demo-replay work.
  - The recorded gripper JOINT value (open/closed) at each of those
    steps -- gripper actuation is a decoupled 1-DOF mechanism, no part
    of the arm's Cartesian pose, so it carries over UNCHANGED to a
    retargeted episode rather than being subject to any SE(3) transform.
  - The source episode's own recorded cube position -- the reference
    pose the eventual retargeting transform will be computed relative
    to.
  - The exact segmentation boundary itself: the first step
    is_holding() -- the same, now-fixed function this project's whole
    demo-seeding investigation already validated -- reports True. This
    is the natural "reach-and-grasp" -> "transport-and-place" subtask
    boundary MimicGen's own methodology calls for (see
    docs/diffusion_policy_data_strategy.md's "Design the segmentation
    point" step) -- everything from here through the rest of the
    episode (the "transport-and-place" segment) needs NO retargeting at
    all, since it's relative to the FIXED placement target, not the
    cube, and can simply be replayed unchanged after the retargeted
    segment splices into it.

Deliberately does NOT yet perform any retargeting or IK -- this script's
only job is producing a clean, validated segment-1 trajectory + boundary
index, saved to disk, for the next stage to consume. Building this
incrementally and validating each piece in isolation (extraction, then
the transform math, then closed-loop IK replay, then combining) rather
than attempting the whole pipeline at once, matching how every other
piece of this project has been built.

Usage:
    ./isaaclab.sh -p /path/to/extract_grasp_segment.py --enable_cameras \\
        --episode /path/to/episode_002.json --out /path/to/segment_002.json
"""

import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Extract a source episode's reach-and-grasp segment for MimicGen-lite retargeting.")
parser.add_argument("--episode", type=str, required=True, help="Path to an episode_*.json file.")
parser.add_argument("--out", type=str, required=True, help="Path to write the extracted segment JSON to.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # isort:skip

# Force line-buffered stdout -- this box's known Kit shutdown-hang means a
# run can be killed before Python's default full-buffering ever flushes,
# silently losing every print() issued so far. See
# replay_demo_to_buffer.py's own module docstring for the full story.
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from envs.pickplace_env import PickPlaceEnv, PickPlaceEnvCfg  # isort:skip

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim/scripts")
from tdmpc2_pickplace_env import PickPlaceTDMPC2Wrapper  # isort:skip

_JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# The 5 ARM joints that actually determine the end-effector's Cartesian
# pose -- excludes "gripper" itself, which is a decoupled open/close
# actuator contributing nothing to the Jacobian/end-effector pose (see
# this module's docstring). This is exactly the joint set the NEXT
# stage's DifferentialIKController needs to be configured over.
_ARM_JOINT_ORDER = _JOINT_ORDER[:-1]


def main():
    with open(args_cli.episode) as f:
        episode = json.load(f)
    print(f"[INFO] Loaded episode with {len(episode)} raw (100Hz) frames from {args_cli.episode}")

    # Downsample to the env's own 50Hz control rate -- same reasoning as
    # replay_demo_to_buffer.py's own module docstring ("timing mismatch"
    # section): the recording is 100Hz, PickPlaceEnv's control rate is
    # 50Hz (decimation=2), so every 2nd frame is one control step.
    downsampled = episode[0::2]
    print(f"[INFO] Downsampled to {len(downsampled)} control-rate (50Hz) frames "
          f"({len(downsampled) - 1} env.step() calls)")

    # wait_for_textures=False -- see replay_demo_to_buffer.py's own
    # docstring for the reproducible Kit texture-streaming stall this
    # works around. Irrelevant here for the same reason: this script
    # never reads a camera image, only body/joint/cube physics state.
    env_kwargs = dict(use_cameras=True, num_envs=1, episode_length_s=10.0, wait_for_textures=False)
    base_env = PickPlaceEnv(PickPlaceEnvCfg(**env_kwargs))
    env = PickPlaceTDMPC2Wrapper(base_env)

    # Body index for the gripper's own IK end-effector frame
    # (gripper_frame_link) -- the same frame grasp_point_world() and
    # jaw_approach_axis_world() already treat as "the gripper," and the
    # frame the next stage's DifferentialIKController will target.
    ee_body_idx = base_env.robot.data.body_names.index("gripper_frame_link")

    env.reset()  # clears persistent per-episode reward-tracking state, restarts the timeout counter -- see replay_demo_to_buffer.py's own docstring for why this matters even for a single-episode script

    # Override cube + robot state to match THIS episode's own recorded
    # start -- identical pattern to replay_demo_to_buffer.py (which this
    # is adapted from) and validate_reward_function.py before it.
    default_root_state = base_env.cube.data.default_root_state.clone()
    root_pose = default_root_state[:, :7].clone()
    source_cube_pos = downsampled[0]["cube_pos"]
    root_pose[0, 0:3] = torch.tensor(source_cube_pos, device=base_env.device) + base_env.scene.env_origins[0]
    base_env.cube.write_root_pose_to_sim(root_pose)
    base_env.cube.write_root_velocity_to_sim(torch.zeros_like(default_root_state[:, 7:]))

    initial_joint_pos = base_env.robot.data.default_joint_pos.clone()
    for i, joint_name in enumerate(base_env.robot.data.joint_names):
        if joint_name in downsampled[0]["joint_pos"]:
            initial_joint_pos[0, i] = downsampled[0]["joint_pos"][joint_name]
    base_env.robot.write_joint_state_to_sim(initial_joint_pos, base_env.robot.data.default_joint_vel)
    base_env.robot.reset()
    base_env.scene.reset()

    # CRITICAL, same reasoning as replay_demo_to_buffer.py: PickPlaceEnv's
    # own action mechanism is an INCREMENTAL delta from _joint_pos_target,
    # not an absolute command -- must match the state override above or
    # the first computed action is a delta from the wrong baseline.
    base_env._joint_pos_target = torch.tensor(
        [[downsampled[0]["joint_pos"][j] for j in _JOINT_ORDER]], device=base_env.device
    )

    def record_ee_pose():
        """Reads back the gripper_frame_link's LIVE world pose this
        step -- this IS the forward-kinematics step (see module
        docstring): no analytic FK is computed by hand, the simulator's
        own physics state is the source of truth. env_origins-corrected
        to the env-local frame (zero here at num_envs=1, but kept
        explicit for the same reason replay_demo_to_buffer.py does --
        confirmed a genuine, non-zero offset in PickPlaceEnv's own
        _reset_idx() for the cube; applying the same correction to the
        robot body pose is the safe, consistent choice even though it's
        a no-op at this env count)."""
        pos_w = (base_env.robot.data.body_pos_w[0, ee_body_idx] - base_env.scene.env_origins[0]).cpu().tolist()
        quat_w = base_env.robot.data.body_quat_w[0, ee_body_idx].cpu().tolist()
        return pos_w, quat_w

    segment = []  # list of {step, ee_pos, ee_quat, gripper_joint, holding}
    ee_pos0, ee_quat0 = record_ee_pose()
    segment.append({
        "step": 0,
        "ee_pos": ee_pos0,
        "ee_quat": ee_quat0,
        "gripper_joint": downsampled[0]["joint_pos"]["gripper"],
        "holding": False,  # can't genuinely be holding at the very first step (pre-any-action)
    })

    boundary_step = None
    for step_idx, frame in enumerate(downsampled[1:], start=1):
        desired_target_raw = torch.tensor([[frame["joint_pos"][j] for j in _JOINT_ORDER]], device=base_env.device)
        # Option 1 (2026-09-08, see replay_demo_to_buffer.py's own
        # docstring for the full reasoning): clip the recorded target to
        # the robot's own physical joint limits before use.
        desired_target = torch.clamp(desired_target_raw, base_env._soft_limits[..., 0], base_env._soft_limits[..., 1])
        # Bypass the normal delta-action velocity clamp -- see
        # replay_demo_to_buffer.py's "ROOT-CAUSE FIX" comment for the
        # full derivation. Only the resulting PHYSICAL TRAJECTORY matters
        # for this script's purpose (extracting real FK poses); the
        # action_label reconstruction replay_demo_to_buffer.py needs for
        # buffer-seeding is irrelevant here, so it's not computed at all.
        base_env._joint_pos_target = desired_target
        zero_action = torch.zeros(6, device=base_env.device)
        obs, reward, done, info = env.step(zero_action)

        ee_pos, ee_quat = record_ee_pose()
        holding = bool(info["holding"])
        segment.append({
            "step": step_idx,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "gripper_joint": frame["joint_pos"]["gripper"],
            "holding": holding,
        })

        if holding and boundary_step is None:
            boundary_step = step_idx
            print(f"[RESULT] Grasp-event boundary (first is_holding()=True) at downsampled step {step_idx} "
                  f"of {len(downsampled) - 1}")
            # Keep tracing a little further even after finding the
            # boundary -- purely diagnostic, confirms holding is
            # genuinely SUSTAINED here, not a single-step flicker, before
            # this episode gets trusted as a source demo for retargeting.
            if step_idx + 10 >= len(downsampled):
                break

        if done:
            print(f"[WARN] episode terminated early at downsampled step {step_idx} "
                  f"(of {len(downsampled) - 1}) -- stopping here.")
            break

    if boundary_step is None:
        print("[RESULT] is_holding() never fired in this episode -- not usable as a MimicGen-lite source demo "
              "(see docs/decisions.md for which of the 8 re-segmented episodes already confirmed ever_holding=True).")
        return

    # Sustained-hold sanity check: confirm holding stayed True for a
    # genuine stretch after the boundary, not one flickering step --
    # directly relevant here since a source demo whose "grasp" was
    # actually a near-miss would poison every episode retargeted from it.
    post_boundary = [s for s in segment if s["step"] >= boundary_step]
    holding_streak = 0
    for s in post_boundary:
        if s["holding"]:
            holding_streak += 1
        else:
            break
    print(f"[RESULT] holding sustained for {holding_streak} consecutive steps immediately after the boundary "
          f"(of {len(post_boundary)} traced)")

    reach_and_grasp_segment = [s for s in segment if s["step"] <= boundary_step]
    output = {
        "source_episode": args_cli.episode,
        "source_cube_pos": source_cube_pos,
        "boundary_step": boundary_step,
        "arm_joint_order": _ARM_JOINT_ORDER,
        "reach_and_grasp_segment": reach_and_grasp_segment,
    }
    with open(args_cli.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[RESULT] Wrote {len(reach_and_grasp_segment)}-step reach-and-grasp segment to {args_cli.out}")
    print("[RESULT] No crash -- segment extraction complete.")


if __name__ == "__main__":
    main()
    simulation_app.close()
