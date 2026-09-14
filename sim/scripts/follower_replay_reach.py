# SPDX-License-Identifier: BSD-3-Clause
"""Second real-hardware step (after follower_writer_test.py's no-op
command test): replay a short, already-recorded, already-validated
teleop episode's REACH-ONLY segment on the real follower arm.

Deliberately NOT the grasp/hold/place portion -- no real cube is
positioned to match the recording yet, so commanding a real grasp
attempt blind isn't meaningful (there's nothing there to actually
close around), and this keeps the very first genuine multi-joint,
multi-step motion this project has ever commanded on real hardware to
plain, purposeless-but-safe arm movement, not a task attempt.

Source: sim/output/teleop_episodes_seed/episode_002.json -- the same
episode used throughout this project's grasp-geometry and MimicGen work
(one of the 5/8 episodes confirmed ever_holding=True in sim, and the
specific one extract_grasp_segment.py found a grasp boundary at
downsampled step 194 of 545 for). Only steps 0 through that same
boundary are replayed here -- the reach, not what comes after.

Unit/sign conversion, all confirmed against real measurements this
session, not assumed:
  - Sim stores the 5 arm joints in RADIANS; the real follower reports/
    accepts DEGREES (SOFollowerConfig's own use_degrees=True default) --
    converted via degrees(radians), matching teleop_bridge.py's existing
    (inverse-direction) leader-to-sim conversion, which never needed a
    sign flip either.
  - elbow_flex and wrist_roll need their sign FLIPPED going from sim to
    the real follower -- see docs/decisions.md's 2026-09-13 joint-sign
    cross-check entry for the full derivation (rendered-image comparison
    for elbow_flex; empirical relative-quaternion axis/angle extraction,
    not a rendered image, for wrist_roll). The other 4 joints (including
    gripper) transfer directly, no flip.
  - Sim's gripper is a continuous radian value in
    [gripper_closed_limit, gripper_open_limit] = [-0.174533, 1.74533]
    (pickplace_reward.py's own documented convention, open=higher);
    the follower's gripper is 0-100 with open=higher too (confirmed by
    the user's own hand test) -- so this is a plain linear rescale, no
    sign flip, inverting teleop_bridge.py's own leader-to-sim formula.

Safety, in order of first-time-caution:
  - `max_relative_target` engaged (limits how far any SINGLE send_action
    call can move a joint, degrees for the arm joints, 0-100 units for
    the gripper).
  - Never jumps directly to the recording's first waypoint -- smoothly
    interpolates there from the arm's OWN actual current position first,
    over several seconds, printing progress.
  - Replays at a deliberately reduced rate (10Hz here, vs. the sim
    recording's native 50Hz control rate) -- more real time per step to
    watch and react, at the cost of a slower, choppier-looking motion
    (acceptable for a first safety-focused test, not a demo).
  - Prints the full converted trajectory's per-joint min/max BEFORE any
    motion happens, so an implausible range is visible up front rather
    than discovered by watching the arm.

Run this interactively, arm powered on, someone physically present and
ready to cut power -- same as every real-hardware script before this one.

Usage:
    python follower_replay_reach.py --episode /path/to/episode_002.json --port /dev/ttyACM0 --id follower1
"""

import argparse
import json
import math
import time

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

parser = argparse.ArgumentParser()
parser.add_argument("--episode", type=str, required=True)
parser.add_argument("--port", type=str, default="/dev/ttyACM0")
parser.add_argument("--id", type=str, default="follower1")
parser.add_argument("--boundary-step", type=int, default=194,
                     help="Downsampled (50Hz) step to stop at -- default matches episode_002's own "
                          "grasp-event boundary from extract_grasp_segment.py, so only the reach "
                          "portion (not grasp/hold) is replayed.")
parser.add_argument("--hz", type=float, default=10.0, help="Replay rate -- deliberately slower than the recording's native 50Hz.")
parser.add_argument("--interp-seconds", type=float, default=4.0, help="Time to smoothly move from the arm's current pose to the trajectory's first waypoint.")
parser.add_argument("--dry-run", action="store_true",
                     help="Convert and print the trajectory summary only -- never connects to the "
                          "follower or moves anything. For verifying the conversion logic itself "
                          "before ever running this for real.")
args = parser.parse_args()

_SIM_GRIPPER_CLOSED = -0.174533
_SIM_GRIPPER_OPEN = 1.74533
_ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# Confirmed 2026-09-13 (docs/decisions.md) -- these two, and only these
# two, need their sign flipped between sim and the real follower.
_FLIP_SIGN = {"elbow_flex", "wrist_roll"}

_MAX_RELATIVE_TARGET = {
    "shoulder_pan": 8.0, "shoulder_lift": 8.0, "elbow_flex": 8.0,
    "wrist_flex": 8.0, "wrist_roll": 8.0, "gripper": 10.0,
}


def sim_joint_pos_to_follower_action(joint_pos_rad: dict) -> dict:
    """Converts one recorded frame's sim joint_pos (radians, 6 joint names)
    into a follower action dict (degrees + gripper 0-100, `.pos`-suffixed
    keys matching SOFollower.send_action()'s expected format directly)."""
    action = {}
    for joint in _ARM_JOINTS:
        deg = math.degrees(joint_pos_rad[joint])
        if joint in _FLIP_SIGN:
            deg = -deg
        action[f"{joint}.pos"] = deg
    gripper_frac = (joint_pos_rad["gripper"] - _SIM_GRIPPER_CLOSED) / (_SIM_GRIPPER_OPEN - _SIM_GRIPPER_CLOSED)
    gripper_frac = max(0.0, min(1.0, gripper_frac))  # defensive -- a recorded value fractionally outside sim's own limits shouldn't become an out-of-range follower command
    action["gripper.pos"] = gripper_frac * 100.0
    return action


def main():
    with open(args.episode) as f:
        episode = json.load(f)
    downsampled = episode[0::2]  # 100Hz recording -> 50Hz sim control rate, same convention as every other script that reads these recordings
    segment = downsampled[: args.boundary_step + 1]
    print(f"[INFO]: Loaded {len(episode)} raw frames -> {len(downsampled)} downsampled -> "
          f"replaying the first {len(segment)} (reach-only, up to boundary step {args.boundary_step}).")

    trajectory = [sim_joint_pos_to_follower_action(frame["joint_pos"]) for frame in segment]

    print("[RESULT] Converted trajectory per-joint range (degrees for arm joints, 0-100 for gripper):")
    for key in trajectory[0]:
        vals = [pt[key] for pt in trajectory]
        print(f"  {key}: [{min(vals):+7.2f}, {max(vals):+7.2f}]")

    if args.dry_run:
        print("[RESULT] --dry-run: stopping here, never connected to the follower or moved anything.")
        return

    cfg = SOFollowerRobotConfig(port=args.port, id=args.id, max_relative_target=_MAX_RELATIVE_TARGET)
    follower = SOFollower(cfg)
    follower.connect(calibrate=True)
    print(f"[INFO]: Follower connected (calibrated={follower.is_calibrated}).")

    try:
        current = follower.get_observation()
        current = {k: v for k, v in current.items() if k.endswith(".pos")}
        print("[RESULT] Current position:")
        for k, v in current.items():
            print(f"  {k} = {v:+.2f}")

        print(f"[INFO]: Smoothly interpolating to the trajectory's first waypoint over "
              f"{args.interp_seconds:.1f}s in 5 seconds... Ctrl+C now to abort.")
        time.sleep(5)

        n_interp_steps = max(1, int(args.interp_seconds * args.hz))
        target0 = trajectory[0]
        for i in range(1, n_interp_steps + 1):
            frac = i / n_interp_steps
            interp = {k: current[k] + frac * (target0[k] - current[k]) for k in current}
            follower.send_action(interp)
            time.sleep(1.0 / args.hz)
        print("[INFO]: Reached the trajectory's first waypoint.")

        after_interp = follower.get_observation()
        print("[RESULT] Position after interpolation (should be close to the printed target range's start):")
        for k in current:
            print(f"  {k} = {after_interp[k]:+.2f} (target was {target0[k]:+.2f})")

        print(f"[INFO]: Replaying the {len(trajectory)}-step reach trajectory at {args.hz:.0f}Hz "
              f"in 3 seconds... Ctrl+C now to abort.")
        time.sleep(3)

        for i, waypoint in enumerate(trajectory):
            follower.send_action(waypoint)
            if i % 20 == 0:
                print(f"[INFO]: step {i}/{len(trajectory) - 1}")
            time.sleep(1.0 / args.hz)

        print("[RESULT] Reach trajectory replay complete.")
        final_pos = follower.get_observation()
        print("[RESULT] Final position:")
        for k in current:
            print(f"  {k} = {final_pos[k]:+.2f}")
    finally:
        follower.disconnect()
        print("[INFO]: Disconnected, torque released.")


if __name__ == "__main__":
    main()
