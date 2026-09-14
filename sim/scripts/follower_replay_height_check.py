# SPDX-License-Identifier: BSD-3-Clause
"""Height-calibration step 2 of 2 (see calibrate_height_sim_predictions.py's
own docstring for the full motivation): command the SAME 4 known joint
configurations to the real follower, one at a time, pausing at each for
you to physically measure the gripper's height above the table.

Measure to the SAME point every time for a fair comparison: the point
where the two open fingertips would meet if closed (i.e. roughly the
center of the gap between the jaws when open) -- this is what
grasp_point_world() computes on the sim side, not gripper_frame_link's
own mounting-bracket origin or the fingertips' own tips.

Poses, in order: neutral (all joints at the "new_calib" zero -- arm
roughly upright/retracted), then reach_start/reach_mid/reach_end (the
same 3 real frames from episode_002's reach segment already replayed
safely on this exact hardware once before -- reach_end is the exact pose
the first real replay stopped at and you measured "about 2-3 inches
above the table" for by eye).

Same unit/sign conversion as follower_replay_reach.py (radians->degrees,
elbow_flex/wrist_roll sign flip, gripper radians->0-100 rescale) --
duplicated here rather than imported, since this is a standalone,
single-purpose diagnostic and importing across scripts that are each
meant to be read start-to-finish on their own isn't worth the coupling
for this.

Run this interactively, arm powered on, someone physically present --
same as every real-hardware script before this one. Have a ruler/tape
measure ready before starting.

Usage:
    python follower_replay_height_check.py --port /dev/ttyACM0 --id follower1
"""

import argparse
import math
import time

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default="/dev/ttyACM0")
parser.add_argument("--id", type=str, default="follower1")
parser.add_argument("--interp-seconds", type=float, default=3.0, help="Time to smoothly move between poses.")
parser.add_argument("--hz", type=float, default=10.0)
args = parser.parse_args()

_SIM_GRIPPER_CLOSED = -0.174533
_SIM_GRIPPER_OPEN = 1.74533
_ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
_FLIP_SIGN = {"elbow_flex", "wrist_roll"}  # confirmed 2026-09-13, docs/decisions.md

_MAX_RELATIVE_TARGET = {
    "shoulder_pan": 8.0, "shoulder_lift": 8.0, "elbow_flex": 8.0,
    "wrist_flex": 8.0, "wrist_roll": 8.0, "gripper": 10.0,
}

# Identical to calibrate_height_sim_predictions.py's own _POSES -- same
# source (episode_002's downsampled steps 0/97/194), same values.
_POSES_RAD = {
    "neutral": {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
                "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 0.0},
    "reach_start": {"shoulder_pan": -1.2965, "shoulder_lift": 0.2562, "elbow_flex": 0.0215,
                    "wrist_flex": 1.1316, "wrist_roll": -1.015, "gripper": 0.5875},
    "reach_mid": {"shoulder_pan": -1.2935, "shoulder_lift": 0.2532, "elbow_flex": 0.0368,
                  "wrist_flex": 1.0779, "wrist_roll": -1.0671, "gripper": 0.586},
    "reach_end": {"shoulder_pan": -1.3165, "shoulder_lift": 0.0875, "elbow_flex": 0.0338,
                  "wrist_flex": 1.2543, "wrist_roll": -1.1254, "gripper": 0.1563},
}


def sim_joint_pos_to_follower_action(joint_pos_rad: dict) -> dict:
    action = {}
    for joint in _ARM_JOINTS:
        deg = math.degrees(joint_pos_rad[joint])
        if joint in _FLIP_SIGN:
            deg = -deg
        action[f"{joint}.pos"] = deg
    gripper_frac = (joint_pos_rad["gripper"] - _SIM_GRIPPER_CLOSED) / (_SIM_GRIPPER_OPEN - _SIM_GRIPPER_CLOSED)
    gripper_frac = max(0.0, min(1.0, gripper_frac))
    action["gripper.pos"] = gripper_frac * 100.0
    return action


def interpolate_to(follower, current, target, seconds, hz):
    n_steps = max(1, int(seconds * hz))
    for i in range(1, n_steps + 1):
        frac = i / n_steps
        interp = {k: current[k] + frac * (target[k] - current[k]) for k in current}
        follower.send_action(interp)
        time.sleep(1.0 / hz)


def main():
    cfg = SOFollowerRobotConfig(port=args.port, id=args.id, max_relative_target=_MAX_RELATIVE_TARGET)
    follower = SOFollower(cfg)
    follower.connect(calibrate=True)
    print(f"[INFO]: Follower connected (calibrated={follower.is_calibrated}).")

    try:
        current = follower.get_observation()
        current = {k: v for k, v in current.items() if k.endswith(".pos")}

        for label, pose_rad in _POSES_RAD.items():
            target = sim_joint_pos_to_follower_action(pose_rad)
            print(f"\n[INFO]: Moving to pose '{label}' over {args.interp_seconds:.1f}s in 3 seconds... "
                  "Ctrl+C now to abort.")
            time.sleep(3)
            interpolate_to(follower, current, target, args.interp_seconds, args.hz)
            current = follower.get_observation()
            current = {k: v for k, v in current.items() if k.endswith(".pos")}
            print(f"[RESULT] Reached pose '{label}'. Measure the gripper's height above the table now "
                  "(to the center of the open jaws -- see module docstring) and note it down.")
            input("[INFO]: Press ENTER when you've recorded the measurement, to move to the next pose...")

        print("\n[RESULT] All 4 poses done.")
    finally:
        follower.disconnect()
        print("[INFO]: Disconnected, torque released.")


if __name__ == "__main__":
    main()
