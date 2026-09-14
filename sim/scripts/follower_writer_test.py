# SPDX-License-Identifier: BSD-3-Clause
"""FIRST script in this project that ever sends a command to a physical
robot -- everything before this (leader_reader.py, follower_reader.py) was
strictly read-only. Deliberately the most conservative possible test: read
the follower's own current joint positions, then command that EXACT SAME
position back -- a true no-op that proves the write path
(SOFollower.send_action()) actually works, with essentially zero risk of
real motion, before anything in this project ever commands a genuinely
different pose.

Uses max_relative_target (see SOFollowerConfig's own docstring) as a
defensive safety clamp regardless of the no-op logic above -- if the
read-back value and the write path somehow disagreed on units or format,
this caps how far any single command could move a joint before it ever
reaches the servo, rather than trusting "we're just sending back what we
just read" alone. 5 degrees for the arm joints, 5 (of 100) for the
gripper -- small in absolute terms for both, not expected to ever
actually engage during this specific no-op test.

Run this interactively, with the arm powered on and someone physically
present and ready to cut power if anything unexpected happens -- this is
the first command a physical motor in this project has ever received.
Not a one-shot SSH command, same reasoning as follower_reader.py.

Usage:
    python follower_writer_test.py --port /dev/ttyACM0 --id follower1
"""

import argparse
import time

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default="/dev/ttyACM0")
parser.add_argument("--id", type=str, default="follower1")
args = parser.parse_args()

_MAX_RELATIVE_TARGET = 5.0


def main():
    # cameras intentionally omitted (empty dict default) -- this is purely
    # a joint-command test, no reason to involve cameras yet.
    cfg = SOFollowerRobotConfig(port=args.port, id=args.id, max_relative_target=_MAX_RELATIVE_TARGET)
    follower = SOFollower(cfg)
    follower.connect(calibrate=True)  # calibration file already exists (follower1.json) -- this just loads it, doesn't re-run the interactive flow
    print(f"[INFO]: Follower connected (calibrated={follower.is_calibrated}).")
    print(f"[INFO]: max_relative_target={_MAX_RELATIVE_TARGET} (defensive clamp, see module docstring).")

    try:
        # Deliberately torque ENABLED here (unlike follower_reader.py,
        # which explicitly disables it for safe by-hand verification) --
        # a command can't do anything with torque off, and this script's
        # entire purpose is testing that the command path itself works.
        before = follower.get_observation()
        print("[RESULT] Current position (before):")
        for k, v in before.items():
            print(f"  {k} = {v:+.2f}")

        print("[INFO]: Commanding this EXACT position back (no-op test) in 3 seconds... "
              "Ctrl+C now to abort.")
        time.sleep(3)

        sent = follower.send_action(before)
        print("[RESULT] Action actually sent (after any max_relative_target clamping):")
        for k, v in sent.items():
            print(f"  {k} = {v:+.2f}")

        time.sleep(1.0)  # let the servo settle/report back accurately
        after = follower.get_observation()
        print("[RESULT] Position after the no-op command:")
        max_abs_diff = 0.0
        for k in before:
            diff = after[k] - before[k]
            max_abs_diff = max(max_abs_diff, abs(diff))
            print(f"  {k} = {after[k]:+.2f} (moved {diff:+.3f} from before)")
        print(f"[RESULT] Largest change on any joint: {max_abs_diff:.3f} "
              f"(should be near zero -- this was a no-op command)")
    finally:
        # ALWAYS release torque on the way out, including on Ctrl+C --
        # disable_torque_on_disconnect defaults to True.
        follower.disconnect()
        print("[INFO]: Disconnected, torque released.")


if __name__ == "__main__":
    main()
