# SPDX-License-Identifier: BSD-3-Clause
"""First real step of sim-to-real deployment: connect to the physical SO-101
FOLLOWER arm (not the leader -- see docs/sim_to_real_checklist.md and
docs/real_arm_setup.md for the leader-only work done so far) and read its
live joint positions in a loop. Deliberately READ-ONLY -- this script never
calls SOFollower.send_action(), so there is no code path here that can
command the arm to move. Runs in the `lerobot` conda env (Python 3.12),
same reasoning as leader_reader.py (LeRobot's source uses 3.12+ syntax
incompatible with Isaac Sim's required 3.11).

No calibration file exists yet for a follower arm anywhere (confirmed:
~/.cache/huggingface/lerobot/calibration/teleoperators/ only has a
`so_leader/leader1.json`, no `robots/so_follower/` directory at all) --
SOFollower.connect(calibrate=True) (the default) will detect this via
`is_calibrated` being False and walk through LeRobot's own interactive
calibration flow automatically:
  1. Disables torque first (the arm goes limp / can be freely moved by
     hand -- this is what makes the whole procedure safe; no commanded
     motion happens at any point during calibration).
  2. Prompts to move the arm to the middle of its range of motion, then
     press ENTER (records homing offsets).
  3. Prompts to move all joints except wrist_roll through their full
     range of motion while it records min/max per joint, then press
     ENTER (records range of motion). wrist_roll is fixed at its full
     4096-step turn range instead of being manually swept, since it's a
     continuous joint with no natural end-of-travel to move to.
  4. Saves the result to ~/.cache/huggingface/lerobot/calibration/
     robots/so_follower/<id>.json for every future connection to reuse.

This needs to be run interactively BY THE USER (not over a one-shot SSH
command) -- the ENTER-key prompts above are timed against actually moving
the physical arm by hand, which only the person standing next to it can
do. Run it directly in a terminal on the Ubuntu box.

Usage:
    python follower_reader.py --port /dev/ttyACM0 --id follower1
"""

import argparse
import time

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default="/dev/ttyACM0")
parser.add_argument("--id", type=str, default="follower1")
parser.add_argument("--hz", type=float, default=2.0, help="Print rate (reads are cheap; this just limits terminal spam).")
args = parser.parse_args()


def main():
    # cameras intentionally omitted (empty dict, SOFollowerRobotConfig's own
    # default) -- this step is purely about joint-position calibration and
    # sign verification, no reason to also bring cameras into the loop yet.
    cfg = SOFollowerRobotConfig(port=args.port, id=args.id)
    follower = SOFollower(cfg)
    follower.connect(calibrate=True)  # see module docstring -- walks through interactive calibration if no file exists yet
    print(f"[INFO]: Follower arm connected (calibrated={follower.is_calibrated}). "
          f"Reading live joint positions -- Ctrl+C to stop. No motion is ever commanded by this script.")

    try:
        while True:
            obs = follower.get_observation()
            joint_str = "  ".join(f"{k}={v:+7.2f}" for k, v in obs.items() if k.endswith(".pos"))
            print(joint_str)
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        follower.disconnect()  # disable_torque_on_disconnect defaults to True -- arm goes limp on exit, not left holding torque


if __name__ == "__main__":
    main()
