# SPDX-License-Identifier: BSD-3-Clause
"""Continuously read the SO-101 leader arm's live joint positions and write
them to a shared JSON file, at high frequency (overwriting each time).

Runs in the `lerobot` conda env (Python 3.12). Exists as a separate process
from the Isaac Sim side because LeRobot's source now uses Python 3.12+
syntax (PEP 695 type aliases), incompatible with Isaac Sim's required
Python 3.11 -- they can't share a process, so this is a simple
file-based IPC bridge instead.

Usage:
    python leader_reader.py --port /dev/ttyACM0 --id leader1
"""

import argparse
import json
import time

from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default="/dev/ttyACM0")
parser.add_argument("--id", type=str, default="leader1")
parser.add_argument("--output", type=str, default="/tmp/leader_state.json")
args = parser.parse_args()


def main():
    cfg = SOLeaderTeleopConfig(port=args.port, id=args.id)
    leader = SOLeader(cfg)
    leader.connect(calibrate=False)
    print(f"[INFO]: Leader arm connected, writing live state to {args.output}")

    try:
        while True:
            action = leader.get_action()
            with open(args.output, "w") as f:
                json.dump({"t": time.time(), "action": action}, f)
    except KeyboardInterrupt:
        pass
    finally:
        leader.disconnect()


if __name__ == "__main__":
    main()
