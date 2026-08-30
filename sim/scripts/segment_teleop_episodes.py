# SPDX-License-Identifier: BSD-3-Clause
"""Segment a continuous teleop_bridge.py recording into individual
pick-and-place episodes, based on the cube's height trajectory.

A session recorded via teleop_bridge.py is one continuous, unbounded list
of steps spanning the whole session (multiple attempts, resets, idle time,
jiggling that never actually lifted the cube, etc.), not pre-split into
episodes. This detects each "cube rises above a real-lift threshold, then
returns to resting height" span as one candidate episode, and discards
spans that never rose high enough to count as a genuine pick (vs. jiggling
the cube around on the table without lifting it).

No Isaac Lab/torch dependency -- plain Python, safe to run in any env.

Usage:
    python segment_teleop_episodes.py --input teleop_recording.json --output-dir episodes/
"""

import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="/home/keerthan/SO-101-WM/sim/output/teleop_recording.json")
parser.add_argument("--output-dir", type=str, default="/home/keerthan/SO-101-WM/sim/output/teleop_episodes")
parser.add_argument(
    "--lift-threshold",
    type=float,
    default=0.05,
    help="Cube height (m) above which we consider it 'lifted' at all (vs. resting on the table).",
)
parser.add_argument(
    "--real-pick-threshold",
    type=float,
    default=0.08,
    help="Minimum peak height (m) during a lifted span to count as a genuine pick, not jiggling.",
)
parser.add_argument("--settle-steps", type=int, default=100, help="Steps below threshold before closing an episode.")
parser.add_argument("--pad-steps", type=int, default=30, help="Extra context steps kept before/after each episode.")
args = parser.parse_args()


def main():
    with open(args.input) as f:
        data = json.load(f)
    print(f"[INFO]: Loaded {len(data)} steps from {args.input}")

    os.makedirs(args.output_dir, exist_ok=True)

    episodes = []
    state = "resting"
    start_idx = None
    below_count = 0

    for i, step in enumerate(data):
        z = step["cube_pos"][2]
        if state == "resting":
            if z > args.lift_threshold:
                state = "lifted"
                start_idx = max(0, i - args.pad_steps)
                below_count = 0
        else:  # state == "lifted"
            if z <= args.lift_threshold:
                below_count += 1
                if below_count >= args.settle_steps:
                    end_idx = min(len(data), i + args.pad_steps)
                    episodes.append((start_idx, end_idx))
                    state = "resting"
                    start_idx = None
            else:
                below_count = 0

    # close a trailing episode if the recording ends mid-lift
    if state == "lifted" and start_idx is not None:
        episodes.append((start_idx, len(data)))

    print(f"[INFO]: Found {len(episodes)} candidate lifted spans")

    kept = 0
    for idx, (start, end) in enumerate(episodes):
        segment = data[start:end]
        peak_height = max(s["cube_pos"][2] for s in segment)
        if peak_height < args.real_pick_threshold:
            print(f"  span {idx}: steps [{start}:{end}], peak={peak_height:.3f}m -- discarded (jiggle, not a real lift)")
            continue
        out_path = os.path.join(args.output_dir, f"episode_{kept:03d}.json")
        with open(out_path, "w") as f:
            json.dump(segment, f)
        print(f"  span {idx}: steps [{start}:{end}], peak={peak_height:.3f}m -- saved to {out_path}")
        kept += 1

    print(f"[RESULT] Kept {kept} genuine pick episodes out of {len(episodes)} candidate spans.")


if __name__ == "__main__":
    main()
