# SPDX-License-Identifier: BSD-3-Clause
"""Bridge the real SO-101 leader arm into the Isaac Sim pick-and-place
scene in real time, so the arm can be physically teleoperated to
demonstrate a grasp while watching the sim (running non-headless, GUI
visible on the physical monitor).

Reads live leader-arm state from a shared JSON file (see leader_reader.py)
instead of importing LeRobot directly -- LeRobot's source uses Python 3.12+
syntax, incompatible with Isaac Sim's required Python 3.11, so they run as
separate processes bridged by this file. Run leader_reader.py (in the
`lerobot` conda env) BEFORE starting this script.

Also records the joint trajectory (and cube pose) to a JSON file,
periodically flushed to disk so a hard kill doesn't lose data -- meant to
be used afterward to (a) extract real grasp parameters (height, gripper
open/closed values) for the scripted demo, and (b) serve as an actual
demonstration for the model-training side of the project later.

Units: the leader arm reports the 5 arm joints in degrees (zero at the
calibrated middle position, matching our sim's URDF "new_calib" convention
of zero-at-middle) and the gripper as a 0-100 range. Converted here to the
radians our sim's ArticulationCfg expects. Direction (sign) of each joint,
and which end of the leader's gripper range is open vs. closed, has NOT
been verified yet -- watch the sim on first run and flag if anything moves
backwards.

Uses the camera-free PickPlaceSceneBaseCfg -- offscreen cameras (scene_camera,
side_camera, top_camera, wrist_camera) render every frame regardless of
whether their output is read, and having 3-4 active was a real, measurable
source of lag during teleop. No --enable_cameras needed here.

Cube position: randomized at session start, and re-randomizable between
reps via a simple trigger file (see RESET_TRIGGER_FILE below) -- both
sampled from the exact same train region PickPlaceEnvCfg uses for TD-MPC2
(sim/envs/pickplace_env.py's cube_x_range/cube_y_range). This was a real
gap, found while planning the diffusion policy's data collection: this
script previously left the cube at the scene's one fixed default spawn
position for an entire session, with its position across "reps" only ever
determined by wherever a human happened to leave it after the previous
pick (already documented as a source of confusion once, in
docs/reward_function.md's validate_reward_function.py section). For a
FAIR comparison against TD-MPC2 later (docs/evaluation_plan.md's Test 2,
novel-position generalization), any demonstrations collected for the
diffusion policy need to respect the same train/held-out position split,
not scatter across whatever positions a human happened to leave the cube.

Usage (run non-headless, on the machine with the monitor + leader arm):
    ./isaaclab.sh -p /path/to/teleop_bridge.py

To re-randomize the cube position between reps during a session, from a
DIFFERENT terminal on the same machine:
    touch /tmp/teleop_reset_cube
This script polls for that file every step and deletes it once handled --
avoids needing a real-time keyboard listener inside the sim loop, same
lightweight file-based IPC pattern as leader_reader.py's shared state file.
"""

import argparse
import json
import math
import os
import random
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleop the real SO-101 leader arm into the Isaac Sim scene.")
parser.add_argument(
    "--leader-state-file", type=str, default="/tmp/leader_state.json", help="Shared file written by leader_reader.py."
)
parser.add_argument(
    "--output", type=str, default="/home/keerthan/SO-101-WM/sim/output/teleop_recording.json", help="Recording path."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from scenes.pickplace_scene import PickPlaceSceneBaseCfg  # isort:skip

# Kept in sync BY HAND with PickPlaceEnvCfg.cube_x_range/cube_y_range
# (sim/envs/pickplace_env.py) -- not imported directly to avoid pulling in
# that env's reward/training machinery into a simple teleop script. Two
# float tuples are a low enough duplication risk to accept; if these ever
# drift apart, TD-MPC2 and diffusion-policy demonstrations would no longer
# share a train region, silently breaking Test 2's fairness guarantee
# (docs/evaluation_plan.md).
_CUBE_X_RANGE = (0.05, 0.25)
_CUBE_Y_RANGE = (-0.20, 0.20)
RESET_TRIGGER_FILE = "/tmp/teleop_reset_cube"

# Leader gripper is a 0-100 range; our sim's gripper joint is in radians,
# URDF range -0.174533 (confirmed CLOSED) to 1.74533 (confirmed OPEN) --
# see run_pickplace_demo.py's comments. Assuming leader 0% = closed,
# 100% = open; flip if this turns out backwards on the real arm.
_SIM_GRIPPER_CLOSED = -0.174533
_SIM_GRIPPER_OPEN = 1.74533

_ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def leader_action_to_sim_joint_pos(action: dict[str, float]) -> dict[str, float]:
    """Convert a leader action dict (degrees + gripper 0-100) to sim joint positions (radians)."""
    sim_pos = {}
    for joint in _ARM_JOINTS:
        sim_pos[joint] = math.radians(action[f"{joint}.pos"])
    gripper_pct = action["gripper.pos"] / 100.0
    sim_pos["gripper"] = _SIM_GRIPPER_CLOSED + gripper_pct * (_SIM_GRIPPER_OPEN - _SIM_GRIPPER_CLOSED)
    return sim_pos


def randomize_cube_position(scene):
    """Teleports the (simulated) cube to a fresh random position sampled
    from the same train region PickPlaceEnvCfg uses -- see _CUBE_X_RANGE/
    _CUBE_Y_RANGE's comment above for why this must stay in sync with it.
    Safe to call at any time (the cube is a simple RigidObject, teleporting
    it doesn't disturb the robot)."""
    cube = scene["cube"]
    default_state = cube.data.default_root_state.clone()
    default_state[0, 0] = random.uniform(*_CUBE_X_RANGE)
    default_state[0, 1] = random.uniform(*_CUBE_Y_RANGE)
    # default_root_state is per-env-LOCAL (same numeric values regardless
    # of which env clone), not world-frame -- env_origins must always be
    # added when writing it back, same convention as
    # PickPlaceEnv._reset_idx() (verified there against IsaacLab's own
    # cartpole_env.py reference). Only ever (0,0,0) here since this script
    # always runs num_envs=1, but kept for correctness/consistency anyway.
    default_state[0, 0:3] += scene.env_origins[0]
    cube.write_root_pose_to_sim(default_state[:, :7])
    cube.write_root_velocity_to_sim(torch.zeros_like(default_state[:, 7:]))


def atomic_save_json(data, path):
    """Write JSON atomically (temp file + rename) so a mid-write interrupt
    (e.g. Ctrl+C / SIGINT arriving during json.dump) can never leave a
    truncated, corrupted file -- the rename only happens once the full
    write has succeeded. Hit this exact corruption once without it: a
    session got interrupted mid-save and the recording was truncated,
    losing the tail of the data (recoverable by hand that time, but not
    guaranteed).
    """
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.replace(tmp_path, path)


def read_leader_state(path, last_t):
    """Read the shared state file if it has a newer timestamp than last seen. Returns (action, t) or (None, last_t)."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None, last_t
    if data["t"] <= last_t:
        return None, last_t
    return data["action"], data["t"]


def main():
    os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = PickPlaceSceneBaseCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    randomize_cube_position(scene)
    if os.path.exists(RESET_TRIGGER_FILE):
        os.remove(RESET_TRIGGER_FILE)  # stale from a previous session -- don't immediately re-trigger
    print(f"[INFO]: Cube randomized within the train region {_CUBE_X_RANGE} x {_CUBE_Y_RANGE}.")
    print(f"[INFO]: To re-randomize between reps, from another terminal: touch {RESET_TRIGGER_FILE}")

    robot = scene["robot"]
    joint_names = robot.data.joint_names
    joint_indices = [joint_names.index(j) for j in _ARM_JOINTS + ["gripper"]]

    print(f"[INFO]: Waiting for leader arm state at {args_cli.leader_state_file} ...")
    print("[INFO]: (make sure leader_reader.py is running in the `lerobot` conda env)")

    sim_dt = sim.get_physics_dt()
    recording = []
    last_save = time.time()
    last_t = 0.0
    step = 0
    current_sim_joint_pos = None

    try:
        while simulation_app.is_running():
            if os.path.exists(RESET_TRIGGER_FILE):
                os.remove(RESET_TRIGGER_FILE)
                randomize_cube_position(scene)
                print(f"[INFO]: Cube re-randomized at step {step}.")

            action, last_t = read_leader_state(args_cli.leader_state_file, last_t)
            if action is not None:
                current_sim_joint_pos = leader_action_to_sim_joint_pos(action)

            if current_sim_joint_pos is not None:
                target = torch.zeros((1, len(joint_indices)), device=sim.device)
                for i, joint in enumerate(_ARM_JOINTS + ["gripper"]):
                    target[0, i] = current_sim_joint_pos[joint]
                robot.set_joint_position_target(target, joint_ids=joint_indices)

            scene.write_data_to_sim()
            # Render every other step, not every step -- physics still runs
            # at full rate (100Hz) so control/contact accuracy is unaffected,
            # but this roughly halves GPU rendering cost. The eye can't tell
            # the difference between 50fps and 100fps anyway.
            sim.step(render=(step % 2 == 0))
            scene.update(sim_dt)

            if current_sim_joint_pos is not None:
                cube_pos = scene["cube"].data.root_pos_w[0].cpu().tolist()
                recording.append(
                    {
                        "t": step * sim_dt,
                        "joint_pos": current_sim_joint_pos,
                        "cube_pos": cube_pos,
                    }
                )
            step += 1

            if time.time() - last_save > 2.0:
                atomic_save_json(recording, args_cli.output)
                last_save = time.time()
    finally:
        atomic_save_json(recording, args_cli.output)
        print(f"[RESULT] Saved {len(recording)} steps to {args_cli.output}")


if __name__ == "__main__":
    main()
    simulation_app.close()
