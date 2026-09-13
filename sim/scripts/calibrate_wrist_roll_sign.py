# SPDX-License-Identifier: BSD-3-Clause
"""Resolve wrist_roll's sign convention specifically, from the EXACT
viewpoint the real-hardware check used: standing behind the wrist, looking
toward the fingertips, is the sim's positive direction clockwise or
counterclockwise?

Deliberately NOT answered by looking at a rendered image and judging by eye
-- a rotation this close to the viewing axis is genuinely hard to read
correctly from a static picture (confirmed by trying). Also NOT answered
by tracking a reference point's raw world position through the joint's
lower/upper limits and computing its swept angle -- tried that too, and
it came back internally inconsistent (moving_jaw_so101_v1_link's radius
from the assumed rotation center varied from 0.014m to 0.042m across
poses, when a point genuinely circling a fixed axis through a fixed
center must have a CONSTANT radius; the likely cause is that wrist_link's
own body origin isn't precisely ON the true rotation axis line, which
corrupts a position-based angle even when the axis DIRECTION is right).

What actually works, used throughout below: read gripper_link's (the
direct child of the wrist_roll joint) ACTUAL world orientation at two
different wrist_roll values, and extract the true rotation axis/angle
straight from the relative quaternion between them (quat_mul + quat_inv,
standard axis-angle extraction) -- no URDF frame composition, no assumed
rotation center, nothing to get subtly wrong. A first attempt at the axis
alone (composing wrist_link's orientation with the joint's local (0,0,1)
axis by hand) got a materially wrong answer -- a 163-degree commanded
rotation only recovered as a ~25-degree angle change -- which is what
motivated re-deriving it this fully empirical way instead.

Usage:
    ./isaaclab.sh -p /path/to/calibrate_wrist_roll_sign.py --headless --enable_cameras
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Empirically resolve wrist_roll's CW/CCW sign convention.")
parser.add_argument(
    "--output-dir", type=str, default="/home/keerthan/SO-101-WM/sim/output/joint_sign_calibration", help="Output dir."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import sys  # isort:skip

# Force line-buffered stdout -- this box's known Kit shutdown-hang means the
# process can sit hung in simulation_app.close() long after the real work
# (including these RESULT prints) is done; without this, Python's default
# full-buffering when stdout isn't a TTY means nothing shows up in a
# redirected log until the buffer happens to fill, and a SIGKILL to clear a
# hung process would lose whatever was still sitting in that buffer. See
# replay_demo_to_buffer.py / train_tdmpc2_pickplace.py's own docstrings for
# the same fix, applied here after hitting exactly this problem directly.
sys.stdout.reconfigure(line_buffering=True)

import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_inv, quat_mul

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from scenes.pickplace_scene import PickPlaceSceneCfg  # isort:skip

_WRIST_ROLL_LOWER = -2.74385
_WRIST_ROLL_UPPER = 2.84121


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = PickPlaceSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    camera = scene["scene_camera"]
    joint_names = robot.data.joint_names
    body_names = robot.data.body_names
    sim_dt = sim.get_physics_dt()

    def hold_pose(joint_pos_dict, steps=90):
        target = torch.zeros((1, len(joint_names)), device=sim.device)
        for name, val in joint_pos_dict.items():
            target[0, joint_names.index(name)] = val
        for _ in range(steps):
            robot.set_joint_position_target(target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

    hold_pose({}, steps=120)

    wrist_link_idx = body_names.index("wrist_link")
    gripper_link_idx = body_names.index("gripper_link")
    wrist_pos_w = robot.data.body_pos_w[0, wrist_link_idx].clone()

    # Derive wrist_roll's world-space rotation axis EMPIRICALLY, from two
    # actually-measured orientations, rather than composing the URDF's
    # parent-link orientation with the joint's own <origin rpy> by hand --
    # the first attempt did exactly that (local_z rotated by wrist_link's
    # own world quaternion alone) and got a materially wrong axis: a
    # 163-degree commanded rotation only showed up as a ~25-degree angle
    # change once measured, meaning most of the real rotation was being
    # invisibly absorbed as motion along the wrong "forward" direction
    # rather than as pure angle change in the projected 2D plane -- the
    # tell that the axis itself, not the angle math around it, was off.
    # This instead reads gripper_link's ACTUAL world orientation at two
    # different wrist_roll values and extracts the true axis from their
    # relative rotation -- no URDF frame composition to get wrong.
    quat_at_zero = robot.data.body_quat_w[0, gripper_link_idx].clone()
    hold_pose({"wrist_roll": _WRIST_ROLL_UPPER}, steps=400)
    quat_at_upper_probe = robot.data.body_quat_w[0, gripper_link_idx].clone()
    hold_pose({}, steps=400)

    q_rel = quat_mul(quat_at_upper_probe.unsqueeze(0), quat_inv(quat_at_zero.unsqueeze(0)))[0]
    # Isaac Lab's quaternion convention is (w, x, y, z) -- axis is the
    # normalized vector part; recovered total angle (2*acos(w)) is printed
    # purely as a sanity check against the known commanded value
    # (_WRIST_ROLL_UPPER, ~162.8 deg) -- if these disagree, the axis
    # extraction itself is untrustworthy and nothing downstream should be
    # believed either.
    w, xyz = q_rel[0].item(), q_rel[1:]
    # Quaternion double-cover consistency fix (see the lower/upper loop's
    # own comment below for the full reasoning) -- didn't happen to matter
    # numerically for this specific ~162.8 deg rotation (w was already
    # positive here), but applied for correctness rather than relying on
    # that.
    if w < 0:
        w, xyz = -w, -xyz
    axis_world = xyz / xyz.norm()
    recovered_angle_deg = math.degrees(2 * math.acos(max(-1.0, min(1.0, w))))
    print(f"[RESULT] wrist_link world pos: {wrist_pos_w.cpu().numpy()}")
    print(f"[RESULT] wrist_roll's EMPIRICALLY-measured world-space rotation axis: {axis_world.cpu().numpy()}")
    print(f"[RESULT] sanity check -- recovered rotation angle from 0 to upper: {recovered_angle_deg:.1f} deg "
          f"(commanded {math.degrees(_WRIST_ROLL_UPPER):.1f} deg -- these must match closely)")

    axis_np = axis_world.cpu()
    wrist_pos_np = wrist_pos_w.cpu()
    eye = (wrist_pos_np - 0.25 * axis_np).unsqueeze(0).to(sim.device)
    look_at = (wrist_pos_np + 0.25 * axis_np).unsqueeze(0).to(sim.device)
    camera.set_world_poses_from_view(eye, look_at)

    def capture(label):
        sim.render()
        rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        out_path = os.path.join(args_cli.output_dir, f"{label}.png")
        Image.fromarray(rgb).save(out_path)
        print(f"[RESULT] Saved {out_path}")

    capture("wrist_roll_from_behind_neutral")

    # Determine each limit's rotation SENSE the same, already-verified-
    # reliable way axis_world itself was derived -- via the relative
    # quaternion from neutral, not by tracking a reference point's raw
    # position (tried that: moving_jaw_so101_v1_link's radius from the
    # assumed axis came back inconsistent across neutral/lower/upper --
    # 0.042m, 0.014m, 0.026m, which should all be equal for a point
    # genuinely tracing a circle around a fixed axis through a fixed
    # center -- meaning wrist_link's own origin isn't precisely ON the
    # true rotation axis line, corrupting both the radius AND the angle
    # computed from it. The quaternion-relative approach never needs an
    # assumed center point at all, so it isn't exposed to that error).
    #
    # "positive" (CCW as viewed with the axis pointing TOWARD the viewer)
    # vs "negative" (CW) here is standard right-hand-rule convention about
    # axis_world specifically -- and since this camera looks ALONG
    # +axis_world (axis pointing AWAY from the viewer, into the screen,
    # per the eye/look_at setup above), a RHR-positive rotation about
    # axis_world appears CLOCKWISE to this camera (viewing from the
    # opposite side of an axis flips the apparent sense -- the standard
    # "clock viewed from behind runs backwards" fact) -- so
    # signed_deg > 0 below means CW as actually seen from behind the
    # wrist looking toward the fingertips, < 0 means CCW.
    quat_at_zero_2 = robot.data.body_quat_w[0, gripper_link_idx].clone()
    wrist_roll_idx = joint_names.index("wrist_roll")
    for label, value in [("lower", _WRIST_ROLL_LOWER), ("upper", _WRIST_ROLL_UPPER)]:
        # wrist_roll's range (~5.6 rad, nearly a full turn) is far larger
        # than any other joint's -- 90 steps (0.9s) was tuned against the
        # other 5 joints' much smaller excursions and was NOT enough for
        # this one to actually finish traveling to the target on the very
        # first attempt (confirmed by that attempt's suspiciously small
        # deltas). 400 steps (4s) comfortably covers the full range.
        hold_pose({"wrist_roll": value}, steps=400)
        achieved = robot.data.joint_pos[0, wrist_roll_idx].item()
        print(f"[RESULT] {label}: commanded wrist_roll={value:+.4f}, ACTUALLY ACHIEVED={achieved:+.4f} "
              f"(must match closely, else the hold didn't converge)")
        capture(f"wrist_roll_from_behind_{label}")

        quat_at_limit = robot.data.body_quat_w[0, gripper_link_idx].clone()
        q_rel_limit = quat_mul(quat_at_limit.unsqueeze(0), quat_inv(quat_at_zero_2.unsqueeze(0)))[0]
        w_l, xyz_l = q_rel_limit[0].item(), q_rel_limit[1:]
        # Quaternion double-cover: q=(w,xyz) and -q=(-w,-xyz) represent the
        # IDENTICAL rotation. To get a CONSISTENT (axis, angle) pair (angle
        # in [0,180], w=cos(angle/2) >= 0), negate BOTH w and xyz together
        # when w<0 -- negating only one (e.g. taking abs(w) while leaving
        # xyz untouched, an earlier, wrong version of this exact check)
        # decouples the angle from its own axis, silently corrupting the
        # direction determination below.
        if w_l < 0:
            w_l, xyz_l = -w_l, -xyz_l
        angle_l = math.degrees(2 * math.acos(max(-1.0, min(1.0, w_l))))
        axis_l = xyz_l / xyz_l.norm()
        # Now that (axis_l, angle_l) consistently describe the same
        # rotation, a dot product against axis_world tells us whether this
        # limit's rotation was in the SAME sense (parallel axis) or
        # OPPOSITE sense (antiparallel axis) as the upper-limit rotation
        # axis_world was itself derived from.
        same_direction = torch.dot(axis_l, axis_world).item() > 0
        signed_deg = angle_l if same_direction else -angle_l
        sense = "CLOCKWISE (CW)" if signed_deg > 0 else "COUNTERCLOCKWISE (CCW)"
        print(f"[RESULT] {label} (wrist_roll={value:+.3f}): rotation from neutral = {signed_deg:+.1f} deg "
              f"about the SAME axis_world used for the camera -> {sense}, "
              f"as seen from behind the wrist looking toward the fingertips "
              f"(magnitude sanity check: |{signed_deg:.1f}| should be close to "
              f"{abs(math.degrees(value)):.1f} = |commanded wrist_roll in degrees|)")
        hold_pose({}, steps=400)

    print("[RESULT] No crash -- wrist_roll's CW/CCW sense computed directly, not eyeballed.")


if __name__ == "__main__":
    main()
    simulation_app.close()
