# SPDX-License-Identifier: BSD-3-Clause
"""Reward function for the SO-101 pick-and-place task.

This is used to train the TD-MPC2-style world model side of the project's
comparison -- not the diffusion policy, which learns purely from
demonstrations and never sees a reward at all. See docs/reward_function.md
for the full design rationale, the alternatives considered, and an
empirical validation against a real captured teleop demonstration
(sim/scripts/validate_reward_function.py).

Deliberately independent of Isaac Lab/torch/omni: every function here
takes plain floats/tuples and returns plain floats/dicts. The (not yet
built) training env is expected to pull its torch tensors down to
CPU/Python at the call site. This keeps the reward function importable,
unit-testable, and inspectable without booting Isaac Sim, and keeps
"what counts as a good outcome" decoupled from "how the simulator
represents state."

Everything here reads PRIVILEGED simulator state (exact cube position,
exact gripper position, contact/grasp state) that a real robot doesn't
have without extra instrumentation. That's fine and standard for
model-based RL reward functions -- this is only ever used to generate the
training signal in sim. The trained policy itself acts only on camera
images and proprioception, never on this function's inputs directly.
"""

import os
import sys
from dataclasses import dataclass

if __name__ == "__main__":
    # only needed to run this file's self-test standalone -- any real
    # caller already has "sim" on sys.path before importing this module
    # (same convention as scenes/pickplace_scene.py, robots/so101.py).
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robots.grasp_geometry import grasp_point_world  # noqa: F401  (re-exported for env convenience)


@dataclass
class PickPlaceRewardConfig:
    """All tunable constants in one place, with the reasoning for each
    default given inline. None of these are claimed to be optimal --
    they're principled starting points, meant to be revisited once real
    training runs expose problems (reward hacking, too-sparse signal,
    wrong scale relative to other terms), not treated as final answers.
    """

    # -- Task geometry --
    # Where the cube should end up. Reuses the exact "place zone" the
    # scripted demo (run_pickplace_demo.py) already established at
    # (-0.15, 0.15, ...) rather than inventing a second, inconsistent
    # target -- one arbitrary place location per project, not two. Height
    # matches pickplace_scene.py's actual cube-resting convention
    # (cube_size / 2 = 0.015), not the demo script's slightly-off 0.02.
    # This will become a randomized value once domain randomization for
    # object/target pose is wired in (see docs/project_plan.md) -- fixed
    # for now, both because that's simpler for this first version and
    # because it's what let this module be validated against fixed-
    # position recorded demonstrations (see docs/reward_function.md).
    target_pos: tuple[float, float, float] = (-0.15, 0.15, 0.015)
    place_tolerance: float = 0.02  # ~cube half-size: "close enough to the target to count as placed"
    place_max_speed: float = 0.05  # m/s -- must be ~at rest at the target, not just passing through it

    # -- Table / failure bounds --
    # Matches pickplace_scene.py: table top at z=0, size=(1.2, 1.2, ...)
    # centered at the origin, so it spans -0.6..+0.6 in both x and y.
    # (Enlarged from 0.6m/0.3 half-extent on 2026-08-30.)
    table_z: float = 0.0
    table_half_extent: float = 0.6
    fall_z_threshold: float = -0.05  # cube center below this -> fell off/through the table

    # -- Grasp / lift detection --
    # Smaller than segment_teleop_episodes.py's 0.05m/0.08m thresholds --
    # those were tuned to reject jittery/noisy TELEOP input. Here we score
    # exact simulated state directly (no human jitter to filter out), so a
    # tighter threshold gives an earlier, more precise "grasped" signal
    # without misfiring on noise, since there isn't any.
    lift_threshold: float = 0.02
    # Gripper joint position (radians) at/below which the gripper counts
    # as "closed enough to be gripping." The URDF's gripper joint range is
    # -0.174533 (closed) to 1.74533 (open) -- this sits in the closed half
    # of that range. A placeholder in the right ballpark (see
    # so101_asset_notes.md); not yet cross-checked against the real grasp
    # width needed to hold this specific cube, since that depends on the
    # cube's physical size relative to the real gripper's actual travel.
    gripper_closed_threshold: float = 0.3
    # How close the gripper's grasp point must be to the cube to count as
    # "actually holding it," on top of the height+joint-angle checks
    # above. Without this, a cube that's merely settling/bouncing slightly
    # above rest height (e.g. right after being released nearby, common
    # right at the start of a segmented episode cut from a longer teleop
    # session) combined with a coincidentally-closed gripper joint
    # elsewhere on the table would falsely register as "grasped" -- this
    # exact false positive was caught empirically while validating this
    # module against real recorded data (episode start: cube at 0.027m,
    # barely above the 0.02m lift_threshold, gripper joint happened to
    # read as closed, but the gripper was physically 0.38m away). See
    # docs/reward_function.md.
    grasp_proximity_threshold: float = 0.05

    # -- Shaping weights --
    # Both phases use a bounded (1 - tanh(d / scale)) shaping term rather
    # than raw negative distance. Two reasons: (1) it saturates smoothly
    # instead of handing out unbounded penalties for being far away, which
    # would let one bad early state dominate an entire trajectory's return
    # and (2) TD-MPC2 trains a neural network to PREDICT this reward --
    # a smooth, bounded target is an easier regression problem than an
    # unbounded one, and this shape (~1 near the goal, ~0 far away,
    # smooth in between) is a standard, well-behaved choice for that.
    reach_weight: float = 1.0
    reach_scale: float = 0.15  # meters -- distance at which reach shaping is ~half-saturated
    grasp_bonus: float = 0.5  # flat per-step bonus while grasped -- rewards closing AND holding, not just touching
    place_weight: float = 1.0
    place_scale: float = 0.15
    success_bonus: float = 10.0  # one-time; must clearly dominate the dense terms so success always wins
    action_penalty_weight: float = 0.01  # small; discourages unrealistic/jerky motion, not a primary objective


def _dist3(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _norm3(v):
    return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5


def _tanh(x):
    # avoid importing math just for this -- also keeps this file
    # dependency-free beyond the stdlib dataclass import.
    if x > 20.0:
        return 1.0
    if x < -20.0:
        return -1.0
    e2x = 2.718281828459045 ** (2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)


def is_grasped(gripper_pos, cube_pos, gripper_joint_pos, cfg: PickPlaceRewardConfig) -> bool:
    """True once the cube is genuinely lifted, the gripper is closed, AND
    the gripper is actually near the cube -- all three together. Height
    and joint-angle alone aren't enough: a cube settling slightly above
    rest height with a coincidentally-closed gripper elsewhere on the
    table would otherwise falsely register as grasped. The proximity
    check is what rules that out.

    This is the STRICT, stateless "was a genuine liftoff just detected"
    check -- it requires height evidence every single call, which makes
    it correct for detecting the initial pickup but WRONG for deciding
    whether the gripper is still holding the cube during a later
    controlled lowering (e.g. onto the place target, whose resting height
    is BELOW lift_threshold by construction -- see is_holding() below,
    which is what compute_reward() actually uses for the phase switch).
    Kept as-is for diagnostics/logging (e.g. "first genuine grasp
    detected at step N"), where the strict liftoff semantics are exactly
    what's wanted."""
    cube_height = cube_pos[2] - cfg.table_z
    close_enough = _dist3(gripper_pos, cube_pos) < cfg.grasp_proximity_threshold
    return (
        cube_height > cfg.lift_threshold
        and gripper_joint_pos <= cfg.gripper_closed_threshold
        and close_enough
    )


def is_holding(gripper_pos, cube_pos, gripper_joint_pos, was_holding: bool, cfg: PickPlaceRewardConfig) -> bool:
    """True if the gripper is currently, physically holding the cube --
    the condition compute_reward() actually uses to pick approach vs.
    transport shaping. This is deliberately NOT the same as is_grasped():
    the place target's resting height (0.015m) is below lift_threshold
    (0.02m) by construction (a successful place always ends with the
    cube back down near table height), so a naive height check on every
    step would flip "grasped" back to False the instant the cube is
    lowered onto the target -- exactly the moment precise "get closer to
    the target" shaping matters most. Caught via a careful manual trace of
    the reward self-test's own numbers, not by the empirical replay
    validation (neither validated episode ever reached a genuine
    lift-and-carry state -- see docs/reward_function.md).

    Fix: require height evidence only to ESTABLISH holding (same as
    is_grasped -- a gripper that's merely closed near the cube without
    ever having lifted it must not count, which also avoids a different
    problem: incentivizing the policy to delay closing the gripper at
    all, since closing prematurely would otherwise cause an immediate,
    unearned switch to the lower-reward transport shaping). Once holding
    has been established, it's allowed to PERSIST across subsequent steps
    purely on proximity + closed-gripper evidence, with no further height
    requirement -- correctly covering the final lowering-to-target
    sequence. Holding still ends immediately if the gripper opens or
    moves away from the cube (proximity or joint-angle check fails),
    which is exactly what should happen on release.

    This is the one deliberate, explicitly-justified exception to this
    module's otherwise fully stateless design -- `was_holding` is an
    ordinary function argument (same inputs always give the same output),
    so this is still a pure function, just one whose result legitimately
    depends on very recent history, which a single-timestep snapshot
    cannot resolve on its own. The caller (the training env) is
    responsible for persisting `was_holding` across steps and passing back
    whatever this function returns."""
    cube_height = cube_pos[2] - cfg.table_z
    close_enough = _dist3(gripper_pos, cube_pos) < cfg.grasp_proximity_threshold
    closed = gripper_joint_pos <= cfg.gripper_closed_threshold
    return closed and close_enough and (cube_height > cfg.lift_threshold or was_holding)


def is_placed(cube_pos, cube_lin_vel, cfg: PickPlaceRewardConfig) -> bool:
    """True once the cube is at rest at the target -- the speed check
    matters: without it, swinging the cube THROUGH the target location
    while carrying it elsewhere would falsely trigger success."""
    d = _dist3(cube_pos, cfg.target_pos)
    speed = _norm3(cube_lin_vel)
    return d < cfg.place_tolerance and speed < cfg.place_max_speed


def is_failed(cube_pos, cfg: PickPlaceRewardConfig) -> bool:
    """True once the episode is unrecoverable -- cube fell through/off
    the table. Used by the (future) env wrapper to terminate early rather
    than wasting steps on a trajectory that can no longer succeed."""
    x, y, z = cube_pos
    if z < cfg.fall_z_threshold:
        return True
    margin = 0.05  # small allowance past the table's physical edge before calling it "off"
    if abs(x) > cfg.table_half_extent + margin or abs(y) > cfg.table_half_extent + margin:
        return True
    return False


def compute_reward(
    gripper_pos,
    cube_pos,
    cube_lin_vel,
    gripper_joint_pos,
    joint_vel,
    was_holding: bool,
    cfg: PickPlaceRewardConfig = PickPlaceRewardConfig(),
):
    """Computes one step's scalar reward plus a diagnostics dict.

    Two-phase dense shaping, switching on the grasp event:
      - Not holding: reward for closing the gripper-to-cube distance
        ("approach"/"reach").
      - Holding: a flat bonus for holding on, plus reward for closing the
        cube-to-target distance ("transport"/"place").

    This phase switch (rather than one continuous "reward height" term)
    is deliberate: a naive reward that just scores cube height would
    conflict with placing, since placing requires eventually LOWERING the
    cube back down at the target -- a policy chasing pure height would
    have no incentive to ever put the cube down. Framing phase 2 as
    "distance to a 3D target point at table height" makes lifting, moving
    sideways, and lowering all part of one continuous, non-conflicting
    signal that naturally decreases to zero exactly when the task is done.

    The phase switch uses is_holding(), not is_grasped() -- see
    is_holding()'s own docstring for why a naive height-gated check on
    every step gets this wrong specifically during the final
    lowering-to-target sequence (a real bug caught by manually tracing
    this function's own self-test, not by the empirical replay
    validation).

    Args:
        gripper_pos: (x, y, z) world position of the actual grasp point
            (use grasp_point_world() on gripper_frame_link's pose -- NOT
            gripper_frame_link's own origin, which is ~8cm away from
            where the fingers actually are).
        cube_pos: (x, y, z) world position of the cube.
        cube_lin_vel: (vx, vy, vz) world linear velocity of the cube.
        gripper_joint_pos: current "gripper" joint position, radians.
        joint_vel: iterable of all actuated joint velocities (rad/s),
            used for the action/energy penalty.
        was_holding: whether is_holding() returned True on the PREVIOUS
            step for this same environment (False on the first step after
            a reset). The caller is responsible for persisting this
            across steps -- see is_holding()'s docstring for why.
        cfg: reward configuration/weights.

    Returns:
        (reward: float, info: dict) -- info carries the boolean
        holding/grasped/placed/failed flags and the individual reward
        components. info["holding"] is what the caller must feed back in
        as next step's `was_holding`. info["grasped"] is the separate,
        strict, stateless liftoff-detection flag (see is_grasped()) --
        useful for logging, not for driving the phase switch.
    """
    holding = is_holding(gripper_pos, cube_pos, gripper_joint_pos, was_holding, cfg)
    grasped = is_grasped(gripper_pos, cube_pos, gripper_joint_pos, cfg)
    placed = is_placed(cube_pos, cube_lin_vel, cfg)
    failed = is_failed(cube_pos, cfg)

    if not holding:
        d_reach = _dist3(gripper_pos, cube_pos)
        dense = cfg.reach_weight * (1.0 - _tanh(d_reach / cfg.reach_scale))
        phase = "approach"
    else:
        d_place = _dist3(cube_pos, cfg.target_pos)
        dense = cfg.grasp_bonus + cfg.place_weight * (1.0 - _tanh(d_place / cfg.place_scale))
        phase = "transport"

    action_penalty = cfg.action_penalty_weight * sum(v * v for v in joint_vel)

    reward = dense - action_penalty
    if placed:
        reward += cfg.success_bonus

    info = {
        "phase": phase,
        "holding": holding,
        "grasped": grasped,
        "placed": placed,
        "failed": failed,
        "dense": dense,
        "action_penalty": action_penalty,
    }
    return reward, info


def _self_test():
    cfg = PickPlaceRewardConfig()
    cube_at_start = (0.28, 0.0, 0.015)
    zero_vel = (0.0, 0.0, 0.0)
    zero_joint_vel = (0.0,) * 6
    gripper_open_joint = 1.0  # above gripper_closed_threshold -> not "closed"
    gripper_closed_joint = -0.1  # below gripper_closed_threshold -> "closed"
    F = False  # was_holding=False -- every single-snapshot test below is a "fresh" call

    # 1. Reach shaping must strictly improve as the gripper approaches the cube.
    far = compute_reward((0.0, 0.0, 0.3), cube_at_start, zero_vel, gripper_open_joint, zero_joint_vel, F, cfg)[0]
    near = compute_reward((0.27, 0.0, 0.02), cube_at_start, zero_vel, gripper_open_joint, zero_joint_vel, F, cfg)[0]
    at_cube = compute_reward(cube_at_start, cube_at_start, zero_vel, gripper_open_joint, zero_joint_vel, F, cfg)[0]
    assert far < near < at_cube, (far, near, at_cube)

    # 2. Touching the cube with an OPEN gripper must NOT count as holding
    #    (the cube hasn't actually left the table -- height gate handles this
    #    regardless of joint angle, but this also checks the joint gate
    #    directly: closed-but-not-lifted should also not count as holding).
    _, info = compute_reward(cube_at_start, cube_at_start, zero_vel, gripper_open_joint, zero_joint_vel, F, cfg)
    assert not info["holding"], info
    lifted_but_open = (0.28, 0.0, 0.10)
    _, info = compute_reward(lifted_but_open, lifted_but_open, zero_vel, gripper_open_joint, zero_joint_vel, F, cfg)
    assert not info["holding"], "lifted with an open gripper should not count as holding"

    # 3. Lifted AND closed must count as holding, and switch to "transport" phase.
    lifted = (0.28, 0.0, 0.10)
    _, info = compute_reward(lifted, lifted, zero_vel, gripper_closed_joint, zero_joint_vel, F, cfg)
    assert info["holding"] and info["phase"] == "transport", info

    # 3b. Regression test for a real bug caught during validation against
    # recorded teleop data: a cube barely above rest height (e.g. still
    # settling) combined with a coincidentally-closed gripper FAR AWAY on
    # the table must NOT count as holding -- height and joint angle alone
    # aren't enough, the gripper must actually be near the cube.
    cube_barely_elevated = (0.28, 0.0, cfg.lift_threshold + 0.005)
    gripper_far_away = (-0.1, 0.3, 0.15)
    _, info = compute_reward(
        gripper_far_away, cube_barely_elevated, zero_vel, gripper_closed_joint, zero_joint_vel, F, cfg
    )
    assert not info["holding"], "closed gripper far from a barely-elevated cube must not count as holding"

    # 3c. Regression test for a SECOND, separate real bug -- caught not by
    # replay validation but by manually tracing this self-test's own
    # numbers during a full correctness audit. lift_threshold (0.02m) sits
    # ABOVE the place target's resting height (0.015m) -- necessarily true,
    # since a successful place ends with the cube back down near table
    # height. The ORIGINAL design (is_grasped-drives-everything, a naive
    # per-step height check) would flip back to "not holding" the instant
    # the cube is lowered onto the target -- exactly the moment the
    # transport-phase's target-distance shaping matters most -- silently
    # replacing it with approach-phase gripper-to-cube shaping instead.
    # Simulate the sequence: establish a genuine hold up high (step 1),
    # then lower the still-held cube onto the target (step 2,
    # was_holding=True from step 1's result) and confirm holding +
    # transport phase PERSIST despite the height drop.
    _, info_step1 = compute_reward(
        (0.28, 0.0, 0.10), (0.28, 0.0, 0.10), zero_vel, gripper_closed_joint, zero_joint_vel, F, cfg
    )
    assert info_step1["holding"], "step 1 should establish a genuine hold while elevated"
    _, info_step2 = compute_reward(
        cfg.target_pos, cfg.target_pos, zero_vel, gripper_closed_joint, zero_joint_vel, info_step1["holding"], cfg
    )
    assert info_step2["holding"] and info_step2["phase"] == "transport", (
        "still-closed gripper lowering an already-held cube onto the target must stay in the "
        "transport phase, not silently fall back to approach shaping"
    )
    # Confirm the OLD (fixed) behavior really was broken -- the strict,
    # stateless is_grasped() check (by design, see its own docstring) still
    # returns False at the exact target height, since 0.015m is not above
    # lift_threshold (0.02m). Expected and correct for is_grasped()
    # specifically -- it's exactly why compute_reward() must not use it
    # for the phase switch.
    assert not is_grasped(cfg.target_pos, cfg.target_pos, gripper_closed_joint, cfg)

    # 3d. Holding must end immediately on release (gripper opens), even
    # with was_holding=True carried in from the previous step.
    _, info_released = compute_reward(
        cfg.target_pos, cfg.target_pos, zero_vel, gripper_open_joint, zero_joint_vel, True, cfg
    )
    assert not info_released["holding"], "an opened gripper must not count as holding regardless of was_holding"

    # 4. Holding reward must strictly improve as the cube approaches the target,
    #    and must NOT depend on cube height once holding (moving sideways at
    #    the same height should not be penalized relative to being higher up --
    #    this is the check that the phase-2 shaping doesn't secretly still
    #    reward pure altitude, which would fight against ever placing the cube).
    far_from_target = compute_reward(
        (0.28, 0.0, 0.10), (0.28, 0.0, 0.10), zero_vel, gripper_closed_joint, zero_joint_vel, F, cfg
    )[0]
    closer_to_target = compute_reward(
        (-0.05, 0.10, 0.10), (-0.05, 0.10, 0.10), zero_vel, gripper_closed_joint, zero_joint_vel, F, cfg
    )[0]
    at_target_same_height = compute_reward(
        (-0.15, 0.15, 0.10), (-0.15, 0.15, 0.10), zero_vel, gripper_closed_joint, zero_joint_vel, F, cfg
    )[0]
    assert far_from_target < closer_to_target < at_target_same_height, (
        far_from_target,
        closer_to_target,
        at_target_same_height,
    )
    # Dense reward must depend ONLY on 3D distance to the target, never on
    # height specifically -- otherwise phase-2 shaping could secretly still
    # reward pure altitude, fighting against ever placing the cube. Tested
    # correctly (not just "these two points happen to be close") by
    # comparing two points at the SAME distance from the target via
    # different decompositions -- one purely vertical offset, one purely
    # horizontal -- and asserting near-exact equality. (An earlier version
    # of this test instead compared two points at genuinely DIFFERENT
    # distances -0.085m via height vs. 0m exactly at target- and asserted
    # they were "close enough," which only passed by coincidence while the
    # 3c bug was still present, silently routing one of the two calls
    # through the wrong formula entirely. Caught during the same
    # correctness audit that found 3c.)
    offset = 0.085
    tx, ty, tz = cfg.target_pos
    vertical_offset_pos = (tx, ty, tz + offset)
    horizontal_offset_pos = (tx + offset, ty, tz)
    _, info_vertical = compute_reward(
        vertical_offset_pos, vertical_offset_pos, zero_vel, gripper_closed_joint, zero_joint_vel, True, cfg
    )
    _, info_horizontal = compute_reward(
        horizontal_offset_pos, horizontal_offset_pos, zero_vel, gripper_closed_joint, zero_joint_vel, True, cfg
    )
    assert info_vertical["holding"] and info_horizontal["holding"], (info_vertical, info_horizontal)
    assert abs(info_vertical["dense"] - info_horizontal["dense"]) < 1e-9, (
        info_vertical["dense"],
        info_horizontal["dense"],
    )

    # 5. Success bonus only fires when actually placed (at rest, at the target).
    reward_at_target_still, info = compute_reward(
        cfg.target_pos, cfg.target_pos, zero_vel, gripper_closed_joint, zero_joint_vel, True, cfg
    )
    assert info["placed"], info
    fast_vel = (1.0, 0.0, 0.0)  # swinging through, not resting
    reward_at_target_moving, info = compute_reward(
        cfg.target_pos, cfg.target_pos, fast_vel, gripper_closed_joint, zero_joint_vel, True, cfg
    )
    assert not info["placed"], "fast-moving cube passing through the target should not count as placed"
    assert reward_at_target_still > reward_at_target_moving + cfg.success_bonus - 0.1

    # 6. Action penalty must reduce reward, all else equal.
    still = compute_reward(cube_at_start, cube_at_start, zero_vel, gripper_open_joint, zero_joint_vel, F, cfg)[0]
    moving = compute_reward(cube_at_start, cube_at_start, zero_vel, gripper_open_joint, (5.0,) * 6, F, cfg)[0]
    assert moving < still, (moving, still)

    # 7. Failure detection.
    assert is_failed((0.0, 0.0, -0.1), cfg)
    assert is_failed((0.8, 0.0, 0.02), cfg)  # beyond the 1.2m table's edge + margin
    assert not is_failed(cube_at_start, cfg)

    print("[OK] pickplace_reward self-test passed")
    print(f"  far={far:.3f} near={near:.3f} at_cube={at_cube:.3f}")
    print(
        f"  far_from_target={far_from_target:.3f} closer={closer_to_target:.3f} "
        f"at_target={at_target_same_height:.3f}"
    )
    print(f"  reward_at_target_still (with success bonus)={reward_at_target_still:.3f}")


if __name__ == "__main__":
    _self_test()
