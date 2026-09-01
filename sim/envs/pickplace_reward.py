# SPDX-License-Identifier: BSD-3-Clause
"""Reward function for the SO-101 pick-and-place task.

This is used to train the TD-MPC2-style world model side of the project's
comparison -- not the diffusion policy, which learns purely from
demonstrations and never sees a reward at all. See docs/reward_function.md
for the full design rationale, the alternatives considered, and an
empirical validation against a real captured teleop demonstration
(sim/scripts/validate_reward_function.py).

Deliberately independent of Isaac Lab/torch/omni: every function here
takes plain floats/tuples and returns plain floats/dicts. The training env
(sim/envs/pickplace_env.py) pulls its torch tensors down to CPU/Python at
the call site. This keeps the reward function importable, unit-testable,
and inspectable without booting Isaac Sim, and keeps "what counts as a
good outcome" decoupled from "how the simulator represents state."

Everything here reads PRIVILEGED simulator state (exact cube position,
exact gripper position, contact/grasp state) that a real robot doesn't
have without extra instrumentation. That's fine and standard for
model-based RL reward functions -- this is only ever used to generate the
training signal in sim. The trained policy itself acts only on camera
images and proprioception, never on this function's inputs directly.

## Reward redesign (2026-08-31): potential-based shaping

The first real TD-MPC2 training run's eval video (watched directly, not
just its printed reward numbers) surfaced genuine reward hacking: the
highest-scoring episode of the run (+97.9) never touched the cube at all.
The mechanism was the old design's dense shaping term
(`reach_weight * (1 - tanh(d / reach_scale))`), which handed out an
ABSOLUTE reward every step based on current distance alone -- a policy
that simply hovered near typical cube positions, without ever tracking
the SPECIFIC cube's actual position that episode, could accumulate
substantial reward over a 500-step episode just by sitting still
somewhere decent.

A first attempted fix (shrinking `reach_scale` from 0.15 to 0.08, tested
in a second training run) addressed the symptom but not the mechanism --
it made the "decent" region narrower, but a policy could still profit
from occupying ANY fixed position with zero real engagement, and the
narrower region also removed most of the useful gradient for an
undertrained policy far from the cube (confirmed by watching THAT run's
eval videos too: consistently negative reward with no directed motion
toward the cube, across 4 checkpoints).

The actual fix is a change in KIND, not degree: potential-based reward
shaping (Ng, Harada & Russell, "Policy Invariance Under Reward
Transformations," ICML 1999). Instead of rewarding the ABSOLUTE value of
a potential function `Phi(state)` every step, reward only the CHANGE,
`Phi(state') - Phi(state)`. This has a proven guarantee: adding a shaping
term of this exact form never changes the optimal policy of the
underlying MDP, no matter how the potential function is chosen -- it can
only affect how fast/easily that policy is found, never bias what "the
best behavior" actually is. Practically, for us: a policy that just
occupies a fixed position (however good) earns EXACTLY ZERO shaping
reward every step it stays there, since the potential isn't changing --
only genuine progress (decreasing distance) earns anything. This closes
the original hacking mechanism at its root rather than by narrowing a
scale, and unlike the reach_scale=0.08 attempt, it's safe to use a
BROADER scale again (reach_scale reverted to its original 0.15) since the
exploit that motivated narrowing it no longer exists under the delta
formulation -- a broader scale gives a smoother, more useful gradient
across the whole workspace for an undertrained policy.

Two more real problems were found and fixed alongside this:

1. No genuine "you touched the cube" signal existed at all, despite this
   being a natural, well-supported idea in the robotics RL literature
   (contact-based rewards are commonly combined with distance shaping
   specifically because contact is a much stronger, harder-to-fake
   grounding signal than geometric proximity). Added `is_touching()` and
   a one-time `touch_bonus` -- fires the first time the gripper reaches
   genuine touch-range of the cube (looser than the full holding
   threshold), rewarding the ACT of reaching all the way to contact,
   distinct from actually grasping. Not a true physics contact sensor
   (still a documented open item, see docs/reward_function.md's known
   limitations) -- this uses the same privileged-distance computation
   already used elsewhere, no new sensor infrastructure required yet.

2. A separate, previously-unnoticed incentive problem: the old
   `grasp_bonus` was a FLAT reward paid EVERY step while holding. Since
   successfully placing the cube ENDS the episode (termination fires on
   is_placed()), this created a perverse incentive -- grasping the cube
   and then just holding it in place, never finishing, could out-earn
   actually completing the task, simply by collecting grasp_bonus for
   every remaining step instead of the one-time success_bonus. Fixed by
   making grasp_bonus ALSO a one-time milestone bonus (fires once, the
   step holding is first established), matching touch_bonus and
   success_bonus. Under the new design, holding the cube without moving
   toward the target earns nothing further at all (the place-phase
   shaping term is also potential-based, so zero movement = zero
   reward) -- there is no longer any reward source that rewards merely
   occupying a state, only genuine progress or milestones.
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
    # Looser than grasp_proximity_threshold on purpose -- represents "the
    # gripper has reached genuine contact range," an earlier, easier
    # milestone than actually being close/closed enough to count as
    # holding. Rewarding this separately (see touch_bonus below) gives a
    # concrete, achievable-before-a-full-grasp target for the touch_bonus
    # milestone.
    touch_threshold: float = 0.08

    # -- Shaping weights --
    # Potential-based: reward = weight * (Phi(new) - Phi(old)), where
    # Phi(d) = 1 - tanh(d / scale) is the SAME bounded (0, 1) shape used
    # before, now applied as a DELTA rather than an absolute value -- see
    # this module's docstring for why (closes the reward-hacking
    # mechanism at its root, per Ng/Harada/Russell's policy-invariance
    # guarantee for potential-based shaping).
    reach_weight: float = 1.0
    # Reverted to its original 0.15 (was briefly tightened to 0.08 to
    # fight the OLD absolute-reward hacking exploit -- no longer
    # necessary under the delta formulation, and a broader scale gives a
    # smoother, more useful gradient for an undertrained policy far from
    # the cube. See this module's docstring for the full story.
    reach_scale: float = 0.15  # meters -- distance at which reach shaping is ~half-saturated
    place_weight: float = 1.0
    place_scale: float = 0.15

    # -- One-time milestone bonuses (sparse), each firing exactly once
    # per episode, the step its condition is first met. Layered on top of
    # the continuous potential-based shaping above, not a substitute for
    # it -- these reward crossing a genuine, discrete threshold of
    # progress (touch range, an established hold, task completion), while
    # the shaping term rewards the continuous progress leading up to each
    # one. Values chosen as a sensible increasing ladder
    # (touch < grasp < success), not derived from any formal tuning.
    touch_bonus: float = 0.3
    # One-time now, not per-step (was 0.5/step) -- see this module's
    # docstring, item 2: a per-step version created a perverse incentive
    # to keep holding the cube forever instead of finishing the task.
    grasp_bonus: float = 2.0
    success_bonus: float = 10.0  # one-time; must clearly dominate everything else so success always wins
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


def _potential(d, scale):
    """The bounded (0, 1) potential function used for both phases' shaping
    -- 1 near d=0, ~0 for d >> scale. Shared helper so both phases and the
    self-test compute it identically."""
    return 1.0 - _tanh(d / scale)


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

    This is the one function in this module whose result depends on more
    than its own instantaneous inputs -- `was_holding` is an ordinary
    function argument (same inputs always give the same output), so this
    is still a pure function, just one whose result legitimately depends
    on very recent history, which a single-timestep snapshot cannot
    resolve on its own. The caller (the training env) is responsible for
    persisting `was_holding` across steps and passing back whatever this
    function returns. (is_touching(), used for the analogous touch_bonus
    milestone, stays fully instantaneous/stateless on purpose -- its
    "have we EVER touched this episode" stickiness is instead implemented
    one level up, in compute_reward()'s own body, since there's no reason
    for is_touching() itself to know about episode history.)"""
    cube_height = cube_pos[2] - cfg.table_z
    close_enough = _dist3(gripper_pos, cube_pos) < cfg.grasp_proximity_threshold
    closed = gripper_joint_pos <= cfg.gripper_closed_threshold
    return closed and close_enough and (cube_height > cfg.lift_threshold or was_holding)


def is_touching(gripper_pos, cube_pos, cfg: PickPlaceRewardConfig) -> bool:
    """True if the gripper has reached genuine touch-range of the cube --
    looser than is_holding()'s proximity requirement, and with no joint-
    angle or height condition at all (this measures "got close enough to
    touch," not "is gripping"). Instantaneous/stateless -- the caller
    tracks whether this has EVER been true this episode (see
    compute_reward()'s `was_touched` parameter) to fire touch_bonus only
    once."""
    return _dist3(gripper_pos, cube_pos) < cfg.touch_threshold


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
    was_touched: bool,
    prev_dist,
    cfg: PickPlaceRewardConfig = PickPlaceRewardConfig(),
):
    """Computes one step's scalar reward plus a diagnostics dict.

    Two-phase POTENTIAL-BASED dense shaping, switching on the grasp event
    (see this module's docstring for the full redesign rationale):
      - Not holding: reward for the CHANGE in gripper-to-cube closeness
        since last step ("approach"/"reach").
      - Holding: reward for the CHANGE in cube-to-target closeness since
        last step ("transport"/"place").

    Only the CHANGE in potential is rewarded, never its absolute value --
    a policy that holds still (anywhere) earns exactly zero shaping
    reward every step it does so. This is what actually closes the
    reward-hacking mechanism found in the first real training run (an
    earlier, absolute-value version let a policy earn substantial reward
    just by hovering near typical cube positions, without ever touching
    the actual cube) -- not merely a smaller scale, a different kind of
    reward entirely (Ng/Harada/Russell's potential-based shaping,
    provably policy-invariant regardless of how the potential is chosen).

    The two-phase split itself (rather than one continuous "reward
    height" term) remains for the same reason as before: a naive reward
    that just scores cube height would conflict with placing, since
    placing requires eventually LOWERING the cube back down at the
    target. Framing phase 2 as "distance to a 3D target point at table
    height" makes lifting, moving sideways, and lowering all part of one
    continuous, non-conflicting signal.

    The phase switch uses is_holding(), not is_grasped() -- see
    is_holding()'s own docstring for why a naive height-gated check on
    every step gets this wrong specifically during the final
    lowering-to-target sequence.

    On top of the continuous shaping, three ONE-TIME milestone bonuses
    fire the step their condition is first met this episode: touch_bonus
    (genuine contact range reached), grasp_bonus (a real hold
    established), success_bonus (task completed). None of these are
    per-step -- an earlier per-step version of grasp_bonus created a
    perverse incentive to hold the cube forever instead of finishing (see
    this module's docstring, item 2).

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
        was_touched: whether is_touching() has EVER returned True this
            episode, up to and not including this step (False on the
            first step after a reset). The caller persists this the same
            way as was_holding.
        prev_dist: the `dist` value THIS function returned in its info
            dict on the PREVIOUS step, or None on the first step after a
            reset OR the first step after a holding/not-holding phase
            transition (in both cases there's no meaningful previous
            value for the CURRENTLY relevant potential, so this function
            treats it as None regardless of what the caller passes --
            see the phase_transition handling below). The caller persists
            whatever this function returns as `info["dist"]`.
        cfg: reward configuration/weights.

    Returns:
        (reward: float, info: dict) -- info carries the boolean
        holding/touched/grasped/placed/failed flags, the individual
        reward components, and `dist` (this step's relevant distance,
        to be passed back as next step's `prev_dist`).
    """
    holding = is_holding(gripper_pos, cube_pos, gripper_joint_pos, was_holding, cfg)
    grasped = is_grasped(gripper_pos, cube_pos, gripper_joint_pos, cfg)
    touching_now = is_touching(gripper_pos, cube_pos, cfg)
    touched = touching_now or was_touched
    placed = is_placed(cube_pos, cube_lin_vel, cfg)
    failed = is_failed(cube_pos, cfg)

    if not holding:
        dist = _dist3(gripper_pos, cube_pos)
        scale, weight = cfg.reach_scale, cfg.reach_weight
        phase = "approach"
    else:
        dist = _dist3(cube_pos, cfg.target_pos)
        scale, weight = cfg.place_scale, cfg.place_weight
        phase = "transport"

    # A phase transition (holding just changed) means `prev_dist` (if any)
    # was computed against a DIFFERENT potential -- gripper-to-cube one
    # step, cube-to-target the next -- so it's meaningless here regardless
    # of what the caller passed. Treat this step's shaping as zero rather
    # than compute a bogus delta between two unrelated quantities.
    phase_transition = holding != was_holding
    effective_prev_dist = None if phase_transition else prev_dist
    if effective_prev_dist is None:
        shaping = 0.0
    else:
        shaping = weight * (_potential(dist, scale) - _potential(effective_prev_dist, scale))

    milestone_bonus = 0.0
    if touching_now and not was_touched:
        milestone_bonus += cfg.touch_bonus
    if holding and not was_holding:
        milestone_bonus += cfg.grasp_bonus

    action_penalty = cfg.action_penalty_weight * sum(v * v for v in joint_vel)

    reward = shaping + milestone_bonus - action_penalty
    if placed:
        reward += cfg.success_bonus

    info = {
        "phase": phase,
        "holding": holding,
        "touched": touched,
        "grasped": grasped,
        "placed": placed,
        "failed": failed,
        "dense": shaping,
        "milestone_bonus": milestone_bonus,
        "action_penalty": action_penalty,
        "dist": dist,
    }
    return reward, info


def _self_test():
    cfg = PickPlaceRewardConfig()
    cube_at_start = (0.28, 0.0, 0.015)
    zero_vel = (0.0, 0.0, 0.0)
    zero_joint_vel = (0.0,) * 6
    gripper_open_joint = 1.0  # above gripper_closed_threshold -> not "closed"
    gripper_closed_joint = -0.1  # below gripper_closed_threshold -> "closed"
    F = False  # was_holding/was_touched=False, prev_dist=None -- a "fresh" (first-step) call

    # 1. With no previous distance (fresh call), shaping is always zero --
    #    there's nothing to compare against yet. This replaces the old
    #    "far < near < at_cube" monotonicity check, which tested the
    #    ABSOLUTE reward's dependence on distance -- meaningless now that
    #    reward depends on the CHANGE in distance, not its current value.
    far_fresh, info_far_fresh = compute_reward(
        (0.0, 0.0, 0.3), cube_at_start, zero_vel, gripper_open_joint, zero_joint_vel, F, F, None, cfg
    )
    # Deliberately just outside touch_threshold (dist ~0.10m > 0.08m), so
    # this isolates the shaping-only property without also tripping
    # touch_bonus -- that's covered separately by test 12 below.
    near_fresh, info_near_fresh = compute_reward(
        (0.18, 0.0, 0.015), cube_at_start, zero_vel, gripper_open_joint, zero_joint_vel, F, F, None, cfg
    )
    assert not info_near_fresh["touched"], "test setup error: this point must be outside touch range"
    assert info_far_fresh["dense"] == 0.0 and info_near_fresh["dense"] == 0.0, (info_far_fresh, info_near_fresh)
    assert far_fresh == near_fresh == 0.0, (far_fresh, near_fresh)

    # 2. THE key hack-closing property: holding perfectly still (identical
    #    distance step to step) earns exactly zero shaping reward,
    #    regardless of whether "still" happens to be close or far. This is
    #    what makes the old exploit (earn reward just by occupying a
    #    decent position) impossible now -- only CHANGING distance pays.
    d = _dist3((0.1, 0.1, 0.1), cube_at_start)
    _, info_still_far = compute_reward(
        (0.1, 0.1, 0.1), cube_at_start, zero_vel, gripper_open_joint, zero_joint_vel, F, F, d, cfg
    )
    assert info_still_far["dense"] == 0.0, "holding still at a FAR distance must earn zero shaping reward"
    d_close = _dist3(cube_at_start, cube_at_start)
    _, info_still_close = compute_reward(
        cube_at_start, cube_at_start, zero_vel, gripper_open_joint, zero_joint_vel, F, F, d_close, cfg
    )
    assert info_still_close["dense"] == 0.0, "holding still EVEN AT THE CUBE must earn zero shaping reward"

    # 3. Genuine progress -- getting closer between steps -- must earn
    #    positive shaping; moving away must earn negative shaping.
    prev_d = _dist3((0.0, 0.0, 0.3), cube_at_start)
    _, info_closer = compute_reward(
        (0.27, 0.0, 0.02), cube_at_start, zero_vel, gripper_open_joint, zero_joint_vel, F, F, prev_d, cfg
    )
    assert info_closer["dense"] > 0.0, "moving closer since last step must earn positive shaping"
    prev_d2 = _dist3((0.27, 0.0, 0.02), cube_at_start)
    _, info_farther = compute_reward(
        (0.0, 0.0, 0.3), cube_at_start, zero_vel, gripper_open_joint, zero_joint_vel, F, F, prev_d2, cfg
    )
    assert info_farther["dense"] < 0.0, "moving farther since last step must earn negative shaping"

    # 4. Touching the cube with an OPEN gripper must NOT count as holding
    #    (the cube hasn't actually left the table -- height gate handles this
    #    regardless of joint angle, but this also checks the joint gate
    #    directly: closed-but-not-lifted should also not count as holding).
    _, info = compute_reward(cube_at_start, cube_at_start, zero_vel, gripper_open_joint, zero_joint_vel, F, F, None, cfg)
    assert not info["holding"], info
    lifted_but_open = (0.28, 0.0, 0.10)
    # This doubles as the "touch" step feeding into test 5 below -- gripper
    # and cube coincide (dist=0, within touch_threshold), so this is also
    # the natural point to establish touched=True before grasping.
    _, info_touch_step = compute_reward(
        lifted_but_open, lifted_but_open, zero_vel, gripper_open_joint, zero_joint_vel, F, F, None, cfg
    )
    assert not info_touch_step["holding"], "lifted with an open gripper should not count as holding"
    assert info_touch_step["touched"] and info_touch_step["milestone_bonus"] == cfg.touch_bonus, info_touch_step

    # 5. Lifted AND closed must count as holding, switch to "transport"
    #    phase, AND fire the one-time grasp_bonus milestone -- and ONLY
    #    grasp_bonus, since touch_bonus already fired on the previous
    #    (still-open) step and was_touched=True is carried forward here
    #    (the realistic staged sequence: touch range (0.08m) is looser than
    #    holding proximity (0.05m), so touching is reached first).
    lifted = (0.28, 0.0, 0.10)
    _, info = compute_reward(
        lifted, lifted, zero_vel, gripper_closed_joint, zero_joint_vel,
        F, info_touch_step["touched"], info_touch_step["dist"], cfg,
    )
    assert info["holding"] and info["phase"] == "transport", info
    assert info["milestone_bonus"] == cfg.grasp_bonus, "first step of holding must fire grasp_bonus exactly, not touch_bonus again"

    # 5b. grasp_bonus must NOT fire again on a second step of already
    #     holding (was_holding=True carried in) -- one-time, not per-step.
    #     This is the actual fix for the completion-timing incentive
    #     problem: since it's one-time, sitting still while holding earns
    #     nothing further (dense=0 per test 2's property, milestone=0
    #     here), removing any reason to delay finishing.
    _, info_still_holding = compute_reward(
        lifted, lifted, zero_vel, gripper_closed_joint, zero_joint_vel, True, info["touched"], info["dist"], cfg
    )
    assert info_still_holding["holding"] and info_still_holding["milestone_bonus"] == 0.0, (
        "grasp_bonus must fire only once, not every step holding continues"
    )
    assert info_still_holding["dense"] == 0.0, "holding still (no progress toward target) must earn zero shaping"

    # 6. Regression test for a real bug caught during validation against
    # recorded teleop data: a cube barely above rest height (e.g. still
    # settling) combined with a coincidentally-closed gripper FAR AWAY on
    # the table must NOT count as holding -- height and joint angle aren't
    # enough, the gripper must actually be near the cube.
    cube_barely_elevated = (0.28, 0.0, cfg.lift_threshold + 0.005)
    gripper_far_away = (-0.1, 0.3, 0.15)
    _, info = compute_reward(
        gripper_far_away, cube_barely_elevated, zero_vel, gripper_closed_joint, zero_joint_vel, F, F, None, cfg
    )
    assert not info["holding"], "closed gripper far from a barely-elevated cube must not count as holding"

    # 7. Regression test for a second real bug -- caught not by replay
    # validation but by manually tracing the (pre-redesign) self-test's
    # own numbers. lift_threshold (0.02m) sits ABOVE the place target's
    # resting height (0.015m) -- necessarily true, since a successful
    # place ends with the cube back down near table height. Confirm
    # holding + transport phase PERSIST when an already-held cube is
    # lowered onto the target, despite the height drop.
    _, info_step1 = compute_reward(
        (0.28, 0.0, 0.10), (0.28, 0.0, 0.10), zero_vel, gripper_closed_joint, zero_joint_vel, F, F, None, cfg
    )
    assert info_step1["holding"], "step 1 should establish a genuine hold while elevated"
    _, info_step2 = compute_reward(
        cfg.target_pos, cfg.target_pos, zero_vel, gripper_closed_joint, zero_joint_vel,
        info_step1["holding"], info_step1["touched"], info_step1["dist"], cfg,
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

    # 8. Holding must end immediately on release (gripper opens), even
    # with was_holding=True carried in from the previous step.
    _, info_released = compute_reward(
        cfg.target_pos, cfg.target_pos, zero_vel, gripper_open_joint, zero_joint_vel, True, True, 0.0, cfg
    )
    assert not info_released["holding"], "an opened gripper must not count as holding regardless of was_holding"

    # 9. Shaping reward must depend ONLY on the scalar 3D distance to the
    # target, never on direction/height specifically, once holding --
    # otherwise phase-2 shaping could secretly still reward pure altitude,
    # fighting against ever placing the cube. Tested with two DIFFERENT
    # end positions that are each at the exact same distance (D1) from the
    # target -- one purely vertically above it, one purely horizontally
    # displaced -- both stepping from the same starting distance (D0, also
    # purely vertical). Both must earn identical shaping, since
    # compute_reward only ever looks at _dist3(cube_pos, target_pos).
    tx, ty, tz = cfg.target_pos
    d0, d1 = 0.20, 0.115
    start_pos = (tx, ty, tz + d0)  # straight above the target, at distance d0
    start_dist = _dist3(start_pos, cfg.target_pos)
    vertical_step_pos = (tx, ty, tz + d1)  # still straight above, now at distance d1
    horizontal_step_pos = (tx + d1, ty, tz)  # off to the side instead, also at distance d1
    assert abs(_dist3(vertical_step_pos, cfg.target_pos) - d1) < 1e-9
    assert abs(_dist3(horizontal_step_pos, cfg.target_pos) - d1) < 1e-9
    _, info_vertical = compute_reward(
        vertical_step_pos, vertical_step_pos, zero_vel, gripper_closed_joint, zero_joint_vel,
        True, True, start_dist, cfg,
    )
    _, info_horizontal = compute_reward(
        horizontal_step_pos, horizontal_step_pos, zero_vel, gripper_closed_joint, zero_joint_vel,
        True, True, start_dist, cfg,
    )
    assert abs(info_vertical["dense"] - info_horizontal["dense"]) < 1e-9, (
        info_vertical["dense"], info_horizontal["dense"],
    )

    # 10. Success bonus only fires when actually placed (at rest, at the target).
    reward_at_target_still, info = compute_reward(
        cfg.target_pos, cfg.target_pos, zero_vel, gripper_closed_joint, zero_joint_vel, True, True, 0.05, cfg
    )
    assert info["placed"], info
    fast_vel = (1.0, 0.0, 0.0)  # swinging through, not resting
    reward_at_target_moving, info = compute_reward(
        cfg.target_pos, cfg.target_pos, fast_vel, gripper_closed_joint, zero_joint_vel, True, True, 0.05, cfg
    )
    assert not info["placed"], "fast-moving cube passing through the target should not count as placed"
    assert reward_at_target_still > reward_at_target_moving + cfg.success_bonus - 0.1

    # 11. Action penalty must reduce reward, all else equal.
    still = compute_reward(
        cube_at_start, cube_at_start, zero_vel, gripper_open_joint, zero_joint_vel, F, F, 0.0, cfg
    )[0]
    moving = compute_reward(
        cube_at_start, cube_at_start, zero_vel, gripper_open_joint, (5.0,) * 6, F, F, 0.0, cfg
    )[0]
    assert moving < still, (moving, still)

    # 12. Touch bonus fires exactly once, the first time touch-range is
    # reached, and not again on a subsequent step even if still touching.
    near_cube = (0.29, 0.0, 0.02)  # within touch_threshold of cube_at_start but not holding (gripper open)
    assert _dist3(near_cube, cube_at_start) < cfg.touch_threshold
    _, info_first_touch = compute_reward(
        near_cube, cube_at_start, zero_vel, gripper_open_joint, zero_joint_vel, F, F, None, cfg
    )
    assert info_first_touch["touched"] and info_first_touch["milestone_bonus"] == cfg.touch_bonus, info_first_touch
    _, info_second_touch = compute_reward(
        near_cube, cube_at_start, zero_vel, gripper_open_joint, zero_joint_vel,
        F, info_first_touch["touched"], info_first_touch["dist"], cfg,
    )
    assert info_second_touch["milestone_bonus"] == 0.0, "touch_bonus must fire only once, not every step touching"

    # 13. Failure detection.
    assert is_failed((0.0, 0.0, -0.1), cfg)
    assert is_failed((0.8, 0.0, 0.02), cfg)  # beyond the 1.2m table's edge + margin
    assert not is_failed(cube_at_start, cfg)

    print("[OK] pickplace_reward self-test passed")
    print(f"  closer_shaping={info_closer['dense']:+.4f} farther_shaping={info_farther['dense']:+.4f}")
    print(f"  grasp_bonus_once={info['holding']} reward_at_target_still={reward_at_target_still:.3f}")


if __name__ == "__main__":
    _self_test()
