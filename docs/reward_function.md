# Reward Function Design & Validation

Documents the reward function for the SO-101 pick-and-place task
(`sim/envs/pickplace_reward.py`), created 2026-08-30: what it's for, how
it's designed, the bugs found and fixed while validating it, and what is
and isn't actually confirmed working.

## What this is and why it only exists for one side of the comparison

This project compares a TD-MPC2-style world model against a diffusion
policy (see project_plan.md). Only the world model needs a reward
function at all:

- The **diffusion policy** learns purely by imitating recorded
  demonstrations. It never sees a reward.
- The **world model** plans: it learns a latent dynamics model
  (`z, a -> z'`) and a latent reward model (`z, a -> r`), then at
  decision time samples many candidate action sequences, rolls each
  forward through the *learned* dynamics+reward models, and executes the
  first action of whichever sequence scored highest (receding-horizon
  MPC). The reward function here is the ground-truth signal used to train
  that learned reward model from sim rollouts.

Because it only runs during sim training, this function is allowed to
read **privileged** simulator state directly (exact cube position, exact
gripper position) that a real robot doesn't have without extra
instrumentation. The trained policy itself never sees any of this -- it
only ever acts on camera images and proprioception. This reward function
is training-time scaffolding, not something that needs to run on
hardware.

## Design

**Note**: the two subsections immediately below describe the *original*
(2026-08-30) design. Both the shaping formula and the grasp bonus changed
in the 2026-08-31 potential-based-shaping redesign -- see that section
further down for what's actually running now. Kept here, clearly marked,
because the *reasoning* in both subsections (why two phases, why a
bounded shape) still applies unchanged to the new formula; only "reward
the absolute value" became "reward the change in value."

### Two-phase shaping, not a single continuous term

The reward switches cleanly at the grasp event:

- **Approach** (ungrasped): reward for closing the distance between the
  gripper's actual grasp point and the cube.
- **Transport** (grasped): (originally) a flat per-step bonus for holding
  on, plus reward for closing the distance between the cube and a fixed
  3D target point. The flat per-step holding bonus was removed in the
  redesign -- see below.

This split is deliberate. An earlier, simpler idea -- just reward cube
height once grasped -- was rejected: height-chasing has no incentive to
ever put the cube back down, since placing means *lowering* it again at
the target. Framing the transport phase as "distance to a target point at
table height" makes lifting, moving sideways, and lowering all part of
one continuous signal that naturally reaches its maximum exactly when the
cube is at rest at the target -- not fighting against the actual goal.

### Bounded (tanh) shaping, not raw negative distance

Both phases' potential function is `1 - tanh(distance / scale)` rather
than `-distance`. Two reasons:
1. It saturates smoothly instead of handing out unbounded penalties for
   being far away, so one bad early state can't dominate an entire
   trajectory's return.
2. TD-MPC2 trains a neural network to *predict* this reward. A smooth,
   bounded target (~1 near the goal, ~0 far away) is a much easier
   regression problem than an unbounded one.

### Grasp detection: three conditions, not one

`is_grasped()` requires all three: the cube is lifted above a threshold,
the gripper joint reads as closed, **and** the gripper is spatially near
the cube. The third condition was added after validation caught a real
bug -- see below. Without it, a cube merely settling slightly above rest
height, combined with a coincidentally-closed gripper joint *anywhere on
the table*, would falsely register as a grasp.

This is a heuristic, not a first-class signal -- there's no contact
sensor on the gripper in the current scene config. Documented explicitly
as a limitation below, not glossed over.

### Reused existing project conventions instead of inventing new ones

- **Place target**: `(-0.15, 0.15, 0.015)` is not a new arbitrary choice
  -- it's the exact "place zone" `run_pickplace_demo.py`'s scripted demo
  already established, with the height corrected to match
  `pickplace_scene.py`'s actual cube-resting convention (the demo script
  itself still has a slightly stale `0.02`). One arbitrary place location
  per project, not two independently-drifting ones.
- **Grasp point**: uses the same `JAW_OFFSET_LOCAL` correction derived
  during the original scripted-grasp debugging (docs/progress.md,
  2026-08-27/28 entry) -- `gripper_frame_link`'s own origin is ~8cm away
  from where the fingers actually meet, and that offset was already
  calibrated both analytically and empirically. It was extracted from
  `run_pickplace_demo.py` into a new dependency-free module,
  `sim/robots/grasp_geometry.py`, since the demo script itself can't be
  safely imported (it boots Isaac Sim as an import side effect) -- this
  is now the single source of truth for that constant instead of two
  independently-hardcoded copies.

### Kept dependency-free on purpose

`pickplace_reward.py` and `grasp_geometry.py` import nothing beyond the
Python standard library. Every function takes plain floats/tuples and
returns plain floats/dicts. This means the reward logic can be imported,
unit-tested, and read without booting Isaac Sim at all -- a real
constraint elsewhere in this project (every script that touches
`isaaclab`/`omni` pays a ~10-20 second Kit boot cost just to import). The
(not yet built) training env is expected to convert its torch tensors to
Python floats at the call site.

### Configuration

All weights/thresholds live in `PickPlaceRewardConfig`, a dataclass with
the reasoning for every default written inline in the code. None of these
are claimed to be optimal -- they're principled starting points sized
from the actual scene geometry (table extent, cube size, existing
segmentation thresholds), meant to be revisited once real training runs
expose problems (reward hacking, too-sparse signal, wrong relative
scale), not treated as final.

## Validation

### Level 1: synthetic self-test (`python sim/envs/pickplace_reward.py`)

Hand-constructed states checking (post-2026-08-31 redesign, see that
section above for what changed): holding still at any distance earns
exactly zero shaping reward, regardless of whether "still" is near or
far (the key hack-closing property); genuine progress between steps
earns positive shaping, regression negative; an open gripper touching
the cube doesn't count as holding; a lifted+closed gripper does, switches
to transport phase, and fires `grasp_bonus` exactly once (not on
subsequent still-holding steps); `touch_bonus` fires exactly once the
first time touch range is reached; transport-phase shaping depends only
on cube-to-target distance, not direction/height (confirming the
phase-2 design goal); the success bonus fires only when actually at rest
at the target, not when swinging through it; the action penalty reduces
reward; failure detection catches a fallen/off-table cube. All pass.
This is fast, deterministic, and needs no simulator -- runs in well under
a second.

### Level 2: replay against real captured teleop demonstrations

This is the more meaningful check: does the reward function produce
sensible values on an *actual* successful pick, not just hand-picked
states? `sim/scripts/validate_reward_function.py` replays a segmented
episode (`sim/output/teleop_episodes/episode_*.json`, real leader-arm
teleop recordings from 2026-08-29) open-loop in sim -- same mechanism as
`replay_episode.py` -- and computes the real reward at every step using
live simulated state (not the recorded JSON's cube trajectory) plus the
gripper's forward-kinematics-derived grasp point.

**Two real bugs were found and fixed this way, not in the synthetic
self-test:**

1. **Wrong initial state.** These episodes are segments cut out of one
   continuous teleop session (`segment_teleop_episodes.py`), so each one
   starts with the cube and arm wherever they physically were after the
   *previous* rep -- not at the scene's default reset state. The first
   version of the validation script reset both to scene defaults before
   replaying, which meant the recorded joint trajectory (aimed at wherever
   the cube actually was) was replayed against a cube in a completely
   different position. Result: the gripper trajectory missed the cube
   entirely, and cube height stayed flat at rest for the whole episode.
   Fixed by initializing both the cube and the robot's joints from the
   episode's own first recorded state.
2. **False-positive grasp detection.** After fixing (1), episode_003's
   very first step showed `grasped=True` with the gripper measured
   0.38m away from the cube -- because that episode happened to start
   with the cube already 0.027m above rest height (mid-settle from the
   prior rep) and the recorded gripper joint value at that instant
   happened to read as "closed," even though the gripper was nowhere
   near it. This is exactly the scenario the design section above
   describes -- `is_grasped()` didn't yet check spatial proximity. Fixed
   by adding a `grasp_proximity_threshold` (5cm) requirement, with a
   regression test added to the self-test locking this in.

**What's confirmed working after both fixes**, replayed across two
episodes (~930 combined simulated steps): no false-positive grasps, no
crashes/NaNs/exploding values, and the approach-phase dense reward rises
and falls sensibly as the gripper's real trajectory moves toward and away
from the cube (e.g. episode_000: 0.023 -> 0.401 as the gripper closes in,
easing back down as it drifts away during a messier stretch of the
teleop rep) -- exactly the shape the shaping function is supposed to
produce given a real, noisy human-driven trajectory. (This specific
absolute-value reading of "dense" predates the 2026-08-31 redesign below,
where `dense` became a per-step delta rather than an absolute value --
the re-run against the same episodes after that redesign is documented
in that section, and confirms the same underlying no-crash/sane-values
result under the new formulation.)

**What is NOT confirmed by real data**: neither tested episode ever
reproduced an actual successful lift in open-loop replay, even the one
with the strongest original recorded peak height (episode_000, originally
lifted to 0.149m; replayed, it only ever reached 0.020m). This held even
after matching both the cube's and the robot's exact recorded starting
state. This is a limitation of **open-loop replay**, not of the reward
function -- `replay_episode.py`'s own docstring already flags this
("replays the exact joint trajectory verbatim, so it only reproduces the
same result under the same starting conditions"); evidently even matching
starting conditions isn't sufficient for a contact-sensitive grasp to
reproduce open-loop, since tiny PD-tracking/contact-timing differences
compound over hundreds of steps with no feedback to correct them.

**Practical consequence**: the grasp-triggering moment itself, the full
transport/place phase, and the sparse success bonus are validated only by
the synthetic self-test in Level 1, not against real recorded data. Real
validation of that half would need either a closed-loop rollout (an
actual policy correcting for drift in real time -- not available until
training starts) or improving open-loop replay fidelity specifically
(neither was pursued further here, to avoid scope creep into a separate,
harder problem).

## Reward-hacking finding and potential-based-shaping redesign (2026-08-31)

### The finding

The first real TD-MPC2 training run (30,000 steps, see
docs/tdmpc2_integration.md) produced its single highest-scoring eval
episode (+97.9) around the middle of training. Watching that episode's
video directly (not just its printed score) showed the arm never touched
the cube at all -- it just moved into and held a plausible-looking
position. The arithmetic confirmed it: the old approach-phase shaping,
`reach_weight * (1 - tanh(d / reach_scale))`, paid out an *absolute*
reward every step based on current distance alone, regardless of whether
that distance had ever changed. At `reach_scale = 0.15`, merely occupying
a position ~0.1-0.15m from *some* plausible cube-adjacent spot -- without
tracking that specific episode's actual cube position -- was worth
roughly +0.2/step, and 0.2 x 500 steps ~= 100, matching the observed score
almost exactly. This is textbook reward hacking: the proxy (distance-based
shaping) was satisfiable without doing the thing it was meant to
approximate (actually reaching the cube).

### First attempt: narrowing reach_scale (superseded)

The first fix tried was shrinking `reach_scale` from 0.15 to 0.08, on the
theory that a sharper falloff would make "vague proximity" worth much
less. A second training run (also 30,000 steps planned, stopped early
after 4 eval checkpoints) showed this addressed the symptom but created a
new problem: eval reward was consistently negative and non-improving
(-3.8, -18.7, -11.2, -33.2 across checkpoints), and extracted video frames
(`sim/output/frame_extract/`, since there's no video-viewing tool
available -- frames were pulled via `ffmpeg -vf
"select=not(mod(n\,100))"` and inspected directly) showed genuinely
undirected motion in both sampled episodes, never engaging the cube. A
scale narrow enough to kill the hovering exploit was also narrow enough
to remove most of the usable gradient for a policy that starts far from
the cube -- the old absolute-value formulation ties "safe from hacking"
and "provides a gradient" to the same one knob (scale), and there was no
setting of that knob that satisfied both.

### The actual fix: potential-based shaping

The real fix is a change in *kind*, not degree: potential-based reward
shaping (Ng, Harada & Russell, "Policy Invariance Under Reward
Transformations," ICML 1999). Instead of paying the *absolute* value of a
potential function every step, pay only the *change*:
`reward = weight * (Phi(state') - Phi(state))`, where `Phi(d) = 1 -
tanh(d / scale)` is the same bounded shape as before. This form has a
proven guarantee -- it never changes the optimal policy of the underlying
MDP, for *any* choice of potential function, so it can only make a good
policy easier to find, never bias what "good" means. Practically: a
policy that holds still anywhere, however "good" that position looks,
earns *exactly zero* shaping reward every step it does so, since the
potential isn't changing. This closes the hacking mechanism at its root,
rather than fighting it by narrowing a scale -- and since the mechanism
that made a broad scale exploitable no longer exists, `reach_scale` was
reverted to its original 0.15 (a broad scale is now purely beneficial: a
smoother, more informative gradient for an undertrained policy far from
the cube, with no hacking downside).

Implementation detail: the shaping delta needs a "previous distance" to
compare against, so `compute_reward()` gained a `prev_dist` parameter
(the caller persists whatever it returns as `info["dist"]`, mirroring the
existing `was_holding` pattern). On the first step after a reset, or the
first step after the approach/transport phase switches, there is no
valid previous value to compare against (a phase switch means the
previous value was measuring a different quantity entirely -- gripper-to-
cube one step, cube-to-target the next), so shaping is defined as exactly
zero for that one step rather than computing a meaningless delta.

### Two more problems fixed alongside it

1. **No genuine touch signal existed.** Added `is_touching()` (a looser
   proximity threshold, 0.08m, than the 0.05m used for `is_holding()`)
   and a one-time `touch_bonus`, rewarding the act of reaching all the way
   to contact as a distinct, earlier milestone than a full grasp. This is
   still a privileged-distance heuristic, not a real contact sensor --
   same caveat as `is_grasped()`, see Known limitations.
2. **The old `grasp_bonus` was a flat per-step reward paid every step
   while holding.** Since a successful place *ends the episode*
   (termination fires on `is_placed()`), this created a real, previously
   unnoticed incentive to grasp the cube and then simply hold it in
   place indefinitely rather than finish -- collecting per-step
   `grasp_bonus` forever instead of one `success_bonus`. Fixed by making
   `grasp_bonus` a one-time milestone too (fires once, the step holding
   is first established), matching `touch_bonus` and `success_bonus`.
   Under the new design there is no reward source left that pays for
   merely *occupying* a state -- only for genuine progress (the shaping
   delta) or crossing a genuine milestone (the three one-time bonuses).

### Verification done before relying on this (2026-08-31)

Mirrored the rigor of the original correctness audit rather than trusting
the redesign by construction alone:

- **Self-test** (`python sim/envs/pickplace_reward.py`), rewritten with
  13 cases, including the property that actually matters most: holding
  perfectly still at any distance (far or already at the cube/target)
  earns exactly zero shaping reward, and the one-time bonuses each fire
  exactly once and not again on a subsequent still-holding/still-touching
  step. Also re-verified every regression case from the original
  correctness audit (the `is_holding()` hysteresis across a
  lowering-onto-target sequence, the false-positive-grasp case) still
  holds under the new signature. Passes both locally and on the training
  machine.
- **Env smoke test** (`sim/scripts/test_pickplace_env.py`), state-only
  (4 envs, 100 random-action steps) and camera-enabled (4 envs, 60
  steps, `--enable_cameras --use_cameras`): both ran end to end with no
  crash, correct observation shapes, and reward magnitudes in a sane
  small range (dominated by the action penalty under pure random
  actions, as expected -- nothing resembling the old +97.9-style
  exploit number).
- **Real-data replay** (`sim/scripts/validate_reward_function.py`)
  against two recorded teleop episodes: `episode_003.json` (374 steps,
  gripper never within touch range, min distance 0.085m -- exercises
  approach-phase shaping only) and `episode_000.json` (556 steps,
  gripper *did* reach touch range at step 0) -- confirming `touch_bonus`
  actually fires on genuine recorded motion, not just synthetic
  self-test states. Neither episode achieves a genuine grasp (a
  pre-existing, already-documented limitation of these specific
  recordings, not something this redesign was expected to fix -- see
  Known limitations), so the transport-phase shaping and `grasp_bonus`
  remain validated only by the self-test, same as before the redesign.

## Correctness audit (2026-08-31): a real bug found by manual tracing, not by replay

Before starting to wire in TD-MPC2/diffusion policy, went back through
`pickplace_reward.py`, `grasp_geometry.py`, and `pickplace_env.py` line
by line rather than assuming the earlier validation above was sufficient
-- and it wasn't. The empirical replay validation (Level 2 above) never
actually exercised the "genuinely holding and lowering the cube toward
the target" code path at all, since neither tested episode ever achieved
a real lift-and-carry. A bug living specifically in that unexercised path
was invisible to it by construction.

**The bug**: `is_grasped()`'s height check (`cube_height >
lift_threshold`, 0.02m) was also driving `compute_reward()`'s
approach/transport phase switch. But the place target's resting height
(0.015m) is *below* `lift_threshold` by construction -- a successful
place always ends with the cube back down near table height. This means
the instant a genuinely-held cube is lowered onto the target,
`cube_height` drops below 0.02m and the old code silently fell back to
approach-phase shaping (gripper-to-cube distance) instead of the intended
transport-phase shaping (cube-to-target distance) -- exactly during the
most precision-critical part of the whole task. Caught by manually
tracing the self-test's own `reward_at_target_still` case by hand
(`cube_pos = target_pos`, height 0.015m) and noticing `is_grasped()`
evaluated to `False` there. The existing self-test didn't catch it
because its pass/fail tolerance was loose enough that the (wrong)
approach-phase number and the (intended) transport-phase number happened
to land close enough together to slip through.

**A second, related flaw** surfaced while fixing the first: one of the
self-test's own assertions ("dense reward shouldn't depend on height once
grasped") compared two points at *different* distances from the target
(0m and 0.085m) and asserted the results were "close enough" -- which
only ever passed by coincidence, because it was unknowingly comparing one
correct transport-phase number against one bugged approach-phase number
that happened to be numerically similar. Replaced with a mathematically
sound comparison: two points at the *same* distance from the target via
different decompositions (pure vertical vs. pure horizontal offset),
asserted equal to within `1e-9`.

**The fix**: added `is_holding()`, a second, deliberately stateful
function (`is_grasped()` is kept unchanged, for strict stateless liftoff
diagnostics). `is_holding()` requires height evidence only to *establish*
holding for the first time -- avoiding a different failure mode, where
dropping the height requirement entirely would let a policy get MORE
reward by leaving the gripper open near the cube than by closing it
before actually being able to move toward the target, discouraging the
gripper from ever closing promptly. Once holding is established, it
persists across subsequent steps on proximity + closed-gripper evidence
alone, correctly covering the final lowering sequence, and still ends
immediately if the gripper opens or moves away. `compute_reward()` now
takes a `was_holding` argument and returns the current state in
`info["holding"]`, which the caller must persist and feed back in next
step -- `PickPlaceEnv` does this via a `_was_holding` per-env tensor,
reset to `False` on every episode reset. This is the one deliberate
exception to this module's otherwise fully stateless design, documented
inline in `is_holding()`'s own docstring.

Re-ran both the self-test (now including explicit regression tests for
both flaws, `_self_test()`'s 3c/3d cases) and the real-data replay
validation against `episode_003.json` afterward: identical numbers to
before the fix, exactly as expected, since that episode never reaches a
genuine hold and so never touches the fixed code path at all -- confirms
the fix didn't disturb the previously-validated approach-phase behavior.

## Between-jaws geometry and gripper-closing shaping (2026-09-03)

Runs 6 and 7 (potential-based reward, plus a fixed-position curriculum
and a raised exploration floor -- see docs/decisions.md) got the arm
reliably reaching and touching the cube for the first time, but not
reliably grasping it. Then a real false positive surfaced: one run7 eval
episode logged `held=True`, but the video showed the gripper closing
fully right BESIDE the cube -- never around it, cube untouched on the
table the whole episode.

**Root cause**: `is_grasped()`/`is_holding()` only ever checked a
SPHERICAL distance from the jaw pivot (`grasp_point_world()`) to the
cube. That distance is identical whether the cube sits directly in front
of the closing jaws or off to one side of them -- there was no notion of
*direction* at all, only *how far*.

**Fix**: added `is_between_jaws()`, which decomposes the cube's position
relative to the jaw pivot along `jaw_approach_axis_world()` -- the
gripper's live world-frame reach direction (`sim/robots/grasp_geometry.py`)
-- into an axial component (roughly within the fingers' own reach) and a
lateral component (genuinely centered, not off to one side). Both
required no new hardware calibration: `jaw_approach_axis_world()` reuses
the exact same already-calibrated `JAW_OFFSET_LOCAL` offset
`grasp_point_world()` has always used, just normalized into a direction
instead of a point. A live measurement (rotating the gripper's moving-jaw
body through its full joint range and reading its world position) ruled
out the naive alternative first -- the two jaw bodies' ORIGINS stay a
constant ~3.6cm apart regardless of joint angle, since the joint rotates
the moving jaw about a pivot rather than translating it, so raw
origin-to-origin distance carries no information about how open the
gripper currently is.

`is_grasped()` now requires `is_between_jaws()` to fire at all.
`is_holding()` requires it only to *establish* holding, not to persist
it -- persistence still only needs proximity + closed, deliberately, to
avoid a new failure mode where minor sway while genuinely carrying the
cube could flicker a real hold back out of these intentionally tight
thresholds (tuned for precisely establishing a grasp, not for tolerating
in-transit jitter).

**Also added**: a `grasp_close_weight` potential-based shaping term
(same delta pattern as reach/place) over the gripper joint's own
closedness, so there's finally a reward signal for the specific act of
closing the gripper once correctly positioned -- previously nothing
rewarded this at all, only gripper-to-cube distance. Explicitly gated on
`is_between_jaws()`, not mere proximity -- a direct, deliberate design
requirement (the user specifically flagged that closing must not be
rewarded just for happening "near" the cube, only for happening with the
cube genuinely positioned to be caught). A policy that snaps the gripper
shut beside the cube earns nothing from this term, closing the exact
incentive gap that produced the run7 false positive in the first place.

Verified via 6 new self-test cases (`sim/envs/pickplace_reward.py`),
including a direct regression test reproducing run7's exact false
positive (a closed gripper positioned beside, not around, the cube --
within the old proximity threshold, above the height threshold, but
failing the new geometric check) and confirming it's now correctly
rejected by both `is_grasped()` and `is_holding()`.

## Lateral-alignment shaping and premature-close penalty (2026-09-03)

Run8 (the fix above, cube moved to the train region's far edge per the
user's own observation of run6/7 video -- see docs/decisions.md) produced
the best behavior yet: the arm reaches for the cube with clear, directed
intent, and `touched=True` fired on 10 of 14 eval checkpoints. But
`is_between_jaws()` never fired once across the entire 70,000-step run --
and direct video review of the best checkpoint showed exactly why. Two
specific, correctable patterns, confirmed by watching the arm approach
across several frames of the episode:

1. **It approaches from directly above and pokes the cube with a single
   fingertip**, never straddling it with both open jaws. A pincer gripper
   closing on an object needs the object centered in the gap *between*
   the two fingers -- a one-fingered poke from above puts the cube off to
   one side of that gap by construction, which is exactly the geometric
   distinction `is_between_jaws()` is designed to require, and exactly
   why it correctly never fired for this behavior. The gate isn't broken;
   the arm simply hasn't learned the harder skill (straddling) yet, only
   the easier one (reaching a single point of contact).
2. **It closes the gripper almost immediately on approach**, well before
   anywhere near correctly positioned -- a separate behavior the user
   specifically flagged after watching the same video.

Neither problem is "just needs more training" on its own -- both trace to
genuine gaps in the reward, not a lack of steps:

**Gap 1: no gradient toward correct alignment.** `is_between_jaws()` is a
binary gate -- a policy that improves its lateral alignment from wildly
off to almost-centered earns exactly the same (zero) reward as one that
doesn't improve at all, right up until the instant it fully qualifies.
Fixed by adding `lateral_align_weight`, a genuine POTENTIAL-BASED shaping
term (same delta pattern as every other shaping term in this module) over
the raw lateral offset itself -- `_jaw_offsets()`, the exact decomposition
`is_between_jaws()` already thresholds, factored out into a shared
helper so both use identical geometry. Gated on being within
`align_activation_range` (deliberately looser than `touch_threshold`, so
the gradient can start pulling the approach into alignment slightly
*before* contact, not only after) and not yet holding.

**Gap 2: nothing discouraged closing at the wrong time.** Before this,
closing the gripper far from the cube was simply NEUTRAL -- no reward,
no cost -- so nothing pushed back against whatever caused the premature
closing (a plausible mechanism: `min_std`'s exploration floor, raised for
run6 onward, applies uniformly across *all* action dimensions including
the gripper, injecting persistent noise into gripper actuation regardless
of position, with nothing counteracting it). Fixed by adding
`premature_close_weight`, a small penalty proportional to closedness,
charged whenever the gripper is closed while NOT `is_between_jaws()` and
not yet holding.

This penalty is deliberately an ABSOLUTE value, not potential-based --
worth confirming explicitly that this does NOT reintroduce the original
reward-hacking mechanism from the 2026-08-31 redesign above. That
mechanism was specifically about an absolute-value *reward* that could be
farmed by occupying a state indefinitely (hover somewhere decent, collect
reward every step). A pure *penalty* has the opposite incentive
structure -- it's minimized by *avoiding* a state, never maximized by
*dwelling* in it, so there's nothing to farm. `premature_close_weight` and
`grasp_close_weight` are mutually exclusive by construction (gated on
`is_between_jaws()` being false vs. true respectively), so they never
fight each other on the same step.

**Also added**: `--resume-from`, a warm-start capability for
`train_tdmpc2_pickplace.py` (`TDMPC2.load()` already existed in the
vendored reference code, just never wired up). Run8 genuinely learned to
reach and touch with real, repeatable intent -- restarting from scratch
to train against these two new terms would discard that and make the
agent re-learn reaching before it could even attempt alignment. Weights
only (`TDMPC2.save()` never persisted the replay buffer), paired with a
much smaller `--seed-episodes` than a from-scratch run, since a resumed
policy that already knows how to act gets little value from a long
pure-random warmup.

Verified via 13 new self-test cases covering both terms (the same
symmetric improve/worsen and no-free-lunch properties already required of
every other shaping term in this module), state-only and camera-enabled
env smoke tests, and real-data replay validation. Caught and fixed two
real bugs while wiring this up, neither in the reward *design* itself:
(1) the self-test's own `compute_reward()` calls passed `cfg` as a bare
positional argument immediately after `prev_joint_pos` in all ~27 call
sites -- correct under the old signature, but silently reinterpreted as
the newly-inserted `prev_lateral` parameter the moment it was added
before `cfg`, caught by an actual test failure rather than by inspection;
fixed by passing `cfg=cfg` explicitly everywhere. (2) the self-test's
`gripper_open_joint` fixture (`1.0`) was only ever "open enough" for the
old binary closed/not-closed cutoff, not the gripper's true physical open
limit -- with the new continuous closedness potential reading the actual
joint angle, that placeholder registered as partially closed and
incorrectly triggered `premature_close_penalty` in unrelated tests; fixed
to use the real `gripper_joint_open_limit`.

## Axial-alignment shaping, added then reverted (2026-09-04/05)

Run9's videos showed the arm pushing the cube with clear directed intent
but never quite grasping it -- close enough that the user asked directly
about making the gripper close at the right moment. Initial frame-reading
concluded (wrongly, corrected directly by the user) that the gripper was
closed during these pushes; re-examined with that correction and found
it genuinely open, but the cube buried near the jaws' pivot/hinge (where
the gap between the fingers is narrowest) rather than out near the
fingertips (where it's widest) -- an axial reach-distance problem.
Plausible mechanism: `reach_weight` rewards driving raw gripper-to-cube
distance toward zero, with no notion of a correct standoff distance, so
it keeps paying out for pulling the pivot itself past the point where
the fingers could actually catch the cube.

Added `axial_align_weight`/`axial_align_scale`, mirroring
`lateral_align_weight` exactly -- same potential-based delta pattern,
same `align_activation_range` gate, same helper-function structure
(`_axial_offset()`/`_axial_potential()` alongside the existing
`_lateral_potential()`) -- but over the axial component of
`_jaw_offsets()` instead of the lateral one. Rewarded any position inside
the valid `[grasp_reach_min, grasp_reach_max]` window, penalized
drifting further outside it in either direction.

Run10 (100,000 steps) showed no improvement once reviewed frame-by-frame
-- the term only concerns position, and the actual remaining gap turned
out to be that the gripper never attempted to close at all (see the next
section). Reverted cleanly via `git revert` rather than deleted in
place, preserving the implementation and its self-test coverage in
history in case the axial positioning question becomes relevant again
independent of the closing question. See docs/decisions.md and
docs/progress.md for the full diagnosis story, including the initial
misdiagnosis and its correction.

## Closing-focused redesign: widened window, retuned weights (2026-09-05)

With the axial term reverted, the real question became why the gripper
never closes at all, across runs 8, 9, and 10. Checked
`is_between_jaws()`'s actual trigger rate against where contact was
really happening: the cube consistently landed right at/beyond the OLD
`grasp_reach_min` (-0.01) boundary, wedged near the pivot rather than out
toward the fingertips -- meaning `is_between_jaws()` was almost never
true at the policy's real contact point. That starved `grasp_close_weight`
of any chance to fire, while `premature_close_weight` (gated on `not
between_jaws`) fired on nearly every step the gripper had any closedness
at all, since "not between jaws" was the overwhelmingly common case.
Reward far more often for NOT closing than for closing is exactly a
"never close" training signal.

Three changes, all targeting this one mechanism, none touching the
reach/lateral-alignment machinery (video review confirmed that part
already works): `grasp_reach_min` widened -0.01 -> -0.04 (covers the
observed contact point), `grasp_close_weight` raised 1.0 -> 2.5 (a
dominant signal once reachable), `premature_close_weight` lowered
0.3 -> 0.1 (no longer strong enough to outweigh the now-reachable
closing signal). One self-test fixture needed updating (`cube_behind`'s
offset, which cleared the OLD boundary but not the new one -- caught
before it could silently pass for the wrong reason, not after).

Run11 (60,000 steps, warm-started from run9's checkpoint specifically,
not run10's -- see docs/decisions.md for why) showed real positioning
improvement (`between_jaws=True` on 4 of 11 eval checkpoints, versus
essentially never before) but still no genuine hold, motivating the next
section.

## premature_close_weight zeroed, --eval-only diagnostic added (2026-09-05/06)

Run11's own video became the first place this project's video-reading
practice was directly corrected: a still-frame read claiming visible
gripper closing turned out (per the user's own direct observation of the
actual video, not stills) to be a foreshortening illusion from the fixed
external `scene_camera` combined with the arm's own rotation. Resolved
by pulling `run_eval_episode()`'s own objective, per-episode
`touched`/`between_jaws`/`held` booleans instead of re-arguing from more
frames: `held` was `False` in all 11 of run11's eval checkpoints, while
`between_jaws` had genuinely flipped `True` in 4 -- positioning had
improved, closing specifically had not.

Added `--eval-only` to `train_tdmpc2_pickplace.py` to find out why --
loads a checkpoint, runs one deterministic eval episode, and writes a
per-step CSV (gripper joint angle, commanded gripper action, axial/
lateral offset, `between_jaws`/`holding`/`touched`) alongside the usual
eval video (see `sim/scripts/train_tdmpc2_pickplace.py`'s own docstring
for the flag, and docs/decisions.md for the full rationale). Replaying
run11's best checkpoint ruled out "brief pass-through, no time to close"
outright: `between_jaws` fired dozens of times across the 500-step
episode, including a 15-consecutive-step window. The gripper barely
moved even during that window, and its one meaningful dip (~0.06 radians
toward closed) reversed within a few steps, climbing back toward
fully-open rather than continuing.

Root cause: `premature_close_weight` is gated on `not between_jaws`, and
`between_jaws` itself is unstable (true only ~15% of the episode, mostly
short bursts) -- so a policy partway through a slow, multi-step close is
one small drift away from the penalty resuming on its still-partly-
closed gripper. Reopening immediately, before the penalty resumes, is
the locally safe strategy. Set `premature_close_weight` to 0.0 as the
cleanest single-variable test of this hypothesis -- the original reason
it existed (run8 closing carelessly far from the cube) is now separately
handled by `grasp_close_weight`'s own `is_between_jaws()` gate, which did
not exist yet when this penalty was first added. Considered a grace
period or `between_jaws` hysteresis instead, deliberately deferred until
this simpler test's result was known.

Run12 (60,000 steps, warm-started from run11) produced this project's
first-ever `held=True` eval result -- verified rather than trusted, see
the next section for what that verification found.

## between_jaws hysteresis, stricter lift_threshold, one-time grasp_bonus (2026-09-06)

Run12's `held=True` result (step 40419) needed independent verification,
having twice now been the site of a corrected claim (the aperture
misdiagnosis, the closing-illusion misdiagnosis). Extended `--eval-only`'s
CSV with `cube_height` and `gripper_cube_dist` specifically so every one
of `is_holding()`'s four establishing conditions -- `cube_height >
lift_threshold`, `gripper_joint_pos <= gripper_closed_threshold`,
`gripper_cube_dist < grasp_proximity_threshold`, `is_between_jaws()` --
could be checked independently from the sim state, not inferred from
video or trusted from the logged flag alone.

Finding: all four conditions genuinely fired -- not a geometric false
positive like the run7 bug -- but the actual lift was only ~9mm on a 3cm
cube (`lift_threshold` required just 5mm above resting height, `0.015m`),
visibly settling back toward resting height over the following several
steps rather than being carried. It recurred 3 separate times in the one
episode (steps ~95-106, ~116, ~140-141), each re-firing the full
`grasp_bonus` (+~2.0 each time, by far the largest reward events in the
episode), because that bonus was gated on `holding and not was_holding`
-- one-time per continuous streak, not one-time per episode. A policy
could cheaply farm the milestone by grazing a razor-thin threshold
repeatedly, with no additional reward for lifting higher or holding
longer than the bare minimum.

The same replay separately showed `is_between_jaws()` flickering False
for single steps in the middle of otherwise-sustained close attempts
(true only ~15% of the episode, mostly short bursts) -- independent of
`premature_close_weight` being 0.0, this still weakened
`grasp_close_weight`'s own gradient during a real attempt, since the
reward zeroed out on flicker steps even while genuine progress toward
closed was being made.

Three changes, all surfacing from this one replay:

1. **`between_jaws_grace_steps` (default 2)**, via a new
   `_between_jaws_effective(raw_between_jaws, prev_miss_streak, cfg)`
   helper -- returns `(effective, new_miss_streak)`, forgiving up to
   `between_jaws_grace_steps` consecutive misses of the strict check
   before `effective` goes back to `False`. Used ONLY to gate
   `grasp_close_weight`'s shaping and (symmetrically, to preserve mutual
   exclusivity if this weight is ever nonzero again) `premature_close_
   penalty` -- deliberately NOT fed to `is_between_jaws()` itself or to
   `is_grasped()`/`is_holding()`'s own establishment logic, which must
   stay exactly as strict as before (loosening the SAME check that gates
   a genuine hold would directly undermine fix #2 below). `prev_miss_streak
   = None` (fresh episode / no history) is treated as "no unearned grace,"
   not as "already earned a full grace window."

2. **`lift_threshold` raised 0.02 -> 0.04** -- roughly 2.5x the observed
   accidental jostle, requiring a genuine lift before `is_grasped()`/
   `is_holding()` can establish at all. Does not affect the place
   target: `is_holding()`'s persistence branch never rechecks height once
   holding is established, specifically so a controlled lowering onto the
   target (whose resting height is always below `lift_threshold` by
   construction) doesn't spuriously end holding -- confirmed every
   existing self-test fixture asserting a genuine lift used heights (0.10m,
   or dynamically `cfg.lift_threshold + 0.005`) comfortably robust to this
   change before touching the value.

3. **`grasp_bonus` regated on a new `was_ever_held` parameter** --
   genuinely episode-sticky (`ever_held = holding or was_ever_held`,
   mirroring `touch_bonus`'s own `was_touched` pattern exactly, which
   never had this bug), instead of `was_holding` (which only says "true
   last step," and so cannot distinguish the actual first hold from the
   first step of a new streak after a brief loss and re-establishment).
   Added as a REQUIRED parameter (no default) to `compute_reward()`
   specifically so a caller forgetting to thread the real persisted value
   through fails loudly (a missing-argument error) rather than silently
   reverting to the old broken behavior -- required updating all ~46
   `compute_reward()` call sites in the self-test (verified
   programmatically, not just by eye, that none were missed) plus both
   real call sites (`pickplace_env.py`, `validate_reward_function.py`).

Verified via 12 new self-test cases: direct unit tests of
`_between_jaws_effective()` (fresh-episode-gets-no-grace, exactly-at-the-
boundary stays effective, one-past-the-boundary stops being effective),
an end-to-end sequence through `compute_reward()` confirming
`grasp_close_weight` genuinely keeps paying out through a forgiven
flicker and genuinely stops once the flicker outlasts the grace window,
and real regression tests for both the lift-threshold graze and the
`grasp_bonus` re-firing bug, built directly from the actual observed
numbers (the ~9mm graze height, the re-establishment-after-loss sequence)
rather than hypothetical ones. Full local + remote self-test pass and a
production smoke test, all clean. Run13 (60,000 steps, warm-started from
run12) is the verification test for these three fixes specifically,
before committing to a full, from-scratch clean retrain against the
finalized design.

## Known limitations / open items

- **Grasp AND touch detection are both heuristics**, not first-class
  sensor signals -- no contact sensor currently exists on the gripper in
  the scene config; `is_touching()` (added in the 2026-08-31 redesign)
  and `is_between_jaws()` (added 2026-09-03, see above) both use the same
  privileged-distance/geometry approach as `is_grasped()`/`is_holding()`.
  `is_between_jaws()`'s reach range (`grasp_reach_max`) in particular is a
  principled guess at the SO-101 gripper's finger length, not a measured
  spec. If misdetection becomes a real problem once training starts,
  adding a `ContactSensorCfg` to the fingertip geometry would be a more
  principled fix than further tuning these thresholds.
- **Target position is fixed**, not yet randomized. This was deliberate
  for this first version (fixed positions are what let this module be
  validated against fixed-position recorded demonstrations at all), but
  the project's plan calls for domain-randomizing object/target pose --
  this will need to be wired in before bulk training data generation.
- **Weights/thresholds are principled guesses**, sized from known scene
  geometry (table extent, cube size, existing segmentation thresholds
  from `segment_teleop_episodes.py`), not tuned against actual training
  runs. Runs 9 through 12 (see the sections above) have since put several
  of these through real iteration -- `grasp_reach_min`/`grasp_close_weight`/
  `premature_close_weight` were each revisited at least once after a
  completed run exposed a problem, and `lift_threshold` was raised after a
  real farming bug was caught directly rather than guessed at. Still not a
  systematic sweep, just problem-driven revision -- expect this to keep
  happening as training continues.
- **The transport-phase hold-and-lower path is still not empirically
  validated against real data** -- only against the synthetic self-test
  (see the correctness audit above). CORRECTED (2026-09-08): this item
  previously claimed no recorded episode ever achieves a sustained
  lift-and-carry, based on only the one or two episodes spot-checked
  early in the project. Re-checked all 8 recorded episodes directly
  (`sim/output/teleop_episodes/episode_000.json` through `episode_007.json`)
  while investigating demonstration-seeded training (see
  docs/decisions.md): every single one reaches a cube height of 8.7cm to
  14.9cm above the table, strongly consistent with a genuine lift-and-carry,
  not a touch or jostle. A real closed-loop replay of one of these episodes
  through `compute_reward()` -- confirming the transport-phase shaping
  actually behaves correctly across a genuine hold, not just the
  hand-constructed self-test cases -- is still not done, and is worth
  doing as part of the demonstration-seeding work now underway.
- Now wired into an actual Gym-style training environment
  (`sim/envs/pickplace_env.py`, see docs/training_env.md) -- this item
  is resolved, kept here only as a pointer for anyone who reads this file
  before that one.

## Files

- `sim/envs/pickplace_reward.py` -- the reward function, config, and
  self-test.
- `sim/robots/grasp_geometry.py` -- the extracted, dependency-free
  `JAW_OFFSET_LOCAL` constant and `grasp_point_world()` helper, now
  shared between `run_pickplace_demo.py` and the reward function. Also
  `JAW_AXIS_LOCAL`/`jaw_approach_axis_world()` (added 2026-09-03), the
  normalized-direction counterpart used by `is_between_jaws()`.
- `sim/scripts/validate_reward_function.py` -- replays a real captured
  episode through the reward function for empirical validation.
