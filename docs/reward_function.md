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

### Two-phase potential-based shaping, not a single continuous term

The reward switches cleanly at the grasp event:

- **Approach** (ungrasped): reward for closing the distance between the
  gripper's actual grasp point and the cube.
- **Transport** (grasped): a flat per-step bonus for holding on, plus
  reward for closing the distance between the cube and a fixed 3D target
  point.

This split is deliberate. An earlier, simpler idea -- just reward cube
height once grasped -- was rejected: height-chasing has no incentive to
ever put the cube back down, since placing means *lowering* it again at
the target. Framing the transport phase as "distance to a target point at
table height" makes lifting, moving sideways, and lowering all part of
one continuous signal that naturally reaches its maximum exactly when the
cube is at rest at the target -- not fighting against the actual goal.

### Bounded (tanh) shaping, not raw negative distance

Both phases use `weight * (1 - tanh(distance / scale))` rather than
`-distance`. Two reasons:
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

Hand-constructed states checking: reach reward strictly improves when
approaching the cube; an open gripper touching the cube doesn't count as
grasped; a lifted+closed gripper does and switches to transport phase;
transport reward depends only on cube-to-target distance, not height
(confirming the phase-2 design goal); the success bonus fires only when
actually at rest at the target, not when swinging through it; the action
penalty reduces reward; failure detection catches a fallen/off-table
cube. All pass. This is fast, deterministic, and needs no simulator --
runs in well under a second.

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
produce given a real, noisy human-driven trajectory.

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

## Known limitations / open items

- **Grasp detection is a heuristic**, not a first-class sensor signal --
  no contact sensor currently exists on the gripper in the scene config.
  If grasp misdetection becomes a real problem once training starts,
  adding a `ContactSensorCfg` to the fingertip geometry would be a more
  principled fix than further tuning the height/joint/proximity
  thresholds.
- **Target position is fixed**, not yet randomized. This was deliberate
  for this first version (fixed positions are what let this module be
  validated against fixed-position recorded demonstrations at all), but
  the project's plan calls for domain-randomizing object/target pose --
  this will need to be wired in before bulk training data generation.
- **Weights/thresholds are principled guesses**, sized from known scene
  geometry (table extent, cube size, existing segmentation thresholds
  from `segment_teleop_episodes.py`), not tuned against actual training
  runs -- there are none yet. Expect to revisit once training exposes
  problems.
- **Not yet wired into an actual Gym-style training environment.** This
  module is a pure, standalone reward function -- it doesn't yet have a
  `reset()`/`step()` env wrapping it, which TD-MPC2 (like most RL
  codebases) expects. That's the next piece of infrastructure needed
  before training can start.

## Files

- `sim/envs/pickplace_reward.py` -- the reward function, config, and
  self-test.
- `sim/robots/grasp_geometry.py` -- the extracted, dependency-free
  `JAW_OFFSET_LOCAL` constant and `grasp_point_world()` helper, now
  shared between `run_pickplace_demo.py` and the reward function.
- `sim/scripts/validate_reward_function.py` -- replays a real captured
  episode through the reward function for empirical validation.
