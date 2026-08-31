# Diffusion Policy: Bulk Demonstration Data Strategy

Documents the discussion and decision around where the diffusion policy's
training data comes from -- the one real blocker on that side of the
project (see docs/algorithms_explained.md's structural asymmetry section:
diffusion policy needs a bulk demonstration dataset before training can
start at all; TD-MPC2 needs no such thing, since it generates its own
data by interacting with `PickPlaceEnv`). Written 2026-08-31, while the
first real TD-MPC2 training run was in progress -- see
docs/tdmpc2_integration.md and the progress.md entry for that run's
results.

**Status: decision on the exact data-generation mechanism deferred by the
user until they return to it.** What *is* decided: the general strategy
(a custom implementation of the MimicGen core idea, see below), and the
three original options this replaces. This doc exists so that decision
doesn't need to be re-derived from scratch later.

## The three original options considered

**Option A: More real human teleop repetitions.** Same mechanism as the
existing 8 episodes (`teleop_bridge.py`, leader arm mirrored into the
simulated follower), now with the cube auto-randomized within the same
train region TD-MPC2 uses (a real gap found and fixed while working
through this -- see "Teleop cube randomization fix" below).
- *Pros*: methodologically the cleanest -- this is the comparison the
  project actually set out to make (planning vs. imitating genuine human
  demonstrations). Approach style can be deliberately varied across reps
  for genuine multi-modality (Test 8, evaluation_plan.md). All
  infrastructure already works.
- *Cons*: needs sustained hands-on human time. Diffusion policy papers
  commonly report usable results from ~50-150 demonstrations for a
  single task -- not the "thousands" that might sound implied by
  "bulk data" -- but still several dedicated sessions, with some
  fraction inevitably filtered out as messy (same as the existing 8).

**Option B: Make the scripted IK demo reliable, then run it
automatically at scale.** `run_pickplace_demo.py`'s two-phase grasp is
still not fully reliable (last documented state; untested since several
physics fixes landed).
- *Pros*: fully automatable, no human bottleneck, fits the "mostly sim,
  bulk automated generation" spirit of the original plan.
- *Cons*: two real costs. Making it reliable is itself an open-ended
  engineering problem with no guaranteed timeline. More fundamentally:
  every episode would follow the *same* scripted kinematic strategy,
  varying only by cube position -- zero natural behavioral diversity,
  directly undermining any ability to test diffusion policy's
  multi-modal handling (Test 8 needs genuine variation in the data to
  test at all).

**Option C: Use TD-MPC2's own successful rollouts as demonstrations.**
Once (if) training produces genuinely successful episodes, extract and
format them as diffusion-policy training data.
- *Pros*: effectively free -- a byproduct of training that's happening
  anyway, easily scaled.
- *Cons*: the one that changes what's actually being measured. Diffusion
  policy would be imitating TD-MPC2's own behavior, not a human's -- the
  comparison stops being "planning vs. human-informed imitation" and
  becomes "planning vs. a frozen, non-replanning copy of itself." Likely
  makes TD-MPC2 look artificially favored on exactly the tests designed
  to probe replanning (Test 3, perturbation recovery), since diffusion
  policy would be imitating a system that only succeeded *because* it
  could replan, then losing that ability in the process of distillation.
  Also contingent on TD-MPC2 actually succeeding often enough to have a
  good pool to draw from -- unknown at time of writing (see the run1
  results below).

Initial lean, before researching further: Option A, on the grounds that
it's the only one that actually answers the project's original question,
even though it's the slowest.

## The question that reframed the decision: can A and B be combined?

Researched how others have approached this -- not a problem unique to
this project. Found a directly relevant, well-established, *named*
technique rather than needing to invent something from scratch.

### MimicGen (NVIDIA, CoRL 2023)

Takes a small number of real human demonstrations and automatically
generates a much larger dataset by adapting them geometrically to new
scene configurations. Reported results: from ~200 human demonstrations,
generated over 50,000 demonstrations across 18 tasks; in one comparison,
data generated from just 10 human demonstrations performed comparably to
200 real human demonstrations.

**Mechanism**: the source human demonstration is parsed into contiguous
*object-centric subtask segments* (e.g. "reach and grasp," then
"transport and place"). For a new scene, the system observes the
object's current pose, and transforms the corresponding segment's
end-effector poses by a rigid SE(3) transformation that preserves the
*relative* pose between end-effector and object exactly as it was in the
source demo -- then interpolates from the robot's current pose into the
start of the transformed segment and replays it. In plain terms: it
re-aims the same demonstrated motion at wherever the object actually is
now, rather than blindly replaying absolute coordinates -- exactly the
failure mode already hit once in this project with `replay_episode.py`'s
open-loop replay (docs/reward_function.md's correctness-audit section).

**Why this fits our task especially well**: MimicGen's core mechanism is
built around exactly the kind of task structure we already have -- a
single object, decomposing naturally into precisely two subtasks
(reach-and-grasp, then transport-and-place). We also already have the
one thing this technique fundamentally needs: precise privileged object
pose at every step, already computed throughout the reward-function
pipeline (`sim/envs/pickplace_reward.py`, `sim/robots/grasp_geometry.py`).

**Practical path considered**: official code is public
(`github.com/NVlabs/mimicgen`), but built on robosuite/robomimic (a
MuJoCo-based ecosystem) -- adapting it to Isaac Lab would be a real
integration project of similar shape to the TD-MPC2 work (see
docs/tdmpc2_integration.md for how much that one actually took, even with
careful upfront verification). Given how much simpler our task is than
MimicGen's general multi-object, multi-subtask framework, **the user
decided to implement just the core idea directly** (segment one of our
existing teleop episodes at the grasp event -- already detected via
`is_holding()`, see docs/reward_function.md -- then SE(3)-transform and
replay against new cube poses) rather than integrate the full published
framework. Smaller, better-scoped, tailored exactly to what's needed;
trades off reusing battle-tested general code for a simpler, more
directly verifiable implementation of the same core idea.

### The other family found: human-in-the-loop hybrid data collection

A related but different line of work (RoboCopilot and the broader
"interactive imitation learning" literature, closer to the classic DAgger
idea) has a human periodically *intervene* on a partially-trained
policy's live rollouts, rather than pre-generating a dataset. Doesn't fit
our current situation directly (no diffusion policy exists yet to
intervene on), but worth knowing about as a different combination axis
for later -- e.g. once a first diffusion policy version exists and needs
targeted correction rather than more raw volume.

## Decision

Collect a smaller, deliberately style-varied batch of clean human teleop
demonstrations (order of 15-30, not 50-150) covering a few genuinely
distinct approach styles, then multiply that into a much larger dataset
via a custom, simplified implementation of MimicGen's core idea:
segment each source demo at the grasp event, then SE(3)-transform and
replay the two resulting subtask segments against new cube poses sampled
from the same train region TD-MPC2 already uses. This keeps genuine human
strategy at the core of the dataset (preserving the project's actual
research question -- planning vs. imitating a human) while getting
Option B's automation and volume, and meaningfully reduces how much of
the user's own hands-on time collection needs.

**Not yet started** -- the user is coming back to this later. When
resumed, the concrete next steps are:

1. Collect the small (~15-30) styled-varied seed demonstration batch via
   `teleop_bridge.py` (already fixed to randomize cube position within
   the train region -- see below).
2. Design the segmentation point -- likely reusing `is_holding()`'s
   transition (already computed, already validated) as the natural
   "reach/grasp" -> "transport/place" subtask boundary, rather than
   inventing a new heuristic.
3. Implement the SE(3) replay-and-adapt step: given a new sampled cube
   pose, compute the transform preserving the seed demo's relative
   gripper-to-cube pose at the segment boundary, transform the rest of
   that segment's trajectory accordingly, and validate the replayed
   trajectory actually succeeds in sim (not just "looks plausible") --
   the same lesson from `validate_reward_function.py`'s correctness audit
   applies here: a geometrically-transformed trajectory needs to be
   empirically confirmed to work, not assumed to.
4. Decide the target total dataset size once (1)-(3) are working and an
   actual generation rate is known empirically, rather than picking a
   number upfront.

## Teleop cube randomization fix (found while working through this)

A concrete gap surfaced while thinking through Option A's actual
execution: `teleop_bridge.py` never moved or randomized the cube's
position at all -- it sat at the scene's one fixed default spawn point
for an entire session, with its position across "reps" only ever
determined by wherever a human happened to leave it after the previous
pick (already a known source of confusion once, see
docs/reward_function.md's correctness-audit section on
`validate_reward_function.py`). For a fair comparison against TD-MPC2
later (Test 2, novel-position generalization,
docs/evaluation_plan.md), any demonstrations collected for the diffusion
policy need to respect the same train/held-out position split, not
scatter across whatever positions a human happened to leave the cube.

Fixed in `sim/scripts/teleop_bridge.py`: the cube is now randomized (same
`cube_x_range`/`cube_y_range` as `PickPlaceEnvCfg`) at session start, and
re-randomizable between reps via a simple trigger file
(`touch /tmp/teleop_reset_cube`, polled once per sim step and consumed) --
avoids needing a real-time keyboard listener inside the live teleop loop,
same lightweight file-based IPC pattern already used for the leader-arm
state bridge. Verified the underlying position-sampling logic directly
(`sim/scripts/test_cube_randomize.py`) -- the isolated test itself hit an
unrelated Kit process-shutdown quirk when run concurrently with the live
TD-MPC2 training job (killed cleanly by PID, training unaffected); given
the fix mirrors the exact already-validated `env_origins` pattern from
`PickPlaceEnv._reset_idx()` (itself checked against Isaac Lab's own
`cartpole_env.py` reference, see docs/reward_function.md), confidence
from code review alone was judged sufficient rather than forcing a second
concurrent test run. Full empirical confirmation will come from the next
real teleop session.

## Context: TD-MPC2's first real training run, running concurrently

Relevant background for whichever data-source path gets picked later --
if TD-MPC2 turns out to reliably succeed, Option C becomes more tempting
despite its comparison-fairness caveat above; if it doesn't, that
caveat becomes moot on its own. First real run (30,000 steps,
`sim/scripts/train_tdmpc2_pickplace.py --steps 30000 --eval-every 5000`)
launched 2026-08-31; see docs/progress.md for the final result once
complete.
