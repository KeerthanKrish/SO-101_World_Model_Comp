# Evaluation Plan: World Model vs. Diffusion Policy

This is the central research question of the project, so this document
gets the same level of care as reward_function.md. Created 2026-08-30,
before any training has started -- deliberately early, because several of
the tests below only work if specific decisions are made *before* bulk
sim data generation or real demonstration collection begins. Getting the
order wrong can silently invalidate a test months from now with no way to
fix it retroactively short of recollecting data.

## The mechanistic difference driving the test design

This isn't "diffusion vs. non-diffusion" -- it's model-based planning vs.
model-free reaction:

- The **world model (TD-MPC2)** plans: at every step, it samples many
  candidate action sequences, rolls each forward through a *learned*
  dynamics+reward model, picks whichever scored best, executes the first
  action, and replans from the new observed state next step. It has an
  explicit, ongoing way to notice "the world isn't where I expected" and
  correct for it.
- The **diffusion policy** reacts: it reproduces a learned motion pattern
  conditioned on the current observation. It has no lookahead and no
  built-in mechanism to notice or correct for a stale expectation, beyond
  whatever robustness happened to come from demonstration variety.

Every test below is designed to expose *that* difference specifically,
not just produce a single leaderboard number. A test that both models
pass or fail identically isn't telling you anything about this
distinction -- the useful tests are the ones where the mechanism should
matter.

## Common success criterion

All tests should share one success/failure definition regardless of which
policy produced the rollout, so scores are actually comparable. Reuse
`is_placed()` / `is_failed()` from `sim/envs/pickplace_reward.py` for
sim evaluation -- it's already a precise, documented definition (cube at
rest within tolerance of the target). For real-hardware evaluation, define
the human-judged equivalent up front (e.g., "cube is stationary and its
center is within ~2cm of a marked target position") so real trials are
graded consistently across sessions and models, not by eyeballing it
differently each time.

## Test 1: Sample efficiency

**Procedure**: Generate one large pool of domain-randomized sim
episodes/rollouts. Train each model on nested subsets of increasing size
(e.g. log-spaced: 1%, 5%, 20%, 50%, 100% of the pool). For every subset
size, evaluate success rate on one **fixed, held-out validation set**
that is never part of any training subset, so every point on the curve
is measured the same way.

**Metric**: success rate (and optionally reward, steps-to-success)
plotted against training data size, log x-axis.

**Why it differentiates**: the world model's dynamics component can, in
principle, learn from *any* rollout -- not just successful demonstrations
-- since dynamics prediction doesn't need a reward label to be useful
training signal. The diffusion policy only learns from demonstrated
behavior. Expect (hypothesis, not a given) a different-shaped curve at
low data budgets, not necessarily just "world model wins."

**Data collection implication**: Decide the total sim episode budget in
advance (order-of-magnitude -- enough that even the smallest subset is
non-trivial and the largest is meaningfully more than that). Store
episodes as individual files/shards with a stable index, not merged into
one blob -- subsampling has to be easy later, and you don't want to
discover this requirement after the data already exists in the wrong
shape.

## Test 2: Novel start-position generalization

**Procedure**: Define, before generating any data, a **training region**
for cube start position (e.g. a box/ellipse within comfortable reach) and
a separate, non-overlapping **held-out region** (e.g. near table edges or
corners) that is *never* sampled during training. After training,
evaluate on (a) fresh in-distribution positions as a sanity baseline and
(b) the held-out region.

**Metric**: success rate in-distribution vs. out-of-distribution -- the
*gap* between the two is the actual generalization metric. Compare which
model's gap is smaller.

**Why it differentiates**: a world model reasons through explicit
(learned) dynamics, which is arguably more likely to extrapolate
predictably to new positions than a policy that's pattern-matching
against what visually resembles its training data.

**Data collection implication -- the most important one in this whole
document**: the train/held-out position split must be decided and
documented *before* bulk data generation starts. If the held-out region
ever leaks into training data, even once, this test is permanently
invalidated and can only be fixed by regenerating data from scratch.
Write the exact boundary down somewhere durable (this doc, or a config
file) the moment it's decided.

## Test 3: Mid-episode perturbation recovery

**Procedure**: run a rollout to a fixed, consistent point (e.g. right
after reaching grasp height, before closing the gripper), then perturb
the cube -- in sim, teleport it by a known offset; on real hardware,
physically nudge it. Measure whether the policy still succeeds.

Do this primarily **in sim** first -- it's exactly repeatable (same
perturbation timing and magnitude every trial), which a hand-nudge on
real hardware can never quite be. Treat the real-hardware version as a
qualitative "does this hold up physically" spot check, not the primary
quantitative measurement.

**Metric**: recovery success rate across a few perturbation magnitudes
(e.g. 2cm/5cm/10cm) -- a robustness curve, not one number.

**Why it differentiates**: this is the most direct test of the
"replanning" hypothesis. A reactive policy has no principled way to
correct after being fed a now-stale internal expectation; a planner
recomputes fresh from the newly observed state every step.

**Data collection implication**: none for training data -- this is purely
an eval-time manipulation. Worth keeping in mind while designing the
(not yet built) training env wrapper: it should support scripted
mid-episode object teleportation as a feature, so this test doesn't
require bolting something unplanned onto the env later.

## Test 4: Visual domain shift

**Procedure**: define the exact randomization ranges used during training
(lighting intensity, table color, cube color, etc.). At eval time, test
on (a) fresh in-range randomizations as a baseline and (b) deliberately
**out-of-range** conditions -- lighting extremes never seen, a cube color
never included in training randomization.

**Metric**: success rate drop from (a) to (b), compared between models.

**Why it differentiates**: tests whether each model's visual
representation actually generalizes or is brittle/overfit to the
specific randomization coverage it happened to see.

**Data collection implication**: document the randomization ranges used
for sim data generation as they're decided, and deliberately carve out
specific values/extremes to reserve exclusively for this test. If "out of
range" isn't tracked precisely, there's no way to later confirm the test
condition was actually novel.

## Test 5: Distractor objects

**Procedure**: add a second object (different shape/color) near the cube
at eval time. Measure whether the policy still targets the correct cube,
or gets confused (attempts the distractor, hesitates, fails to
discriminate).

**Metric**: success rate with vs. without a distractor present.

**Why it differentiates**: probes whether either model learned a general
"find and grasp the red cube" concept, versus a positional shortcut
("grasp whatever's roughly in this region of the workspace").

**Data collection implication**: proposed as an **eval-only** addition,
not part of training data, to test generalization to an unseen scene
element. Two things worth deciding now, while the (not yet built) env/
observation pipeline is still being designed, rather than as a surprise
later: (1) confirm the observation and reward computation won't break
or misbehave when an extra object exists in the scene that they weren't
built expecting, and (2) decide whether this stays a pure generalization
test (zero distractor exposure ever) or whether you also want a separate
"trained with distractor variety" run as its own comparison -- the latter
would require deliberately generating a chunk of distractor-augmented
training data, a bigger commitment than this document currently assumes.

## Test 6: Zero-shot sim-to-real transfer

**Procedure**: **before any real-world fine-tuning happens**, run both
sim-trained policies on the real follower arm across a small, fixed,
repeatable set of real cube starting positions (physically marked --
tape marks or measured coordinates -- so they're reproducible across
sessions and model checkpoints). Record success rate for each, zero-shot.
*Then* proceed with the planned real-data fine-tuning phase, and re-run
the exact same fixed eval set afterward to measure the before/after
delta.

**Metric**: zero-shot real success rate (arguably the headline result of
the entire project), plus the fine-tuning improvement delta.

**Why it matters most**: this is the actual deliverable the whole
sim-to-real comparison is for -- not sim performance in isolation.

**Data collection implication -- also critical**: reserve a small, fixed
set of real-world evaluation configurations (e.g. 5-10 marked cube
positions) for evaluation *only*, and keep them **out of the real-arm
fine-tuning dataset** entirely. If the same positions used for eval later
get folded into the fine-tuning demonstrations, every post-fine-tuning
number collected on them stops being a fair test (the model will have
been trained on exactly what it's being tested on). Decide and mark these
reserved positions *before* collecting the real fine-tuning batch, not
after -- once fine-tuning data collection is underway it's easy to
forget which positions were supposed to stay untouched.

## Test 7: Inference latency / control frequency

**Procedure**: measure wall-clock time per action decision for each
model on the actual deployment hardware, under realistic conditions
(single-sample inference, matching the real control loop rate).

**Metric**: decision latency (ms) and achievable control frequency (Hz),
compared against the real servo control loop's required rate (already an
open item in `sim_to_real_checklist.md` under "control interface /
frequency").

**Why it matters**: a model that scores well in an unconstrained offline
evaluation but can't run fast enough for real-time control isn't actually
deployable -- this grounds the comparison in practical viability, not just
an abstract benchmark number.

**Data collection implication**: none. Purely a systems measurement,
performed once trained models exist.

## Test 8: Multi-modal demonstration handling

**Procedure**: if demonstrations include genuinely varied valid approach
styles (e.g. approaching the cube from the left some reps, from the right
others), check whether the diffusion policy's repeated samples from the
same start state produce diverse-but-valid distinct approaches (a known
strength of diffusion models), versus whether the world model/planner
collapses to one consistent mode or occasionally blends modes into an
invalid trajectory (a known failure mode of naive planners and value
functions on multi-modal action distributions).

**Metric**: qualitative -- categorize trajectory diversity and validity,
e.g. by clustering resulting end-effector paths. Not a single success-
rate number.

**Why it differentiates**: representing multi-modal distributions well is
a specific, known theoretical strength of diffusion models; this is one
of the few tests that could plausibly favor the diffusion side.

**Data collection implication -- also critical, and easy to miss**: if
every human teleop demonstration uses the *same* rigid approach style
every single time, there is no multi-modality in the data to begin with,
and this test becomes meaningless by construction -- both models will
trivially look "unimodal" simply because the data was. **If this
comparison matters, demonstrations need to deliberately vary approach
style across reps during collection** (some from the left, some from the
right, different grip depths) -- this is a concrete action for whoever is
teleoperating, not something that can be fixed in post-processing.

## Data collection pre-planning checklist

Consolidating every decision above that needs to be made *before* bulk
sim data generation or real demonstration collection starts -- get these
wrong and some tests can only be fixed by recollecting data from scratch:

1. **Cube start-position split**: define and document the training
   region vs. the held-out generalization-test region (Test 2) before
   generating sim rollouts. **In progress 2026-08-30**: running
   `sim/scripts/reachability_sweep.py` to empirically map the arm's
   actual IK-reachable workspace on the enlarged 1.2m table, rather than
   guessing bounds -- see docs/progress.md for the result once complete.
2. **Target position randomization**: currently fixed in
   `pickplace_reward.py` (documented open item). **Decided 2026-08-30**:
   fixed for the first actual training run (isolates pipeline bugs from
   generalization questions), randomized for the real bulk-data-
   generation run once that's validated -- see decisions.md.
3. **Visual randomization ranges**: decide and document lighting/color/
   texture ranges used in training, with specific values deliberately
   reserved as out-of-range for the domain-shift test (Test 4). **Cube
   color specifically decided 2026-08-30**: stays a fixed, constant red
   during training; a different, never-trained-on color is reserved for
   Test 4 -- see decisions.md. Lighting/table-color ranges still pending,
   need Isaac Lab's domain randomization tooling (not yet set up).
4. **Distractor object handling**: decide whether it's a pure eval-time
   addition (default assumption here) or needs its own training data
   variant; either way, check the observation/reward pipeline tolerates
   an unexpected extra object without breaking.
5. **Data storage shape**: save sim episodes as individually addressable
   files/shards with a stable index from the start, so the sample-
   efficiency experiment (Test 1) can subsample later without a data
   migration.
6. **Demonstration style variety**: if multi-modal handling (Test 8)
   matters, deliberately vary teleop approach style across reps during
   collection -- this is a live decision for whoever is at the leader arm,
   not something fixable afterward. **Decided 2026-08-30**: yes, future
   teleop sessions should vary approach style -- see decisions.md.
7. **Reserved real-world eval set**: mark a small, fixed set of real cube
   positions for zero-shot/before-after evaluation (Test 6) *before*
   collecting the real fine-tuning demonstration batch, and keep the
   real-arm demo collection away from those exact configurations.

## Eval harness design note

Given how many of these tests share infrastructure (run a rollout, log
trajectory + success/failure + timing, optionally perturb/randomize
something first), the eval harness is worth building once, generally, to
support all eight tests from day one -- rather than bolting on bespoke
one-off logging for each test as it comes up later. Not urgent to build
yet (no trained models exist), but worth keeping in mind when that
harness is designed.
