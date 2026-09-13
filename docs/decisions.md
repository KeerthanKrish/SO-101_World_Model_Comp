# Decisions Log

Record of choices made and the reasoning behind them, kept separate from the
project plan so the plan stays clean and the "why" doesn't get lost.

---

**Decision**: Compare a TD-MPC2-style world model against LeRobot's diffusion
policy, rather than comparing a diffusion world model against a diffusion
policy.

**Why**: The more interesting research axis is model-based (explicit
predicted consequences, replanning) vs. model-free (direct reflex policy),
not "diffusion vs. non-diffusion." A TD-MPC2-style world model is also
practical for online replanning during real-arm control (cheap per-step
imagination), whereas a diffusion world model is expensive per imagined step
(full denoising loop per step) and better suited to offline high-fidelity
simulation than live planning. A diffusion-based world model remains a
possible stretch goal, not core scope.

---

**Decision**: Use NVIDIA Isaac Lab as the simulator.

**Why**: GPU-parallelized, strong domain-randomization tooling for sim-to-real
work, and a good fit given the available NVIDIA GPU. Alternatives considered:
Genesis (faster/lighter but less mature tooling), ManiSkill3 (manipulation-
focused, good middle ground), MuJoCo MJX (lightweight but less
domain-randomization/manipulation tooling built in).

---

**Decision**: Keep project scope to one task (pick-and-place a single cube),
no fixed calendar timeline.

**Why**: User wants a research-quality comparison, not a multi-task benchmark
sweep. Avoids scope creep while still supporting a genuine sample-efficiency /
robustness comparison.

---

**Decision**: Documentation lives in `docs/` inside the project root (travels
with the code), not as a separate top-level location.

**Why**: User preference — docs and code should move together as one unit.

---

**Decision**: Use NVIDIA driver 580 branch (580.173.02, `nvidia-driver-580-open`)
instead of the 595 branch initially installed.

**Why**: Isaac Sim 5.1.0's RTX renderer segfaults on Blackwell GPUs (this
machine's RTX 5060 Ti included) with the 595.x driver branch -- a known,
widely-reported issue (NVIDIA developer forums, isaac-sim/IsaacLab GitHub
issues all show the same `librtx.scenedb.plugin.so` crash signature). The
580 branch is NVIDIA's officially validated driver for Isaac Sim 5.1.0.
Confirmed fix empirically: camera-enabled rendering worked immediately after
downgrading and rebooting.

---

**Decision**: Do not use Isaac Sim's WebRTC streaming for live remote viewing;
stick with offscreen image/video capture instead.

**Why**: Tried three ways to get the Isaac Sim WebRTC Streaming Client
working over Tailscale to `100.71.12.16:8011` -- (1) plain connection to a
generic empty streaming instance (connected then immediately disconnected,
per `NVST_CCE_DISCONNECTED` in the server log), (2) explicit
`--/app/livestream/publicEndpointAddress` set to the Tailscale IP (same
result), (3) streaming our actual SO-101 scene directly via Isaac Lab's
`--livestream 2 --kit_args` (this time no connection reached the server log
at all). This matches widely-documented unreliability of Isaac Sim's WebRTC
streaming outside NVIDIA's own cloud infrastructure (see docs/progress.md
for sources). NoMachine remote desktop was identified as the reliable
alternative but declined for now -- offscreen rendering (screenshot/video
capture, already working) is the standing approach for visual checks. Revisit
NoMachine if live interactive viewing becomes important later.

---

**Decision**: Local (Windows laptop) and remote (Ubuntu, `keerthan@100.71.12.16`)
copies of the project are kept in sync via `scp`, on-demand only (when the
user explicitly asks), not automatically.

**Why**: Avoids unnecessary token usage from rewriting files on both sides,
and avoids syncing on a timer for no reason. Implementation work (code,
running experiments, robots, GPU) all happens on the Ubuntu machine.

**Update (2026-09-02)**: Superseded for CODE specifically -- the user
migrated their local machine (Windows -> a new MacBook Pro) and switched
code sync to git (Ubuntu pushes, local machine pulls) instead of scp.
The underlying reasoning here (Ubuntu is where implementation actually
happens, local machine is a consumer/mirror) is unchanged and is exactly
why git's one-directional flow fits -- only the sync MECHANISM for code
changed. scp/rsync is still used, on-demand, for the large binary
artifacts `.gitignore` deliberately excludes and git will never carry
(`sim/output/`, `assets/`). See docs/preferences.md for the current
arrangement.

---

**Decision**: The pick-and-place reward function (`sim/envs/pickplace_reward.py`)
uses two-phase potential-based shaping (approach -> transport, switching at
the grasp event) with tanh-bounded distance terms, rather than a single
continuous reward (e.g. cube height) or raw negative distance.

**Why**: A pure height-based reward has no incentive to ever place the cube
back down, since placing requires lowering it again at the target -- it would
fight against the actual goal. Framing the post-grasp phase as "distance to a
3D target point at table height" makes lift -> move -> lower one continuous,
non-conflicting signal. tanh-bounding (rather than raw distance) keeps the
reward smooth and bounded, which matters because TD-MPC2 trains a neural
network to predict this value -- a bounded target is an easier regression
problem than an unbounded one. See docs/reward_function.md for the full
design writeup.

---

**Decision**: Grasp detection requires cube height + gripper joint angle +
gripper-to-cube proximity, all three together -- not just height and joint
angle.

**Why**: Caught empirically, not by code review -- validating the reward
against a real captured teleop episode showed a false positive where a cube
barely above rest height (still settling from a previous rep) plus a
coincidentally-closed gripper joint elsewhere on the table registered as
"grasped." No contact sensor exists on the gripper yet, so this is a
heuristic; a proper `ContactSensorCfg` would be a more principled fix if
misdetection becomes a real problem once training starts.

---

**Decision**: The reward function's place target reuses `run_pickplace_demo.py`'s
existing place-zone location, `(-0.15, 0.15, ...)`, rather than defining a
new one.

**Why**: One arbitrary place location per project, not two independently-
drifting ones. Fixed (not yet randomized) for this first version, since a
fixed position is what allowed empirical validation against real recorded
(fixed-position) demonstrations -- will need to become randomized once
domain randomization for object/target pose is wired in, per the project
plan.

---

**Decision**: The place target will be randomized eventually, but stays
fixed for the first actual training run once the training pipeline is
built.

**Why**: Isolates infrastructure bugs (does the training loop even work)
from generalization questions (does the policy handle novel targets) --
validate the simple, fixed-target case works end-to-end first, then add
target randomization for the real bulk-data-generation run. Made ahead of
time while planning docs/evaluation_plan.md's data-collection checklist,
not yet acted on since no training pipeline exists yet.

---

**Decision**: Cube color/appearance stays a fixed, constant red during
training. A different color (not used anywhere in training) is reserved
exclusively for the visual domain-shift evaluation test
(docs/evaluation_plan.md, Test 4).

**Why**: This is a single known-object task, not a general color-invariant
grasping system -- randomizing cube color during training isn't needed
for this scope and would add noise without benefit. Keeping it constant
also means a genuinely novel color at test time is a clean, meaningful
out-of-distribution probe, rather than something partially already seen
during training randomization.

---

**Decision**: Future teleop demonstration sessions should deliberately
vary approach style (e.g. sometimes approaching the cube from the left,
sometimes the right, varying grip depth) rather than repeating the same
motion every rep.

**Why**: Keeps the option open to test multi-modal grasp handling later
(docs/evaluation_plan.md, Test 8) -- diffusion policies are known to
represent multi-modal action distributions well, but there's nothing to
test if every demonstration used one identical rigid style. Low cost to
start now; expensive (impossible without recollecting data) to add
retroactively.

---

**Decision**: Enabled self-collision checking on the SO-101 articulation
(`enabled_self_collisions=True`, was `False`); tightened the reward's
`reach_scale` from 0.15 to 0.08; added direct agent checkpoint saving to
`train_tdmpc2_pickplace.py`.

**Why**: Reviewing the first real TD-MPC2 training run's eval video
directly (not just the printed reward numbers) surfaced two real problems
the summary stats alone hid. First, the arm was contorting into
self-intersecting poses -- physically impossible on the real hardware --
because self-collision had been left at its inherited default (`False`)
with no one having deliberately decided it should be off. Second, an eval
episode that scored the highest reward of the run (+97.9) had never
touched the cube at all: the dense approach-phase reward (`reach_scale`
governing how quickly it falls off with distance) was broad enough that
merely hovering in the general vicinity of typical cube positions, without
tracking any specific episode's actual cube, could accumulate substantial
reward over a 500-step episode -- genuine reward hacking, not learning
progress. Checkpoint saving was added because the trained policy from that
run couldn't be reloaded for further inspection once the video raised
these concerns -- every future run now saves one.

**How to apply**: A shorter (15,000-step) run was launched immediately
after these fixes, specifically so the fixes themselves can be verified
quickly (no reward hacking, no self-clipping, physics still stable)
before committing to a longer run. See docs/tdmpc2_integration.md for
results once available. `reach_scale=0.08` is not claimed to be the
final right value -- if the shorter run still shows hacking, it needs
tightening further; if it can no longer learn *any* useful gradient
(reward is now too sparse to provide signal from far away), it needs
loosening. Judge from the next run's actual behavior, not from re-deriving
the value analytically.

**Superseded by the decision below** -- that run showed the opposite
problem (too sparse, no gradient), which led to identifying `reach_scale`
narrowing as the wrong kind of fix in the first place.

---

**Decision**: Replaced absolute-value dense shaping with potential-based
shaping (`reward = weight * (Phi(new dist) - Phi(old dist))`, not
`weight * Phi(current dist)`); added a one-time `touch_bonus`; converted
`grasp_bonus` from a continuous per-step reward into a one-time bonus;
reverted `reach_scale` from 0.08 back to its original 0.15.

**Why**: The `reach_scale=0.08` run above (decision immediately prior)
showed consistently negative, non-improving reward across all 4 eval
checkpoints, and extracted video frames confirmed genuinely undirected
motion, never engaging the cube -- the narrower scale killed most of the
useful gradient for a policy that starts far away, without actually
fixing the underlying mechanism (a policy could still profit from
occupying any fixed position, just a smaller one). Potential-based reward
shaping (Ng, Harada & Russell, ICML 1999) fixes the actual mechanism:
rewarding only the *change* in a potential function is provably
policy-invariant (never changes the MDP's optimal policy, for any choice
of potential), and under this formulation, holding still anywhere earns
exactly zero shaping reward every step -- closing the original hacking
exploit without needing a narrow, gradient-starved scale to do it. Once
the mechanism no longer depends on scale, there is no more reason to keep
`reach_scale` narrow, so it reverted to 0.15. Separately, the old
`grasp_bonus` (a flat reward paid every step while holding) was found to
create its own, previously-unnoticed incentive to hold the cube
indefinitely rather than finish the task, since finishing ends the
episode -- fixed by making it fire once, like the new `touch_bonus`. See
docs/reward_function.md's "Reward-hacking finding and potential-based-
shaping redesign" section for the full writeup, including the
verification suite run before trusting this (self-test, state-only and
camera-enabled env smoke tests, real-data replay against two recorded
teleop episodes).

**How to apply**: A new ~15,000-step run was launched after this redesign
and its verification suite, matching the eval/checkpoint/video cadence of
the previous (killed) run. Judge success the same way as before --
watch the actual eval videos, not just the printed reward trend --
specifically checking that the arm now engages the cube (touch_bonus/
grasp_bonus firing in the logs) rather than just occupying a static pose.
See docs/tdmpc2_integration.md for results once available.

---

**Decision**: Raised TD-MPC2's seed-episode exploration budget from
tdmpc2's own default (5 episodes, ~2500 steps) to 30 episodes (~15,000
steps), exposed via a new `--seed-episodes` CLI flag.

**Why**: Both the 15,000-step and 50,000-step runs under the new
potential-based-shaping reward converged to the arm holding one fixed
idle pose for the entire episode, regardless of checkpoint or the cube's
randomized position -- confirmed by direct frame inspection at dense
sampling across multiple checkpoints in each run, not just the reward
numbers (which stayed small and non-hacky, but that turned out to mean
"honestly near-zero" rather than "learning"). The likely mechanism: the
new reward correctly gives exactly zero net reward for holding still
anywhere (that's what closes the original hacking exploit), which also
removes any reward pressure pushing an untrained policy to move at all,
unless the replay buffer already contains a genuine touch/grasp
trajectory for the value function to learn from. tdmpc2's own default
exploration budget (a heuristic tuned for its own benchmark tasks, not
this one) apparently never produced such a trajectory against a small,
randomly-positioned target in either run. This is presented as a targeted
experiment based on the likely root cause, not a proven fix -- running
the SAME 50,000-step total budget again with only this one variable
changed, specifically so any behavioral difference can be attributed to
the exploration-budget change rather than confounded with other changes.

**How to apply**: Judge success the same way as every run before it --
watch the actual eval videos/frames at dense sampling across multiple
checkpoints, not just the printed reward trend, since a small/near-zero
reward has now been shown twice to be consistent with either genuine
non-hacky learning OR a static idle policy that happens to score
similarly. If this run still shows the same fixed-pose behavior, the
next hypothesis to test would be the visual encoder's ability to
represent the cube's position at all (a diagnostic, not a design change),
before trying a further reward-design change. See
docs/tdmpc2_integration.md for results once available.

**Update**: this run (the fifth overall) showed the identical fixed-pose
behavior -- raising the seed-episode budget alone was not the fix. See
the decision below for what actually moved the needle.

---

**Decision**: Added two curriculum/exploration levers, `--cube-pos` (fix
the cube to one point instead of randomizing it every episode) and
`--min-std` (raise the floor under TD-MPC2's own CEM planning noise from
0.05 to 0.5), and used both together for a sixth training run.

**Why**: Raising the random-exploration seed budget 6x (the previous
decision) didn't change anything -- run5 reproduced run4's exact
fixed-idle-pose behavior. Rather than keep scaling that same knob further
on faith, traced TD-MPC2's own planning code (`_plan()` in tdmpc2.py)
directly. Two things stood out: (1) random exploration has to relocate a
small, randomly-placed target from scratch every single episode, which is
a genuinely hard search problem on its own, independent of how much of it
there is; (2) TD-MPC2's CEM planner samples 512 candidate action
sequences and narrows to 64 elites over 6 iterations every single step --
and CEM is known to over-confidently narrow its own sampling std even
when the value estimates it's ranking by are pure noise (no real learned
signal yet), which is exactly the mechanism that would produce a
falsely-precise, repeatably-idle action. The actual training-time
exploration noise (`a = a + std * randn(...)`, added only when
`eval_mode=False`) uses exactly that potentially-falsely-converged std.
`--cube-pos` attacks the first problem by making the target trivially
easy to find (curriculum learning: solve the easier version -- can this
learn to reach/grasp at all -- before asking the harder one -- can it
generalize across positions). `--min-std` attacks the second directly by
putting a floor under how confidently-wrong CEM's std collapse can get,
with zero effect on eval-time behavior (`eval_mode` skips the
noise-injection line entirely, so this can't be "cheating" by making
eval look artificially better -- it only changes what the training
rollout itself explores).

**How to apply**: This run (the sixth) is a real result, not just an
experiment sent off to run -- `touched=True` fired on every eval
checkpoint from step 20,459 onward (5 in a row), the first time any run
has shown reliable, repeated contact with the cube rather than either
static idling or a single lucky-looking frame. `held` never went true and
no episode succeeded. Both levers were changed together, so this doesn't
tell us which one mattered more, or whether both were needed -- not worth
disentangling yet given neither had been tried at all before. A seventh
run at 70,000 steps with identical settings was launched immediately
(per explicit standing instruction: launch the next run automatically
once the current one finishes, without waiting for confirmation) to see
whether more of the same training converts reliable touching into
reliable holding. See docs/tdmpc2_integration.md for results as they
land.

**Update**: run7 (still in progress at time of the next decision below)
logged one eval episode's `held=True` -- checked directly on video, since
this project always watches before trusting a success flag, and it was
the right call: the gripper closed fully BESIDE the cube, never around
it. A real detection false positive, not progress. See the next decision.

---

**Decision**: Added `is_between_jaws()` (a directional geometry check
replacing the old spherical-distance-only grasp/hold detection) and a new
`grasp_close_weight` potential-based shaping term for the specific act of
closing the gripper, gated on that same check. Also moved the training
cube position from `(0.15, 0.0)` (the train region's geometric center) to
`(0.25, 0.0)` (the far edge of the same region) for run8.

**Why**: The run7 false positive above traced to a real, specific gap --
`is_grasped()`/`is_holding()` only ever checked "is the jaw pivot within
some radius of the cube," which is identical whether the cube is in
front of the closing jaws or off to the side of them. Fixed by
decomposing the cube's position relative to the jaw pivot along the
gripper's live reach direction (`jaw_approach_axis_world()`,
`sim/robots/grasp_geometry.py`) into an axial (within-reach) and lateral
(centered) component -- reusing the ALREADY-calibrated `JAW_OFFSET_LOCAL`
offset as a direction rather than deriving new hardware calibration. A
live measurement first ruled out the naive alternative (raw distance
between the two jaw bodies' origins) -- that distance stays constant
(~3.6cm) across the gripper's full joint range, since the joint rotates
the moving jaw about a pivot rather than translating it, so it carries no
information about openness at all.

Separately, no reward signal existed for the act of closing the gripper
once well-positioned -- reach-shaping only ever measured gripper-to-cube
distance, never the gripper's own closedness. The user's explicit,
specific requirement for this term: it must NOT reward closing just for
happening near the cube (that would reproduce the run7 incentive at the
shaping level even after fixing detection) -- only for happening with the
cube genuinely positioned to be caught, which is exactly what gating on
`is_between_jaws()` enforces.

The cube position moved to the region's far edge based on the user's own
direct observation from run6/7 video review: the cube felt too close to
the base for a clean grasp angle. Checked against
`sim/output/reachability_sweep.json` rather than picked freely --
`(0.25, 0.0)` turned out to have the BEST empirically-measured IK
convergence of any point checked in the entire train region (error
0.0009, tied for lowest), not just "further away," so this isn't a
tradeoff against reachability.

**How to apply**: Verified via 6 new self-test cases in
`pickplace_reward.py`, including a direct regression test reproducing
run7's exact false positive (cube positioned beside, not around, a
closed gripper -- within the old proximity threshold, above the height
threshold, failing only the new geometric check) and confirming both
`is_grasped()` and `is_holding()` now correctly reject it. An eighth
run using both new levers, with the far-edge cube position, is the next
real test -- judge the same way as every run before it: watch the actual
eval videos, not just the printed flags, especially now that `held=True`
has been shown capable of a false positive once already. See
docs/tdmpc2_integration.md for results as they land.

---

**Decision**: Add `lateral_align_weight` (continuous potential-based
alignment shaping) and `premature_close_weight` (an absolute penalty for
closing at the wrong time) to `pickplace_reward.py`, and add
`--resume-from` warm-start support to `train_tdmpc2_pickplace.py`.

**Why**: Run8 (the `is_between_jaws()` fix, cube moved to the train
region's far edge) produced the best behavior yet -- reliable, purposeful
reaching and touching -- but zero `between_jaws=True` across all 14 eval
checkpoints. Direct video review of the best checkpoint showed two
specific, correctable patterns rather than "just needs more training":
the arm approaches from directly above and pokes the cube with a single
fingertip, never straddling it with both jaws, and separately closes the
gripper almost immediately on approach, well before correctly positioned
(the second one specifically flagged by the user from the same video).

Two different gaps, not one. `is_between_jaws()` is a binary gate with no
gradient leading up to it -- improving alignment from wildly-off to
almost-centered earned the same (zero) reward as not improving at all.
`lateral_align_weight` fixes this the same way every other shaping term
in this reward works: potential-based, over the raw lateral offset itself
(reusing `is_between_jaws()`'s own decomposition, factored into a shared
`_jaw_offsets()` helper), gated on being within a slightly looser
activation range than `touch_threshold` so the gradient can start before
actual contact.

Separately, nothing discouraged closing at the wrong time -- it was
simply neutral (no reward, no cost), plausibly compounded by `min_std`'s
exploration floor (raised for run6 onward) applying uniformly across all
action dimensions including the gripper, injecting persistent noise into
gripper actuation with nothing counteracting it. `premature_close_weight`
adds a small, deliberately ABSOLUTE (not potential-based) penalty for
being closed while not correctly positioned and not yet holding.
Explicitly confirmed this does NOT reintroduce the original reward-
hacking mechanism (docs/reward_function.md, 2026-08-31 redesign): that
was specifically an absolute-value REWARD farmable by occupying a state
indefinitely; a pure PENALTY has the opposite incentive structure
(minimized by avoiding a state, never maximized by dwelling in it), so
there's nothing to exploit. Mutually exclusive with `grasp_close_weight`
by construction (opposite gating condition), so the two never compete on
the same step.

`--resume-from` warm-starts training from a saved checkpoint (`TDMPC2.load()`
already existed in the vendored reference code, simply never wired up)
rather than training from scratch -- run8 genuinely learned to reach and
touch with real, repeatable intent, and restarting from zero to train
against these two new terms would discard that skill and make the agent
re-learn reaching before it could even attempt alignment. Weights only
(`TDMPC2.save()` never persisted the replay buffer), paired with a much
smaller `--seed-episodes` than any from-scratch run, since a resumed
policy that already knows how to act gets little value from a long
pure-random warmup.

**How to apply**: Verified via 13 new self-test cases (the same
symmetric improve/worsen and no-free-lunch properties already required of
every other shaping term in this module), a full local + remote self-test
pass, state-only and camera-enabled env smoke tests, and real-data replay
validation -- all pass with no crash. Caught and fixed two real bugs
while wiring this up, neither in the reward design itself: the self-test's
own `compute_reward()` calls were passing `cfg` positionally in a slot
that silently became `prev_lateral` the moment that parameter was
inserted (caught by an actual test failure, fixed by passing `cfg=cfg`
explicitly everywhere), and a stale test fixture (`gripper_open_joint`)
that was only ever "open enough" for the old binary check, not the true
physical open limit, which registered as partially closed under the new
continuous potential. A ninth run, warm-started from run8's final
checkpoint with both new terms active, is the next real test -- judge it
the same way as every run before it: watch the actual eval videos. See
docs/tdmpc2_integration.md for results once available.

---

**Decision**: Every TD-MPC2 training run now writes its eval videos and
checkpoints into its own `run_name` subfolder (`--run-name`, defaulting
to an auto-generated timestamp) under `sim/output/tdmpc2_eval_videos/`
and `tdmpc2_checkpoints/`, instead of directly into those directories.

**Why**: Asked to organize existing run videos into per-run folders for
easier navigation, and discovered why that was hard in the first place --
eval checkpoint step numbers depend only on episode length and eval
cadence, which most runs share, so most runs produced identical
checkpoint filenames. Every run had been writing flat into the same two
directories since run1, meaning each later run's save silently
overwrote an earlier run's file at that exact name, with nothing
surfacing an error either way.

Reconstructed the damage from the training logs that still exist (runs
6-9) plus file timestamps: **runs 2, 3, 4, 6, and 7's raw video and
checkpoint files are unrecoverable**, overwritten by whichever run came
right after them. Partial visual evidence for runs 3, 4, and 7 survives
only because individual frames happened to be extracted and saved as
PNGs during live analysis of those runs at the time (now under
`sim/output/frame_extract/runN/`); run 6 has no surviving images at all,
only the written notes already in docs/. One video (run7's step-20459
false positive) was recovered from an early manual copy made before the
overwrite happened, and restored to its rightful place. Runs 1, 5, 8, and
9 are intact -- each happened to be the last run to use its particular
step-number range. The reward numbers and interpretations already
written into docs/ are unaffected by any of this, since they were
recorded from live logs at the time, not derived from these files after
the fact.

**How to apply**: Existing surviving files reorganized into `runN/`
subfolders on both machines (an `_ambiguous_runs2-3-4/` folder holds
files that collided among exactly those three runs and can no longer be
attributed with certainty; `_verification_smoke_tests/` holds ad-hoc test
runs that were never part of the numbered sequence -- see the README.txt
in each directory). Going forward, every run's `--run-name` is printed at
startup (`[INFO] run_name=...`) -- worth glancing at when launching a
run, since nothing currently checks that an explicitly-passed name
doesn't collide with an existing one (the auto-generated timestamp
default always will avoid this; only a manually-chosen `--run-name`
could still collide).

---

**Decision**: Add `axial_align_weight` (continuous potential-based
shaping over the axial/depth component of `_jaw_offsets()`), then fully
revert it one round later after it failed to help.

**Why**: The user watched run9's videos and flagged two near-miss
episodes -- the gripper pushing the cube along with clear directed
intent, close enough that closing at the right moment "would've basically
picked it up." My first read of the frames was wrong: I initially
concluded the gripper was CLOSED during these pushes and began
implementing gripper-aperture pre-shaping. The user corrected this
directly ("no hang on, the gripper is open when its pushing it") --
re-examining the same frames with that correction showed the gripper
genuinely open, but the cube buried near the jaws' pivot/hinge (where the
open gap is narrowest) rather than out near the fingertips (where it's
widest). Hypothesized root cause: `reach_weight` rewards driving raw
gripper-to-cube distance toward zero, with no notion of a correct
standoff distance, so it keeps paying out for pulling the pivot itself
past the point where the fingers could actually catch the cube.
`axial_align_weight` mirrored `lateral_align_weight` exactly (same
potential-based delta, same `align_activation_range` gate) but over the
axial component instead of the lateral one, rewarding any position inside
the valid `[grasp_reach_min, grasp_reach_max]` window and penalizing
drifting further outside it in either direction.

Run10 (100,000 steps, warm-started from run9, the longest single run to
date) showed no improvement once actually reviewed frame-by-frame: full
open-loop "reach once and freeze" behavior across every sampled episode
-- the arm made contact within 1-2 seconds, then held a static pose for
the remainder of a 25-second episode regardless of what happened next
(the cube sometimes stayed wedged near the pivot, sometimes got pushed
completely out of reach with the arm never re-engaging), and the gripper
never closed even once, in any sampled frame, across three full episodes
inspected in detail. Since axial_align_weight only concerns *position*,
not the actual close decision, and the video showed the real remaining
gap was the gripper never attempting to close at all, keeping the term
would only have added complexity without addressing the actual problem.
Reverted cleanly via `git revert` (not deleted-in-place), preserving the
implementation and its self-test coverage in history in case the axial
positioning question becomes relevant again later.

**How to apply**: The corrected diagnosis technique -- re-examining
earlier frames with fresh eyes after a direct user correction, rather
than defending the first read -- is the reusable lesson here, not the
specific reward term. See docs/progress.md for the full narrative
including the misdiagnosis and correction, and the entry below for what
actually turned out to matter (the closing incentive itself).

---

**Decision**: Widen `grasp_reach_min` (-0.01 -> -0.04), raise
`grasp_close_weight` (1.0 -> 2.5), lower `premature_close_weight`
(0.3 -> 0.1) -- a "closing-focused" redesign replacing the reverted
axial-alignment attempt above.

**Why**: With axial_align_weight reverted, the real question became "why
does the gripper never close at all, across runs 8, 9, and 10." Checking
`is_between_jaws()`'s actual trigger rate against where contact was
really happening (the cube consistently landing right at/beyond the OLD
`grasp_reach_min` boundary, wedged near the pivot, not out toward the
fingertips) showed it was almost never true at the policy's real contact
point -- starving `grasp_close_weight` of any chance to fire, while
`premature_close_weight` (gated on `not between_jaws`) fired on nearly
every step the gripper had any closedness at all, since "not between
jaws" was the overwhelmingly common case. Reward far more often for NOT
closing than for closing is exactly a "never close" training signal.

Three narrow changes targeting only this mechanism, deliberately leaving
the reach/lateral-alignment machinery untouched (video review confirmed
that part already works): widen the window so it actually covers the
observed contact point, raise the closing reward so it's a dominant
signal once reachable, lower the penalty so it no longer drowns that
signal out.

**How to apply**: Verified via an updated self-test (one boundary-
dependent fixture needed adjusting once the window widened) and a full
production-pipeline smoke test. Launched run11 (60,000 steps,
warm-started from run9's checkpoint specifically -- NOT run10's, since
run10's extra 100,000 steps were spent reinforcing the counterproductive
"freeze" habit under the old incentive structure, and building on top of
a more deeply entrenched bad habit seemed worse than restarting from the
less-contaminated run9 checkpoint). Run11's frame-by-frame video review
showed real improvement in the arm's dynamism (more exploratory,
re-approaching behavior rather than "dive and freeze") but the objective
per-episode logging (`between_jaws`/`held`, already emitted by
`run_eval_episode()`) was the actual source of truth -- see the next two
entries for why trusting that logging over a video read mattered here.

---

**Decision**: Zero out `premature_close_weight` entirely (0.1 -> 0.0) as
a single-variable experiment, rather than combining it with a grace
period.

**Why**: The user watched run11's `eval_step_050399.mp4`/
`eval_step_055389.mp4` and disputed my claim (based on reading extracted
still frames) that the gripper was visibly closing -- correctly. Rather
than re-litigate the same frame-reading approach, pulled the objective,
ground-truth `touched`/`between_jaws`/`held` booleans `run_eval_episode()`
already logs once per eval episode straight from the training log:
`held` was `False` in every one of run11's 11 eval checkpoints, while
`between_jaws` had flipped `True` in 4 of them -- something that
essentially never happened in runs 9/10. So positioning had genuinely
improved; closing specifically had not.

To find out why, added `--eval-only` (see the next entry) and replayed
run11's best checkpoint with new per-step logging (gripper joint angle,
commanded gripper action, axial/lateral offset, `between_jaws`). That
data ruled out "brief pass-through, no time to close": `between_jaws`
fired dozens of times across one 500-step episode, including windows
lasting 10+ consecutive steps. The gripper still barely moved even
during those long windows, and the one place it dipped meaningfully
(~0.06 radians toward closed) reversed within a few steps, climbing
back to ~fully-open rather than continuing to close or settling.
Mechanism: `premature_close_weight` is gated on `not between_jaws`, and
`between_jaws` itself is unstable (true only ~15% of that episode,
mostly in short bursts) -- so a policy partway through a slow,
multi-step close is one small drift away from the penalty resuming on
its still-partly-closed gripper. Reopening immediately, before the
penalty can resume, is the locally safe strategy. The original reason
this penalty existed (run8 closing carelessly far from the cube,
regardless of position) is now separately handled by
`grasp_close_weight`'s own `is_between_jaws()` gate, which did not exist
yet when this penalty was first introduced.

Considered combining the fix with a grace period (only charge the
penalty after several consecutive out-of-window steps) or giving
`between_jaws` itself hysteresis, but deliberately zeroed the weight
outright first as the cleanest single-variable test of the underlying
hypothesis -- if removing the penalty alone did not produce a real
close, that would argue the instability theory needed one of those
instead, not just less punishment.

**How to apply**: Verified via self-test (the mechanism itself stays
covered through a local nonzero-weight config, `premature_test_cfg`,
even though the production default is 0.0) and a production smoke test.
Launched run12 (60,000 steps, warm-started from run11's final
checkpoint). Result: run12 produced this project's first-ever `held=True`
eval result (step 40419) -- see the entries below for how that claim was
verified rather than taken at face value, and what it actually turned
out to mean.

---

**Decision**: Added `--eval-only` to `train_tdmpc2_pickplace.py` -- loads
a checkpoint via `--resume-from`, runs exactly one deterministic eval
episode, and writes a per-step diagnostic CSV alongside the usual eval
video, then exits (no buffer/logger/training loop constructed at all).

**Why**: `run_eval_episode()`'s existing once-per-episode
`touched`/`between_jaws`/`held` booleans could say WHETHER something
fired during an episode, but not for how many consecutive steps, nor
what the gripper was actually doing during that window -- not enough to
tell a genuine timing problem (the cube only correctly positioned for a
handful of steps) apart from the closing incentive itself still being
too weak, and not enough to independently verify a `held=True` result
against is_holding()'s own four establishing conditions rather than
trusting the flag at face value. Deliberately reads gripper/cube state
directly from the env/robot (`_grasp_points_local()`, `cube.data.root_pos_w`)
and calls `pickplace_reward.py`'s own pure helpers (`_jaw_offsets`,
`_dist3`) rather than modifying `pickplace_env.py` to expose new fields
-- keeps the diagnostic fully outside the classes actual training
depends on. `step_log_path` defaults to `None` everywhere in
`run_eval_episode()`, so this changes nothing about a real training
run's own periodic eval calls.

Extended once more (2026-09-06) with `cube_height`/`gripper_cube_dist`
specifically to audit run12's `held=True` result against all four of
`is_holding()`'s own conditions independently -- see the entry below for
what that audit found.

**How to apply**: `./isaaclab.sh -p sim/scripts/train_tdmpc2_pickplace.py
--headless --enable_cameras --eval-only --resume-from <checkpoint.pt>
--cube-pos 0.25 0.0 --min-std 0.5 --run-name <diagnostic-run-name>`.
Always give it its own `--run-name` (e.g. `run12_diag`) distinct from
the run being diagnosed, so the diagnostic video/CSV never lands in the
same folder as that run's real output.

---

**Decision**: Add `between_jaws_grace_steps` hysteresis (via a new
`_between_jaws_effective()` helper, gating ONLY `grasp_close_weight`/
`premature_close_penalty`), raise `lift_threshold` (0.02 -> 0.04), and
regate `grasp_bonus` on a genuinely episode-sticky `was_ever_held`
instead of `was_holding`.

**Why**: run12's `held=True` result (step 40419) needed verifying, not
just accepting -- the same lesson as the video-misread entries above,
applied to a claim built on logged data this time rather than my own
eyes. Extended `--eval-only`'s CSV with `cube_height`/`gripper_cube_dist`
and replayed the exact checkpoint. Finding: all four of `is_holding()`'s
establishing conditions genuinely fired -- not a geometric false
positive like the run7 bug -- but the actual lift was only ~9mm on a 3cm
cube (`lift_threshold` required just 5mm above resting height), visibly
settling back toward resting height within a few steps rather than being
carried. It recurred 3 separate times in the one episode, each re-firing
the full `grasp_bonus`, because that bonus was gated on "holding now but
wasn't the previous step" (one-time per continuous streak), not "first
time this episode" -- a policy could cheaply farm the milestone by
grazing a razor-thin threshold repeatedly, with no additional reward for
lifting higher or holding longer.

Separately, the same replay showed `is_between_jaws()` flickering False
for single steps in the middle of otherwise-sustained close attempts
(true only ~15% of the episode, mostly short bursts) -- since
`grasp_close_weight` is gated on it being true THIS step, those
drop-outs zeroed the reward for steps where real progress toward closed
was still being made, independent of `premature_close_weight` (already
0.0) being a factor at all.

Three changes, addressed together since all three surfaced from the same
replay: `between_jaws_grace_steps` (default 2) forgives up to 2
consecutive flicker-outs for the closing/premature-penalty gate
specifically -- deliberately NOT touching `is_between_jaws()` itself or
`is_grasped()`/`is_holding()`'s own establishment logic, which must stay
exactly as strict as before (loosening the same check that gates a
genuine hold would directly undermine the very next fix). `lift_threshold`
raised to 0.04 (roughly 2.5x the observed accidental jostle). `grasp_bonus`
regated on `was_ever_held`, a new parameter mirroring `touch_bonus`'s own
`was_touched` pattern exactly, which never had this bug.

**How to apply**: `was_ever_held` was added as a required (no default)
parameter to `compute_reward()`, deliberately -- a silently-wrong default
would let a caller forget to thread the real persisted value through
without any error, exactly the kind of mistake this project's own
`cfg=cfg`-positional bug (see the lateral-alignment entry above) already
demonstrated is easy to make. Required updating and auditing all ~46
`compute_reward()` call sites in the self-test (verified programmatically
that none were missed) plus both real call sites (`pickplace_env.py`,
`validate_reward_function.py`). Verified via 12 new self-test cases
(direct unit tests of `_between_jaws_effective`, an end-to-end hysteresis
sequence through `compute_reward()`, and real regression tests for both
the lift and grasp_bonus farming bugs using the actual observed numbers),
a full local + remote self-test pass, and a production smoke test.
Launched run13, warm-started from run12's final checkpoint, as a
relatively cheap verification that these specific fixes work as intended
before committing to the much larger investment of a full, from-scratch
clean retrain against the finalized reward design.

---

**Decision**: `--eval-only` replays are NOT reproductions of a
checkpoint's original logged episode -- documenting this as a real tool
limitation, discovered while trying to verify run13's fixes, rather than
letting it stand as an unstated assumption.

**Why**: replayed two of run13's `between_jaws=True` checkpoints
(020459's own step number, and 055389) to check whether the raised
`lift_threshold` was specifically blocking a would-be false hold. The
first replay came back as an entirely different, much less-engaged
episode than what training-time eval had logged for that exact
checkpoint (no `between_jaws`, cube never left resting height) --
inconsistent with byte-for-byte reproduction. Root cause: TD-MPC2's CEM
planner samples candidate trajectories from a random distribution at
every planning step, and nothing in this pipeline seeds that RNG (the
`isaaclab.envs.direct_rl_env` "Seed not set" warning that appears in
every log flags the same underlying issue for env creation, but the
planner's own sampling is a separate, unstated instance of it).
`eval_mode=True` only suppresses the FINAL chosen action's added
exploration noise -- it does not make the planner's own internal search
deterministic. Each `--eval-only` invocation is therefore a genuine,
independent fresh sample from the checkpoint's policy, not a replay of
one specific past episode.

**How to apply**: does not undermine the earlier run12 audit's core
finding (the lift/grasp_bonus farming pattern was established by
examining many steps/events within one representative episode, a
property robust to which exact episode gets sampled) -- but any FUTURE
`--eval-only` diagnostic must be read as "a representative sample of
this checkpoint's typical behavior," never as "the exact episode that
produced a specific past log line." Seeding TD-MPC2's own RNG (e.g. a
`torch.manual_seed()` call before `agent.act()` inside
`run_eval_episode()`) would fix this if exact reproducibility is ever
needed again, but has not been done -- not required for this
diagnostic's actual purpose (characterizing typical behavior), and
seeding a planner that samples every step across a 500-step episode
would need care to get genuinely right, not a one-line fix.

---

**Decision**: Launched run14 -- a full, clean, from-scratch retrain (no
`--resume-from`, default 30-episode/15,000-step seed phase, 100,000
steps) against the current, most-refined reward design, and run15 -- an
80,000-step continuation warm-started from run14's own checkpoint.

**Why**: after watching run13's videos, the user observed that the arm
no longer seemed to be even getting positioned between the jaws as
reliably as it had appeared to earlier in the project, and asked whether
recovering that -- and only then layering the closing-leniency mechanics
on top -- was the right move. Agreed, and connected it to the run13
`between_jaws` numbers actually declining across runs 11->12->13 (4/11
-> 4/11 -> 3/11) -- consistent with five generations of continuous
warm-starting through several different, sometimes-conflicting reward
regimes since run9 having left the checkpoint's value function a
patchwork, not a clean optimum, for positioning specifically (the
`reach_weight`/`lateral_align_weight` terms actually responsible for
positioning had not been touched since run8/9 -- everything since had
been about closing). Since there is no way to surgically recover just
the positioning-relevant learning from within an already-warm-started
network, training fresh seemed like the direct way to let positioning
re-develop without the inherited drift, with every closing-leniency fix
already active from step zero.

**Result -- the theory was wrong, not confirmed**: run14 (clean) got
`touched=True` reliably (74% of checkpoints, matching run8's own
from-scratch benchmark) but `between_jaws=True` on only 1 of 19
checkpoints -- WORSE than any warm-started run, not better. Run15 (80k
more steps warm-started from run14, directly testing "just needs more
time on top of solid touching," deliberately mirroring the run8-to-run9
pattern) got 0 of 15 checkpoints, the entire run. Reassessed rather than
forced the original theory: `lateral_align_weight` has only ever been
shown to work refining an ALREADY-touching-reliably policy (run8 into
run9), never discovering touching and precise lateral straddling
simultaneously from a random or under-trained one -- the warm-start
chain likely was not degrading positioning after all; it may instead
have been the only reason positioning ever worked in the first place.

**How to apply**: seven runs of pure RL exploration against this reward
design (runs 9 through 15) have now never produced a genuine sustained
hold -- only ever momentary, sub-centimeter grazes (see run12's
`held=True` audit above). Rather than continue iterating on reward
shaping or attempting a longer/differently-seeded from-scratch run,
pivoting to demonstration-seeded training as the next direction -- see
the entry immediately below.

---

**Decision**: Investigating demonstration-seeded training -- replaying
the project's 8 recorded real teleop episodes
(`sim/output/teleop_episodes/episode_000.json` through `episode_007.json`)
through the actual simulator to capture full (observation, action,
reward, next-observation) transitions, and inserting them into TD-MPC2's
replay buffer before training starts, rather than continuing to rely on
the agent's own exploration to ever discover a genuine grasp.

**Why**: seven consecutive RL runs (9 through 15) against the current,
fully-refined reward design have never produced a real sustained hold --
`between_jaws` positioning itself even regressed under a clean retrain
meant to improve it (see the entry above). TD-MPC2 learns its world
model (dynamics, reward, value) from whatever is in its replay buffer,
regardless of who generated it -- currently that buffer starts empty and
fills only with the agent's own, largely unsuccessful rollouts, so the
value function has never had a real example of "a sustained grasp
happened, and it was worth a lot" to learn from. Checked the actual
recorded demonstrations before committing to this direction rather than
assuming they would help: an earlier docs/reward_function.md note had
flagged that no validated episode reached a genuine lift-and-carry state
-- but that note only reflected the one or two episodes checked early in
the project. Re-checked all 8 directly: every one reaches a cube height
of 8.7cm to 14.9cm above the table (versus the ~2.75cm the best RL
rollout ever accidentally achieved), consistent with genuine deliberate
lifts, not touches or jostles. That earlier note is now corrected in
docs/reward_function.md.

**How to apply**: prototyping in progress -- extending
`validate_reward_function.py`'s existing replay machinery (already
replays these exact episodes through the real simulator and computes
real rewards) to also capture the camera observations
(`wrist_rgb`/`top_rgb`) it currently skips, and converting the
episodes' recorded raw joint-POSITION trajectories into the
normalized per-joint-DELTA action format `pickplace_env.py`'s action
space actually uses, before packaging the result into TD-MPC2's buffer
format. A full clean-retrain-style checkpoint of the project (this
documentation pass, plus confirming the git tree is clean and tagging
the current commit) was taken immediately before starting this work, at
the user's explicit request, given how different a direction this is
from every run so far.

---

**Decision**: While building the demo-to-buffer replay prototype
(`sim/scripts/replay_demo_to_buffer.py`), found and fixed three
independent bugs that had been silently preventing every recorded
teleop episode from registering as a real grasp: (1) episode
segmentation cut off almost the entire pre-lift approach/grasp phase,
(2) `is_between_jaws()`'s reach thresholds were calibrated around an
unmeasured guess at roughly half the gripper's true physical reach, and
(3) `jaw_approach_axis_world()` pointed the wrong physical direction
(pivot toward the wrist, not pivot toward the fingertips).

**Why / evidence for each**:
- *Segmentation*: `segment_teleop_episodes.py` only starts an episode
  when cube height first crosses `lift_threshold` (0.05m), with just 30
  raw (0.3s) frames of pre-padding -- but the cube sits completely
  motionless throughout the entire approach+grasp phase, so that padding
  never reaches it. Checked all 8 original episodes directly: every
  single one starts with the gripper already closed (recorded gripper
  joint value below `gripper_closed_threshold` at frame 0). Fixed by
  re-running segmentation with `--pad-steps 400` against the still-extant
  raw `teleop_recording.json` into a new `sim/output/teleop_episodes_v2/`
  (original directory left untouched) -- 7 of 8 episodes now genuinely
  start with the gripper open. Verified via replay: episode_002's max
  cube height went from 0.0229m (broken) to 0.0872m, matching the
  original recording's own 0.0873m almost exactly.
- *Thresholds*: even after fixing segmentation, a confirmed genuine
  rigid hold (constant ~0.0896m gripper-to-cube distance while both
  moved together) still failed every detection check. `grasp_reach_max`
  (0.05), `grasp_proximity_threshold` (0.05), and `touch_threshold`
  (0.08) were all originally set from an unmeasured guess. Verified the
  true reach two independent ways: parsing `moving_jaw_so101_v1.stl`'s
  binary mesh data directly gives ~8.2cm jaw extent from the pivot;the
  live replay's own measurement during the hold was ~8.96cm. Corrected
  (see `sim/envs/pickplace_reward.py` commit `be4e466`): grasp_reach_max
  0.05->0.09, grasp_proximity_threshold 0.05->0.10, touch_threshold
  0.08->0.14, align_activation_range 0.10->0.18 (cascading, to preserve
  its documented looser-than-touch_threshold ordering). grasp_reach_min
  left unchanged -- that evidence was specifically about the far/positive
  direction.
- *Axis direction*: still failed even with corrected thresholds --
  `is_between_jaws()`'s `axial` component came out consistently NEGATIVE
  (~-0.087) throughout the confirmed hold, meaning the cube registered as
  behind the pivot. `JAW_AXIS_LOCAL` had been defined as the normalized
  `JAW_OFFSET_LOCAL` itself, on the assumption that vector (pivot's
  position offset from `gripper_frame_link`'s origin) already pointed in
  the fingers' reach direction -- backwards, since that offset actually
  points from the pivot back toward the wrist. Independently confirmed by
  computing the TRUE fingertip position each traced step (moving jaw
  link's live world pose plus a mesh-measured local offset) and finding
  it measurably CLOSER to the held cube than the pivot was (~0.055-0.059m
  vs ~0.088m) -- exactly what a positive axial reading is supposed to
  mean. Fixed by negating `JAW_AXIS_LOCAL` relative to `JAW_OFFSET_LOCAL`
  (`sim/robots/grasp_geometry.py` commit `0920adc`); `JAW_OFFSET_LOCAL`
  and `grasp_point_world()` themselves were untouched, since the pivot
  position was independently confirmed correct.

**Result**: replaying all 8 re-segmented episodes through the real
production reward function, 5 of 8 (episodes 000, 002, 003, 005, 006)
now correctly register `ever_holding=True` for a genuine sustained
grasp -- the first time this project has ever detected a real hold, from
any source, RL or replay. The other 3 fail for separate, already-
understood reasons unrelated to this fix: episode_004 still starts with
the gripper closed even at 400 pad-steps (needs more padding or a
different segmentation signal); episodes 001 and 007 both diverge from
their own recordings under Option 1's joint-limit clipping (their
replays never reach the recording's own peak cube height), suggesting
clipping the recorded target rather than the robot's actual velocity is
losing something for these two specifically.

Also hit and worked around an unrelated Kit/Omniverse engine issue while
running these replays: `DirectRLEnv.reset()`'s texture-streaming wait
loop occasionally spins forever (confirmed via `py-spy` against the live
process and `/proc/<pid>/io` showing zero ongoing disk reads -- nothing
is actually still loading, an internal Kit busy-flag is just stuck).
Worked around with `wait_for_textures=False`, scoped to this replay
script only (it never reads a camera image, only physics state) --
production training keeps the framework default.

**How to apply**: this closes out the fidelity/detection half of the
demo-seeding investigation -- the replay pipeline now produces genuine,
buffer-compatible transitions where `ever_holding` correctly fires for
real grasps. Remaining before this becomes usable for actual training:
investigate episode_004 further (or accept 5/8 usable episodes as
enough to seed with); decide whether to also revisit the 6-of-8
wrist_flex joint-limit-exceedance finding now that its Option 1 clipping
workaround looks like the likely cause of episodes 001/007's divergence;
and wire the validated replay logic into `train_tdmpc2_pickplace.py`
itself (currently only a standalone diagnostic script) so seeded
transitions actually populate the buffer before online training starts.

---

**Decision**: Wired the validated replay logic into `train_tdmpc2_pickplace.py`
itself as `--seed-demos <dir>` (commit `c1cc5e8`), then ran a sequence of
four real training runs (16 through 19) iterating on what it actually
takes for demo-seeding to produce a genuine sustained hold, rather than
stopping once the pipeline merely ran without crashing.

**run16** (40k steps, seed-demos, cube position fully randomized):
`ever_holding` never fired at any of 7 eval checkpoints, `between_jaws`
never fired either -- seeding alone, with a randomized target, was not
enough.

**run17** (40k steps, seed-demos, cube FIXED to episode_002's own
recorded position 0.113/0.240, plus periodic re-injection of the seed
episodes -- see below): `between_jaws=True` fired at 2 of 7 checkpoints
(25449, 30439) -- the first time ANY training run, RL or seeded, had
ever gotten the gripper correctly positioned. Investigated via torchrl's
own `SliceSampler` source (not assumed): it picks trajectories UNIFORMLY
BY EPISODE COUNT, not weighted by length, so a one-time seeding batch
becomes a shrinking fraction of what gets sampled as real episodes
accumulate (5/85 ≈ 5.9% by run16's end). `--seed-demos-min-fraction`
(default 0.15) periodically re-adds the already-replayed seed episodes
(free -- no new env stepping) to hold their share roughly constant
instead of fading. A per-step `--eval-only` CSV trace of run17's own
checkpoint then diagnosed WHY `holding` still never fired despite
`between_jaws` working: `gripper_joint_pos` was genuinely, steadily
trending toward closed over ~15-20 steps, but the arm's own lateral
position was independently drifting in and out of the `between_jaws`
window on a similar timescale (a real 5-consecutive-step miss streak
observed) -- a timing race the close was losing, not a policy that
never attempted to close at all.

**Reward tuning in response** (commit `2b00af4`): `between_jaws_grace_steps`
2 -> 5 (the existing tolerance mechanism for `grasp_close_weight`'s gate
was sized from run12-era evidence of 1-2 step flickers, not the 5-step
drift actually observed here) and `grasp_close_weight` 2.5 -> 4.0 (this
potential-based term's total payout is the same regardless of how many
steps closing takes, so raising the weight doesn't directly reward
speed, but does make closing more valuable relative to other shaping and,
through TD-MPC2's own discounting, makes finishing sooner comparatively
more attractive).

**run18** (120k steps, same reward tuning, NO `--min-std`): a clear
regression -- `between_jaws` never fired across all 24 checkpoints, and
`touched` became sporadic in the run's second half after being
consistent in the first. Video review (contact-sheet frame grids, not
just the logged booleans) showed the arm reaching toward the cube then
retreating into a small, fixed, tucked pose and simply freezing there
for the rest of the episode -- the exact signature of TD-MPC2's CEM
planner over-confidently collapsing its own exploration std toward a
falsely-precise idle action, a mechanism this project already diagnosed
and fixed once before (runs 4/5, `--min-std`), but had left out of every
seed-demos run so far specifically to isolate that variable. Also
discovered mid-run18: the buffer's automatically-inflated capacity
(122,458, to fit seed transitions + steps) pushed storage from GPU to
CPU memory (a `2.5*bytes_required < free_gpu_memory` heuristic in
`common/buffer.py`), making this run take 7.1 hours instead of the
estimated 4.5 -- purely a wall-clock/throughput effect, not a
correctness issue.

**Separately, mid-investigation: a genuine machine-level networking bug
was found and fixed on the Ubuntu box.** A `--seed-demos`+`--min-std`
smoke test failed with `FileNotFoundError` fetching IsaacLab's default
ground-plane asset (`GroundPlaneCfg`'s stock fallback `usd_path`, a
`https://omniverse-content-production.s3-us-west-2.amazonaws.com/...`
URL -- NVIDIA's own Nucleus/Omniverse cloud content server, not
anything this project's code requests). Root-caused via `curl -v`
(`Immediate connect fail ... Network is unreachable`) and `ip route
show` (returned empty for IPv4): the box's WiFi interface
(`wlx001325ae639d`) had NO IPv4 default gateway route at all, only an
IPv6 one -- explaining why `google.com` worked (IPv6 fallback) while the
IPv4-only S3 endpoint didn't. Confirmed `nmcli device show` recorded
`IP4.GATEWAY: --` (never received/applied). Fixed by manually adding the
missing routes (`ip route add 10.0.0.0/24 dev wlx001325ae639d scope
link`, then `ip route add default via 10.0.0.1 dev wlx001325ae639d`) --
a pure addition, not touching the live WiFi association at all, chosen
specifically to avoid any risk to the existing SSH/Tailscale session
(confirmed intact throughout). This fix is runtime-only (`ip route add`,
not persisted to NetworkManager config) -- will need reapplying if this
machine reboots or the WiFi interface reconnects.

**run19** (55k steps, ~2.8 hours, min-std restored to 0.5, same reward
tuning as run18, same fixed cube position + re-injection as run17): by
far the best result this investigation has produced. `touched=True` at
all 10/10 eval checkpoints (no gaps, no sign of run18's freeze pattern).
`between_jaws=True` at 6 of 10 checkpoints (15469, 20459, 30439, 40419,
45409, 50399) -- both more frequent and earlier-appearing than run17's
2/7. Video review (contact-sheet grids of steps 30439 and 50399)
confirmed this wasn't just a logged-boolean artifact: the arm stays
visibly, continuously engaged with the cube for the large majority of
each episode (roughly 80% of the 50399 episode specifically), a
qualitatively different, much more sustained pattern than either
run17's brief hover-then-retreat or run18's reach-then-freeze.
`ever_holding` still never fired.

**How to apply**: this is the strongest checkpoint this project has
reached -- tagged `run19-close-timing` (see below) specifically so it
can be returned to regardless of what the next experiment does. The
remaining, now well-characterized gap is purely about CLOSING SPEED:
positioning is frequent and sustained, but the gripper still doesn't
close decisively enough within the window it has. `grasp_close_weight`
and `between_jaws_grace_steps` were the first, lower-risk, INDIRECT
levers tried (make closing more valuable / more tolerant of brief
misses) -- a more direct mechanical or reward-shaping fix aimed
specifically at closing speed (not just closing eventually) is the
natural next step; see the entry immediately below for that
investigation.

---

**Decision**: Added `close_speed_bonus` (commit `bf8107e`) -- a one-time
reward, riding `grasp_bonus`'s exact anti-farming gate, crediting how
fast the gripper's joint was moving in the closing direction at the
instant a hold is first established (capped via
`close_speed_bonus_max_vel`). Full derivation and the two rejected
alternative designs (a continuous/per-step velocity reward, both a
farm-proof symmetric version -- mathematically redundant with the
existing `grasp_close_weight` term -- and an asymmetric one -- reopens
oscillation-farming) are in that commit's own message and the config's
docstring; not repeated here.

**Result -- run20** (warm-started from run19's checkpoint,
`--seed-episodes 2`, otherwise identical settings): the first genuine,
detected hold this project has ever produced, at the very first eval
checkpoint (step 5489). Across all 10 checkpoints: `touched=True` 10/10,
`between_jaws=True` 8/10 (up from run19's 6/10), `held=True` 1/10.
Positioning consistency kept improving; actual holding remained rare.

**New diagnostic finding**: a fresh per-step `--eval-only` trace
(reusing the same tooling that diagnosed run17) on a `between_jaws=True,
held=False` episode from run20's final checkpoint showed the gripper
closing genuinely fast now (~0.045 rad/step, ~3x run17's rate) -- but
stopping at roughly 42% of the way to `gripper_closed_threshold` and
then REVERSING, opening back up, right as `lateral` drifted past its
threshold. `gripper_cube_dist` stayed smooth throughout (no spike),
ruling out an actual physical knock-away -- this is the POLICY deciding
to abandon the attempt, not a collision. Best-supported explanation:
the value function has had very few genuine "finishing pays off"
examples to learn from -- only the 5 demo episodes reliably show a
complete close, and evaluation episodes (where recent successes have
shown up, including this run's own step-5489 hold) are never added back
into the training buffer, so a lucky eval success doesn't directly
teach the model anything.

**Response -- run21** (warm-started from run20, 75k steps,
`--seed-demos-min-fraction` raised 0.15 -> 0.20 to give the
demonstrations' complete closures more weight, otherwise identical):
a genuinely mixed result. Overall `between_jaws=True` only 4 of 14
checkpoints (~29%) -- worse than run20's 80% -- and `held=True` never
fired at all this run. But the LAST 3 of 4 checkpoints all showed
`between_jaws=True` with steadily climbing reward (-7.08 -> +2.57 ->
+7.65), the two best total-reward episodes this project has produced,
right as the run ended. Can't yet cleanly attribute this to the demo-
weight change specifically versus just more steps/variance -- the
middle of the run was worse than run20 on positioning, for reasons not
yet understood.

**How to apply**: whether to keep warm-starting on the assumption this
upward trend continues, or intervene more directly (e.g. something that
specifically discourages abandoning a close once started), is an open
decision -- see the entry immediately below for the diagnostic run
against run21's own final checkpoint used to inform it.

---

**Decision**: Diagnosed run21's own final checkpoint (`agent_step_075001_final.pt`)
directly via `--eval-only`, rather than assuming its own training-log
numbers were representative. Result: 4 of 4 fresh draws failed to even
reach `touched=True` -- a real, consistent regression, not noise (4
independent stochastic samples all missing is far more than chance).
The SAME run's earlier checkpoint at step 70359 (`agent_step_070359.pt`)
immediately produced `touched=True, between_jaws=True`, reward +11.06 --
even better than that checkpoint's own training-log figure. **Concrete
implication for any future warm start from run21: resume from
`agent_step_070359.pt`, not the final checkpoint** -- training is not
monotonic, and the very last snapshot saved is not necessarily the best
one a run produced.

Analyzed the full per-step CSV from that +11.06 episode in detail and
found the reward number itself was misleading: the gripper cycled
through the same approach-dip-retreat pattern roughly 15 separate times
across the 500-step episode, each dip bottoming out in a narrow band
(joint potential corresponding to roughly 35-45% closed, never
approaching `gripper_closed_threshold`) before reversing -- and
`cube_height` never once rose above resting height anywhere in the
entire episode. The high total reward was earned entirely by
re-collecting ordinary reach/align/close shaping on each fresh approach
leg, not by getting closer to an actual grasp. User independently
confirmed this exact pattern on video before any fix was attempted.

Root cause: `grasp_close_weight`'s shaping has no memory -- redoing an
identical shallow dip pays exactly what it paid the first time, so nothing
makes genuinely exceeding a prior attempt's depth worth more than safely
repeating it. A secondary, mechanical contributor considered but not
acted on: CEM's 3-step planning horizon can't directly "see" a 15-20+
step full commitment, so it must trust the value function's own
(still-uncertain) longer-horizon estimate, making the short, verified-safe
dip-and-retreat reward easier to have confidence in than a longer,
riskier commitment.

**Fix -- `deepest_close_weight`** (commit `27e7c1e`): a new potential-based
shaping term, measured against a MONOTONIC high-water mark (the deepest
point reached so far THIS episode) rather than just the previous step --
repeating an already-reached depth earns nothing further, only a
genuinely new record does. Summed over a whole episode this telescopes
to exactly `weight * (deepest point ever reached - fully open)`,
regardless of how many shallower repeats or retreats happen in between --
verified directly in the self-test (not just argued), and set equal to
`grasp_close_weight` (4.0) so genuinely new depth is worth double the
base closing rate. Full design reasoning, including why a plain
continuous velocity-style reward was rejected again here for the same
farming-risk reasons as `close_speed_bonus`, is in the commit message
and the config's own docstring.

**How to apply**: next test should warm-start from
`run21_more_demo_weight/agent_step_070359.pt` specifically (see above,
not the final checkpoint), with `deepest_close_weight` now active.
Launched as run22 (48,000 steps), confirmed healthy, left running.

## MimicGen-lite for the diffusion-policy side (2026-09-11)

With no current teleop access, started the diffusion-policy side of the
project (docs/diffusion_policy_data_strategy.md's already-decided plan)
rather than sit idle on the world-model side. Chose the SE(3)
transform-and-replay mechanism as the first thing to validate, over
collecting more raw teleop or scaling scripted IK, since it's the
mechanism most likely to hide a hard blocker (if the retargeting math or
the closed-loop IK replay can't actually reproduce a valid grasp at a
new cube position, the whole MimicGen-lite plan needs rethinking before
any further demo-collection effort is worth spending).

**Framework decision**: Isaac Lab ships its own MimicGen port,
`isaaclab_mimic`, but its full data-generation framework
(`isaaclab_mimic/datagen/data_generator.py`) requires environments
authored against `ManagerBasedRLMimicEnv` -- a different env-authoring
paradigm than this project's `PickPlaceEnv(DirectRLEnv)`. Porting the
env to that paradigm just to use the framework would be substantially
more work than the retargeting logic itself. Decision: reuse only the
framework's underlying, dependency-free math utilities directly
(`isaaclab.utils.math.make_pose`/`unmake_pose`/`pose_inv`/
`pose_in_A_to_pose_in_B` -- plain 4x4 tensor operations with no env
coupling) in a custom script, not adopt the framework. Confirmed via
reading `data_generator.py` and `isaaclab/utils/math.py` directly.

**IK decision**: driving the retargeted Cartesian end-effector poses
back into joint-space commands needs inverse kinematics. Rather than
hand-deriving SO-101-specific IK, `isaaclab.controllers.
DifferentialIKController` is a generic Jacobian pseudo-inverse
controller (`dq = J+dx`, configured via `DifferentialIKControllerCfg
(command_type="pose", use_relative_mode=False, ik_method="dls")`) that
works for any articulated robot Isaac Lab can simulate, SO-101 included
-- confirmed via `IsaacLab/scripts/tutorials/05_controllers/
run_diff_ik.py`, which gives the exact wiring pattern (Jacobian via
`robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, joint_ids]`
with a `-1` body-index offset for a fixed-base robot; ee pose converted
to the robot's root frame via `subtract_frame_transforms()`; desired
joint positions from `diff_ik_controller.compute(...)`; applied via
`robot.set_joint_position_target(...)`).

**First implemented piece -- `extract_grasp_segment.py`** (new script):
adapted from `replay_demo_to_buffer.py`'s proven state-override replay
pattern (override cube pose, robot joint state, `_joint_pos_target`
directly, bypass the velocity clamp via a zero action). Replays a
source episode and records, at every step, the gripper's own live
world pose (`gripper_frame_link`'s `body_pos_w`/`body_quat_w`, corrected
by `env_origins` -- the same live-simulator-as-ground-truth reasoning
used throughout this project's grasp-geometry work, not an analytically
computed FK) plus the recorded gripper joint value. Uses the reward
function's own `info["holding"]` -- the same, now axis-bug-fixed
function this project's whole demo-seeding investigation already
validated -- to find the first step a genuine hold is established: the
natural reach-and-grasp / transport-and-place segmentation boundary
MimicGen's own methodology calls for. Deliberately does no retargeting
or IK yet -- built incrementally, one validated piece at a time, matching
how every other piece of this project has been built.

Two launch-flag issues surfaced getting this running on the Ubuntu box,
neither a logic bug: (1) the SSH command needed
`source ~/miniforge3/etc/profile.d/conda.sh && conda activate
env_isaaclab` first -- confirmed by checking the actual interpreter path
of the already-running run22 training process
(`/home/keerthan/miniforge3/envs/env_isaaclab/bin/python`) rather than
guessing; (2) the script's `use_cameras=True` env config (kept to match
`replay_demo_to_buffer.py`'s own buffer-format-compatible observation
shape) requires the `--enable_cameras` CLI flag or Isaac Lab's camera
sensor init raises immediately -- added to match
`replay_demo_to_buffer.py`'s own documented usage.

**Result**: ran against `episode_002` (one of the 5/8 episodes already
confirmed `ever_holding=True`; also the episode run17 fixed the cube
position to, so its `source_cube_pos` readback -- (0.113, 0.240, 0.015)
-- served as a consistency check against that prior finding). Grasp
boundary found at downsampled step 194 of 545. Holding sustained for
222 of the 306 steps traced after the boundary -- confirms this is a
genuine, non-flickering grasp, not a one-step flicker that would poison
every episode retargeted from it. Extracted a 195-step reach-and-grasp
segment (written to `sim/output/segment_002.json`), spanning only
~2.5cm of end-effector travel (this recording's segmentation already
cut close to the final approach) with the gripper joint closing from
0.588 to 0.156 over the segment.

**How to apply**: next step is validating the SE(3) retargeting math
itself (`pose_in_A_to_pose_in_B`) against a new target cube position
using this segment, then closed-loop `DifferentialIKController` replay,
then splicing into the source episode's own unchanged
transport-and-place tail, then confirming `is_holding()` still fires at
the new position -- the same empirical validation standard used
throughout this project, not "looks plausible."

## run22's regression, root-caused (2026-09-12)

run22 (48,000 steps, warm-started from `run21_more_demo_weight/
agent_step_070359.pt`, `deepest_close_weight` newly active) finished
worse than its own starting point: `held=True` 0/9 eval checkpoints,
`between_jaws=True` only 1/9, ending on its two weakest checkpoints. Dug
into why via `--eval-only` per-step CSV traces at three points --
run21's step-70359 starting checkpoint, run22's step-20459 (its one
`between_jaws` hit), and run22's final checkpoint -- rather than
guessing from the aggregate booleans alone.

**What the traces actually show**, in `gripper_joint_pos` amplitude
(higher = more open) and how often `lateral` dips under the tight
`grasp_lateral_threshold` (0.02m) `is_between_jaws()` needs:

| Checkpoint | joint_pos stdev (steps 100+) | mean abs step-delta | % steps lateral<0.02 | between_jaws hits |
|---|---|---|---|---|
| run21 step 70359 (pre-deepest-close) | full-range swings (0.74-1.74) | large | 30.6% (153/500) | 153/500 |
| run22 step 20459 (mid-run) | 0.102 | 0.033 | 4.5% | 42/500 |
| run22 final (048001) | 0.077 | 0.015 | 0% (min 0.0201 -- misses by 0.1mm, never crosses) | 0/500 |

A clean, monotonic decay across the run: the gripper's oscillation
amplitude and the frequency of dipping into the tight lateral window
both shrink together, in lockstep, the longer training continues. The
approach phase itself is unaffected (first 15 steps of the final
checkpoint's trace show a smooth, monotonic dist reduction 0.378m ->
0.197m, same as always) -- this is specifically a collapse in the
close-range "commit to it" behavior, not a broken policy overall.

**Root cause -- an unintended interaction between two independently-fine
mechanisms.** `deepest_close_weight` is correctly farm-proof against the
old exploit (verified again here: unlike `grasp_close_weight`, its
record persists across the `between_jaws_for_closing` gate turning off
and on, not just across steps -- so timing when the gate happens to be
open/closed can't manufacture free reward the way it could before). But
farm-proofing it this way has a side effect nothing in this project had
reason to anticipate until now: the reward bar it sets RISES every time
it's cleared, making genuine reward strictly sparser as an episode (and
a training run) goes on. Meanwhile `action_penalty_weight` (0.002 *
sum of squared joint velocities, all 6 joints, unconditional) has been
in this reward function since early on, quietly doing nothing much
because the OLD exploit's reward was large enough to swap for it many
times over. Once `deepest_close_weight` closes that exploit and makes
new reward genuinely hard to earn, the same small, constant
action-penalty cost of attempting (again) to close and correct position
starts to outweigh the shrinking expected payoff -- and gradient
descent, correctly following that signal, trains the policy toward
smaller, safer, lower-energy adjustments that increasingly fail to
reach the millimeter-tight lateral window needed to even get another
shot at the record.

This also retroactively explains HOW run21's oscillation loop earned
reward in the first place, beyond "no memory": `close_shaping` (like
`align_shaping`) is only computed while `between_jaws_for_closing` is
true -- a real potential-based term, but one that stops being evaluated
during the ungated part of the cycle. Ng/Harada/Russell's policy-
invariance guarantee for potential-based shaping (the theoretical basis
this whole reward function is built on -- see this file's earlier
entries) requires the potential to be evaluated every single step; a
term that's gated off for part of a trajectory no longer has that exact
telescoping guarantee, because the "opening" portion of a cycle that
happens to occur outside the gate is never charged, while the
"closing" portion that happens to occur back inside the gate always
is. That asymmetry, not just a lack of memory, is the concrete
mechanism that let run21 farm real (if misleading) reward from pure
oscillation. `deepest_close_weight` avoids this specific hole by
persisting the record independent of the gate -- correctly -- but doing
so is also what makes the reward landscape sparser over time, which is
the new problem.

**Proposed fix -- phase-gate `action_penalty_weight` down (not to zero)
during the pre-hold approach/grasp phase, full strength once holding**:
this is a penalty, not a reward, so reducing it can't reopen a farming
exploit the way adding a new positive term could (nothing to gain by
lingering in a low-penalty state -- the existing `premature_close_weight
= 0.0` precedent already established this same reasoning). It targets
the diagnosed mechanism directly (the fixed cost of attempting now
outweighs deepest_close_weight's shrinking expected payoff) without
touching deepest_close_weight itself, which is doing its job correctly.
Full-strength action_penalty during holding/transport is kept, since
smoothness while actually carrying the cube is still worth protecting.
Not yet implemented -- checking with the user on this direction (versus
alternatives considered and set aside: a decayed, still-nonzero
repeat-reward on `deepest_close_weight` itself, which would need new
farm-proofing work to avoid reopening exactly the gate-timing exploit
above; and reverting the demos/warm-start chain further back, which
doesn't address the mechanism found here at all) before making another
reward-function change and spending another multi-hour run on it.

**Implemented** (user confirmed): added `action_penalty_approach_scale:
float = 0.2` to `PickPlaceRewardConfig`, applied in `compute_reward()` as
`action_penalty_scale = 1.0 if holding else cfg.action_penalty_approach_scale`,
multiplying `action_penalty_weight` before it scales the summed squared
joint velocities. No new required parameter to `compute_reward()` --
confirmed via `grep -rn 'compute_reward(' sim/` that this touches no
other call site (`pickplace_env.py`, `validate_reward_function.py`)
beyond the config default they already inherit.

Verified, in order: (1) 4 new self-test cases (20a-20d) -- the reduced
scale applies while not holding, full (unscaled) weight applies once
holding, the same motion costs strictly less pre-hold than post-hold, and
the production default is a genuine partial reduction (0 < scale < 1,
not accidentally 0 or >=1) -- passing both locally (Mac) and on the
Ubuntu training box; (2) a full `--smoke-test` run of
`train_tdmpc2_pickplace.py` (301 env steps, 202 agent.update() calls) --
no crash, full pipeline runs end to end through the ACTUAL call site
(`pickplace_env.py`'s `_get_dones()`) this fix needs to work through, not
just the reward function in isolation. `validate_reward_function.py`
(a separate, lower-level verification path that builds its own
`InteractiveScene` directly rather than going through `PickPlaceEnv`) was
also attempted against `episode_002`, but hung for 13+ minutes in Isaac
Sim's own scene/asset-loading phase, before ever reaching the step loop
or calling `compute_reward()` even once -- confirmed via the log
(nothing beyond the `PhysxCfg` warning that prints during `sim.reset()`)
and process CPU/GPU profile (steady CPU use, 0% GPU, far below the other
scripts' memory footprint). This is upstream of anything this change
touches, so it can't be evidence about the fix either way; killed it
rather than wait further, since the smoke test already exercises the
real call site. Not otherwise investigated -- if this script is needed
again, its scene-loading path may itself be worth a fresh look (possibly
the same texture/asset-streaming stall `replay_demo_to_buffer.py` hit and
worked around, which this script never adopted since it doesn't build
its scene through `PickPlaceEnv` at all).

**How to apply**: ready for a real training run to test whether this
actually restores the oscillation/exploration behavior and recovers
`between_jaws`/`held` rates, warm-started from the same
`run21_more_demo_weight/agent_step_070359.pt` checkpoint run22 used, with
both `deepest_close_weight` and `action_penalty_approach_scale` active.

## Stepping back: why does every fix look like it only half-works? (2026-09-12)

run23 (the fix above) finished on schedule (~3h, confirmed from the
training script's own "Ran 48001 env steps ... in 10693.2s" summary line
-- an earlier alarm about a 7-hour runtime was a misread of the log
file's mtime, which kept advancing for hours after training actually
finished because of the already-known Kit shutdown-hang continuing to
emit harmless internal warnings; not a new problem). Eval results:
`between_jaws=True` 4/9 (vs run22's 1/9) opening with three CONSECUTIVE
hits (run22 never had two in a row) and closing on a positive-reward
`between_jaws=True` checkpoint rather than run22's two weakest -- a real,
different, better shape. But `held=True` still never fired, and a
5-checkpoint mid-run dip (steps 20459-40419) persisted.

Rather than treat this as "close but needs fix #4," stepped back to ask
why this project keeps landing here: implement a fix, get a partial/
ambiguous result, diagnose a new issue, implement another fix, repeat.
Two structural things this investigation had never actually checked,
despite bearing directly on every conclusion drawn since run17:

**1. TD-MPC2's planner is NOT deterministic even in eval mode, and the
env's own tolerances are tight enough for this to matter a lot.** Read
`tdmpc2.py`'s `_plan()` directly rather than assuming eval_mode makes
this a non-issue. `eval_mode` only skips ONE thing:
`a = a + std * torch.randn(...)`, the post-selection noise injection.
Everything upstream of that still runs identically in eval mode: MPPI
resamples `num_samples=512` candidate trajectories per planning call
across `iterations=6` refinement rounds, narrows toward `num_elites=64`
elites each round (never below `min_std=0.5`'s floor -- exactly the
same floor raised project-wide to stop CEM's own confidence from
collapsing prematurely), and then the ACTUALLY EXECUTED action is
`torch.index_select(elite_actions, 1, gumbel_softmax_sample(score))` --
a genuine categorical sample over the surviving elite set, weighted by
`exp(temperature=0.5 * (elite_value - max_value))` (not sharply peaked
at this temperature -- multiple elites retain real weight). So
`min_std`'s floor, which exists specifically to keep the elite set from
collapsing to a single point, also means the elite set retains genuine
diversity every single eval planning call -- and every step's executed
action is a real stochastic draw from that diversity, EVEN IN EVAL
MODE. This isn't a hypothesis; it's what the code does. Every single
`between_jaws`/`held`/reward number this whole project has ever reported
for a checkpoint has been exactly ONE such draw.

**Confirmed empirically, not just from the code**: re-ran
`run21_more_demo_weight/agent_step_070359.pt` (the shared warm-start
ancestor of both run22 and run23) multiple times with nothing at all
changed -- same checkpoint, same cube position, same everything. Reward
across draws: +12.51 (the original single draw this whole
investigation's plan was built on), +20.17, +22.84, +18.61, .... a >80%
relative spread between the highest and lowest, on the IDENTICAL
network weights. `between_jaws=True` on every draw so far -- but the
reward magnitude swinging this much on a fixed policy means the
between_jaws/held BOOLEANS for less reliable checkpoints could easily
be flipping between True and False from draw to draw too, not from any
real change in policy quality.

**Added `--eval-only-repeats N`** to `train_tdmpc2_pickplace.py` (loops
`run_eval_episode()` N times in one Kit boot, reusing the same env --
`env.reset()` at the top of each call already clears every persisted
per-episode state, so successive draws are genuinely independent;
confirmed via reading `run_eval_episode()` itself, not assumed) so this
is actually checkable going forward instead of continuing to judge
single draws. Prints a per-draw line plus an aggregate hit-rate summary;
video/CSV filenames get a `_drawN` suffix when N>1, byte-identical
unsuffixed behavior at the N=1 default so nothing existing changes.

**2. Every warm-start hop AFTER the first has underperformed its own
immediate predecessor; the first one didn't.** Laid out side by side
from docs/progress.md's own numbers:

| Hop | between_jaws rate | held rate | vs predecessor |
|---|---|---|---|
| run19 (baseline) | 6/10 | 0/10 | -- |
| run19 -> run20 | 8/10 | 1/10 | improved |
| run20 -> run21 | 4/14 (~29%) | 0/14 | regressed mid-run, recovered late |
| run21(070359) -> run22 | 1/9 | 0/9 | regressed, no recovery |
| run21(070359) -> run23 | 4/9 | 0/9 | partial recovery vs run22, still below run21's own start |

Only the very FIRST warm start (19->20, introducing close_speed_bonus)
came out ahead of its starting point. Every hop since has cost
something relative to where it started, regardless of which specific
reward term changed alongside it (`--seed-demos-min-fraction` for 21,
`deepest_close_weight` for 22, `action_penalty_approach_scale` for 23).
That pattern -- consistent across three different reward changes -- is a
real signal that the REWARD TERM being tuned each time may not be the
actual variable driving the outcome. The common thread across every one
of these hops instead: `--resume-from` restores network weights only,
never the replay buffer (`TDMPC2.save()` never persisted it, a design
decision from run9, unchanged since) -- every single one of these runs
restarts training with a value function that was calibrated against a
large, diverse, converged buffer, now bootstrapping against a buffer
containing only 5 demo episodes plus whatever this run collects fresh.
This is the same "offline-to-online RL" destabilization documented in
the literature (e.g. Nakamoto et al., "Cal-QL", NeurIPS 2023) -- a
value function fine-tuned online against a buffer that doesn't match
the distribution it converged on offline tends to take an initial
quality hit before (if ever) recovering, and the more times this
transition is repeated in a chain, the more chances for it to compound
or fail to fully recover.

**What this changes about the plan going forward**: stop treating each
run's eval numbers as ground truth from a single draw, and stop
adding a new reward term per run without first checking whether the
LAST one's outcome was even real signal. Concretely:

1. Finish the 3-checkpoint x 6-draw variance study already running
   (`run21_070359`, `run22_final`, `run23_045409`) to get real,
   multi-draw hit rates for the two "before/after this fix" checkpoints,
   before drawing any conclusion about whether
   `action_penalty_approach_scale` helped.
2. If the warm-start-chain-degrades-regardless-of-reward-term pattern
   holds up under multi-draw scrutiny, the next experiment should target
   the ACTUAL shared mechanism -- e.g. a real comparison of continuing
   training UNINTERRUPTED for longer from a single verified-good
   checkpoint (no further warm-start hop at all) versus another
   warm-start hop -- rather than a fourth reward-shaping term.
3. Separately worth a controlled check with the new repeats tooling:
   whether a LOWER `min_std` at eval/inference time (not necessarily
   during training) changes the between_jaws/held hit rate on an
   otherwise-identical checkpoint -- directly testing whether the
   planner's own noise floor, not the reward function, is what's
   capping precision on the tight `grasp_lateral_threshold` window.

## The variance study results, and what they actually show (2026-09-12)

All three planned checks finished. Full results, 6 draws each unless
noted:

| Checkpoint | min_std | touched | between_jaws | held | mean reward |
|---|---|---|---|---|---|
| run21 step70359 (ancestor, shared by run22 & run23) | 0.5 (default) | 1.00 | **1.00** | 0.00 | +14.11 |
| run21 step70359 (same checkpoint, noise only) | 0.2 | 1.00 | **1.00** | 0.00 | +22.31 |
| run22 final (deepest_close_weight only) | 0.5 | 1.00 | **0.00** | 0.00 | -1.11 |
| run23 step45409 (+ action_penalty_approach_scale) | 0.5 | 0.83 | **0.67** | 0.00 | +1.08 |

Four real conclusions follow directly from this table, not from a single
draw each:

**1. This wasn't noise -- the diagnosed regression and the fix were both
real.** The ancestor checkpoint genuinely, reliably gets between the
jaws (6/6, then another 6/6 at a different min_std -- 12/12). run22's
collapse to 0/6 is a real, fully reproducible failure, not an unlucky
sample -- matching the per-step CSV/video evidence already gathered.
`action_penalty_approach_scale` genuinely, substantially recovered this
-- 0.67 vs 0.00, a real and large effect, not noise. The single training-
time eval draws that made this fix look "ambiguous" (4/9 checkpoints
during the run, ending on a good one) understated how much it actually
helped -- a proper multi-draw check of its best checkpoint shows a much
clearer win than the single-draw training log suggested. **Keep this
fix. It worked.**

**2. It's still not fully recovered** -- 0.67 vs the ancestor's 1.00 is a
real, remaining gap, not just measurement noise (95% CI on a 4/6 binomial
sample is wide, but 0.67 is a full 6-draw run below a checkpoint that
scored 12/12 across two separate 6-draw runs at two different min_std
settings -- a meaningfully different regime, not the same number
restated).

**3. `min_std` is NOT the lever holding back `between_jaws` or `held` --
ruling out that hypothesis directly, not just deprioritizing it.**
Lowering it from 0.5 to 0.2 on the identical, already-good ancestor
checkpoint left `between_jaws` at 1.00 (no room to improve -- already
maxed) and, more importantly, left `held` at 0.00 too. If planner noise
were what was preventing a hold, cutting it by more than half should
have shown SOME movement on `held` for a checkpoint that already nails
positioning every time. It didn't move at all. (It did raise mean reward
noticeably, +22.31 vs +14.11 -- plausibly less wasted energy/action
penalty from reduced search noise -- a real but secondary effect, not
chased further here.)

**4. The actual headline finding: `held=True` is 0 out of 18 draws,
across every checkpoint tested, including the best one available.**
This is the one that reframes the whole investigation. Every reward
change in this project's recent history --
`close_speed_bonus` -> `deepest_close_weight` -> `action_penalty_approach_scale`
-- has been designed and judged against single training-time eval draws
that occasionally showed `held=True` (run20: 1/10 checkpoints, one time,
ever, in this project's entire history). Under an honest, repeated,
multi-draw test, NOT ONE of the three checkpoints examined here --
including the very checkpoint every one of these runs warm-started
from, the strongest one this whole lineage has -- ever produced a held
episode. That single run20 result has never been reproduced by any
descendant checkpoint under rigorous testing. The reward-shaping chain
has been iterating on symptoms of "doesn't commit to closing decisively
enough" on top of a foundation that has not been shown, even once under
honest measurement, to reliably finish a grasp.

**What this means for "why does every fix only half-work"**: it's not
that each fix was wrong -- `action_penalty_approach_scale` demonstrably
was a real, correct, verified improvement (finding 1). It's that the
target being aimed at (recover `between_jaws`/`held` rates) was always
being measured through a single noisy draw, so "did this help" was
answerable only approximately, and the actual hardest part of the task
(reliably finishing a hold, not just reaching good position) has never
had a genuine success to build FROM in this specific lineage -- run20's
one recorded hold was likely itself a low-probability event under the
same unseeded-planner stochasticity documented above, not a skill this
family of checkpoints ever reliably possessed.

**Recommendation going forward** (a change in approach, not a fourth
reward term):

1. Keep `action_penalty_approach_scale` -- verified working, don't
   revert or second-guess it further.
2. Adopt `--eval-only-repeats` (>=6) as the standard for judging any
   future checkpoint or change -- a single draw is no longer treated as
   evidence of anything, in either direction.
3. Stop adding new reward-shaping terms reactively to the most recent
   symptom. The actual bottleneck now is getting this lineage to
   reliably finish a grasp even once under honest testing -- more
   training TIME on the current best checkpoint (uninterrupted, no new
   warm-start hop, no new reward term) is the next thing to actually try
   and then MEASURE properly with the new tooling, before concluding
   another reward change is needed at all.
4. If a longer uninterrupted run still shows `held_rate=0` under
   multi-draw testing, that's the point to reconsider something
   structural (buffer persistence across warm starts; concentrating more
   seed-demo weight specifically on the hold-and-carry portion of the
   5 available demo episodes) -- not before, since neither has been ruled
   in or out yet by real evidence.

## run24: the "just train longer" experiment, tested properly (2026-09-12)

Ran the recommended experiment: `run24_extended_training`, 96,000 steps
(double run22/23's budget), resumed from `run23_action_penalty_fix/
agent_step_045409.pt` -- the checkpoint actually VERIFIED at 67%
`between_jaws` in the variance study, not just the last one saved (the
same run21 lesson applied deliberately this time). No new reward term,
no other setting changed -- the only variable was more uninterrupted
time. Finished in 21469s (~5h58m, matching the ~6h estimate). 19
single-draw training-time evals, 4 scattered `between_jaws=True` hits,
no `held=True` -- deliberately NOT interpreted from single draws this
time (per the new standard).

Ran `--eval-only-repeats 6` on the two checkpoints that mattered: the
best-looking single-draw checkpoint (step 65369, +2.87 reward,
between_jaws=True) and the final checkpoint (096001, what would
otherwise get used by default). Results:

| Checkpoint | touched | between_jaws | held | mean reward |
|---|---|---|---|---|
| run24 step65369 (best-looking, 48k steps into the extension) | 1.00 | **0.67** | 0.00 | +1.37 |
| run24 final (096001, end of the full 96k-step run) | **0.00** | **0.00** | 0.00 | -0.66 |

**Two real conclusions, not single-draw noise:**

1. **"Just train longer" plateaued rather than improved.** 0.67 at step
   65369 is statistically indistinguishable from run23's own already-
   verified 0.67 -- 48,000 additional steps of uninterrupted training,
   with nothing else changed, produced no further recovery toward the
   ancestor checkpoint's 1.00. The hypothesis this run was built to test
   did not pan out: more time alone, on this same lineage, isn't closing
   the remaining gap.

2. **The final checkpoint regressed hard -- confirmed a second time,
   not a fluke of run21.** 0/6 touched, let alone between_jaws -- a
   complete failure, worse than even run22's collapsed final checkpoint
   (which still touched 6/6). This is now the SECOND independent
   instance (after run21) of a run's own final saved checkpoint being
   meaningfully worse than an earlier one from the same run -- strong,
   now-replicated confirmation that this isn't a one-off: whatever a
   training run's last checkpoint happens to look like should never be
   assumed to be its best, on this codebase, full stop.

**Running total across every checkpoint this investigation has now
verified with multi-draw testing: `held=True` in 0 of 36 independent
draws**, spanning 4 distinct checkpoints across 3 different training
runs (run21's ancestor at two `min_std` settings, run22, run23, run24 --
twice). This is no longer a small-sample curiosity -- it's a solid null
result. Continuing to add more training time to this exact lineage,
unmodified, is not the way past it.

**How to apply**: per the plan this was set up to test, this is the
point to try something structurally different rather than a 5th
variation on "warm-start + train some more":
- Persist and reload the replay buffer across warm starts (currently
  `TDMPC2.save()`/`--resume-from` never does -- weights only, a design
  decision from run9, unchanged since) -- directly addresses the
  offline-to-online destabilization mechanism already documented above,
  rather than hoping more steps outruns it.
- Re-weight the 5 seed demo episodes' OWN sampling specifically toward
  their hold-and-carry portions (right now `--seed-demos-min-fraction`
  controls how often whole episodes get re-injected, not which PART of
  an episode gets sampled more) -- if the buffer is systematically
  under-representing the rare "finish the hold" transitions relative to
  the much more common "approach/reach" ones, that would directly
  explain a persistent 0/36 on `held` regardless of how long training
  runs.
- Not yet tried, and not recommended yet either -- worth deciding
  together which of these (or something else) to test next, rather than
  picking one unilaterally given how much this investigation has already
  shown that assumptions here need checking before committing more
  training hours.

## `--seed-demos-hold-segments`, designed and implemented (2026-09-12)

Chose demo re-weighting over buffer persistence. The deciding evidence:
the ancestor checkpoint (100% `between_jaws` across 12 draws, no warm-
start instability in play at all) still never held. Buffer persistence
targets warm-start destabilization specifically -- a real but separate
problem, since `between_jaws`/`touched` do partially self-correct with
more training (67% recovered in both run23 and run24) even without it.
`held` never recovers, in a checkpoint that was never destabilized in
the first place -- pointing at a persistent data/signal gap, not
transient instability, which is what this option targets.

**Corrected mechanism** (my first-pass framing, given to the user
verbally, was wrong and worth recording as a caught error rather than
memory-holed): initially reasoned the hold phase is under-represented
because it's a small FRACTION of each demo episode's own length. Checked
this against the real data before implementing anything and it doesn't
hold up -- `extract_grasp_segment.py` against all 5 seed episodes shows
the hold phase is actually LARGER than the approach phase by step count
(boundary at steps 194-200 of ~500-step episodes). The actual mechanism
is different: only 5 demo episodes exist, and `--seed-demos-min-fraction`
guarantees them at least a floor of the buffer's total EPISODE count --
not of how many TRANSITIONS are genuinely post-grasp. The other ~80%+ of
the buffer is self-generated RL experience, and since the policy rarely
succeeds (0/36 in the variance study), that 80% contributes close to
zero hold-phase transitions of its own. So hold-phase signal in the
whole buffer comes almost entirely from the 5 demos' own segments, a
small, fixed, non-growing source, regardless of how much self-generated
(non-holding) experience piles up around it. Confirmed the mechanism
this depends on by reading torchrl's `SliceSampler._sample_slices()`
directly: `get_traj_idx()` samples episode index uniformly via
`torch.randint(maxval, ...)` where `maxval` is the number of distinct
episodes -- confirms episode selection really is uniform by COUNT, not
weighted by length, so a SHORT, hold-concentrated pseudo-episode gets
exactly the same per-draw selection odds as a full-length one. That's
what makes this approach well-suited to this specific sampler.

**Design**: `--seed-demos-hold-segments` (new flag, requires
`--seed-demos`). For each seed episode, extracts the segment starting
`HOLD_SEGMENT_LOOKBACK_STEPS` (50 steps = 1 second at 50Hz) before its
own first `is_holding()=True` step through its end, and injects it as
an ADDITIONAL pseudo-episode alongside (not instead of) the whole
episode. The 50-step lookback isn't arbitrary -- checked against all 5
episodes' real boundaries (194-200, remarkably consistent) before
picking it, specifically so the injected segment captures the critical
transition-INTO-holding moment itself, not just its aftermath (a clean
cut exactly at the boundary would only teach "how to continue a hold
already established," missing the harder, rarer event of establishing
one). Hold segments share the exact same `--seed-demos-min-fraction`
floor and re-injection cadence as whole episodes -- deliberately not
given their own separate fraction, since there's no evidence yet for
what that should be; doubling the "seed group" (10 pseudo-episodes
instead of 5, at the current 5 real episodes) under the same target
floor already meaningfully shifts the mix toward hold-concentrated
content without adding an untuned new knob. If this isn't enough,
giving hold segments independent weight is the natural next lever --
not done pre-emptively.

**Implementation**: `replay_episode_to_tds()` now tracks
`first_holding_step` (the tds-index of the first `is_holding()=True`
step) as part of its existing single replay pass -- no second replay,
no extra Isaac Sim cost, just one more field recorded from data already
being read every step for `ever_holding`. A small, pure, dependency-free
`hold_segment_start(first_holding_step, lookback_steps)` does the
boundary arithmetic in isolation (clamped at 0, returns None if the
episode never held) -- kept separate from the actual tds slicing at the
call site specifically so the one thing worth getting wrong here stays
checkable without booting Isaac Sim, even though this file as a whole
can't be (`AppLauncher` boots at import time, so there's no formal
`_self_test()` here the way `pickplace_reward.py` has -- verified
against real data instead, see below). Hold segments feed into the SAME
`seed_tds_concat` list already used for periodic re-injection, so that
existing machinery needed no changes at all -- it just re-injects more,
now hold-concentrated, episodes.

**Verified, in order, before considering this done**:
1. Real-data extraction against all 5 actual seed episodes (`--steps 600`,
   full 500-step episode length so no artificial truncation before the
   boundary): every segment's start/end/length matches hand-computed
   expectations exactly (e.g. episode_000: boundary step 198, segment
   148-499, length 352 = lookback of 50 + 254 = 499-148+1, confirmed
   arithmetically for all 5). Buffer log confirms 10 total seed episodes
   (5 whole + 5 hold-segment), re-injection message correctly reflects
   the new count.
2. The `None`/never-held edge case, via `--smoke-test`'s 100-step env
   timeout (well before any real boundary at 194-200): all 5 episodes
   correctly fall back to the `[WARN] ... never reached is_holding()=True`
   path, "0 hold segment(s)" correctly reflected in the summary line, no
   crash.
3. Regression check: `--seed-demos` WITHOUT the new flag reproduces the
   exact same per-episode stats as before this change, byte-for-byte
   (same reward sums, same transition counts) -- confirms this is a
   pure addition, nothing about the existing mechanism changed.

All three passed cleanly on the first real attempt. Ready for a real
training run using this mechanism.

## run25: hold segments tested for real (2026-09-13)

72,000 steps, warm-started from the ancestor checkpoint (`run21_more_demo_weight/
agent_step_070359.pt`, the most stable available -- 100% `between_jaws`
across 12 prior draws), `--seed-demos-hold-segments` newly active,
otherwise identical to run22/23/24. Finished in 16021.6s (~4h27m).

Single-draw training-time evals aren't conclusions on their own, but
step 45409 stood out: +5.077 reward, the highest single-draw reward this
entire investigation has recorded. Verified it and the final checkpoint
properly with `--eval-only-repeats 6`:

| Checkpoint | touched | between_jaws | held | mean reward |
|---|---|---|---|---|
| run25 step45409 | **1.00** | **1.00** | 0.00 | +3.13 |
| run25 final (072001) | 0.00 | 0.00 | 0.00 | -0.68 |

**Step 45409 is the first descendant checkpoint since the ancestor
itself to fully match its 100% `between_jaws` rate** -- run22 collapsed
to 0%, run23 and run24's best both plateaued at 0.67. This is a real,
substantial recovery, not a single lucky draw (6/6). Whether
`--seed-demos-hold-segments` is specifically what did it (versus this
particular warm-start hop just landing better than the last three) isn't
provable from one run -- but it's the strongest positioning result any
descendant of the ancestor has produced, and worth taking as a genuine
positive signal for the approach rather than noise, given 6/6 is not an
ambiguous number.

`held` is still 0/6 on even this best checkpoint -- the core goal this
change targeted isn't solved yet. The final checkpoint (072001) is
ANOTHER instance of the by-now-familiar "last checkpoint saved is not
the best" pattern -- the THIRD confirmed case (after run21 and run24) of
a run's own final checkpoint being meaningfully worse than an earlier
one from the same run. This is no longer worth re-litigating per run;
it should just be treated as a standing fact about this codebase: never
deploy or warm-start from a "final" checkpoint without checking earlier
ones first.

**How to apply**: step45409 is now the best-verified checkpoint this
project has produced since the original ancestor. If continuing to
iterate on `--seed-demos-hold-segments` (e.g. giving hold segments their
own independent re-injection weight rather than sharing
`--seed-demos-min-fraction` with whole episodes, per this feature's own
"next lever" note), warm-start from HERE, not from the ancestor again --
this is real, verified progress worth building on rather than
discarding.

## Independent hold-segment weighting (2026-09-13)

Implemented the "next lever" flagged above: `--seed-demos-hold-segments-fraction`
(new flag, default 0.20), giving hold segments their own target buffer-
episode-count fraction, re-injected on their own schedule, fully
decoupled from `--seed-demos-min-fraction` (which now stays at whatever
it's set to, e.g. 0.20 in practice, without being diluted by sharing
with hold segments the way run25's first version did).

**Implementation**: replaced the single `num_seed_episodes`/
`seed_tds_concat`/`reinject_every` triple with two fully independent
ones -- `whole_tds_concat`/`whole_reinject_every`/
`real_episodes_since_reinject_whole` and the hold-segment equivalents --
each computed via the same `n / (reinject_every + n)` steady-state math
as before (now a small named helper, `_reinject_every(n, frac)`, instead
of inlined once), and each checked completely independently in the main
loop's re-injection block. No other part of the pipeline needed to
change -- buffer capacity sizing already summed both groups' transition
counts before this, unaffected by how their re-injection cadence is
split.

**Verified, in order**:
1. A deliberately extreme test (`--seed-demos-hold-segments-fraction 0.9`,
   `--seed-demos-min-fraction` left at its own 0.15 default) to make the
   two groups' cadences clearly different and directly observable within
   a short run: hold segments (reinject_every=1) re-injected after
   EVERY real episode (confirmed at steps 499, 998, 1497, 1996, buffer
   episode count climbing by exactly +6 each cycle -- 5 hold-segment
   re-injections + 1 new real episode), while whole episodes
   (reinject_every=28) correctly never fired in this ~4-episode-long
   test. Exactly the independent behavior this was designed to produce.
2. Confirmed the actual-practice values compute correctly:
   `--seed-demos-min-fraction 0.20` (explicit) and
   `--seed-demos-hold-segments-fraction` left at its own new 0.20
   default both produce `reinject_every=20`, matching the hand-computed
   expectation (`round(5*(1-0.2)/0.2) = 20`) for both groups
   independently.

No crashes in either test. Ready for a real training run, warm-started
from run25's verified step45409 checkpoint (not the ancestor again --
that's now the best-verified starting point this project has).
