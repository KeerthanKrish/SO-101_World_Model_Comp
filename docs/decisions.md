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
