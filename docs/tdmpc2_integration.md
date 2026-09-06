# TD-MPC2 Integration

Documents wiring the official TD-MPC2 codebase into `PickPlaceEnv`,
2026-08-31. This was a genuinely involved integration -- five real,
previously-unexercised bugs were found in the process, each verified
against the actual source before being fixed, none guessed. Full
pipeline smoke test passes: real episodes, real replay buffer storage,
real gradient updates with sensible loss trends.

## Setup

Cloned the official reference implementation
(`github.com/nicklashansen/tdmpc2`) into `tdmpc2/` at the project root
(gitignored, same convention as `IsaacLab/`/`lerobot/` -- an external
dependency, not part of this repo's own history, never modified in
place).

**Python/dependency compatibility**: TD-MPC2 wants Python 3.11 -- the
same version `env_isaaclab` already runs, unlike the earlier LeRobot
situation (which needed a separate conda env and a file-based IPC
bridge, see docs/real_arm_setup.md). No IPC needed here, since the
training loop must call our env in-process anyway.

Installed TD-MPC2's other pip dependencies into the *existing*
`env_isaaclab` environment, deliberately protecting `torch` and `numpy`
from being touched: `torch==2.7.0+cu128` is the specific Blackwell-GPU
build this project fought hard to get working (see docs/progress.md,
2026-08-27), and letting pip silently reinstall TD-MPC2's generic
`torch==2.7.1` pin would have broken Isaac Sim entirely. Installed the
pure-Python dependencies normally, then `tensordict`/`torchrl`
specifically with `--no-deps` to prevent them from pulling in a
different torch. This did downgrade `gymnasium` (1.2.1 -> 0.29.1, a
dependency of Isaac Lab's `DirectRLEnv`) and `tensordict` (0.14.0 ->
0.8.3, also an Isaac Lab dependency) -- verified both were safe by
re-running the existing `PickPlaceEnv` smoke tests (state-only and
camera-enabled) immediately after each install step, before proceeding
further. Both passed cleanly.

## The core mismatch: what TD-MPC2 expects vs. what PickPlaceEnv is

Verified directly against TD-MPC2's source (not assumed) that its
reference training loop (`trainer/online_trainer.py`) expects a single,
non-batched, classic-old-gym-API environment: `reset() -> obs` (no info
tuple), `step(action) -> (obs, reward, done, info)` (one combined `done`,
not separate `terminated`/`truncated`), with `info['success']` and
`info['terminated']` required keys. `PickPlaceEnv` is natively
GPU-batched (`DirectRLEnv`, built for `num_envs` in parallel) and returns
the modern Gymnasium 5-tuple.

**Resolved by running `PickPlaceEnv` with `num_envs=1`** specifically for
TD-MPC2. This is not a compromise -- TD-MPC2's whole design point is
sample efficiency from a single environment stream (see
docs/algorithms_explained.md), so this is the intended usage pattern, not
a workaround forced on it.

## Bug 1: the encoder only supports one observation modality at a time

`common/world_model.py`'s `WorldModel.encode()` docstring literally says
*"This implementation assumes a single state-based observation"* -- and
the code confirms it: `self._encoder[self.cfg.obs](obs)` picks exactly
one modality via a single fixed config string (`cfg.obs`, either
`'state'` or `'rgb'`), never both together. This is why every single
example task in the codebase -- including ManiSkill, their one
manipulation-with-cameras benchmark -- asserts `cfg.obs == 'state'`. None
of the shipped examples actually combine modalities, despite the paper
describing multi-modal support and the per-modality encoders being built
correctly for every key in `cfg.obs_shape` regardless (`common/layers.py`'s
`enc()`).

Our task needs both proprioception (`'state'`) and both cameras
simultaneously -- required, not optional, since the real deployed policy
has no privileged state, only proprioception plus its two actual
cameras (`docs/pickplace_env.py`'s own module docstring; an earlier draft
of that env only wired in one camera, an oversight caught before real
training started).

**Fix**: `sim/scripts/tdmpc2_fusion_patch.py` monkey-patches
`WorldModel.encode` at runtime (never editing the cloned `tdmpc2/` files)
to encode each modality separately through its own already-built encoder,
then sum the results. Traced the full call graph first (`act()`,
`_plan()`, `_estimate_value()`, `_update()`, the replay buffer) to confirm
this is the *only* place needing a fix -- everything downstream only ever
consumes the resulting latent vector `z`, agnostic to how many modalities
produced it.

**Related, load-bearing constraint found while designing the fix**: the
conv encoder (`common/layers.py`'s `conv()`) has no final projection
layer -- its output size is whatever falls out of the raw 4-layer conv
stack for a 64x64 input, which for the default `num_channels=32` is
exactly 512. This only matches `cfg.latent_dim` for the `model_size=5`
preset (`latent_dim=512`); `model_size=1` (`latent_dim=128`) would
silently mismatch and crash. **Any task using `'rgb'` at all must use
`model_size=5`** (or a hand-tuned `num_channels` matching a different
`latent_dim`) -- not a preference, a hard architectural constraint.

## Two required camera-side fixes, folded into the fusion strategy

- **Their conv encoder hard-asserts `in_shape[-1] == 64`** -- both
  cameras (native 480x270 and 960x540, matching the real hardware, see
  docs/real_camera_setup.md) are resized to 64x64 before reaching it.
- **Only one `'rgb'` key is supported at all** -- both resized images are
  channel-stacked into one `(6, 64, 64)` tensor under the single `'rgb'`
  key ("early fusion," a standard multi-camera technique, not a hack).
  Confirmed mechanically compatible with zero source modification: the
  conv encoder's first `Conv2d` derives its input channel count from the
  given shape rather than hardcoding 3.

Both handled in `sim/scripts/tdmpc2_pickplace_env.py`'s
`PickPlaceTDMPC2Wrapper`.

## Bug 2: `agent.act()` cannot accept a raw dict observation

`TDMPC2.act()`'s first line is `obs.to(self.device,
non_blocking=True).unsqueeze(0)` -- requires `obs` to support `.to()`
and `.unsqueeze()`. Their own reference loop
(`OnlineTrainer.train()`/`eval()`) calls `agent.act(obs, ...)` with
whatever the env directly returned -- for a dict observation, a plain
Python dict, which has neither method. Checked `OfflineTrainer.eval()`
too (their multi-task path): same pattern, and their multi-task task set
(`common/__init__.py`'s `TASK_SET`) is entirely DMControl/Meta-World
state-based tasks -- no image task ever exercises this either. **This
call site has apparently never actually been run with a genuine Dict
observation in the shipped codebase.**

**Fix**: rather than patch `act()` itself, wrap the observation in a
`TensorDict` before calling it -- exactly what their own `to_td()` helper
already does for replay-buffer storage. `TensorDict` supports `.to()`
and `.unsqueeze()` uniformly across all its keys, so `act()` needs no
modification at all once its input is the right kind of object. Applied
in our own training loop (`sim/scripts/train_tdmpc2_pickplace.py`),
which is a corrected version of `OnlineTrainer.train()` rather than a
reuse of it unmodified, since that reference loop has this exact gap for
any dict-observation task.

## Bug 3: the reference `to_td()` helper's own nested-TensorDict construction is wrong

Found by actually running the pipeline, not by reading: `to_td()` builds
the inner `obs` `TensorDict` with `batch_size=()`, then nests it inside
an outer `TensorDict` with `batch_size=(1,)`. `tensordict` requires a
nested `TensorDict`'s batch size to be a compatible prefix of its
parent's -- `()` cannot nest inside `(1,)` at all. Runtime error: `"the
Tensor state has shape torch.Size([12]) which is incompatible with the
batch-size torch.Size([1])"`. Same root cause as Bug 2 -- every example
task takes the plain-tensor branch of this same function and never
exercises the dict branch. Fixed by unsqueezing each modality tensor to
add the matching leading batch dimension before constructing the inner
`TensorDict` with `batch_size=(1,)` to match the outer.

## Two smaller, unrelated fixes made along the way

- **`sim/envs/pickplace_env.py`'s `extras` dict was never actually
  populated.** `DirectRLEnv`'s base `step()` never touches `self.extras`,
  and our own `_get_dones()` computed `info["placed"]`/`info["failed"]`
  locally per-env but discarded them after only using them to set
  `terminated`. This meant no external caller could ever distinguish a
  *successful* termination from a *failed* one -- exactly what the
  adapter's `info['success']` needs. Fixed by stashing both as per-env
  boolean tensors on `self.extras` inside `_get_dones()`.
- **`hydra.utils.get_original_cwd()` requires an active `@hydra.main`
  CLI context**, which this script deliberately bypasses entirely (per
  the same reasoning as bypassing `envs.make_env()` -- neither applies to
  a custom, non-benchmark task). Patched narrowly (returns a fixed path)
  since it's a pure working-directory lookup unrelated to the actual
  algorithm, not worth standing up a real Hydra context for.

## The `envs` namespace collision

`sim/envs/` and `tdmpc2/tdmpc2/envs/` are both packages literally named
`envs`. Having both on `sys.path` at once means whichever gets imported
(and cached in `sys.modules`) first wins the name for the rest of the
process. Resolved two ways: (1) import order in
`train_tdmpc2_pickplace.py` deliberately imports `envs.pickplace_env`
*before* `tdmpc2/tdmpc2` is even added to `sys.path`, so the name is
already claimed; (2) `tdmpc2_pickplace_env.py`'s adapter folds
`envs/wrappers/tensor.py`'s small amount of logic (numpy<->torch
conversion, `rand_act()`) directly into itself rather than importing it,
so nothing from `tdmpc2/tdmpc2/envs/` needs importing at all.

## Files

- `sim/scripts/tdmpc2_fusion_patch.py` -- the `encode()` monkey-patch.
- `sim/scripts/tdmpc2_pickplace_env.py` -- the adapter
  (`PickPlaceTDMPC2Wrapper`): single-env, classic-gym API, camera resize
  + channel-stack fusion, torch-native throughout.
- `sim/scripts/train_tdmpc2_pickplace.py` -- the training entry point:
  builds cfg from TD-MPC2's own `config.yaml` (reusing its
  `parse_cfg()` defaulting logic rather than re-deriving it by hand),
  and implements a corrected version of `OnlineTrainer.train()`.
- `sim/scripts/test_tdmpc2_adapter.py` -- smoke test for the adapter in
  isolation (obs shapes/dtypes, `info` keys).

## Smoke test results (2026-08-31)

`--smoke-test` shrinks episode length (2s/100 steps vs. the real 10s/500),
batch size (8 vs. 256), and buffer size, so a full pipeline pass can be
verified quickly. 300 env steps: 3 full episodes correctly added to the
replay buffer, 202 real `agent.update()` calls, all producing finite
losses with sensible trends (`termination_loss` dropped from 0.51 to
0.0002 as the network correctly learned that random actions almost never
trigger a real termination; `reward_loss`/`value_loss` trended down from
~2.7 to ~0.3-0.5 as the networks began fitting). 32.7 seconds total. No
crash anywhere in env -> adapter -> `agent.act()` (real MPC planning,
once past `seed_steps`) -> buffer -> `agent.update()`.

**What this confirms**: the full pipeline is mechanically correct end to
end. **What this does NOT confirm**: task performance -- 300 steps of
mostly-random actions is nowhere near enough to expect the agent to
learn anything about actually picking up the cube. That's real training,
not yet started.

## First real training run (2026-08-31)

30,000 steps, full 500-step episodes, default batch size (256), `num_envs=1`
(required -- see the core-mismatch section above). Periodic evaluation
added on top of the smoke-test design: every `--eval-every` real training
steps (checked at episode boundaries only), one deterministic
(`eval_mode=True`) episode runs using the SAME env instance (a second
`SimulationContext` isn't possible in this process), is NOT added to the
replay buffer, and has `scene_camera` frames captured into an mp4 for a
visual checkpoint -- deliberately not tdmpc2's own `save_video` mechanism,
which only activates when wandb is enabled (`common/logger.py`'s
`VideoRecorder`), and real wandb logging was deliberately not stood up.

**Results**: 3935s (~66 min) total, 60 episodes collected, 27,501 real
gradient updates, no crash. Eval episode rewards across the run: +14.6,
+90.3, +12.6, +54.1, +97.9 -- noisy (each is a single episode at one
random cube position, not an average) but trending upward overall. No
successful placement in any eval checkpoint (`success=False` throughout).
~~**Interpretation**: real learning signal (the agent is doing something
meaningfully better than random, reflected in the reward trend), but not
enough environment interaction yet to fully solve the task -- consistent
with TD-MPC2's own benchmark tasks typically needing substantially more
steps even for simpler continuous control problems. Not a red flag on its
own; the natural next experiment is simply more training time, not a
different design.~~

**Correction (2026-08-31, later the same day)**: the interpretation above
was wrong, and watching the actual +97.9 episode's video (rather than
trusting the printed number) is what caught it -- the arm never touched
the cube. The old dense reward paid the *absolute* value of a
distance-based potential every step, so simply occupying a
plausible-looking position was enough to accumulate that score over 500
steps, with no relationship to actual task progress. This was genuine
reward hacking, not learning. See docs/reward_function.md's "Reward-
hacking finding and potential-based-shaping redesign" section and
docs/decisions.md for the full investigation and fix. Left the original
paragraph struck through rather than deleted, since the mistaken
interpretation and how it was caught is itself part of this project's
record.

## Second training run (2026-08-31): reach_scale=0.08 (superseded)

An interim fix (narrowing `reach_scale` from 0.15 to 0.08 to make vague
proximity worth less) was tried next, launched as a shorter 15,000-step
run to check the fix quickly. Stopped early after 4 eval checkpoints:
-3.8, -18.7, -11.2, -33.2 -- consistently negative and non-improving.
Extracted video frames (`sim/output/frame_extract/`, since there's no
video-viewing tool available) confirmed genuinely undirected motion in
both sampled episodes, never engaging the cube. Diagnosis: narrowing the
scale fixed the symptom (less reward for vague proximity) but not the
mechanism (a policy could still profit from occupying any fixed
position, just a smaller one) -- and also removed most of the usable
gradient for a policy starting far from the cube. This is what motivated
looking for a fix to the *mechanism* rather than further tuning the
scale -- see docs/reward_function.md for the potential-based-shaping
redesign this led to.

## Third training run (2026-08-31): potential-based shaping

15,000 steps, same `--eval-every 2500` cadence as the second run, launched
immediately after implementing and verifying the potential-based-shaping
redesign (self-test, env smoke tests, real-data replay -- see
docs/reward_function.md). 2115s (~35 min) total, 31 episodes collected,
12,501 gradient updates, no crash.

**Results**: eval episode rewards across the run: -5.7, -20.5, -3.0,
-2.7, -1.4 -- and, more informatively, the *training*-episode reward sums
(one per completed episode, 31 total) went from a -30 to -80 range in the
first 5 episodes down to consistently -1 to -12 by the last 10, a much
cleaner improving trend than the eval numbers alone suggest. No
successful placement (`success=False` throughout, expected at this
scale). Critically, no episode scored anywhere near the old +97.9-style
exploit magnitude -- the hacking mechanism appears genuinely closed, not
just harder to trigger.

**Direct video/frame inspection** (both the first and last eval
checkpoints, since the numbers alone are exactly what missed the run1
exploit): the arm reaches partway out from its rest pose, then folds into
a compact, low-effort static pose and stays there for most of the
episode, in both the first (step 2994) and last (step 12976) checkpoints
-- never approaching the cube's actual position. No self-collision/
clipping visible in either. **Interpretation**: the exploit is fixed --
near-zero reward now honestly means "no net progress," not "found a way
to fake progress" -- but 15,000 steps has not yet been enough for the
planner to discover genuine cube-directed reaching. Given the new reward
has essentially zero gradient once the arm is idle (shaping only pays for
*changing* distance), and the arm found a locally cheap idle pose early,
it's plausible the policy needs either more steps to stumble into
reward-earning exploration, or a stronger push to keep exploring rather
than settle. Not yet resolved -- the natural next step (per the
established short-run-then-long-run workflow) is a longer run to see
whether more steps alone resolves this, before concluding anything more
drastic is needed.

At the time, the user separately reviewed `eval_step_010481.mp4` (a
checkpoint in between the two sampled above) and flagged it as visually
the closest the arm got to the cube -- denser frame sampling of that
specific video did show the gripper hovering right at the cube's position
for a good stretch of the episode. **This is now believed to have been
coincidental**, not learned reaching -- see the fourth run below, where
the *same* fixed idle pose recurred across multiple checkpoints and
runs while only the cube's randomized spawn position differed, meaning a
"close" eval episode most likely just means the cube happened to spawn
near wherever the idle pose already rests, not that the policy moved
toward it.

## Fourth training run (2026-09-01): same reward, 50,000 steps

50,000 steps, `--eval-every 5000`, launched to test whether more training
time alone resolves run3's lack of visible reaching, per the user's own
suggestion after confirming no reward hacking or self-collision issues in
run3. 8503s (~2h22m) total, 100 episodes collected, 47,501 gradient
updates, no crash.

**Results**: eval episode rewards across the run: -7.6, -11.1, -3.0,
-1.2, -2.9, -4.2, -1.5, -1.2, -1.9 -- still noisy, still no exploit-scale
numbers, but also no longer a clearly improving trend the way run3's
*training*-episode sums were (those hover in a -2 to -50 range throughout
without a clean late-run tightening this time).

**Direct frame inspection across three checkpoints** (steps 20459, 40419,
45409, sampled every 25 frames instead of every 60 this time, after
under-sampling missed real detail in run3's review): the arm holds the
**exact same folded resting pose** in every single sampled frame across
all three checkpoints, regardless of the cube's (randomized, clearly
visible and differently-positioned each time) location. **Interpretation
-- corrects the run3 interpretation above**: the policy has not been
exploring toward the cube at all, in either run. It converges to one
fixed, low-effort static configuration and stays there for the whole
episode; the eval reward's small variance is almost entirely explained by
the brief settling transient at episode start (and pure chance in how
close that fixed pose happens to land relative to that episode's random
cube spawn), not by any real cube-tracking. Likely mechanism: the
potential-based shaping correctly gives *zero* net reward for holding
still anywhere (the whole point of the fix), which also means there is no
reward pressure at all pushing an untrained policy to move, unless its
experience already contains a genuine touch/grasp trajectory for the
value function to learn from -- and tdmpc2's default exploration budget
(5 random episodes, 2500 steps) apparently never produced one against a
small, randomly-positioned target. "Just run longer" is therefore a
weaker bet than it looked after run3 alone, since a second, 3.3x-longer
run reproduced the identical qualitative behavior rather than showing
progress toward reaching. See docs/decisions.md for the resulting fix
(raising the seed-episode exploration budget) tried next.

## Fifth training run (2026-09-01): seed_episodes raised 5 -> 30

50,000 steps, `--eval-every 5000`, same reward as run4 but with the
random-exploration seed budget raised 6x (2,500 -> 15,000 steps / 5 -> 30
episodes) -- the fix decided after run4 (see docs/decisions.md). 7090s
(~2h) total, 101 episodes collected, 35,001 gradient updates, no crash.

**Results**: eval episode rewards across the run: -5.6, -5.4, -8.9, -1.3,
-4.3, -2.8, -1.1, -0.7, -3.1 -- similar range to run4, no exploit-scale
numbers. **Direct frame inspection** (checkpoints 020469, 040429,
045419, every 25 frames): identical finding to run4 -- the arm holds one
fixed resting pose across every sampled frame, regardless of checkpoint
or the cube's randomized position. **This is the run that ruled out
"just needs more random exploration"**: 6x more random-seed budget
produced the qualitatively identical failure mode, which is what
motivated actually tracing TD-MPC2's own planning code (`_plan()` in
tdmpc2.py) rather than continuing to scale the seed budget further --
see docs/decisions.md for what that tracing found (a CEM std-collapse
mechanism, not just insufficient random-exploration volume) and the two
levers it led to.

This run's results were not written up at the time (the project moved
directly into the CEM investigation and then a device migration) --
added retroactively while documenting run6 below, so the record stays
complete.

## Sixth training run (2026-09-02): fixed cube position + raised min_std

50,000 steps, `--eval-every 5000 --cube-pos 0.15 0.0 --min-std 0.5` --
the two curriculum/exploration levers from tracing `_plan()` (see
docs/decisions.md): fixing the cube to one point (the geometric center of
the train region) instead of randomizing it every episode, and raising
the floor under CEM's training-time exploration noise from 0.05 to 0.5.
7273s (~2h1m) total, 100 episodes collected, 35,001 gradient updates, no
crash. Also the first run to log `touched`/`held` directly
(extras["touched"]/["holding"], added just before this run -- see the
commit adding this) instead of relying on sampled-frame guessing.

**Results**: eval episode rewards: -14.0, -19.9, -2.7, -4.4, -2.3, -1.6,
-2.4, -2.3, -2.1. **touched=True from step 20459 onward, on every single
remaining checkpoint (5 in a row)** -- a genuine qualitative break from
every prior run, where touched was never observed to be reliably true
across consecutive checkpoints (run4/run5 showed it essentially never via
frame inspection; run3 showed one plausible-looking but likely-lucky
frame given the cube was randomized then). `held` never went true and no
episode succeeded, so the arm reaches and touches the cube reliably but
hasn't learned to close the gripper and complete a hold yet. Reward also
settled into a visibly tighter band (-1.6 to -2.4) in the second half of
the run, consistent with a policy that has learned a real, repeatable
behavior rather than one dominated by episode-to-episode luck.

**Interpretation**: this is the first run since the reward-hacking fix
where there is direct, repeated, logged evidence of real learned
progress (not exploit-driven, not static-idle-driven). Both levers were
changed together, so this doesn't isolate which one mattered more (or
whether both were needed) -- not a priority to disentangle yet given
neither has been tried alone. Launched a seventh run at 70,000 steps
with the identical settings immediately after this one finished, per the
user's standing instruction, to see whether more time converts reliable
touching into reliable holding/success -- see below once it completes.

## Seventh training run (2026-09-03)

70,000 steps, identical settings to run6 (`--cube-pos 0.15 0.0 --min-std
0.5`), launched automatically the moment run6 finished per explicit
standing instruction from the user ("start a 70k run once its done,
don't wait for me to confirm"). Purpose: isolate whether run6's
touched-but-not-held plateau resolves with more of the exact same
training, before considering any further design changes. 12671.2s
(~3h31m) total, 140 episodes collected, 55,001 gradient updates, no
crash. Two leftover hung processes from an earlier, unrelated jaw-
geometry measurement script were found still running (and competing for
GPU) the entire time this run trained, discovered only once the run
finished -- a real oversight, not cleaned up promptly after their own
results were captured; killed by exact PID once noticed.

**Results**: eval episode rewards: -9.1, -8.5, -2.1, -4.4, -13.4, -1.2,
-2.2 (steps 5489 through 65369, `--eval-every 5000`). Two checkpoints
logged `held=True` (steps 20459 and 30439) -- both watched on video
before trusting them (same standing practice that caught run1's exploit),
and BOTH turned out to be false positives, in two different specific
geometries: step 20459's gripper closed fully beside the cube (the one
already root-caused -- see docs/decisions.md and docs/reward_function.md
for the `is_between_jaws()` fix this produced); step 30439's gripper
closed near the cube's base/pivot area rather than around its body,
a related but distinct misalignment the same geometric fix also covers.
`touched=True` fired on the same 5 of 7 checkpoints as before touching
became reliable in run6, confirming that behavior held under a full
70,000-step budget rather than being a run6-specific fluke.

**Interpretation**: more of the exact same training did NOT resolve the
touched-but-not-held plateau -- it surfaced a second, related false
positive instead, and no genuinely new behavior emerged from the extra
20,000 steps over run6. This is the evidence that "just run longer" had
already been exhausted as a strategy for this specific gap (unlike
earlier in the project, where it hadn't yet been tried before being
dismissed) -- motivating the `is_between_jaws()` geometric fix that run8
tests below, rather than a fourth uniform-length repeat.

## Eighth training run (2026-09-03)

70,000 steps, `--cube-pos 0.25 0.0 --min-std 0.5` -- the `is_between_jaws()`
fix plus the cube moved to the train region's far edge (see
docs/decisions.md for why that specific point). 10972.4s (~3h3m) total,
140 episodes collected, 55,001 gradient updates, no crash.

**Results**: eval episode rewards across all 14 checkpoints (steps 5489
through 65369): -3.4, -3.4, -3.1, -1.8, -2.5, -2.9, -7.1, -3.4, -7.9,
-2.9, -1.1, -3.5, -3.4. `touched=True` on 10 of 14 checkpoints --
consistent with run6/7's reliable-touching finding, now confirmed under
the new detection logic too. Critically: **`between_jaws=True` and
`held=True` on ZERO checkpoints, all 14** -- meaning the fix is
demonstrably not creating any new false positives (a real risk with any
tightened detection logic), at the cost of not yet seeing the milestone
it exists to enable either.

**Direct video review** (best checkpoint, step 55389) explains why,
concretely: the arm approaches the cube from directly above and pokes it
with a single fingertip, sustained and clearly purposeful, but never
straddles it with both open jaws -- exactly the geometric distinction
`is_between_jaws()` is built to require, and exactly why it correctly
never fires for this specific behavior. Separately, the user flagged
from the same video that the gripper closes almost immediately on
approach, well before anywhere near correctly positioned. Full mechanism
and the two-part fix (`lateral_align_weight` continuous alignment
shaping, `premature_close_weight` penalty) in docs/reward_function.md
and docs/decisions.md.

**Interpretation**: real, visible qualitative progress (purposeful,
repeatable single-point contact) with a correctly-strict detector
withholding credit for the harder skill (straddling) it hasn't learned
yet. Not evidence the fix is wrong -- evidence it's precisely rejecting
exactly the behavior it was built to reject.

## Ninth training run (2026-09-03/04)

40,000 steps, `--cube-pos 0.25 0.0 --min-std 0.5 --seed-episodes 2
--resume-from <run8's final checkpoint>` -- WARM-STARTED from run8
rather than trained from scratch, specifically so the reach/touch skill
run8 already learned isn't discarded while training against the two new
terms (`lateral_align_weight`, `premature_close_weight`) added after
reviewing run8's video. Shorter than the preceding from-scratch runs
(40k vs. 70k) since this is refinement of an existing skill, not primary
skill acquisition -- confirmed loading correctly (`[INFO] Resumed agent
weights from ...`) and using a much smaller seed-exploration phase
(1,000 steps, the floor) than any prior run, since a resumed policy that
already knows how to act gets little value from a long pure-random
warmup.

**Results**: `touched=True` on the reviewed checkpoints, no
`between_jaws=True`/`held=True`. Direct video review (`eval_step_036427.mp4`,
`eval_step_032435.mp4`) showed real, specific progress -- the arm
touching the cube with the stationary part of the gripper and pushing it
along, close enough that closing the gripper at the right moment "would've
basically picked it up" (the user's own read, later confirmed accurate).

**Interpretation**: initially misdiagnosed as a gripper-aperture problem
(briefly implemented, then fully reverted after direct user correction --
see docs/progress.md and docs/decisions.md for the full story); correctly
re-diagnosed as an axial positioning problem, the cube buried near the
jaws' pivot rather than out near the fingertips. Motivated
`axial_align_weight`, tested in run10 below.

## Tenth training run (2026-09-04/05)

100,000 steps (the longest single run to date), `--cube-pos 0.25 0.0
--min-std 0.5 --seed-episodes 2 --resume-from <run9's final checkpoint>`
-- adds `axial_align_weight` on top of run9's reward.

**Results**: no crash. Frame-by-frame review of three full eval episodes
(steps 85329, 90319, 95309) showed an open-loop "reach once and freeze"
reflex -- contact within 1-2 seconds of a 25-second episode, then a
completely static pose for the remainder regardless of what happens
next. **The gripper never closed even once, in any sampled frame, across
all three episodes traced in full.**

**Interpretation**: `axial_align_weight` did not address the actual
bottleneck (it only concerns position, not the close decision) and was
reverted via `git revert` rather than built on further -- see
docs/decisions.md. The real finding motivated a closing-focused redesign
(`grasp_reach_min` widened, `grasp_close_weight` raised,
`premature_close_weight` lowered), tested in run11 below, deliberately
warm-started from run9's checkpoint rather than run10's -- run10's extra
100,000 steps had been spent reinforcing the counterproductive "freeze"
habit, a worse starting point than the less-contaminated run9 checkpoint.

## Eleventh training run (2026-09-05)

60,000 steps, `--cube-pos 0.25 0.0 --min-std 0.5 --seed-episodes 2
--resume-from <run9's final checkpoint>` -- the closing-focused redesign
(`grasp_reach_min` -0.01 -> -0.04, `grasp_close_weight` 1.0 -> 2.5,
`premature_close_weight` 0.3 -> 0.1).

**Results**: no crash. Objective per-episode logging across all 11 eval
checkpoints: `touched=True` on 10 of 11, **`between_jaws=True` on 4 of 11**
(steps 20459, 35429, 50399, 55389) -- something that essentially never
happened in runs 9/10 -- but `held=True` on zero. A `--eval-only`
per-step diagnostic replay of the best checkpoint (step 55389) showed
`between_jaws` flickering true/false dozens of times across the 500-step
episode (including a 15-consecutive-step window), with the gripper
joint barely moving even during that long window, and its one
meaningful dip (~0.06 radians toward closed, around steps 368-371)
reversing within a few steps rather than continuing.

**Interpretation**: positioning had genuinely improved (the widened
window works); closing specifically still had not. Root-caused to
`premature_close_weight` firing on every step `between_jaws` drops out
(which it does constantly, being unstable), making "reopen immediately"
the locally safe strategy versus committing to a multi-step close.
Motivated zeroing `premature_close_weight` entirely, tested in run12
below. Also the run this project's video-reading practice was directly
corrected on: initial claims about visible gripper closing (from still
frames) turned out to be a foreshortening illusion from the fixed
external `scene_camera` combined with the arm's own rotation -- resolved
by pulling the objective logged booleans instead of re-arguing from more
stills. See docs/progress.md for the full narrative.

## Twelfth training run (2026-09-05/06)

60,000 steps, `--cube-pos 0.25 0.0 --min-std 0.5 --seed-episodes 2
--resume-from <run11's final checkpoint>` -- `premature_close_weight`
zeroed (0.1 -> 0.0).

**Results**: no crash. **`held=True` fired for the first time in this
project's history**, at step 40419's eval episode (`touched=True,
between_jaws=True, held=True`). `between_jaws=True` on 4 of 11
checkpoints total.

**Interpretation**: verified rather than taken at face value -- a
`--eval-only` replay of the exact checkpoint, extended with
`cube_height`/`gripper_cube_dist`, confirmed all four of `is_holding()`'s
establishing conditions genuinely fired (not a geometric false positive
like the run7 bug), but the actual lift was only ~9mm on a 3cm cube
(`lift_threshold` required just 5mm above resting height), settling back
toward resting height within a few steps rather than being carried --
and it recurred 3 separate times in the one episode, each re-firing the
full `grasp_bonus` due to a bonus-gating bug (see docs/decisions.md).
Motivated three fixes (`between_jaws_grace_steps` hysteresis,
`lift_threshold` raised to 0.04, `grasp_bonus` regated on a genuinely
episode-sticky `was_ever_held`), tested in run13 below.

## Thirteenth training run (2026-09-06, in progress)

60,000 steps, `--cube-pos 0.25 0.0 --min-std 0.5 --seed-episodes 2
--resume-from <run12's final checkpoint>` -- the between_jaws hysteresis/
lift_threshold/grasp_bonus fixes, all three verified via self-test and a
production smoke test before launch. A relatively cheap verification
that these specific fixes work as intended, deliberately kept at the
same 60k scale as runs 11/12 for direct comparability, before committing
to a full, from-scratch clean retrain against the finalized reward
design. Full results once complete.

## Known limitations / not yet done

- Thirteen training runs done so far (30,000 / 15,000 / 15,000 / 50,000 /
  50,000 / 50,000 / 70,000 / 70,000 / 40,000 / 100,000 / 60,000 / 60,000
  / 60,000 steps, under ten different reward/exploration/curriculum
  configurations, six of them warm-started rather than from-scratch --
  see the run sections above), still no systematic sweep over training
  duration, hyperparameters, or fusion strategy. A full clean (non-warm-
  started) retrain against the current, most-refined reward design is
  planned as the next major run once run13 confirms the latest fixes
  hold up -- warm-starting continuously since run9 means the current
  checkpoint's value function is a patchwork of adaptations to several
  different, sometimes-conflicting reward regimes, not a clean optimum
  for the current one.
- Checkpointing/logging beyond console output + eval videos is not wired
  up (`save_agent`/`enable_wandb` both `False`).
- The fusion strategy (elementwise sum of two SimNorm-normalized
  encodings) is a deliberately minimal first choice, not tuned or
  compared against alternatives (e.g. concatenation plus a learned
  projection) -- worth revisiting once more training results exist to
  motivate the added complexity.
- `torch.compile` is disabled (`compile: false`) for fast, debuggable
  iteration -- real training would likely want it re-enabled for speed,
  which may surface its own compile-time issues given how much of this
  integration required runtime monkey-patching.
