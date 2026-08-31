# Training Environment (`PickPlaceEnv`)

Documents the Gym-style training environment (`sim/envs/pickplace_env.py`),
created 2026-08-30: the piece of infrastructure that was missing after
`pickplace_reward.py` -- something with an actual `reset()`/`step()`
interface a training loop (TD-MPC2, or anything else) can drive.

## What it is

An Isaac Lab `DirectRLEnv` subclass wrapping the existing scene configs
(`PickPlaceSceneBaseCfg`/`PickPlaceSceneCfg`) and the already-validated
`pickplace_reward.py`. Chosen over Isaac Lab's other workflow
(`ManagerBasedRLEnv`, a heavier declarative-config style) because this
whole project has consistently used imperative scripts driving
`InteractiveScene` directly -- `DirectRLEnv` keeps that same style while
adding the standard vectorized reset/step machinery for free, rather than
hand-rolling multi-env bookkeeping that Isaac Lab already gets right.

## Observations: policy vs. critic, and why cameras are not optional

Two separate observation groups, Isaac Lab's built-in asymmetric
actor-critic pattern:

- **"policy"**: proprioception (joint positions + velocities, 12 values)
  plus, when `use_cameras=True`, both `wrist_camera` and `top_camera` RGB
  images. This is **everything and exactly** what a real deployed policy
  would have -- the real robot has no other sensing. Both cameras are
  included deliberately, not just the wrist view: an earlier version of
  this env only wired in `wrist_camera`, an oversight caught before any
  real training started (the whole point of matching both cameras to the
  real hardware earlier in this project would be undermined by only using
  one of them).
- **"critic"**: privileged ground truth (cube position/velocity, gripper
  grasp point, target position -- 12 values). Allowed only because this
  is sim-training-only scaffolding; never part of "policy," never
  deployed. Same justification as `pickplace_reward.py`'s own privileged
  state use (see docs/reward_function.md).

**Privileged state is a training-time convenience, not an alternative to
cameras.** Nothing about this environment makes cameras optional for the
actual trained policy -- the two are not interchangeable, and there is no
version of the plan where the final policy runs without both cameras.

## Actions

Joint-space, not Cartesian/IK -- matching how the real arm is already
driven elsewhere in this project (`teleop_bridge.py` converts the leader
arm's action into sim joint positions directly). Each action is a
normalized per-joint delta in `[-1, 1]`, scaled by that joint's actual
velocity limit (read from the articulation config, ultimately from the
URDF/STS3215 datasheet estimates) times the control step duration -- so
`1.0` means "move this joint as fast as the real servo actually can, for
one control step." This is a sim-to-real-grounded action scale, not an
arbitrary one, and directly addresses the "torque/velocity limits" open
item in `sim_to_real_checklist.md`.

## Reward/termination

Calls `pickplace_reward.compute_reward()` verbatim, in a per-environment
Python loop rather than a reimplemented batched-torch version. This is a
deliberate simplicity/performance tradeoff, not an oversight: the reward
function was specifically built and empirically validated as a small,
dependency-free, scalar function (docs/reward_function.md), and
reimplementing that same logic a second time in vectorized form risks
exactly the kind of silent-drift bug this project already hit once (see
`JAW_OFFSET_LOCAL`'s extraction into `grasp_geometry.py`). Not worth the
risk at the modest `num_envs` this project actually runs (see below).
Revisit only if `num_envs` needs to scale into the thousands and this
loop is measured to be a real bottleneck -- it isn't, at 16.

## Cube position sampling

Reset samples cube position uniformly from `cube_x_range=(0.05, 0.25)`,
`cube_y_range=(-0.20, 0.20)` -- the TRAIN region from the empirical
reachability sweep (`sim/scripts/reachability_sweep.py`). The held-out
region for generalization testing (docs/evaluation_plan.md, Test 2) is
deliberately never sampled here.

## `num_envs`: measured, not guessed

Ran `sim/scripts/num_envs_scaling_test.py`, sweeping `num_envs` both
without cameras (a diagnostic ceiling only) and with both cameras enabled
(the number that actually governs real training).

**State-only (`use_cameras=False`, diagnostic only -- never how the real
policy trains):**

| num_envs | env-steps/sec |
|---|---|
| 16 | 1198.7 |
| 64 | 2013.8 |
| 256 | 2711.8 |
| 1024 | 2874.8 |

Clear diminishing returns past 256 -- consistent with the per-env Python
reward loop (see above) becoming the bottleneck rather than physics
itself, exactly as anticipated in that code's own comments.

**Both cameras enabled (the config that matters):**

| num_envs | Result |
|---|---|
| 4 | 39.9 env-steps/sec |
| 16 | 45.9 env-steps/sec -- confirmed clean, zero errors |
| 24 | **fails** |
| 32 | **fails** |

24 and 32 both fail immediately and identically: the RTX renderer spams
`[Error] [gpu.foundation.plugin] Unable to allocate descriptor sets.` /
`Failed to allocate ParameterBlock resources` every frame, forever,
without crashing cleanly -- this is what made it look like a hang at
first (high CPU, zero progress) rather than a resource-limit failure.
This is a fixed internal Vulkan/RTX descriptor pool size being exceeded
by `num_envs * 2 cameras` simultaneous render products, not a gradual
slowdown -- there's a hard wall somewhere between 16 and 24 on this GPU
(RTX 5060 Ti), not a soft degradation curve.

**Decision**: `num_envs=16` is the confirmed-safe default, set directly
in `PickPlaceEnvCfg`. Going higher would need raising Kit's RTX
descriptor pool size via a renderer config setting -- not pursued now,
since 16 is already a meaningful throughput multiple over the original
default of 4 and this isn't a large-cluster setup. Revisit only if
training wall-clock time at 16 turns out to be a real practical problem.

## Verified working (smoke tests)

- State-only, 4 envs, 60 steps: no crash, cube resets land inside the
  configured train region, reward values sane.
- State-only, 4 envs, 550 steps (past the 500-step episode timeout): all
  4 envs correctly truncated and reset with a freshly sampled cube
  position -- confirms the reset-on-termination path, the trickiest part
  of a vectorized env, actually works.
- Both cameras, 2 envs: correct observation shapes (`wrist_rgb`:
  `(N, 270, 480, 3)`, `top_rgb`: `(N, 540, 960, 3)`), no crash.
- Recorded a 300-step random-action rollout video
  (`sim/scripts/record_pickplace_env_video.py`) for a visual sanity check
  -- shows reset/episode mechanics working under random actions (no
  trained policy exists yet, so the motion itself is meaningless).

## Known limitations / not yet done

- No trained policy exists yet -- this is purely the environment
  infrastructure both TD-MPC2 and the diffusion policy will eventually
  train through.
- Target position is still fixed (matches `pickplace_reward.py`'s own
  documented open item) -- not yet randomized.
- The per-env Python reward loop caps effective throughput scaling past
  a few hundred envs (state-only) -- irrelevant at the current
  camera-limited `num_envs=16`, but would need revisiting if the RTX
  descriptor limit is ever raised enough to make it matter again.
