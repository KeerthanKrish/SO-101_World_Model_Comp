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

## Known limitations / not yet done

- Real training (the full 500-step episode length, default
  hyperparameters, many more steps) has not been run -- only the
  smoke test's shrunk, fast configuration.
- `eval()` is not wired up (disabled via a very high `eval_freq` in the
  smoke-test config) -- it needs its own separate env instance (parallel
  to the training env), not yet built.
- Checkpointing/logging beyond console output is not wired up
  (`save_agent`/`enable_wandb` both `False` for the smoke test).
- The fusion strategy (elementwise sum of two SimNorm-normalized
  encodings) is a deliberately minimal first choice, not tuned or
  compared against alternatives (e.g. concatenation plus a learned
  projection) -- worth revisiting once real training results exist to
  motivate the added complexity.
- `torch.compile` is disabled (`compile: false`) for fast, debuggable
  iteration -- real training would likely want it re-enabled for speed,
  which may surface its own compile-time issues given how much of this
  integration required runtime monkey-patching.
