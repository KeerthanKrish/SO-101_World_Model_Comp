# SPDX-License-Identifier: BSD-3-Clause
"""TD-MPC2 training entry point for the SO-101 pick-and-place task.

Deliberately does NOT go through tdmpc2/train.py's @hydra.main decorator
or envs.make_env()'s benchmark-suite dispatch -- neither applies to a
custom, non-benchmark environment. Instead: loads tdmpc2's own
config.yaml as a base (reusing its correct defaults/parse_cfg() logic
rather than re-deriving them by hand), overrides task-specific fields
explicitly, and implements its own bounded training loop -- a corrected
version of trainer/online_trainer.py's OnlineTrainer.train(), fixing one
real gap found while integrating: the reference loop calls
`agent.act(obs, ...)` with the RAW dict observation returned by env.step(),
but TDMPC2.act()'s first line (`obs.to(device).unsqueeze(0)`) requires
something with .to()/.unsqueeze() -- which a plain Python dict doesn't
have. Confirmed (see docs/tdmpc2_integration.md) that every example task
in the codebase uses a single plain-tensor observation, so this call site
has apparently never actually been exercised with a genuine Dict
observation. The fix: wrap obs in a TensorDict (exactly what their own
common/buffer.py already does for buffer storage) before calling act() --
TensorDict supports .to()/.unsqueeze() uniformly across all its keys, so
act() itself needs no modification at all.

Also applies tdmpc2_fusion_patch.py's WorldModel.encode() replacement
before constructing the agent -- see that module's docstring for why
(the shipped encode() only supports one observation modality at a time,
and this task needs 'state' + 'rgb' together, both required for a real
deployed policy).

Also does NOT import anything from tdmpc2/tdmpc2/envs/ (e.g. its
TensorWrapper) -- sim/envs/ and tdmpc2/tdmpc2/envs/ are both packages
literally named "envs", a genuine namespace collision when both are on
sys.path at once. Resolved by importing our own envs.pickplace_env FIRST
(before tdmpc2/tdmpc2 is even added to sys.path, so the name "envs" is
already claimed and cached) and folding TensorWrapper's small amount of
logic directly into tdmpc2_pickplace_env.py's adapter instead of
importing it -- see that module's own docstring.

--smoke-test shrinks episode length and buffer/batch sizes so a full
pipeline pass (collect an episode, sample from the buffer, compute a real
update) can be verified quickly -- confirmed working, see
docs/tdmpc2_integration.md. Without it, this runs REAL training: full
500-step episodes, default batch size, seed_steps set from
--seed-episodes (default 30, overriding tdmpc2's own 5-episode default --
see build_cfg()'s docstring for why).

Periodic evaluation is interleaved with training in the SAME env
instance, not a separate one -- Isaac Sim only allows one
SimulationContext per process (DirectRLEnv's own __init__ raises if one
already exists), so a second env instance for eval isn't possible here.
Every --eval-every real steps (checked at episode boundaries, never
mid-episode), one eval episode runs with agent.act(eval_mode=True) --
deterministic, no exploration noise -- instead of random/exploratory
actions, is NOT added to the replay buffer, and has scene_camera frames
captured and stitched into an mp4 for visual sanity-checking of what the
CURRENT policy actually does (not just "does the pipeline run"). This is
deliberately NOT tdmpc2's own built-in save_video mechanism
(common/logger.py's VideoRecorder) -- that path only activates when
wandb is enabled (`self._video = VideoRecorder(...) if self._wandb and
cfg.save_video else None`), and we're deliberately not standing up real
wandb logging for this.

Checkpointing is also handled directly here, not through
common/logger.py's Logger.save_agent() -- that method is only ever called
from trainer/offline_trainer.py (the multi-task path we don't use);
trainer/online_trainer.py's OnlineTrainer.train() (whose logic this
script's loop is based on) never calls it at all, checkpointed or not.
Calls TDMPC2.save() directly instead -- a small, simple method
(`torch.save({"model": self.model.state_dict()}, fp)`) with no
wandb/Logger coupling. Added after the first real training run
(2026-08-31) produced a policy that could never be reloaded for further
inspection, once eval video review raised real concerns about it (see
docs/tdmpc2_integration.md's results section and docs/decisions.md) --
every future run saves a checkpoint at each eval point plus a final one.

Usage:
    ./isaaclab.sh -p /path/to/train_tdmpc2_pickplace.py --headless --enable_cameras --smoke-test
    ./isaaclab.sh -p /path/to/train_tdmpc2_pickplace.py --headless --enable_cameras --steps 30000 --eval-every 5000
    # Curriculum experiment (2026-09-02, after runs 4/5 converged to a
    # static idle pose regardless of checkpoint) -- see build_cfg()'s
    # docstring for the full reasoning behind these two:
    ./isaaclab.sh -p /path/to/train_tdmpc2_pickplace.py --headless --enable_cameras \\
        --steps 50000 --eval-every 5000 --cube-pos 0.15 0.0 --min-std 0.5
"""

import argparse
import os
import subprocess
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true", help="Short episode/buffer for a fast pipeline check.")
parser.add_argument("--steps", type=int, default=None, help="Override total env steps.")
parser.add_argument("--eval-every", type=int, default=5000, help="Real training steps between eval+video episodes.")
parser.add_argument(
    "--seed-episodes", type=int, default=None,
    help="Episodes of pure-random exploration before the agent's own policy starts acting. "
    "Overrides tdmpc2's own default (5 episodes) -- see build_cfg()'s docstring for why.",
)
parser.add_argument(
    "--cube-pos", type=float, nargs=2, default=None, metavar=("X", "Y"),
    help="Fix the cube's spawn position to one point (both training AND eval episodes, since "
    "they share the same env instance) instead of randomizing across PickPlaceEnvCfg's default "
    "train region. A curriculum lever, not a permanent setting -- see build_cfg()'s docstring "
    "for why this was added. Omit for the original fully-randomized behavior.",
)
parser.add_argument(
    "--min-std", type=float, default=None,
    help="Floor on TD-MPC2's own CEM planning std (tdmpc2/config.yaml's default: 0.05). "
    "See build_cfg()'s docstring for why this was raised. Omit to keep tdmpc2's own default.",
)
parser.add_argument(
    "--horizon", type=int, default=None,
    help="CEM/MPPI planning AND training rollout length (tdmpc2/config.yaml's default: 3). Added "
    "2026-09-14 to test whether the planner's short 3-step lookahead is why a policy that reliably "
    "gets between the jaws (100% verified, run25) has still never once genuinely held the cube "
    "(0% across 40+ multi-draw-verified episodes spanning many checkpoints) -- a real hold needs "
    "committing through many more steps than CEM can directly see, so it has to trust the value "
    "function's own (still-uncertain) longer-horizon bootstrap instead of evaluating the commitment "
    "directly. Confirmed compatible with warm-starting from an existing checkpoint trained at a "
    "DIFFERENT horizon: horizon is used purely as a loop trip-count during both planning (tdmpc2.py's "
    "_plan()) and the training rollout/loss computation (update()), never as a fixed tensor shape "
    "baked into any network layer -- the only horizon-shaped state (_prev_mean, a CEM warm-start "
    "buffer) belongs to the TDMPC2 wrapper object itself, not self.model, and TDMPC2.save()/load() "
    "only ever touch self.model.state_dict() -- confirmed by reading tdmpc2.py directly, then "
    "verified empirically by actually loading run25's checkpoint at a different horizon with no "
    "shape-mismatch error. Increases both per-step planning cost (CEM unrolls more steps every env "
    "step, i.e. every action) and per-update training cost (the consistency/reward/value losses "
    "unroll further, and Buffer.sample()'s effective batch grows since it draws horizon+1 length "
    "windows) roughly in proportion to the new/old horizon ratio -- a training run at a larger "
    "horizon will complete fewer real env steps in the same wall-clock budget than one at the "
    "default. Omit to keep tdmpc2's own default (3).",
)
parser.add_argument(
    "--video-dir", type=str, default="/home/keerthan/SO-101-WM/sim/output/tdmpc2_eval_videos",
    help="Base eval video directory -- each run writes into its own --run-name subfolder under this, "
    "never directly into it. See --run-name.",
)
parser.add_argument(
    "--checkpoint-dir", type=str, default="/home/keerthan/SO-101-WM/sim/output/tdmpc2_checkpoints",
    help="Base checkpoint directory -- same --run-name subfolder convention as --video-dir.",
)
parser.add_argument(
    "--run-name", type=str, default=None,
    help="Name of the subfolder under --video-dir/--checkpoint-dir this run writes into (e.g. 'run10'). "
    "Defaults to an auto-generated timestamp (run_YYYYMMDD_HHMMSS) if omitted, which is guaranteed unique "
    "-- pass an explicit name for something more readable, but only if you're sure it won't collide with "
    "an existing run's name (nothing checks that for you). Added 2026-09-04 after discovering that eval "
    "steps 2994/5489/... 45409 etc. recur across many runs sharing the same episode length/eval cadence, "
    "which had been silently overwriting earlier runs' videos and checkpoints at those exact filenames for "
    "days -- runs 2, 3, 4, 6, and 7's raw output files were lost this way before it was caught. See "
    "docs/decisions.md.",
)
parser.add_argument(
    "--resume-from", type=str, default=None,
    help="Warm-start from a saved agent checkpoint (TDMPC2.save()'s own format) instead of training from "
    "scratch. Loads model weights only -- the replay buffer is NOT restored (TDMPC2.save() never saved "
    "it), so this run still starts with an empty buffer and collects its own fresh experience; only the "
    "encoder/dynamics/reward/value/policy networks carry over. Pair with a much smaller --seed-episodes "
    "than a from-scratch run -- the resumed policy already knows how to act, so there's little value in "
    "a long pure-random warmup, and every step of it is wasted opportunity to build on what's already "
    "learned.",
)
parser.add_argument(
    "--eval-only", action="store_true",
    help="Skip training entirely. Requires --resume-from. Loads that checkpoint, runs "
    "--eval-only-repeats eval episode(s) (agent.act(eval_mode=True), same as a normal "
    "training-time eval), and writes both the usual eval video AND a per-step diagnostic CSV "
    "(step, the raw commanded gripper action, reward, gripper_joint_pos, axial/lateral offset, "
    "between_jaws/holding/touched) to <run_video_dir>/eval_only_<checkpoint-name>[_<draw>]_"
    "steps.csv, then exits -- no buffer, no logger, no training loop constructed at all. Added "
    "2026-09-06 specifically to answer one question run11's own eval logs couldn't: "
    "is_holding() never fired even in the episodes where between_jaws briefly did -- this "
    "checks whether that's a genuine TIMING problem (the cube only in the correctly-positioned "
    "zone for a handful of steps, not enough time to close) versus the gripper simply never "
    "trending toward closed even while it had time to. See docs/decisions.md.",
)
parser.add_argument(
    "--eval-only-repeats", type=int, default=1,
    help="Only used with --eval-only. Number of independent eval episodes to draw against the "
    "SAME checkpoint, in one Kit boot (re-using the same env instance -- see run_eval_episode()'s "
    "own docstring for why a second SimulationContext isn't possible in this process, so this "
    "loops rather than relaunching). Added 2026-09-12: CEM planning is never seeded (confirmed "
    "genuinely non-reproducible even against an identical checkpoint and starting state -- see "
    "docs/decisions.md), so every eval boolean this project has ever reported (between_jaws, "
    "held, touched) has been a SINGLE stochastic draw per checkpoint. run22/run23's diagnostic "
    "traces couldn't tell apart 'this checkpoint genuinely regressed' from 'this one draw was "
    "unlucky' -- this makes that distinguishable by drawing a real sample. Prints a per-draw line "
    "plus an aggregate hit-rate summary at the end; video/CSV filenames get a `_drawN` suffix when "
    "this is >1 (unsuffixed, exactly the original behavior, when left at the default of 1).",
)
parser.add_argument(
    "--seed-demos", type=str, default=None,
    help="Directory of recorded teleop episode_*.json files (see "
    "sim/scripts/segment_teleop_episodes.py) to replay through the real env and add to the "
    "buffer before training starts, seeding it with genuine successful-grasp transitions "
    "instead of relying solely on RL exploration to ever discover one -- seven straight runs "
    "(9 through 15) against this reward design never did. See docs/decisions.md. Every "
    "episode_*.json found in this directory is replayed and added, so point this at a "
    "directory containing only the episodes worth seeding with (e.g. a curated subset "
    "confirmed via replay_demo_to_buffer.py to reach a genuine ever_holding=True), not "
    "necessarily the full raw segmentation output. Buffer capacity is enlarged to fit these "
    "transitions PERMANENTLY, never evicted for the life of the run -- see the buffer_cfg "
    "comment in main() for why a one-time bootstrap that fades was deliberately rejected. "
    "Omit for the original behavior (empty buffer, pure RL exploration).",
)
parser.add_argument(
    "--seed-demos-min-fraction", type=float, default=0.15,
    help="Only used with --seed-demos. TD-MPC2's own buffer sampler (torchrl SliceSampler) "
    "picks each training batch's trajectories UNIFORMLY BY EPISODE COUNT, not weighted by "
    "episode length or recency -- confirmed by reading its source directly (torchrl/data/"
    "replay_buffers/samplers.py's _sample_slices()), not assumed. A one-time seeding batch "
    "therefore becomes a SHRINKING fraction of what gets sampled as more RL episodes "
    "accumulate (run16: 5 seed episodes out of 85 total by the end, ~5.9% -- confirmed present "
    "in every batch throughout, but not a large signal, and would keep shrinking further on a "
    "longer run). This periodically re-adds the already-replayed seed episodes (no new env "
    "stepping needed -- just another buffer.add() of already-computed transitions, effectively "
    "free) whenever their share of total buffered episodes would otherwise drop below this "
    "fraction, keeping their presence roughly constant rather than one-time-and-fading. See "
    "docs/decisions.md.",
)
parser.add_argument(
    "--seed-demos-hold-segments", action="store_true",
    help="Only used with --seed-demos. Added 2026-09-12 after a multi-draw variance study "
    "(--eval-only-repeats) found held=True in 0 of 36 honestly-measured draws across every "
    "checkpoint this project has tested, including its most stable one -- see docs/decisions.md. "
    "--seed-demos-min-fraction (above) only guarantees a floor on WHOLE seed episodes' share of "
    "total episode count; it says nothing about how many of the actual TRANSITIONS sampled are "
    "genuinely post-grasp, and since SliceSampler picks episodes uniformly BY COUNT (not weighted "
    "by length), the ~80%+ of the buffer that's self-generated RL experience contributes close to "
    "zero hold-phase transitions of its own (the policy essentially never succeeds), leaving "
    "genuine hold-phase signal thin and non-growing regardless of how long training runs. This "
    "flag extracts, from each seed episode, the segment starting HOLD_SEGMENT_LOOKBACK_STEPS "
    "before its own first is_holding()=True step through its end (see that constant's own "
    "comment), and injects it as an ADDITIONAL pseudo-episode alongside (not instead of) the "
    "whole episode. Re-injected on its OWN independent cadence -- see "
    "--seed-demos-hold-segments-fraction below -- not the whole episodes' "
    "--seed-demos-min-fraction (run25's first version of this shared one combined floor between "
    "the two groups, which meant adding hold segments quietly diluted the whole episodes' own "
    "proven representation to make room; verified via --eval-only-repeats that run25 still worked "
    "well despite that, but there's no reason to keep paying that cost once it's this cheap to "
    "avoid). Since episodes are sampled uniformly by COUNT, a short hold-focused pseudo-episode "
    "gets the exact same per-draw selection probability as a full-length one -- it costs nothing "
    "in selection weight to be short. Requires --seed-demos.",
)
parser.add_argument(
    "--seed-demos-hold-segments-fraction", type=float, default=0.20,
    help="Only used with --seed-demos-hold-segments. Added 2026-09-13, alongside decoupling hold "
    "segments' re-injection cadence from --seed-demos-min-fraction (see --seed-demos-hold-segments' "
    "own help text) -- the independent target share of total buffer EPISODE COUNT hold-segment "
    "pseudo-episodes maintain, computed and re-injected completely separately from the whole "
    "episodes' own --seed-demos-min-fraction floor (which stays at its own value, unchanged, no "
    "longer diluted by sharing with hold segments). Same math as --seed-demos-min-fraction "
    "(steady-state share of total episode insertions converges to n / (reinject_every + n), solved "
    "for reinject_every given this target). Defaulted to 0.20 -- the same value "
    "--seed-demos-min-fraction has actually been run at in practice (run22-25), chosen here so "
    "hold segments get a genuinely independent, equally-substantial floor rather than reusing the "
    "OTHER flag's own separate 0.15 default, which was tuned (loosely) for a different purpose "
    "(whole episodes, well before hold segments existed) -- not a claim that 0.20 is specifically "
    "correct for hold segments, just a reasonable, comparable starting point given no evidence yet "
    "points to a better one.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import glob
import json
import shutil
from datetime import datetime
from time import time

# Force line-buffered stdout -- this box's known Kit shutdown-hang (real
# work completes, including the final [RESULT] summary, but the process
# hangs afterward in simulation_app.close()) means a run can sit hung for
# a long time before it's noticed and killed. Every backgrounded run here
# redirects stdout to a file, which Python fully block-buffers by
# default -- without this, whatever happened after the last flush
# (potentially the entire run, including [RESULT]) is invisible in the
# log until the buffer happens to fill, making a finished-but-hung run
# indistinguishable from a genuinely stuck one. Confirmed necessary the
# hard way (2026-09-09): a seed-demos smoke test hung in close() with
# nothing printed past step 125 in the log, even though loss/step
# progress had clearly continued (confirmed via nvidia-smi/py-spy against
# the live process) well past that point. Already applied to
# replay_demo_to_buffer.py for the same reason. `sys` itself is already
# imported at the top of this file, before AppLauncher.
sys.stdout.reconfigure(line_buffering=True)

import torch
from omegaconf import OmegaConf
from PIL import Image
from tensordict.tensordict import TensorDict

# IMPORT ORDER MATTERS: sim/envs/ and tdmpc2/tdmpc2/envs/ are both
# packages literally named "envs". Whichever gets imported (and cached in
# sys.modules) FIRST wins the name for the rest of the process -- Python
# won't re-resolve an already-imported module name even after a new
# sys.path entry is added. Importing ours first, before tdmpc2/tdmpc2 is
# even added to sys.path, avoids the collision (confirmed necessary: an
# earlier ordering hit "ModuleNotFoundError: No module named
# 'envs.pickplace_env'" because tdmpc2's envs package had already claimed
# the name). Nothing used from tdmpc2/tdmpc2 in this script imports from
# ITS OWN envs package anyway (we bypass envs.make_env entirely), so this
# reordering costs nothing.
sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from envs.pickplace_env import PickPlaceEnv, PickPlaceEnvCfg  # isort:skip
# _jaw_offsets/_dist3 -- only used by run_eval_episode()'s optional
# step_log_path diagnostic (2026-09-06), to report the raw axial/lateral
# offset and gripper-cube distance each step rather than just the binary
# between_jaws/holding flags already in `info`.
from envs.pickplace_reward import _dist3, _jaw_offsets  # isort:skip

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim/scripts")
from tdmpc2_pickplace_env import PickPlaceTDMPC2Wrapper  # isort:skip

sys.path.insert(0, "/home/keerthan/SO-101-WM/tdmpc2/tdmpc2")
from tdmpc2_fusion_patch import apply_fusion_patch  # isort:skip

apply_fusion_patch()  # MUST happen before constructing any WorldModel/TDMPC2 instance.

import hydra.utils  # isort:skip
from common.parser import parse_cfg  # isort:skip
from common.buffer import Buffer  # isort:skip
from common.logger import Logger  # isort:skip
from tdmpc2 import TDMPC2  # isort:skip

# parse_cfg() calls hydra.utils.get_original_cwd(), which only works
# inside an active @hydra.main CLI context -- this script deliberately
# bypasses that decorator entirely (see module docstring: neither it nor
# envs.make_env() applies to a custom, non-benchmark environment), so
# there's no such context to query. This has nothing to do with the
# actual algorithm -- it's purely a working-directory lookup used to
# build cfg.work_dir -- so patched narrowly rather than standing up a
# real Hydra context just to satisfy one path computation. Our own
# build_cfg() below sets cfg.work_dir explicitly afterward anyway, so the
# exact value returned here doesn't matter.
hydra.utils.get_original_cwd = lambda: "/home/keerthan/SO-101-WM"


def episode_length_s() -> float:
    """Single source of truth for the episode length, used both for
    PickPlaceEnvCfg (the REAL env timeout) and TD-MPC2's own cfg.episode_length
    bookkeeping -- a real bug caught by actually running this: an earlier
    version computed this value only for cfg.episode_length and forgot to
    also pass it to PickPlaceEnvCfg, which kept its own default (10s/500
    steps) -- the smoke test then ran for 301 steps and never reached a
    single episode timeout, so 0 episodes ever reached the buffer."""
    return 2.0 if args_cli.smoke_test else 10.0  # 100 vs 500 steps at 0.02s/step


def build_cfg():
    base = OmegaConf.load("/home/keerthan/SO-101-WM/tdmpc2/tdmpc2/config.yaml")
    episode_length = int(episode_length_s() / 0.02)
    # tdmpc2's own default is max(1000, 5*episode_length) -- see
    # envs.make_env() in tdmpc2/tdmpc2/envs/__init__.py -- 5 episodes of
    # pure-random exploration before the agent's own (learned) policy
    # starts acting. That default is a reasonable heuristic for TD-MPC2's
    # own benchmark tasks, but proved NOT NEARLY enough here: both the
    # second (15k-step) and third (50k-step) training runs under the
    # potential-based-shaping reward converged to the arm holding one
    # fixed idle pose for the entire episode, regardless of checkpoint or
    # the cube's (randomized) position -- confirmed by direct frame
    # inspection, not just the reward numbers (see docs/tdmpc2_integration.md).
    # Likely mechanism: the new reward correctly gives zero net reward for
    # holding still (that's the whole point -- it closed the original
    # reward-hacking exploit), which also means there's no reward pressure
    # pushing an untrained policy to move at all UNLESS its experience
    # already contains a genuine touch/grasp/reward-earning trajectory for
    # the value function to learn from. With only 5 random episodes (2500
    # steps) to find one by chance against a small, randomly-positioned
    # target, none apparently ever occurred. `--seed-episodes` lets this
    # be overridden independently of tdmpc2's own formula -- default here
    # raised 6x to 30 episodes, giving random exploration a substantially
    # larger, still-bounded budget to stumble into a rewarding trajectory
    # before the learned policy takes over. Not guaranteed to fix it --
    # this is a real experiment, not a proven solution -- but directly
    # targets the likely root cause rather than just running longer with
    # the same 5-episode budget again (see docs/decisions.md).
    default_seed_episodes = 5 if args_cli.smoke_test else 30
    seed_episodes = args_cli.seed_episodes or default_seed_episodes
    seed_steps = 5 if args_cli.smoke_test else max(1000, seed_episodes * episode_length)

    # Raising seed_episodes 6x (above) was NOT enough on its own -- runs 4
    # and 5 (30k/50k steps, 5 and 30 seed episodes respectively) both
    # converged to the arm holding one fixed idle pose for essentially the
    # whole episode, confirmed by direct frame inspection across multiple
    # checkpoints in each run (see docs/tdmpc2_integration.md). Tracing
    # TD-MPC2's own planning code (tdmpc2.py's _plan()) surfaced a second,
    # more precise mechanism on top of "random exploration rarely finds a
    # small randomized target": at every single env step, its CEM planner
    # samples 512 candidate action sequences, narrows to the best 64 over
    # 6 refinement iterations, and picks one -- but CEM is well known to
    # over-confidently narrow its own std even when the underlying value
    # estimates it's ranking by are pure noise (no real learned signal yet
    # to distinguish a genuinely good sequence from a bad one). The one
    # line that matters most: `if not eval_mode: a = a + std *
    # torch.randn(...)` -- the actual exploration noise added to the
    # action during TRAINING rollouts uses exactly this same std, which
    # can (and, empirically, does) collapse toward min_std well before any
    # real signal justifies that confidence, producing a falsely-precise,
    # repeatably-idle action. Raising min_std puts a hard floor under how
    # confidently-wrong that collapse can get. Deliberately NOT touching
    # max_std (2, unchanged) -- CEM should still be ALLOWED to narrow
    # toward something tight if it ever does find a real signal to
    # exploit, just not below this floor. This change is safe on eval:
    # `eval_mode` skips the `a = a + std * randn(...)` line entirely, so
    # eval-time behavior (and the underlying planning/mean-action quality)
    # is completely unaffected -- this only changes how much the TRAINING
    # rollout itself explores. Kept as an explicit opt-in override
    # (--min-std) rather than a new permanent default, since it's a real
    # experiment, not a proven fix.
    #
    # Separately, --cube-pos fixes the cube to one specific point instead
    # of randomizing across the full train region for both training and
    # eval (they share one env instance -- see run_eval_episode()'s
    # docstring). This is a curriculum lever attacking the OTHER half of
    # the same underlying problem: even with better exploration noise,
    # random exploration still has to relocate a small, randomly-placed
    # target from scratch every single episode. Fixing the target for a
    # run isolates "can this reward/architecture learn to reach and grasp
    # AT ALL" from "can it generalize across positions" -- the harder
    # question we haven't earned the right to ask yet, given no run has
    # solved the easier one. (0.15, 0.0) -- the geometric center of
    # PickPlaceEnvCfg's existing (0.05-0.25, -0.20-0.20) train region --
    # is a principled, unbiased choice: the middle of the already-
    # validated reachable core, not a position picked by looking at where
    # a previous run's idle pose happened to rest (which would bias this
    # experiment toward a false positive). Also opt-in, not a new default,
    # for the same reason as min_std above.

    overrides = {
        "task": "so101-pickplace",
        "obs": "rgb",  # informational only post-patch (encode() no longer branches on this), kept for parse_cfg/logging
        "episodic": True,  # REQUIRED: our task has real terminations (success/failure) -- see online_trainer.py
        "model_size": 5,  # REQUIRED for any task using 'rgb' -- see tdmpc2_fusion_patch.py's docstring
        # 30k steps (~60 real episodes) is a bounded FIRST real run, not
        # training to convergence -- enough to see genuine learning signal
        # and produce a few eval videos, not an open-ended commitment.
        # Override with --steps for a longer/shorter run.
        "steps": args_cli.steps or (300 if args_cli.smoke_test else 30_000),
        "batch_size": 8 if args_cli.smoke_test else 256,
        "buffer_size": 5_000 if args_cli.smoke_test else 1_000_000,  # auto-capped at min(buffer_size, steps) anyway
        "seed_steps": seed_steps,
        "eval_freq": 10_000_000,  # tdmpc2's own eval() path unused -- see module docstring, we run our own eval loop
        "eval_episodes": 1,
        "compile": False,  # see module/file docstrings -- kept off for a debuggable first real run, not just the smoke test
        "enable_wandb": False,
        "save_video": False,  # tdmpc2's own video path is wandb-gated -- we capture eval video ourselves instead
        "save_agent": False,
        "exp_name": "smoke_test" if args_cli.smoke_test else "run1",
        "data_dir": "/home/keerthan/SO-101-WM/sim/output/tdmpc2_data",
    }
    # Conditional, not baked into `overrides` unconditionally like the
    # keys above -- when --min-std isn't passed, we want tdmpc2's own
    # config.yaml default (0.05) to pass through `base` untouched, not a
    # second hardcoded copy of that default here that could silently drift
    # from the real one.
    if args_cli.min_std is not None:
        overrides["min_std"] = args_cli.min_std
    if args_cli.horizon is not None:
        overrides["horizon"] = args_cli.horizon
    cfg = OmegaConf.merge(base, overrides)
    cfg.obs_shape = {"state": (12,), "rgb": (6, 64, 64)}
    cfg.action_dim = 6
    cfg.episode_length = episode_length
    # cfg.work_dir gets set by parse_cfg() itself (via the patched
    # get_original_cwd() above) -- no need to set it here too.
    return parse_cfg(cfg)


def to_td(obs, action=None, reward=None, terminated=None, action_dim=6):
    """Adapted from online_trainer.py's OnlineTrainer.to_td() -- copied
    rather than imported since it's a small method on a class we're not
    otherwise using wholesale. One real fix applied here, found by
    actually running this: the reference version builds the inner obs
    TensorDict with batch_size=(), then nests it inside an outer
    TensorDict with batch_size=(1,) -- but tensordict requires a nested
    TensorDict's batch_size to be a compatible prefix of its parent's, so
    batch_size=() cannot nest inside batch_size=(1,) at all (confirmed by
    the actual runtime error: "Tensor state has shape [12] which is
    incompatible with the batch-size [1]"). Every example task in the
    codebase uses a single plain-tensor observation, which takes the
    `else` branch below and never exercises this dict path at all -- so,
    consistent with the other gaps found while integrating (see this
    file's own module docstring), this exact code has apparently never
    actually been run with a genuine Dict observation before now. Fixed
    by unsqueezing each modality tensor to add the matching leading batch
    dim before constructing the inner TensorDict with batch_size=(1,).
    """
    if isinstance(obs, dict):
        obs = TensorDict({k: v.unsqueeze(0) for k, v in obs.items()}, batch_size=(1,), device="cpu")
    else:
        obs = obs.unsqueeze(0).cpu()
    if action is None:
        action = torch.full((action_dim,), float("nan"))
    if reward is None:
        reward = torch.tensor(float("nan"))
    if terminated is None:
        terminated = torch.tensor(float("nan"))
    return TensorDict(
        obs=obs, action=action.unsqueeze(0), reward=reward.unsqueeze(0), terminated=terminated.unsqueeze(0),
        batch_size=(1,),
    )


_JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Added 2026-09-12 for --seed-demos-hold-segments (see its own help text
# for the full derivation: held=True was 0/36 across every checkpoint
# this project has honestly, repeatedly measured -- see docs/decisions.md
# -- and the buffer only ever guarantees a floor on whole SEED EPISODES'
# share of total episode count, not on how many of the TRANSITIONS within
# them are genuinely post-grasp). How many steps of "final approach" to
# include BEFORE a demo episode's first is_holding()=True step when
# extracting its hold-and-carry segment, so the segment captures the
# critical transition-into-a-hold moment itself, not just its aftermath.
# 50 steps = 1 second at this env's 50Hz control rate. Checked against
# all 5 of this project's current seed episodes (episode_000/002/003/
# 005/006) via extract_grasp_segment.py before picking this: their own
# first-hold boundaries land at steps 194-200 with remarkable
# consistency, so 50 is a generous "final approach" window comfortably
# inside every one of them -- not deeply tuned, just checked against the
# real data rather than guessed.
HOLD_SEGMENT_LOOKBACK_STEPS = 50


def hold_segment_start(first_holding_step, lookback_steps=HOLD_SEGMENT_LOOKBACK_STEPS):
    """Pure index arithmetic for --seed-demos-hold-segments (see
    HOLD_SEGMENT_LOOKBACK_STEPS' own comment) -- deliberately kept as a
    tiny, dependency-free function separate from the actual tds slicing
    at its call site (which needs torch/tensordict), so the one thing
    actually worth getting wrong here -- the boundary arithmetic itself --
    stays trivially checkable independent of Isaac Sim.

    Returns None (no segment to extract) if this episode never held at
    all (first_holding_step is None) -- a caller must not extract a
    "hold segment" from an episode that never established one. Otherwise
    clamps to 0 rather than going negative, for the (currently
    unobserved, but not impossible for some future seed episode) case
    where a hold is established within the first `lookback_steps` of the
    episode -- extracting from the true start of the episode in that
    case rather than an invalid negative index.
    """
    if first_holding_step is None:
        return None
    return max(0, first_holding_step - lookback_steps)


def replay_episode_to_tds(base_env, env, episode_path):
    """Replays one recorded teleop episode (sim/output/teleop_episodes_v2/
    episode_*.json format, or any directory produced the same way) through
    the REAL env/wrapper this run already constructed -- the exact
    env/observation/reward pipeline real training uses, not a separate
    instance -- producing genuine TD-MPC2-format transitions. Distilled
    from sim/scripts/replay_demo_to_buffer.py, which validated this
    approach directly (5 of 8 re-segmented episodes reach a genuine
    ever_holding=True when replayed this way -- see docs/decisions.md for
    the full derivation, evidence, and remaining caveats). Strips that
    prototype's investigation-only diagnostics (step traces, fingertip-
    position verification against the STL mesh) -- those answered
    questions already resolved; keeps only the per-episode saturation/
    clipping summary, still useful for auditing a specific seed episode's
    replay fidelity when this actually gets used.

    Calls env.reset() at the start -- REQUIRED even though the state gets
    immediately overridden below, since multiple episodes may be replayed
    in sequence on this same persistent env instance (unlike the
    prototype, which only ever replayed one episode per process): reset()
    is what clears PickPlaceEnv's own per-episode reward-tracking state
    (_was_holding/_prev_dist/_ever_held/etc, see pickplace_env.py's
    _reset_idx()) and restarts its internal episode-length timeout
    counter. Skipping it would let one episode's leftover state (e.g.
    _ever_held=True from a genuine hold) silently leak into the next
    replayed episode's reward computation.

    Note: replayed episodes are subject to the SAME episode-length
    timeout as real training rollouts (episode_length_s, ~500 steps at
    50Hz) -- a demonstration whose own downsampled length exceeds that
    gets truncated, matching every real training episode's own hard cap
    rather than injecting anomalously long episodes into a buffer/sampler
    built around that fixed length. Confirmed via replay_demo_to_buffer.py
    that this did not cost any of the 5 successful episodes their
    holding-worthy moment (each occurs well within the first 500 steps).

    Returns (tds, stats): tds is a list of to_td()-format TensorDicts
    (torch.cat(tds) is directly buffer.add()-able); stats is a dict of
    the episode's own outcome for the caller to log/audit, including
    (added 2026-09-12) `first_holding_step`, the index into `tds` of the
    first step this episode's own is_holding() became True (None if it
    never did) -- see --seed-demos-hold-segments/hold_segment_start()'s
    own docstrings for why. Computed for every replay regardless of
    whether that flag is passed, since it costs nothing extra (the
    replay loop already checks info["holding"] every step for
    ever_holding) -- only ACTED on when the flag is set.
    """
    with open(episode_path) as f:
        episode = json.load(f)
    downsampled = episode[0::2]  # 100Hz recording -> the env's own 50Hz control rate

    env.reset()  # see docstring -- clears prior-episode persistent reward state, restarts the timeout counter

    default_root_state = base_env.cube.data.default_root_state.clone()
    root_pose = default_root_state[:, :7].clone()
    root_pose[0, 0:3] = torch.tensor(downsampled[0]["cube_pos"], device=base_env.device) + base_env.scene.env_origins[0]
    base_env.cube.write_root_pose_to_sim(root_pose)
    base_env.cube.write_root_velocity_to_sim(torch.zeros_like(default_root_state[:, 7:]))

    initial_joint_pos = base_env.robot.data.default_joint_pos.clone()
    for i, joint_name in enumerate(base_env.robot.data.joint_names):
        if joint_name in downsampled[0]["joint_pos"]:
            initial_joint_pos[0, i] = downsampled[0]["joint_pos"][joint_name]
    base_env.robot.write_joint_state_to_sim(initial_joint_pos, base_env.robot.data.default_joint_vel)
    base_env.robot.reset()
    base_env.scene.reset()

    # PickPlaceEnv's action mechanism is INCREMENTAL (each action is a
    # delta from _joint_pos_target, not absolute -- see
    # PickPlaceEnv._pre_physics_step()); must match the state override
    # above or the first computed action below is a delta from the wrong
    # baseline.
    base_env._joint_pos_target = torch.tensor(
        [[downsampled[0]["joint_pos"][j] for j in _JOINT_ORDER]], device=base_env.device
    )
    obs = env._build_obs(base_env._get_observations())  # env.reset()'s own obs is stale after the override above

    tds = [to_td(obs, action_dim=6)]
    max_cube_height = downsampled[0]["cube_pos"][2]
    ever_touched = ever_between_jaws = ever_holding = False
    first_holding_step = None  # index into `tds` -- see hold_segment_start()'s own docstring
    reward_sum = 0.0
    clamp_events = 0
    limit_clip_events = 0
    n_steps_run = 0

    for frame in downsampled[1:]:
        desired_target_raw = torch.tensor([[frame["joint_pos"][j] for j in _JOINT_ORDER]], device=base_env.device)
        # OPTION 1 (chosen after discussion with the user, 2026-09-08):
        # clip the recorded target to the robot's own physical joint
        # limits before using it for anything below -- see
        # replay_demo_to_buffer.py's module docstring for the full
        # reasoning, and docs/decisions.md for the two episodes (001, 007)
        # whose replay fidelity this clipping appears to cost the most.
        desired_target = torch.clamp(desired_target_raw, base_env._soft_limits[..., 0], base_env._soft_limits[..., 1])
        if not torch.equal(desired_target_raw, desired_target):
            limit_clip_events += 1
        desired_delta = desired_target - base_env._joint_pos_target
        raw_ratio = (desired_delta / base_env._max_delta)[0]  # unclamped -- kept only as a label/diagnostic
        action_label = raw_ratio.clamp(-1.0, 1.0)
        if bool((raw_ratio.abs() > 1.0).any()):
            clamp_events += 1

        # Bypasses the normal delta-action velocity clamp -- overriding
        # _joint_pos_target directly and passing a ZERO action makes
        # _pre_physics_step()'s own delta math a no-op on top of it, so
        # the actual PD command becomes exactly the recorded target,
        # matching how the original recording was made (no per-tick
        # velocity ceiling -- see replay_demo_to_buffer.py's "ROOT-CAUSE
        # FIX" comment for the full derivation). action_label above (what
        # a normal delta-action would have needed, clamped for
        # representability) is still the action stored in the buffer --
        # an honest, direction-consistent label for what happened, even
        # on the rare saturated step where it does not exactly reconstruct
        # the commanded target (checked against the 5 seed-worthy
        # episodes directly: each joint saturates on at most ~0.8% of
        # steps -- see docs/decisions.md).
        base_env._joint_pos_target = desired_target
        zero_action = torch.zeros(6, device=base_env.device)
        obs, reward, done, info = env.step(zero_action)
        tds.append(to_td(obs, action_label.cpu(), reward, torch.tensor(float(info["terminated"])), action_dim=6))

        n_steps_run += 1
        cube_height = base_env.cube.data.root_pos_w[0, 2].item()
        max_cube_height = max(max_cube_height, cube_height)
        ever_touched = ever_touched or info["touched"]
        ever_between_jaws = ever_between_jaws or info["between_jaws"]
        ever_holding = ever_holding or info["holding"]
        # First is_holding()=True step this episode -- tds.append() just
        # above already happened, so len(tds)-1 is the index of the entry
        # this step's `info` actually belongs to. Only the FIRST such step
        # is recorded (later steps where holding flickers back to True
        # after a brief loss don't move this) -- --seed-demos-hold-segments
        # wants the genuine first transition into holding, not a later one.
        if first_holding_step is None and info["holding"]:
            first_holding_step = len(tds) - 1
        reward_sum += reward.item()
        if done:
            break

    stats = dict(
        n_transitions=len(tds), n_steps_run=n_steps_run, n_steps_available=len(downsampled) - 1,
        reward_sum=reward_sum, max_cube_height=max_cube_height,
        recorded_max_cube_height=max(s["cube_pos"][2] for s in episode),
        ever_touched=ever_touched, ever_between_jaws=ever_between_jaws, ever_holding=ever_holding,
        first_holding_step=first_holding_step,
        clamp_events=clamp_events, limit_clip_events=limit_clip_events,
    )
    return tds, stats


def run_eval_episode(env, base_env, agent, cfg, video_path, step_log_path=None):
    """Runs ONE episode with the CURRENT policy (eval_mode=True -- no
    exploration noise), using the SAME env instance training already
    uses (a second Isaac Sim SimulationContext isn't possible in this
    process -- see module docstring). NOT added to the replay buffer.
    Captures scene_camera frames and stitches them into an mp4 at
    video_path -- a visual check on what the policy actually does, not
    just whether the pipeline runs. Returns (episode_reward, success,
    ever_touched, ever_held, ever_between_jaws).

    ever_touched/ever_held (whether pickplace_reward.py's touch_bonus/
    holding state ever fired at any point this episode, via
    extras["touched"]/["holding"] -- see pickplace_env.py's _get_dones())
    were added after run4/run5 (2026-09-01): judging whether the gripper
    ever actually engaged the cube by eye, from a handful of sampled
    video frames, turned out to be error-prone and slow (see
    docs/tdmpc2_integration.md) -- this gives an exact, cheap, objective
    answer straight from the reward function's own state instead.

    ever_between_jaws (added 2026-09-03 alongside is_between_jaws() --
    see pickplace_reward.py's module docstring) separates two otherwise-
    indistinguishable failure modes: "never got the cube positioned
    correctly at all" vs. "got it positioned but never closed in time" --
    the run7 false positive that motivated adding this check in the first
    place was exactly a case where naive proximity alone couldn't tell
    these apart.

    step_log_path (added 2026-09-06, --eval-only): when given, writes a
    per-step CSV of (step, the raw commanded gripper action, reward,
    gripper_joint_pos, axial, lateral, cube_height, gripper_cube_dist,
    between_jaws, holding, touched) to this path. The ever_* booleans
    above already say WHETHER between_jaws/holding ever fired this
    episode, but not for how many consecutive steps, nor what the
    gripper was actually doing (already trending closed but out of time,
    vs. not moving toward closed at all) during that window -- exactly
    the distinction needed to tell a genuine TIMING problem (cube only
    correctly positioned for a handful of steps) apart from the closing
    incentive itself still being too weak. cube_height/gripper_cube_dist
    (added after a `holding=True` eval result didn't match what direct
    video review showed -- see docs/decisions.md) additionally let every
    one of is_holding()'s four establishing conditions (cube_height >
    lift_threshold, gripper_joint_pos <= gripper_closed_threshold,
    gripper_cube_dist < grasp_proximity_threshold, between_jaws) be
    checked independently from the logged CSV, rather than trusting the
    `holding` boolean at face value -- this project has caught more than
    one is_holding()/is_between_jaws() false positive before by checking
    the underlying geometry directly instead of the flag alone. All five
    of axial/lateral/cube_height/gripper_cube_dist are deliberately
    recomputed independently via _jaw_offsets()/_dist3() on the same
    gripper_pos/gripper_quat/cube_pos values pickplace_env.py's own
    _get_dones() already reads (grasp_points_local(), cube.data.root_pos_w)
    rather than modifying pickplace_env.py to expose them -- keeps this
    diagnostic fully outside the reward/env classes actual training
    depends on, at the cost of only re-deriving values pickplace_reward.py
    already computes internally every step anyway. None of this runs (nor
    is `step_log_path` ever passed) during a real training run's own
    periodic eval calls -- default None leaves every real call site's
    behavior byte-for-byte unchanged.
    """
    frames_dir = video_path + "_frames"
    os.makedirs(frames_dir, exist_ok=True)

    obs = env.reset()
    scene_cam = base_env.scene["scene_camera"]
    info = {"success": False, "touched": False, "holding": False, "between_jaws": False}
    ep_reward, t, done = 0.0, 0, False
    ever_touched, ever_held, ever_between_jaws = False, False, False
    step_rows = [] if step_log_path is not None else None

    while not done:
        obs_for_act = TensorDict(obs, batch_size=(), device="cpu") if isinstance(obs, dict) else obs
        action = agent.act(obs_for_act, t0=(t == 0), eval_mode=True)
        obs, reward, done, info = env.step(action)
        ep_reward += reward.item()
        ever_touched = ever_touched or info["touched"]
        ever_held = ever_held or info["holding"]
        ever_between_jaws = ever_between_jaws or info["between_jaws"]

        if step_rows is not None:
            gripper_pos, gripper_quat = base_env._grasp_points_local()
            gripper_pos_l = gripper_pos[0].cpu().tolist()
            cube_pos = (base_env.cube.data.root_pos_w - base_env.scene.env_origins)[0].cpu().tolist()
            gripper_joint_pos = base_env.robot.data.joint_pos[0, base_env._joint_indices[-1]].item()
            axial, lateral = _jaw_offsets(gripper_pos_l, gripper_quat[0].cpu().tolist(), cube_pos)
            action_gripper = action.reshape(-1)[-1].item()
            # cube_height/gripper_cube_dist -- added alongside the
            # is_holding()/is_grasped() sanity-check this diagnostic is
            # for (2026-09-06): `between_jaws`/`holding` alone don't say
            # WHY holding became true -- is_holding() requires FOUR things
            # together to establish it (cube_height > lift_threshold,
            # gripper_joint_pos <= gripper_closed_threshold, spherical
            # gripper-cube distance < grasp_proximity_threshold, AND
            # between_jaws), so all four need to be independently
            # verifiable from this log, not just trusted from the
            # boolean, given this project's own history of more than one
            # is_holding()/is_grasped() false positive caught only by
            # directly checking the underlying geometry (see
            # is_between_jaws()'s and is_holding()'s own docstrings).
            cube_height = cube_pos[2]
            gripper_cube_dist = _dist3(gripper_pos_l, cube_pos)
            step_rows.append(
                f"{t},{action_gripper:+.4f},{reward.item():+.4f},{gripper_joint_pos:+.4f},"
                f"{axial:+.4f},{lateral:+.4f},{cube_height:+.4f},{gripper_cube_dist:+.4f},"
                f"{info['between_jaws']},{info['holding']},{info['touched']}"
            )

        base_env.sim.render()
        rgb = scene_cam.data.output["rgb"][0, ..., :3].cpu().numpy()
        Image.fromarray(rgb).save(os.path.join(frames_dir, f"frame_{t:05d}.png"))
        t += 1

    if step_rows is not None:
        with open(step_log_path, "w") as f:
            f.write(
                "step,action_gripper,reward,gripper_joint_pos,axial,lateral,"
                "cube_height,gripper_cube_dist,between_jaws,holding,touched\n"
            )
            f.write("\n".join(step_rows) + "\n")

    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", "20", "-i", os.path.join(frames_dir, "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", video_path,
        ],
        check=True, capture_output=True,
    )
    shutil.rmtree(frames_dir)
    return ep_reward, bool(info["success"]), ever_touched, ever_held, ever_between_jaws


def main():
    cfg = build_cfg()
    print(f"[INFO] task={cfg.task} model_size={cfg.model_size} latent_dim={cfg.latent_dim} "
          f"episode_length={cfg.episode_length} steps={cfg.steps} seed_steps={cfg.seed_steps} "
          f"min_std={cfg.min_std} horizon={cfg.horizon} cube_pos={args_cli.cube_pos}")

    # --cube-pos fixes cube_x_range/cube_y_range to a single (degenerate,
    # zero-width) range -- sample_uniform(x, x, ...) returns exactly x
    # every call (torch.rand(...) * (x - x) + x = x, regardless of the
    # random draw), so this needed no changes to PickPlaceEnv itself, just
    # feeding it a single-point range instead of the default wide one. See
    # build_cfg()'s docstring for why this run might use it.
    env_cfg_kwargs = dict(use_cameras=True, num_envs=1, episode_length_s=episode_length_s())
    if args_cli.cube_pos is not None:
        cube_x, cube_y = args_cli.cube_pos
        env_cfg_kwargs["cube_x_range"] = (cube_x, cube_x)
        env_cfg_kwargs["cube_y_range"] = (cube_y, cube_y)
    base_env = PickPlaceEnv(PickPlaceEnvCfg(**env_cfg_kwargs))
    env = PickPlaceTDMPC2Wrapper(base_env)

    # Aimed once -- same third-person view as record_pickplace_env_video.py.
    # Only used during eval episodes (see run_eval_episode), not training steps.
    base_env.scene["scene_camera"].set_world_poses_from_view(
        torch.tensor([[0.6, -0.6, 0.5]], device=base_env.device), torch.tensor([[0.15, 0.0, 0.05]], device=base_env.device)
    )
    # Every run gets its own subfolder -- never written directly into
    # --video-dir/--checkpoint-dir. See --run-name's own help text for
    # why: eval steps recur across runs sharing the same episode length/
    # eval cadence (most of them do), and writing flat silently overwrote
    # earlier runs' videos/checkpoints at those exact filenames for days
    # before it was caught -- runs 2, 3, 4, 6, and 7's raw outputs were
    # lost this way.
    run_name = args_cli.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_video_dir = os.path.join(args_cli.video_dir, run_name)
    run_checkpoint_dir = os.path.join(args_cli.checkpoint_dir, run_name)
    print(f"[INFO] run_name={run_name} (video_dir={run_video_dir} checkpoint_dir={run_checkpoint_dir})")
    os.makedirs(run_video_dir, exist_ok=True)
    os.makedirs(run_checkpoint_dir, exist_ok=True)

    agent = TDMPC2(cfg)
    if args_cli.resume_from is not None:
        # Weights only -- see --resume-from's own help text for why the
        # buffer isn't (and can't be) restored. Must happen AFTER
        # TDMPC2(cfg) constructs the network (agent.load() loads INTO the
        # existing model, matching TDMPC2.save()/.load()'s own save-format
        # -- see tdmpc2/tdmpc2/tdmpc2.py) and BEFORE the training loop
        # below starts using the agent at all.
        agent.load(args_cli.resume_from)
        print(f"[INFO] Resumed agent weights from {args_cli.resume_from}")

    if args_cli.eval_only:
        # No buffer/logger/training loop at all -- see --eval-only's own
        # help text. Requires --resume-from since there's nothing
        # meaningful to evaluate about a freshly/randomly initialized
        # agent for this diagnostic's purpose.
        if args_cli.resume_from is None:
            raise ValueError("--eval-only requires --resume-from (nothing meaningful to evaluate otherwise).")
        ckpt_name = os.path.splitext(os.path.basename(args_cli.resume_from))[0]
        n_repeats = args_cli.eval_only_repeats
        draws = []
        for draw_idx in range(n_repeats):
            # Unsuffixed at the default (n_repeats=1) -- exactly the
            # original filenames, so any existing tooling/paths pointed at
            # a specific eval_only_<ckpt>.mp4 keeps working unchanged.
            suffix = "" if n_repeats == 1 else f"_draw{draw_idx}"
            video_path = os.path.join(run_video_dir, f"eval_only_{ckpt_name}{suffix}.mp4")
            step_log_path = os.path.join(run_video_dir, f"eval_only_{ckpt_name}{suffix}_steps.csv")
            eval_reward, eval_success, eval_touched, eval_held, eval_between_jaws = run_eval_episode(
                env, base_env, agent, cfg, video_path, step_log_path=step_log_path
            )
            draws.append((eval_reward, eval_success, eval_touched, eval_held, eval_between_jaws))
            print(f"[INFO] eval-only draw {draw_idx}/{n_repeats - 1} -- reward={eval_reward:+.3f} "
                  f"success={eval_success} touched={eval_touched} between_jaws={eval_between_jaws} "
                  f"held={eval_held}")
            print(f"[INFO] video={video_path}")
            print(f"[INFO] steps_csv={step_log_path}")
        if n_repeats > 1:
            # Aggregate hit-rate summary -- see --eval-only-repeats' own
            # help text for why a single draw can't be trusted to tell a
            # genuine regression apart from an unlucky sample against
            # CEM's own unseeded, per-call stochastic planning.
            n = len(draws)
            touched_rate = sum(d[2] for d in draws) / n
            between_jaws_rate = sum(d[4] for d in draws) / n
            held_rate = sum(d[3] for d in draws) / n
            mean_reward = sum(d[0] for d in draws) / n
            print(f"[RESULT] eval-only aggregate over {n} draws of {ckpt_name}: "
                  f"touched_rate={touched_rate:.2f} between_jaws_rate={between_jaws_rate:.2f} "
                  f"held_rate={held_rate:.2f} mean_reward={mean_reward:+.3f}")
        # NOT simulation_app.close() here -- the `if __name__ ==
        # "__main__":` block at the bottom of this file already does
        # that exactly once after main() returns, on every code path.
        # Calling it a second time here would be redundant at best.
        return

    if args_cli.seed_demos_hold_segments and args_cli.seed_demos is None:
        raise ValueError("--seed-demos-hold-segments requires --seed-demos (nothing to extract a segment from).")

    # --seed-demos: replay real recorded teleop episodes through this
    # exact env/wrapper instance BEFORE the buffer is even constructed --
    # see replay_episode_to_tds()'s own docstring and docs/decisions.md
    # for the full derivation. Collected into a plain list first (not
    # added to a buffer yet) specifically so the total transition count
    # is known before Buffer(cfg) runs, needed for the capacity
    # adjustment below.
    seed_episodes = []
    # --seed-demos-hold-segments (see its own help text): (path, segment_tds)
    # for each seed episode's extracted hold-and-carry segment, kept
    # separate from seed_episodes above (rather than merged into it) so
    # the per-episode logging/stats below stays about the WHOLE episode's
    # own replay outcome, uncluttered by the derived segment's numbers.
    hold_segments = []
    total_demo_transitions = 0
    if args_cli.seed_demos is not None:
        episode_paths = sorted(glob.glob(os.path.join(args_cli.seed_demos, "episode_*.json")))
        print(f"[INFO] --seed-demos: found {len(episode_paths)} episode file(s) in {args_cli.seed_demos}")
        for path in episode_paths:
            tds, stats = replay_episode_to_tds(base_env, env, path)
            seed_episodes.append((path, tds, stats))
            total_demo_transitions += len(tds)
            truncated = stats["n_steps_run"] < stats["n_steps_available"]
            print(f"[INFO] seed-demo replay {os.path.basename(path)}: "
                  f"transitions={stats['n_transitions']} reward_sum={stats['reward_sum']:+.3f} "
                  f"max_cube_height={stats['max_cube_height']:.4f} "
                  f"(recorded={stats['recorded_max_cube_height']:.4f}) "
                  f"ever_touched={stats['ever_touched']} ever_between_jaws={stats['ever_between_jaws']} "
                  f"ever_holding={stats['ever_holding']} clamp_events={stats['clamp_events']} "
                  f"limit_clip_events={stats['limit_clip_events']} truncated_by_timeout={truncated}")
            if args_cli.seed_demos_hold_segments:
                start_idx = hold_segment_start(stats["first_holding_step"])
                if start_idx is None:
                    print(f"[WARN] --seed-demos-hold-segments: {os.path.basename(path)} never reached "
                          f"is_holding()=True -- no hold segment extracted from it")
                else:
                    segment_tds = tds[start_idx:]
                    hold_segments.append((path, segment_tds))
                    total_demo_transitions += len(segment_tds)
                    print(f"[INFO] --seed-demos-hold-segments: extracted a {len(segment_tds)}-step hold "
                          f"segment from {os.path.basename(path)} (steps {start_idx}-{start_idx + len(segment_tds) - 1}, "
                          f"starting {HOLD_SEGMENT_LOOKBACK_STEPS} steps before first is_holding()=True "
                          f"at step {stats['first_holding_step']})")
        print(f"[INFO] --seed-demos: {total_demo_transitions} total transitions from "
              f"{len(episode_paths)} episode(s)"
              + (f" plus {len(hold_segments)} hold segment(s)" if hold_segments else ""))

    # Buffer's own capacity is min(cfg.buffer_size, cfg.steps) -- see
    # common/buffer.py. buffer_size already defaults to 1,000,000 (never
    # the binding constraint at any step count used so far), so cfg.steps
    # alone determines capacity -- exactly the number of RL steps this run
    # takes. A copy of cfg with `steps` inflated by the demo transition
    # count -- used ONLY for sizing Buffer's capacity, never cfg.steps
    # itself, which still drives the main loop/eval schedule/logging
    # completely unchanged -- gives the FIRST seeding batch a little extra
    # headroom before anything needs to be evicted for it. This is now a
    # minor nicety, not the main mechanism: since TD-MPC2's own sampler
    # picks trajectories uniformly BY EPISODE COUNT (see
    # --seed-demos-min-fraction's help text), a one-time batch still
    # becomes a shrinking fraction of what gets sampled as more RL
    # episodes accumulate regardless of whether it's ever evicted from
    # storage -- the periodic re-injection below (not this capacity bump)
    # is what actually keeps the seed episodes' SAMPLING share from
    # fading over a long run.
    buffer_cfg = cfg
    if total_demo_transitions > 0:
        buffer_cfg = OmegaConf.merge(cfg, {"steps": cfg.steps + total_demo_transitions})
    buffer = Buffer(buffer_cfg)
    logger = Logger(cfg)
    print(agent.model)

    for path, tds, stats in seed_episodes:
        seed_ep_idx = buffer.add(torch.cat(tds))
        print(f"[INFO] seed-demo episode {seed_ep_idx} added to buffer from {os.path.basename(path)} "
              f"(len={len(tds)}, reward_sum={stats['reward_sum']:+.3f})")

    for path, segment_tds in hold_segments:
        seed_ep_idx = buffer.add(torch.cat(segment_tds))
        print(f"[INFO] hold-segment episode {seed_ep_idx} added to buffer from {os.path.basename(path)} "
              f"(len={len(segment_tds)})")

    # Cache the already-replayed transitions (torch.cat once, reused many
    # times below) -- re-injection needs no new Isaac Sim stepping at all,
    # just another buffer.add() of data already computed above, so it's
    # effectively free to do often. buffer.add() re-tags `episode` on
    # whatever TensorDict it's given to the buffer's own current counter
    # before copying it into storage, so reusing the same cached object
    # across many add() calls is exactly how it's designed to be called
    # repeatedly (confirmed by reading common/buffer.py directly) -- no
    # aliasing/staleness risk.
    whole_tds_concat = [torch.cat(tds) for _, tds, _ in seed_episodes]
    hold_tds_concat = [torch.cat(segment_tds) for _, segment_tds in hold_segments]

    def _reinject_every(n, frac):
        """Steady-state share of total episode INSERTIONS (not
        necessarily of whatever happens to still be resident after
        eviction, a subtly different and harder-to-track quantity)
        converges to n / (reinject_every + n) -- solve for reinject_every
        given the target fraction. None if there's nothing to re-inject
        (n == 0 -- e.g. --seed-demos-hold-segments not passed, or no
        seed episode ever held at all)."""
        if n == 0:
            return None
        return max(1, round(n * (1 - frac) / frac))

    # Added 2026-09-13: whole episodes and hold segments now maintain
    # their OWN independent target fraction / re-injection cadence, each
    # computed and tracked completely separately -- see
    # --seed-demos-hold-segments-fraction's own help text for why this
    # replaced run25's first version (one shared floor across both
    # groups, which meant adding hold segments quietly diluted the whole
    # episodes' own already-proven representation to make room for them).
    whole_reinject_every = _reinject_every(len(seed_episodes), args_cli.seed_demos_min_fraction)
    if whole_reinject_every is not None:
        print(f"[INFO] --seed-demos-min-fraction={args_cli.seed_demos_min_fraction}: re-injecting all "
              f"{len(seed_episodes)} whole seed episode(s) every {whole_reinject_every} "
              f"real episode(s) added")
    hold_reinject_every = _reinject_every(len(hold_segments), args_cli.seed_demos_hold_segments_fraction)
    if hold_reinject_every is not None:
        print(f"[INFO] --seed-demos-hold-segments-fraction={args_cli.seed_demos_hold_segments_fraction}: "
              f"re-injecting all {len(hold_segments)} hold-segment episode(s) every "
              f"{hold_reinject_every} real episode(s) added")
    real_episodes_since_reinject_whole = 0
    real_episodes_since_reinject_hold = 0

    step, ep_idx, done = 0, 0, True
    tds = None
    start = time()
    num_updates_done = 0
    next_eval_at = args_cli.eval_every

    while step <= cfg.steps:
        if done:
            if step > 0:
                ep_idx = buffer.add(torch.cat(tds))
                print(f"[INFO] step {step}: episode {ep_idx} added to buffer "
                      f"(len={len(tds)}, reward_sum={sum(td['reward'].item() for td in tds[1:]):.3f})")

                # Whole episodes and hold segments re-inject on their own
                # independent schedules (see whole_reinject_every/
                # hold_reinject_every's own comment) -- both checked here,
                # completely independently of each other.
                if whole_reinject_every is not None:
                    real_episodes_since_reinject_whole += 1
                    if real_episodes_since_reinject_whole >= whole_reinject_every:
                        for demo_td in whole_tds_concat:
                            ep_idx = buffer.add(demo_td)
                        print(f"[INFO] step {step}: re-injected {len(seed_episodes)} whole "
                              f"seed-demo episode(s) to maintain sampling share (buffer.num_eps={ep_idx})")
                        real_episodes_since_reinject_whole = 0

                if hold_reinject_every is not None:
                    real_episodes_since_reinject_hold += 1
                    if real_episodes_since_reinject_hold >= hold_reinject_every:
                        for demo_td in hold_tds_concat:
                            ep_idx = buffer.add(demo_td)
                        print(f"[INFO] step {step}: re-injected {len(hold_segments)} hold-segment "
                              f"episode(s) to maintain sampling share (buffer.num_eps={ep_idx})")
                        real_episodes_since_reinject_hold = 0

            if not args_cli.smoke_test and step >= next_eval_at:
                video_path = os.path.join(run_video_dir, f"eval_step_{step:06d}.mp4")
                eval_reward, eval_success, eval_touched, eval_held, eval_between_jaws = run_eval_episode(
                    env, base_env, agent, cfg, video_path
                )
                print(f"[INFO] step {step}: EVAL episode -- reward={eval_reward:+.3f} "
                      f"success={eval_success} touched={eval_touched} between_jaws={eval_between_jaws} "
                      f"held={eval_held} video={video_path}")
                ckpt_path = os.path.join(run_checkpoint_dir, f"agent_step_{step:06d}.pt")
                agent.save(ckpt_path)
                print(f"[INFO] step {step}: checkpoint saved to {ckpt_path}")
                next_eval_at += args_cli.eval_every

            obs = env.reset()
            # THE FIX: wrap in TensorDict before calling agent.act() -- see
            # module docstring. Their reference OnlineTrainer.train() does
            # NOT do this and would crash here for any dict observation.
            tds = [to_td(obs, action_dim=cfg.action_dim)]

        obs_for_act = TensorDict(obs, batch_size=(), device="cpu") if isinstance(obs, dict) else obs
        if step > cfg.seed_steps:
            action = agent.act(obs_for_act, t0=len(tds) == 1)
        else:
            action = env.rand_act()

        obs, reward, done, info = env.step(action)
        tds.append(to_td(obs, action, reward, torch.tensor(float(info["terminated"])), cfg.action_dim))

        if step >= cfg.seed_steps and buffer.num_eps > 0:
            train_metrics = agent.update(buffer)
            num_updates_done += 1
            if num_updates_done % 5 == 1:
                print(f"[INFO] step {step}: update #{num_updates_done} -- "
                      f"total_loss={train_metrics['total_loss'].item():.4f} "
                      f"consistency_loss={train_metrics['consistency_loss'].item():.4f} "
                      f"reward_loss={train_metrics['reward_loss'].item():.4f} "
                      f"value_loss={train_metrics['value_loss'].item():.4f} "
                      f"termination_loss={float(train_metrics['termination_loss']):.4f}")

        step += 1

    elapsed = time() - start
    label = "smoke test" if args_cli.smoke_test else "training run"
    if not args_cli.smoke_test:
        final_ckpt = os.path.join(run_checkpoint_dir, f"agent_step_{step:06d}_final.pt")
        agent.save(final_ckpt)
        print(f"[RESULT] Final checkpoint saved to {final_ckpt}")
    print(f"\n[RESULT] === TD-MPC2 {label} summary ===")
    print(f"[RESULT] Ran {step} env steps, {ep_idx} episodes added to buffer, "
          f"{num_updates_done} agent.update() calls, in {elapsed:.1f}s.")
    print("[RESULT] No crash -- full pipeline (env -> adapter -> agent.act -> buffer -> agent.update) runs end to end.")


if __name__ == "__main__":
    main()
    simulation_app.close()
