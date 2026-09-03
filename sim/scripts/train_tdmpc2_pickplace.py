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
    "--video-dir", type=str, default="/home/keerthan/SO-101-WM/sim/output/tdmpc2_eval_videos", help="Eval video output dir."
)
parser.add_argument(
    "--checkpoint-dir", type=str, default="/home/keerthan/SO-101-WM/sim/output/tdmpc2_checkpoints",
    help="Where to save agent checkpoints (one per eval checkpoint, plus a final one).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import shutil
from time import time

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


def run_eval_episode(env, base_env, agent, cfg, video_path):
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
    """
    frames_dir = video_path + "_frames"
    os.makedirs(frames_dir, exist_ok=True)

    obs = env.reset()
    scene_cam = base_env.scene["scene_camera"]
    info = {"success": False, "touched": False, "holding": False, "between_jaws": False}
    ep_reward, t, done = 0.0, 0, False
    ever_touched, ever_held, ever_between_jaws = False, False, False

    while not done:
        obs_for_act = TensorDict(obs, batch_size=(), device="cpu") if isinstance(obs, dict) else obs
        action = agent.act(obs_for_act, t0=(t == 0), eval_mode=True)
        obs, reward, done, info = env.step(action)
        ep_reward += reward.item()
        ever_touched = ever_touched or info["touched"]
        ever_held = ever_held or info["holding"]
        ever_between_jaws = ever_between_jaws or info["between_jaws"]

        base_env.sim.render()
        rgb = scene_cam.data.output["rgb"][0, ..., :3].cpu().numpy()
        Image.fromarray(rgb).save(os.path.join(frames_dir, f"frame_{t:05d}.png"))
        t += 1

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
          f"min_std={cfg.min_std} cube_pos={args_cli.cube_pos}")

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
    os.makedirs(args_cli.video_dir, exist_ok=True)
    os.makedirs(args_cli.checkpoint_dir, exist_ok=True)

    agent = TDMPC2(cfg)
    buffer = Buffer(cfg)
    logger = Logger(cfg)
    print(agent.model)

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

            if not args_cli.smoke_test and step >= next_eval_at:
                video_path = os.path.join(args_cli.video_dir, f"eval_step_{step:06d}.mp4")
                eval_reward, eval_success, eval_touched, eval_held, eval_between_jaws = run_eval_episode(
                    env, base_env, agent, cfg, video_path
                )
                print(f"[INFO] step {step}: EVAL episode -- reward={eval_reward:+.3f} "
                      f"success={eval_success} touched={eval_touched} between_jaws={eval_between_jaws} "
                      f"held={eval_held} video={video_path}")
                ckpt_path = os.path.join(args_cli.checkpoint_dir, f"agent_step_{step:06d}.pt")
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
        final_ckpt = os.path.join(args_cli.checkpoint_dir, f"agent_step_{step:06d}_final.pt")
        agent.save(final_ckpt)
        print(f"[RESULT] Final checkpoint saved to {final_ckpt}")
    print(f"\n[RESULT] === TD-MPC2 {label} summary ===")
    print(f"[RESULT] Ran {step} env steps, {ep_idx} episodes added to buffer, "
          f"{num_updates_done} agent.update() calls, in {elapsed:.1f}s.")
    print("[RESULT] No crash -- full pipeline (env -> adapter -> agent.act -> buffer -> agent.update) runs end to end.")


if __name__ == "__main__":
    main()
    simulation_app.close()
