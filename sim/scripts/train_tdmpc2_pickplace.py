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
    just whether the pipeline runs. Returns (episode_reward, success).
    """
    frames_dir = video_path + "_frames"
    os.makedirs(frames_dir, exist_ok=True)

    obs = env.reset()
    scene_cam = base_env.scene["scene_camera"]
    ep_reward, t, done, info = 0.0, 0, False, {"success": False}

    while not done:
        obs_for_act = TensorDict(obs, batch_size=(), device="cpu") if isinstance(obs, dict) else obs
        action = agent.act(obs_for_act, t0=(t == 0), eval_mode=True)
        obs, reward, done, info = env.step(action)
        ep_reward += reward.item()

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
    return ep_reward, bool(info["success"])


def main():
    cfg = build_cfg()
    print(f"[INFO] task={cfg.task} model_size={cfg.model_size} latent_dim={cfg.latent_dim} "
          f"episode_length={cfg.episode_length} steps={cfg.steps} seed_steps={cfg.seed_steps}")

    base_env = PickPlaceEnv(PickPlaceEnvCfg(use_cameras=True, num_envs=1, episode_length_s=episode_length_s()))
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
                eval_reward, eval_success = run_eval_episode(env, base_env, agent, cfg, video_path)
                print(f"[INFO] step {step}: EVAL episode -- reward={eval_reward:+.3f} "
                      f"success={eval_success} video={video_path}")
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
