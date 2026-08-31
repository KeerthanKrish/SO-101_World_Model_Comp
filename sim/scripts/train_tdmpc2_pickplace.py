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

This is a SMOKE TEST launcher (--smoke-test), not a real training run:
shrinks episode length and buffer/batch sizes so a full pipeline pass
(collect an episode, sample from the buffer, compute a real update) can
be verified quickly. Real training would use the full 500-step episode
length and default hyperparameters.

Usage:
    ./isaaclab.sh -p /path/to/train_tdmpc2_pickplace.py --headless --enable_cameras --smoke-test
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true", help="Short episode/buffer for a fast pipeline check.")
parser.add_argument("--steps", type=int, default=None, help="Override total env steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from time import time

import torch
from omegaconf import OmegaConf
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

    overrides = {
        "task": "so101-pickplace",
        "obs": "rgb",  # informational only post-patch (encode() no longer branches on this), kept for parse_cfg/logging
        "episodic": True,  # REQUIRED: our task has real terminations (success/failure) -- see online_trainer.py
        "model_size": 5,  # REQUIRED for any task using 'rgb' -- see tdmpc2_fusion_patch.py's docstring
        "steps": args_cli.steps or (300 if args_cli.smoke_test else 1_000_000),
        "batch_size": 8 if args_cli.smoke_test else 256,
        "buffer_size": 5_000 if args_cli.smoke_test else 1_000_000,
        "seed_steps": 5,  # low on purpose for the smoke test -- real training should use max(1000, 5*episode_length)
        "eval_freq": 10_000_000,  # effectively disabled -- eval() needs its own separate env instance, not built here yet
        "eval_episodes": 1,
        "compile": False,  # skip torch.compile for a fast, debuggable smoke test
        "enable_wandb": False,
        "save_video": False,
        "save_agent": False,
        "exp_name": "smoke_test" if args_cli.smoke_test else "default",
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


def main():
    cfg = build_cfg()
    print(f"[INFO] task={cfg.task} model_size={cfg.model_size} latent_dim={cfg.latent_dim} "
          f"episode_length={cfg.episode_length} steps={cfg.steps}")

    base_env = PickPlaceEnv(PickPlaceEnvCfg(use_cameras=True, num_envs=1, episode_length_s=episode_length_s()))
    env = PickPlaceTDMPC2Wrapper(base_env)

    agent = TDMPC2(cfg)
    buffer = Buffer(cfg)
    logger = Logger(cfg)
    print(agent.model)

    step, ep_idx, done = 0, 0, True
    tds = None
    start = time()
    num_updates_done = 0

    while step <= cfg.steps:
        if done:
            if step > 0:
                ep_idx = buffer.add(torch.cat(tds))
                print(f"[INFO] step {step}: episode {ep_idx} added to buffer "
                      f"(len={len(tds)}, reward_sum={sum(td['reward'].item() for td in tds[1:]):.3f})")
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
    print("\n[RESULT] === TD-MPC2 pipeline smoke test summary ===")
    print(f"[RESULT] Ran {step} env steps, {ep_idx} episodes added to buffer, "
          f"{num_updates_done} agent.update() calls, in {elapsed:.1f}s.")
    print("[RESULT] No crash -- full pipeline (env -> adapter -> agent.act -> buffer -> agent.update) runs end to end.")


if __name__ == "__main__":
    main()
    simulation_app.close()
