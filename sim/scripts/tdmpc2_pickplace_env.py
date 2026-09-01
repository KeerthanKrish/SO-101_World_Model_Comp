# SPDX-License-Identifier: BSD-3-Clause
"""Adapter wrapping PickPlaceEnv (sim/envs/pickplace_env.py) into the
interface the official TD-MPC2 codebase (tdmpc2/, cloned separately,
untouched) actually expects -- verified directly against its source
(envs/mujoco.py's MuJoCoWrapper is the established pattern for adapting a
modern-Gymnasium custom env, envs/wrappers/tensor.py and
trainer/online_trainer.py for the exact calling convention), not guessed:

1. Single, non-batched env, classic old-gym API: reset() -> obs (no info
   tuple), step(action) -> (obs, reward, done, info) (a single combined
   done, not separate terminated/truncated). PickPlaceEnv is natively
   GPU-batched (DirectRLEnv, built for num_envs in parallel) and returns
   the modern 5-tuple. Fixed by running PickPlaceEnv with num_envs=1 for
   TD-MPC2 specifically -- not a compromise: TD-MPC2's whole design
   point is sample efficiency from a single env stream (see
   docs/algorithms_explained.md), so this is the intended usage pattern,
   not a workaround.
2. info['success'] and info['terminated'] are required keys, accessed
   unconditionally by TensorWrapper.step() and the online trainer.
   Populated directly from our reward function's own info dict
   (info["placed"] -> success, the terminated flag -> terminated) --
   no new success/failure logic invented, reusing what's already
   validated (docs/reward_function.md).
3. TD-MPC2's encoder factory (common/layers.py's enc()) dispatches
   STRICTLY by literal key name -- only 'state' and 'rgb' are recognized
   at all, and only ONE 'rgb' key is supported (raises
   NotImplementedError for anything else). This directly conflicts with
   our two-camera requirement (wrist_camera + top_camera, both required
   for the real deployed policy -- see pickplace_env.py's own docstring).
   Resolved via channel-stacking: both images are resized to 64x64
   (their conv encoder hard-asserts `in_shape[-1] == 64`) and
   concatenated along the channel dimension into one (6, 64, 64) tensor
   under the single 'rgb' key -- "early fusion," a legitimate, standard
   multi-camera technique, not a hack, and importantly requires ZERO
   modification to the external tdmpc2/ codebase (its conv encoder
   derives input channel count from the given shape rather than
   hardcoding 3, confirmed by reading common/layers.py's conv()
   directly).

Deliberately does NOT use tdmpc2's own envs/wrappers/tensor.py
TensorWrapper for the numpy<->torch conversion, even though that's the
reference pattern -- `sim/envs/` and `tdmpc2/tdmpc2/envs/` are both
packages literally named "envs", and having both on sys.path
simultaneously is a genuine, confirmed Python namespace collision
(whichever gets imported first wins the name for the whole process).
Rather than fight that with import-order tricks, TensorWrapper's small
amount of logic (numpy<->torch conversion, rand_act()) is folded directly
into this class instead -- this wrapper returns torch tensors and accepts
a torch action directly, so nothing from tdmpc2/tdmpc2/envs/ needs
importing at all.

Usage: import PickPlaceTDMPC2Wrapper from a script that already has
`sim/` on sys.path.
"""

import numpy as np
import torch
import torch.nn.functional as F

import gymnasium as gym

_IMG_SIZE = 64  # hard requirement of tdmpc2's conv encoder, see module docstring


def _resize_to_64(img_hw3_uint8: torch.Tensor) -> torch.Tensor:
    """(H, W, 3) uint8 -> (3, 64, 64) float32 in [0, 255], matching the
    channel-first convention tdmpc2's conv encoder expects (its own
    PixelPreprocess layer handles the /255 normalization internally, so
    values are kept in [0, 255] here, not pre-normalized)."""
    chw = img_hw3_uint8.permute(2, 0, 1).float().unsqueeze(0)  # (1, 3, H, W)
    resized = F.interpolate(chw, size=(_IMG_SIZE, _IMG_SIZE), mode="bilinear", align_corners=False)
    return resized.squeeze(0)  # (3, 64, 64)


class PickPlaceTDMPC2Wrapper(gym.Wrapper):
    """Adapts a num_envs=1 PickPlaceEnv to TD-MPC2's expected single-env,
    classic-gym, Dict(state, rgb) interface. See module docstring for the
    three specific mismatches this resolves and why each fix was chosen.
    """

    def __init__(self, env, success_bonus_threshold: float = 0.0):
        super().__init__(env)
        assert env.num_envs == 1, "TD-MPC2's online trainer expects a single, non-batched env -- see module docstring."
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
                "rgb": gym.spaces.Box(low=0, high=255, shape=(6, _IMG_SIZE, _IMG_SIZE), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    def rand_act(self) -> torch.Tensor:
        """Matches tdmpc2's TensorWrapper.rand_act() exactly (folded in
        here rather than imported -- see module docstring)."""
        return torch.from_numpy(self.action_space.sample().astype(np.float32))

    def _build_obs(self, obs_dict) -> dict:
        """Returns CPU torch tensors (not numpy) -- see module docstring
        on why this wrapper skips tdmpc2's separate TensorWrapper."""
        proprio = obs_dict["policy"]["proprio"][0].cpu().float()  # (12,)
        wrist = _resize_to_64(obs_dict["policy"]["wrist_rgb"][0])  # (3, 64, 64), still on GPU
        top = _resize_to_64(obs_dict["policy"]["top_rgb"][0])  # (3, 64, 64), still on GPU
        rgb = torch.cat([wrist, top], dim=0).cpu()  # (6, 64, 64)
        return {"state": proprio, "rgb": rgb}

    def reset(self, **kwargs):
        obs_dict, _extras = self.env.reset()
        return self._build_obs(obs_dict)

    def step(self, action: torch.Tensor):
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self.env.device).reshape(1, -1)  # (1, 6)
        obs_dict, reward, terminated, truncated, extras = self.env.step(action_t)

        done = bool(terminated[0].item() or truncated[0].item())
        info = {
            "terminated": bool(terminated[0].item()),
            # info["success"] must reflect whether THIS episode ultimately
            # succeeded -- our reward's "placed" flag from the same step
            # that produced `terminated`, exposed via extras (see
            # pickplace_env.py's _get_dones(), which stashes it there
            # specifically for callers like this one). On a timeout
            # (truncated, not terminated) "placed" is correctly False.
            "success": bool(extras["placed"][0].item()),
            # Forwarded for eval-loop logging only (e.g. run_eval_episode()
            # in train_tdmpc2_pickplace.py) -- not required by tdmpc2 itself,
            # unlike terminated/success above. Lets an eval loop report
            # whether the gripper ever actually touched/held the cube
            # during an episode, instead of that having to be inferred by
            # eye from sampled video frames (see docs/tdmpc2_integration.md's
            # run4/run5 investigation).
            "touched": bool(extras["touched"][0].item()),
            "holding": bool(extras["holding"][0].item()),
        }
        return self._build_obs(obs_dict), torch.tensor(float(reward[0].item())), done, info
