# SPDX-License-Identifier: BSD-3-Clause
"""Monkey-patches WorldModel.encode() (from the cloned, UNTOUCHED tdmpc2/
codebase) to properly fuse multiple observation modalities.

Why this is needed: verified directly against tdmpc2/tdmpc2/common/world_model.py
that the shipped encode() picks exactly ONE modality --
`self._encoder[self.cfg.obs](obs)`, where cfg.obs is a single fixed string
('state' or 'rgb') -- its own docstring literally says "This implementation
assumes a single state-based observation." Every example task in the
codebase (mujoco.py, maniskill.py, dmcontrol.py) asserts `cfg.obs == 'state'`
for exactly this reason: none of them actually combine modalities. Our task
needs both proprioception ('state') AND both cameras (channel-stacked into
one 'rgb' tensor, see tdmpc2_pickplace_env.py) simultaneously -- required,
not optional, since a real deployed policy has no privileged state, only
proprioception plus its two actual cameras.

Confirmed by tracing the full call path (see docs/algorithms_explained.md
and docs/tdmpc2_integration.md for the full writeup) that this is the ONLY
place a fix is needed:
  - The per-modality encoders already exist for every key in cfg.obs_shape
    (common/layers.py's enc() builds one per key regardless) -- only the
    forward-pass dispatch in encode() needs extending.
  - Everything downstream (dynamics, reward, value function, planner,
    training update, replay buffer) only ever consumes the resulting
    latent vector z, agnostic to how many modalities produced it -- so
    fixing encode() alone is sufficient, nothing else needs patching.

Fusion strategy: encode each modality separately through its own
already-built encoder, then sum the results. Both are the same
dimension (cfg.latent_dim, since the state MLP path projects there
explicitly, and the rgb conv path's implicit output size must already
equal latent_dim for the unpatched single-modality case to work at all --
see this module's own note on model_size below) and already SimNorm-
normalized (each per-key encoder ends with a SimNorm activation). This is
a deliberately minimal, well-justified first fusion strategy -- summing
two already-normalized representations, adding zero new trainable
parameters. Concatenation plus a learned projection layer would be a
natural refinement to consider once training results exist to justify
the added complexity; not pursued now to avoid unverified speculative
complexity per this project's own stated engineering principles.

IMPORTANT model_size constraint, found while designing this fix: the
conv encoder (common/layers.py's conv()) has NO final projection layer --
its output size is whatever falls out of the raw conv stack for a 64x64
input, which for num_channels=32 (the default) is exactly 512. This only
matches cfg.latent_dim for the `model_size=5` preset (latent_dim=512) --
model_size=1 (latent_dim=128) would silently mismatch and crash. Any task
using 'rgb' at all MUST use model_size=5 (or a custom num_channels tuned
to match a different latent_dim) for this reason -- not a preference,
a hard architectural constraint.
"""

import torch

from common.world_model import WorldModel


def _fused_encode(self: WorldModel, obs, task):
    """Replacement for WorldModel.encode() -- see module docstring."""
    if self.cfg.multitask:
        obs = self.task_emb(obs, task)

    latents = []
    for k in self.cfg.obs_shape.keys():
        o = obs[k]
        if k == "rgb" and o.ndim == 5:
            latents.append(torch.stack([self._encoder[k](x) for x in o]))
        else:
            latents.append(self._encoder[k](o))

    if len(latents) == 1:
        return latents[0]
    z = latents[0]
    for extra in latents[1:]:
        z = z + extra
    return z


def apply_fusion_patch():
    """Call once, before constructing any TDMPC2/WorldModel instance."""
    WorldModel.encode = _fused_encode
