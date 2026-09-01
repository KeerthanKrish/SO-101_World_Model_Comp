# SPDX-License-Identifier: BSD-3-Clause
"""Gym-style (Isaac Lab DirectRLEnv) training environment for the SO-101
pick-and-place task -- the piece of infrastructure that was missing after
pickplace_reward.py: something with an actual reset()/step() interface
that a TD-MPC2 (or any other RL) training loop can drive.

Actions are joint-space, matching how the real arm is already driven
elsewhere in this project (teleop_bridge.py converts the leader arm's
action into sim joint positions directly, not Cartesian/IK targets) --
not IK end-effector targets, to stay consistent and avoid adding IK where
it isn't needed. Each action is a normalized per-joint delta in [-1, 1],
scaled by that joint's actual velocity limit (from the URDF/articulation
config) times the env's step duration -- so "1.0" means "move this joint
as fast as the real STS3215 servo actually can, for one control step," a
sim-to-real-grounded action scale rather than an arbitrary one.

Observations split into "policy" (proprioception, plus both camera images
if use_cameras=True -- wrist_camera and top_camera, the exact two cameras
physically mounted on the real arm/workspace, see
docs/real_camera_setup.md -- this is everything and exactly what a real
deployed policy would actually have) and "critic" (privileged cube/
gripper/target positions -- allowed here since this only matters for sim
training, see pickplace_reward.py's own module docstring on why; never
part of "policy", never deployed). This is Isaac Lab's built-in asymmetric
actor-critic observation pattern (DirectRLEnvCfg.state_space), not a
custom addition.

Reward/termination reuse the already-validated pickplace_reward.py
functions verbatim, called in a per-environment Python loop rather than
reimplemented in batched torch. Three pieces of per-env state are
persisted across steps beyond what Isaac Lab already tracks -- all three
reset on every env reset, all three required by pickplace_reward.py's own
compute_reward() signature:
  - `_was_holding`: is_holding() hysteresis (a real bug, found during a
    full correctness audit: the naive stateless height check it replaces
    incorrectly drops out of the transport phase the instant an
    already-held cube is lowered onto the target, since the target's
    resting height is below the lift-detection threshold by construction
    -- see docs/reward_function.md).
  - `_was_touched`: whether is_touching() has ever fired this episode,
    gating the one-time touch_bonus milestone (added in the 2026-08-31
    potential-based-shaping redesign -- see pickplace_reward.py's module
    docstring).
  - `_prev_dist`: the relevant distance (gripper-to-cube or cube-to-
    target, whichever phase was active) from the PREVIOUS step, needed to
    compute this step's potential-based shaping delta. NaN represents
    "no valid previous value" (right after a reset) -- converted to
    Python None at the compute_reward() call site, since NaN can't be
    stored in a bool/uniform way inside a plain torch tensor alongside a
    real "no value yet" sentinel otherwise. Critically, this must be
    reset to NaN on every env reset even though compute_reward() ALSO
    independently zeroes shaping across any holding-state phase
    transition -- without an explicit reset here, a new episode that
    happens to start in the same phase as the previous episode's last
    step (the overwhelmingly common case: gripper starts open, so
    holding=False on both sides) would silently carry over the previous
    episode's final distance and compute a bogus shaping delta between
    two entirely unrelated episodes.

This is a deliberate simplicity/performance tradeoff: pickplace_reward.py
was specifically built and empirically validated as a small,
dependency-free, scalar function (see docs/reward_function.md) --
reimplementing that same logic a second time in vectorized form would
risk exactly the kind of silent-drift bug this project already hit once
(see JAW_OFFSET_LOCAL's extraction into grasp_geometry.py) and isn't
worth it at the modest num_envs this project runs (a single desktop GPU,
not a large cluster). Revisit only if num_envs actually needs to scale
into the thousands and this loop is measured to be a real bottleneck.

Multi-env note: cube/gripper positions read from the simulator are in
true world coordinates, which include each cloned environment's origin
offset once num_envs > 1. pickplace_reward.py's target/table-bounds
constants are defined per-env-local (matching how this project used the
scene when num_envs was always 1). Every position is converted
world <-> local by subtracting/adding `scene.env_origins` at the
boundary -- easy to get wrong silently, so every call site below does it
explicitly rather than assuming it out.
"""

from __future__ import annotations

import math
import sys

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from envs.pickplace_reward import PickPlaceRewardConfig, compute_reward  # isort:skip
from robots.grasp_geometry import grasp_point_world  # isort:skip
from scenes.pickplace_scene import PickPlaceSceneBaseCfg, PickPlaceSceneCfg  # isort:skip

_JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@configclass
class PickPlaceEnvCfg(DirectRLEnvCfg):
    # -- timing --
    decimation = 2  # physics steps per env step: 0.01s physics dt -> 0.02s / 50Hz control rate
    # 500 steps at 0.02s/step = 10s. This is now the authoritative episode
    # length -- supersedes PickPlaceRewardConfig.max_episode_steps, which
    # was only ever a documented placeholder for whichever env eventually
    # got built (this one).
    episode_length_s = 10.0
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=0.01, render_interval=2)

    # -- scene (built in __post_init__, depends on use_cameras/num_envs) --
    # IMPORTANT: pass these to the PickPlaceEnvCfg(...) constructor, e.g.
    # PickPlaceEnvCfg(use_cameras=True, num_envs=8) -- do NOT set them as
    # attributes on an already-constructed cfg. __post_init__ builds
    # self.scene from these values but only runs once, at construction;
    # mutating them afterward silently has no effect on the scene actually
    # used (caught during this env's own smoke test).
    use_cameras = False
    # 16 is a measured hard ceiling for use_cameras=True on this GPU (RTX
    # 5060 Ti), not a guess -- 24 and 32 both failed immediately with RTX
    # descriptor-set/ParameterBlock exhaustion (a fixed internal resource
    # pool, not a graceful slowdown). See
    # sim/scripts/num_envs_scaling_test.py and docs/training_env.md for
    # the full sweep. State-only (use_cameras=False) scales far higher
    # (~1000+) but is a diagnostic number only -- the real deployed policy
    # always needs both cameras, so 16 is the number that actually governs
    # real training runs.
    num_envs = 16
    env_spacing = 2.0

    # -- spaces --
    # Normalized per-joint delta in [-1, 1] -- see module docstring for scaling.
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,))
    # "policy": what a real deployed policy would see (proprio [+ wrist image]).
    observation_space = {"proprio": 12}
    # "critic": privileged ground truth, sim-training-only.
    state_space = {"privileged": 12}

    reward_cfg = PickPlaceRewardConfig()

    # Cube start-position sampling -- the TRAIN region from the empirical
    # reachability sweep (sim/scripts/reachability_sweep.py, 2026-08-30):
    # a rectangle solidly inside the confirmed-reachable core. The
    # held-out region for generalization testing is deliberately NOT
    # sampled here -- see docs/evaluation_plan.md checklist item 1 for the
    # reserved test points.
    cube_x_range = (0.05, 0.25)
    cube_y_range = (-0.20, 0.20)

    def __post_init__(self):
        scene_cls = PickPlaceSceneCfg if self.use_cameras else PickPlaceSceneBaseCfg
        self.scene = scene_cls(num_envs=self.num_envs, env_spacing=self.env_spacing, replicate_physics=True)
        if self.use_cameras:
            # Both cameras the real arm actually carries -- see
            # docs/real_camera_setup.md. A real deployed policy has no
            # privileged state, only what its cameras see, so both need to
            # be part of "policy", not just the wrist view (an earlier
            # version of this env only wired in wrist_camera, an oversight
            # caught before any real training started).
            # Resolutions from pickplace_scene.py: wrist_camera 480x270,
            # top_camera 960x540.
            self.observation_space = dict(self.observation_space)
            self.observation_space["wrist_rgb"] = [270, 480, 3]
            self.observation_space["top_rgb"] = [540, 960, 3]


class PickPlaceEnv(DirectRLEnv):
    cfg: PickPlaceEnvCfg

    def __init__(self, cfg: PickPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        self.cube = self.scene["cube"]
        self._joint_indices = [self.robot.data.joint_names.index(j) for j in _JOINT_ORDER]

        ee_cfg = SceneEntityCfg("robot", body_names=["gripper_frame_link"])
        ee_cfg.resolve(self.scene)
        self._ee_body_id = ee_cfg.body_ids[0]

        # Per-joint max delta per control step, grounded in the real
        # servo's actual velocity limit -- see module docstring.
        self._max_delta = self.robot.data.joint_vel_limits[0, self._joint_indices] * self.step_dt
        # (num_envs, 6, 2), order [lower, upper].
        self._soft_limits = self.robot.data.soft_joint_pos_limits[:, self._joint_indices]

        self._joint_pos_target = self.robot.data.default_joint_pos[:, self._joint_indices].clone()

        self._reward_cfg = self.cfg.reward_cfg
        self._target_pos_local = torch.tensor(self._reward_cfg.target_pos, device=self.device)

        # Cached each step in _get_dones() (called first in the base
        # class's step()), reused by _get_rewards() -- both need the same
        # PRE-reset snapshot. _get_observations() runs its own fresh
        # computation afterward, since by then some envs may have just
        # been reset -- see module docstring.
        self._cached_reward = torch.zeros(self.num_envs, device=self.device)

        # Per-env persisted state required by pickplace_reward.compute_reward()
        # -- see module docstring for what each one is and why it's needed.
        self._was_holding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._was_touched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_dist = torch.full((self.num_envs,), float("nan"), device=self.device)

    def _setup_scene(self):
        # Everything (robot, cube, table, ground, light, optional cameras)
        # is already declared in the scene cfg (PickPlaceSceneBaseCfg /
        # PickPlaceSceneCfg) and spawned automatically by the base class's
        # InteractiveScene(self.cfg.scene) call before this runs -- nothing
        # extra to add here.
        pass

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        delta = actions.clamp(-1.0, 1.0) * self._max_delta
        target = self._joint_pos_target + delta
        self._joint_pos_target = torch.clamp(target, self._soft_limits[..., 0], self._soft_limits[..., 1])

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._joint_pos_target, joint_ids=self._joint_indices)

    def _grasp_points_local(self) -> torch.Tensor:
        """Per-env grasp point, in EACH ENV'S OWN LOCAL frame -- (num_envs, 3)."""
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_id]
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_id]
        out = torch.zeros_like(ee_pos_w)
        for i in range(self.num_envs):
            out[i] = torch.tensor(
                grasp_point_world(ee_pos_w[i].tolist(), ee_quat_w[i].tolist()), device=self.device
            )
        return out - self.scene.env_origins

    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos[:, self._joint_indices]
        joint_vel = self.robot.data.joint_vel[:, self._joint_indices]
        policy_obs = {"proprio": torch.cat([joint_pos, joint_vel], dim=-1)}
        if self.cfg.use_cameras:
            policy_obs["wrist_rgb"] = self.scene["wrist_camera"].data.output["rgb"][..., :3]
            policy_obs["top_rgb"] = self.scene["top_camera"].data.output["rgb"][..., :3]

        gripper_pos_local = self._grasp_points_local()
        cube_pos_local = self.cube.data.root_pos_w - self.scene.env_origins
        cube_lin_vel = self.cube.data.root_lin_vel_w
        target_pos = self._target_pos_local.expand(self.num_envs, 3)
        critic_obs = {
            "privileged": torch.cat([cube_pos_local, cube_lin_vel, gripper_pos_local, target_pos], dim=-1)
        }

        return {"policy": policy_obs, "critic": critic_obs}

    def _get_states(self):
        # Not called by this Isaac Lab version's step()/reset() (verified
        # against the installed source) -- "critic" is already included
        # directly in _get_observations() above, which is what's actually
        # used. Implemented anyway for forward/external-tool compatibility,
        # at zero extra cost (delegates to the same computation).
        return self._get_observations().get("critic")

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        gripper_pos_local = self._grasp_points_local()
        cube_pos_local = self.cube.data.root_pos_w - self.scene.env_origins
        cube_lin_vel = self.cube.data.root_lin_vel_w
        joint_pos = self.robot.data.joint_pos[:, self._joint_indices]
        joint_vel = self.robot.data.joint_vel[:, self._joint_indices]

        rewards = torch.zeros(self.num_envs, device=self.device)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        placed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for i in range(self.num_envs):
            prev_d = self._prev_dist[i].item()
            reward, info = compute_reward(
                gripper_pos_local[i].tolist(),
                cube_pos_local[i].tolist(),
                cube_lin_vel[i].tolist(),
                joint_pos[i, -1].item(),  # "gripper" joint is last in _JOINT_ORDER
                joint_vel[i].tolist(),
                bool(self._was_holding[i].item()),
                bool(self._was_touched[i].item()),
                None if math.isnan(prev_d) else prev_d,
                self._reward_cfg,
            )
            rewards[i] = reward
            terminated[i] = info["placed"] or info["failed"]
            self._was_holding[i] = info["holding"]
            self._was_touched[i] = info["touched"]
            self._prev_dist[i] = info["dist"]
            placed[i] = info["placed"]
            failed[i] = info["failed"]

        self._cached_reward = rewards
        # Exposed via self.extras so external callers (e.g. the TD-MPC2
        # adapter, sim/scripts/tdmpc2_pickplace_env.py) can distinguish a
        # SUCCESSFUL termination from a FAILED one -- terminated alone
        # doesn't say which, and this was a real gap caught while writing
        # that adapter (self.extras is never otherwise populated by
        # DirectRLEnv's base step(), it stays an empty dict by default).
        self.extras["placed"] = placed
        self.extras["failed"] = failed

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        return self._cached_reward

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)
        default_joint_pos = self.robot.data.default_joint_pos[env_ids]
        default_joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, None, env_ids)
        self._joint_pos_target[env_ids] = default_joint_pos[:, self._joint_indices]
        self._was_holding[env_ids] = False
        self._was_touched[env_ids] = False
        # NaN, not 0.0 -- 0.0 is a real, meaningful distance (already at
        # the cube/target) and must never be mistaken for "no previous
        # value yet." See module docstring for why this reset is required
        # even though compute_reward() also independently zeroes shaping
        # across a holding-state phase transition -- a fresh episode
        # starting in the SAME phase as the last episode's final step
        # (the common case) needs this explicit reset to avoid comparing
        # against a stale, unrelated distance.
        self._prev_dist[env_ids] = float("nan")

        cube_x = sample_uniform(self.cfg.cube_x_range[0], self.cfg.cube_x_range[1], (n,), self.device)
        cube_y = sample_uniform(self.cfg.cube_y_range[0], self.cfg.cube_y_range[1], (n,), self.device)

        default_cube_root_state = self.cube.data.default_root_state[env_ids].clone()
        # x/y overwritten with the sampled train-region position; z (resting
        # height) kept from the scene's own default -- no need to duplicate
        # that constant here. All three then get the per-env world origin
        # offset added, matching how every multi-env Isaac Lab reset works
        # (see cartpole_env.py in isaaclab_tasks for the same pattern).
        default_cube_root_state[:, 0] = cube_x
        default_cube_root_state[:, 1] = cube_y
        default_cube_root_state[:, 0:3] += self.scene.env_origins[env_ids]

        self.cube.write_root_pose_to_sim(default_cube_root_state[:, :7], env_ids)
        self.cube.write_root_velocity_to_sim(torch.zeros_like(default_cube_root_state[:, 7:]), env_ids)
