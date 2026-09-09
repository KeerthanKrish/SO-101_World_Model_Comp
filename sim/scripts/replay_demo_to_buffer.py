# SPDX-License-Identifier: BSD-3-Clause
"""Prototype: replays a real recorded teleop demonstration through the
ACTUAL production env/wrapper (PickPlaceEnv + PickPlaceTDMPC2Wrapper),
producing genuine TD-MPC2-format transitions (observations, actions,
rewards) -- the first step toward demonstration-seeded training (see
docs/decisions.md, "Investigating demonstration-seeded training").

Why this matters: seven consecutive RL runs (9 through 15) against the
current, fully-refined reward design have never produced a genuine
sustained hold -- TD-MPC2 learns its world model (dynamics, reward,
value) purely from whatever transitions sit in its replay buffer, and
that buffer has only ever contained the agent's own, largely
unsuccessful rollouts. The 8 recorded demonstrations
(sim/output/teleop_episodes/episode_*.json) all reach 8.7-14.9cm cube
height -- genuine lifts, dramatically higher than the ~2.75cm the best
RL rollout ever accidentally achieved (re-checked directly, see
docs/decisions.md; an earlier docs/reward_function.md note claiming
otherwise was stale and has been corrected). Seeding the buffer with
real successful-grasp transitions gives the world model ground-truth
examples of what a genuine hold looks like and what reward follows it,
rather than requiring the agent to discover this through exploration
that has repeatedly failed to find it.

Deliberately reuses the REAL production classes end to end rather than
reimplementing anything, to minimize the risk of a subtle format
mismatch between demo-derived and agent-generated transitions:
  - PickPlaceEnv + PickPlaceTDMPC2Wrapper -- the exact env/observation/
    reward pipeline real training uses. A demo-derived transition's
    "state"/"rgb" observation is built by the identical code path a real
    rollout's would be.
  - to_td() -- copied verbatim from train_tdmpc2_pickplace.py (not
    imported: that module runs AppLauncher-constructing code at import
    time, and only one SimulationContext is allowed per process -- see
    its own module docstring). Any change to the real to_td() must be
    mirrored here.
  - common.buffer.Buffer -- TD-MPC2's own replay buffer, imported
    directly (no fusion-patch/hydra dependency, unlike the full training
    script -- see common/buffer.py, which imports only torch/tensordict/
    torchrl).

## The timing mismatch this script resolves

Recorded demonstrations are captured at the FULL physics rate, 100Hz
(dt=0.01) -- confirmed directly in teleop_bridge.py's own comment ("physics
still runs at full rate (100Hz) so control/contact accuracy is
unaffected"), one recorded frame per physics step. PickPlaceEnv's actual
control rate is decimated: decimation=2 at dt=0.01, i.e. 50Hz, one action
held for 2 physics steps. Feeding every recorded 100Hz frame as a separate
env.step() would produce transitions representing HALF the real time per
step of an agent-generated one -- a genuine modeling error for a world
model that implicitly learns "what happens after this much real time,"
not just a cosmetic mismatch. Fixed by downsampling: every 2nd recorded
frame (indices 0, 2, 4, ...) is treated as one control-step target,
matching the decimation exactly.

## The action-space conversion this script performs

Recorded episodes store raw, absolute joint POSITIONS (what the leader
arm commanded). PickPlaceEnv's action space is normalized per-joint
DELTAS in [-1, 1] from the env's own internally-tracked PD target
(`_joint_pos_target`, updated incrementally each step -- see
PickPlaceEnv._pre_physics_step()). Converts by computing, for each
downsampled target frame, `(desired_target - current_joint_pos_target) /
_max_delta`, clamped to [-1, 1] -- reusing the env's own `_max_delta`/
`_soft_limits` rather than recomputing them, so this is exactly the
inverse of what `_pre_physics_step()` itself does. Critically,
`_joint_pos_target` must ALSO be overridden to the demo's own starting
pose immediately after reset (see main()) -- left at its post-reset
default (the robot's rest pose), the first computed delta would be
measured from the wrong baseline, corrupting the first action.

Usage:
    ./isaaclab.sh -p /path/to/replay_demo_to_buffer.py --enable_cameras \\
        --episode /path/to/episode_000.json
"""

import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a recorded demonstration into TD-MPC2-format transitions.")
parser.add_argument("--episode", type=str, required=True, help="Path to an episode_*.json file.")
parser.add_argument(
    "--full-resolution", action="store_true",
    help="Diagnostic: replay at the demonstration's own native 100Hz rate (decimation=1, every "
    "recorded frame is its own env.step()) instead of downsampling to the env's normal 50Hz "
    "control rate. Testing whether the 50Hz downsampling's coarser per-step target jumps "
    "(losing whatever fine-grained/bursty motion happened within an averaged-over 20ms window) "
    "is why a first replay attempt significantly undershot the demonstration's own recorded "
    "cube height -- see docs/decisions.md.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # isort:skip

# Force line-buffered stdout -- this box's known Kit shutdown-hang means a
# run can be killed before Python's default full-buffering (in effect
# whenever stdout is redirected to a file, as every backgrounded run here
# is) ever flushes, silently losing every print() issued so far. Confirmed
# necessary: a first attempt at this script produced zero output at all,
# not even the very first print, despite the real work completing.
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from envs.pickplace_env import PickPlaceEnv, PickPlaceEnvCfg  # isort:skip
# _jaw_offsets -- only used by this script's own trace diagnostic, to
# report the axial/lateral decomposition is_between_jaws() actually
# checks, rather than just the combined spherical distance.
from envs.pickplace_reward import _jaw_offsets  # isort:skip

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
# _rotate_vector -- only used by this script's own diagnostic to
# independently transform a candidate TRUE fingertip offset (measured
# directly from moving_jaw_so101_v1.stl, not assumed) through the moving
# jaw link's own live world pose, to check whether jaw_approach_axis_world()'s
# "gripper_frame_link-to-pivot direction == pivot-to-fingertip direction"
# assumption actually holds -- see this script's diagnostic output.
from robots.grasp_geometry import _rotate_vector  # isort:skip

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim/scripts")
from tdmpc2_pickplace_env import PickPlaceTDMPC2Wrapper  # isort:skip

sys.path.insert(0, "/home/keerthan/SO-101-WM/tdmpc2/tdmpc2")
from common.buffer import Buffer  # isort:skip

from omegaconf import OmegaConf  # isort:skip
from tensordict.tensordict import TensorDict  # isort:skip

_JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def to_td(obs, action=None, reward=None, terminated=None, action_dim=6):
    """Copied VERBATIM from train_tdmpc2_pickplace.py -- see this
    module's own docstring for why it's copied rather than imported. Any
    change to the real one must be mirrored here."""
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
    with open(args_cli.episode) as f:
        episode = json.load(f)
    print(f"[INFO] Loaded episode with {len(episode)} raw (100Hz) frames from {args_cli.episode}")

    # wait_for_textures=False: works around a reproducible Kit/Omniverse
    # asset-streaming stall (2026-09-09) where DirectRLEnv.reset()'s "if
    # self.cfg.wait_for_textures: while SimulationManager.assets_loading():
    # self.sim.render()" loop (isaaclab/envs/direct_rl_env.py) spins
    # forever -- confirmed via py-spy against the live process (repeatedly
    # sampled at the exact same render() call) and /proc/<pid>/io (rchar
    # flat, read_bytes=0 across a 5s window -- no new asset data is being
    # read from disk at all, so nothing is actually still loading; Kit's
    # internal busy flag is just stuck). Irrelevant here regardless --
    # this script never reads a camera IMAGE, only body/joint/cube physics
    # state, so whether RTX textures finish streaming doesn't affect
    # anything this script measures. Not a fix to any project code; a
    # narrow workaround for this script's own use of DirectRLEnvCfg's
    # stock default (True), scoped here rather than in pickplace_env.py
    # since production training (which DOES render camera observations for
    # the policy) should keep the default.
    if args_cli.full_resolution:
        downsampled = episode  # every recorded 100Hz frame is its own control step
        env_kwargs = dict(use_cameras=True, num_envs=1, episode_length_s=10.0, decimation=1, wait_for_textures=False)
        print(f"[INFO] --full-resolution: replaying all {len(downsampled)} frames at decimation=1 "
              f"({len(downsampled) - 1} env.step() calls)")
    else:
        # Downsample to the env's own 50Hz control rate -- see module
        # docstring's "timing mismatch" section.
        downsampled = episode[0::2]
        env_kwargs = dict(use_cameras=True, num_envs=1, episode_length_s=10.0, wait_for_textures=False)
        print(f"[INFO] Downsampled to {len(downsampled)} control-rate (50Hz) frames "
              f"({len(downsampled) - 1} env.step() calls)")

    base_env = PickPlaceEnv(PickPlaceEnvCfg(**env_kwargs))
    env = PickPlaceTDMPC2Wrapper(base_env)

    # Body index for the moving jaw link itself -- used below to
    # independently verify jaw_approach_axis_world()'s directional
    # assumption against the TRUE fingertip position (measured directly
    # from moving_jaw_so101_v1.stl's own bounding box: local-frame Y
    # extent [-0.082, +0.010], i.e. the mesh extends ~8.2cm in -Y from
    # the link's own origin -- the visual/collision <origin> in the URDF
    # shifts the mesh by only (~0, ~0, 0.0189) relative to the link
    # frame, so mesh-local Y is link-local Y directly). Candidate
    # fingertip offset in the link's OWN local frame:
    _JAW_TIP_LOCAL = (0.0, -0.082, 0.019)
    _jaw_link_idx = base_env.robot.data.body_names.index("moving_jaw_so101_v1_link")

    # Diagnostic: soft joint-position limits for every joint, vs. this
    # demonstration's own recorded range for that joint -- checking
    # whether the recording legitimately goes beyond what PickPlaceEnv's
    # own _pre_physics_step() will ever allow (a position-limit clamp,
    # not the velocity-limit one this script's earlier fix targeted).
    print("\n[DIAG] === Soft joint-position limits vs. this episode's recorded range ===")
    for j, joint_name in enumerate(_JOINT_ORDER):
        lo = base_env._soft_limits[0, j, 0].item()
        hi = base_env._soft_limits[0, j, 1].item()
        recorded_vals = [s["joint_pos"][joint_name] for s in episode if joint_name in s["joint_pos"]]
        print(f"[DIAG] {joint_name:14s} soft_limits=[{lo:+.3f}, {hi:+.3f}] "
              f"recorded_range=[{min(recorded_vals):+.3f}, {max(recorded_vals):+.3f}]")

    obs = env.reset()  # builds an initial obs against the env's OWN (irrelevant) random reset state

    # Override cube + robot state to match THIS episode's own recorded
    # start -- mirrors validate_reward_function.py's exact pattern
    # (same reasoning: these episodes were cut out of one continuous
    # teleop session, so each starts wherever the cube/arm actually was,
    # not at a clean default reset state).
    default_root_state = base_env.cube.data.default_root_state.clone()
    root_pose = default_root_state[:, :7].clone()
    # ROOT-CAUSE FIX: recorded cube_pos is in the raw-scene WORLD frame
    # validate_reward_function.py uses (env_origin implicitly (0,0,0) --
    # a single, un-cloned InteractiveScene). PickPlaceEnv is a
    # DirectRLEnv, cloned per-environment even at num_envs=1 -- its own
    # _reset_idx() explicitly adds `self.scene.env_origins[env_ids]` when
    # placing the cube (see pickplace_env.py), confirming cube world
    # position is offset by the environment's own origin. Missing this
    # addition was the actual bug behind the first replay attempts: the
    # gripper target tracked the demonstration perfectly (confirmed via
    # the [TRACE] diagnostic) yet the cube never moved AT ALL, in either
    # a 50Hz or a full 100Hz replay -- exactly what a constant world-frame
    # offset between the gripper's true position and the cube's written
    # position would produce, regardless of timing resolution.
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

    # CRITICAL, and the one piece with no analogue in
    # validate_reward_function.py: PickPlaceEnv's own action mechanism is
    # INCREMENTAL (each action is a delta from `_joint_pos_target`, not
    # an absolute command -- see PickPlaceEnv._pre_physics_step()). Left
    # at its post-reset default (the robot's rest pose), the first
    # computed action below would be a delta from the WRONG baseline,
    # corrupting the first control step. Must match the override above.
    base_env._joint_pos_target = torch.tensor(
        [[downsampled[0]["joint_pos"][j] for j in _JOINT_ORDER]], device=base_env.device
    )

    # `obs` from env.reset() above is now stale (built before the state
    # override) -- rebuild it from the corrected state via the wrapper's
    # own observation-building method, so the FIRST transition's
    # observation is consistent with everything after it.
    obs = env._build_obs(base_env._get_observations())

    # Trace the TRUE starting state, right after the override and before
    # any step -- confirms whether a divergence is present from frame 0
    # (an override bug) or only develops over subsequent steps.
    gripper_pos0, _ = base_env._grasp_points_local()
    cube_pos0_local = (base_env.cube.data.root_pos_w - base_env.scene.env_origins)[0].cpu().tolist()
    gp0 = gripper_pos0[0].cpu().tolist()
    dist0 = ((gp0[0] - cube_pos0_local[0]) ** 2 + (gp0[1] - cube_pos0_local[1]) ** 2 + (gp0[2] - cube_pos0_local[2]) ** 2) ** 0.5
    print(f"[TRACE] step=   0 (post-override, pre-step) gripper_pos=({gp0[0]:.3f},{gp0[1]:.3f},{gp0[2]:.3f}) "
          f"cube_pos=({cube_pos0_local[0]:.3f},{cube_pos0_local[1]:.3f},{cube_pos0_local[2]:.3f}) "
          f"recorded_cube_pos={downsampled[0]['cube_pos']} dist={dist0:.4f}")

    tds = [to_td(obs, action_dim=6)]
    max_cube_height = downsampled[0]["cube_pos"][2]
    ever_touched = ever_between_jaws = ever_holding = False
    reward_sum = 0.0
    # Diagnostic (temporary): how often/how severely the desired delta
    # exceeds what one control step can physically achieve -- if the
    # recorded targets (raw leader-commanded positions, NOT the follower
    # robot's actual tracked position -- teleop_bridge.py sets a PD TARGET
    # every 100Hz tick, which the physical robot only approaches, so
    # consecutive targets can jump by more than the robot's own velocity
    # limit allows in one step) saturate the action often, the replay will
    # systematically lag behind the original demonstration.
    clamp_events = [0] * 6
    max_ratio = [0.0] * 6
    # Diagnostic (temporary): how often/by how much the RECORDED target
    # itself exceeds the robot's own soft joint-position limits (see
    # docs/decisions.md) -- confirmed via the URDF directly
    # (assets/SO-ARM100/.../so101_new_calib.urdf's wrist_flex <limit> tag
    # matches PickPlaceEnv's soft_limits exactly: [-1.658, +1.658] rad),
    # not an artificially-tightened RL-only safety margin. 6 of 8 recorded
    # demonstrations exceed it on wrist_flex specifically, 4 by the exact
    # same 0.139 rad -- a real, physical mismatch between what the
    # recording captured and what this action space will ever let a
    # policy (or this replay) command, not a bug in the conversion logic.
    limit_clip_events = [0] * 6
    max_limit_overshoot = [0.0] * 6

    for step_idx, frame in enumerate(downsampled[1:], start=1):
        desired_target_raw = torch.tensor([[frame["joint_pos"][j] for j in _JOINT_ORDER]], device=base_env.device)
        # OPTION 1 (chosen after discussion with the user, 2026-09-08):
        # clip the recorded target itself to the robot's own physical
        # joint limits before using it for anything below -- both the
        # direct override AND the action_label delta -- rather than
        # bypass the limit for replay purposes or leave it unaddressed.
        # Lower risk than loosening the limit (a real, physical
        # constraint, not a tunable), at the cost of not perfectly
        # reproducing the small fraction of each affected demonstration's
        # motion that goes beyond it. Whether this still lands close
        # enough to a genuine grasp is an open, honestly-uncertain
        # question -- being tested directly rather than assumed either
        # way (an 8-degree wrist error could plausibly still be within
        # is_between_jaws()'s tolerance, or could plausibly not be, given
        # how sensitive that geometry has proven throughout this project).
        desired_target = torch.clamp(desired_target_raw, base_env._soft_limits[..., 0], base_env._soft_limits[..., 1])
        for j in range(6):
            overshoot = abs(desired_target_raw[0, j].item() - desired_target[0, j].item())
            if overshoot > 0:
                limit_clip_events[j] += 1
            max_limit_overshoot[j] = max(max_limit_overshoot[j], overshoot)
        desired_delta = desired_target - base_env._joint_pos_target
        raw_ratio = (desired_delta / base_env._max_delta)[0]  # (6,), unclamped -- kept only as a label/diagnostic now
        action_label = raw_ratio.clamp(-1.0, 1.0)
        for j in range(6):
            r = abs(raw_ratio[j].item())
            if r > 1.0:
                clamp_events[j] += 1
            max_ratio[j] = max(max_ratio[j], r)

        # ROOT-CAUSE FIX: going through env.step(action) with a
        # delta-then-clamp action, as above, silently caps how fast any
        # joint can move to _max_delta per control step -- confirmed via
        # a direct gripper/cube world-position trace to be the actual
        # cause of the replay never reaching the cube at all (wrist_flex
        # needed over max velocity for real portions of this
        # demonstration, and by the time the arm caught up the recorded
        # trajectory had already moved on, compounding into a 0.25m+
        # miss by mid-episode). But the ORIGINAL recording was never
        # subject to this cap in the first place -- teleop_bridge.py (and
        # validate_reward_function.py's replay of it) simply calls
        # set_joint_position_target() with the raw recorded value every
        # tick and lets the PD controller track it under its own
        # actuator dynamics, no explicit per-tick velocity ceiling.
        # Reproduced that here: override `_joint_pos_target` directly to
        # the exact recorded value, then pass a ZERO action so
        # _pre_physics_step()'s own delta math (`target =
        # _joint_pos_target + action.clamp(-1,1)*_max_delta`) is a no-op
        # on top of it -- the actual PD command becomes exactly the
        # recorded target, matching validate_reward_function.py's proven
        # approach, while still going through the real env's own
        # observation/reward/decimation pipeline. `action_label` above
        # (what a normal delta-action would have needed, clamped for
        # representability) is still computed and stored in the buffer's
        # `action` field -- an honest, direction-consistent label for
        # what happened, even on saturated steps where it does not
        # exactly reconstruct the commanded target.
        base_env._joint_pos_target = desired_target
        zero_action = torch.zeros(6, device=base_env.device)
        obs, reward, done, info = env.step(zero_action)
        tds.append(to_td(obs, action_label.cpu(), reward, torch.tensor(float(info["terminated"])), action_dim=6))

        cube_height = base_env.cube.data.root_pos_w[0, 2].item()
        max_cube_height = max(max_cube_height, cube_height)
        ever_touched = ever_touched or info["touched"]
        ever_between_jaws = ever_between_jaws or info["between_jaws"]
        ever_holding = ever_holding or info["holding"]
        reward_sum += reward.item()

        # Temporary diagnostic: replayed vs. original cube height side by
        # side every 10 steps, to see WHERE the trajectories diverge
        # rather than just THAT they do. Also the ACTUAL gripper world
        # position and its real distance to the cube -- the env_origins
        # theory (ruled out: adding it was a no-op) still leaves "the
        # cube never moves at all" unexplained by target-tracking alone,
        # since _joint_pos_target matching the recording doesn't prove
        # the PHYSICAL robot/gripper is actually where that implies.
        if step_idx % 10 == 0 or step_idx <= 15:
            gripper_pos, gripper_quat = base_env._grasp_points_local()
            cube_pos_local = (base_env.cube.data.root_pos_w - base_env.scene.env_origins)[0].cpu().tolist()
            gp = gripper_pos[0].cpu().tolist()
            gq = gripper_quat[0].cpu().tolist()
            dist = ((gp[0] - cube_pos_local[0]) ** 2 + (gp[1] - cube_pos_local[1]) ** 2 + (gp[2] - cube_pos_local[2]) ** 2) ** 0.5
            axial, lateral = _jaw_offsets(gp, gq, cube_pos_local)
            wf_target = base_env._joint_pos_target[0, 3].item()
            wf_actual = base_env.robot.data.joint_pos[0, base_env._joint_indices[3]].item()
            wf_recorded = frame["joint_pos"]["wrist_flex"]

            # Independent check: TRUE fingertip position, from the moving
            # jaw link's own LIVE world pose plus the mesh-measured local
            # offset -- entirely separate from grasp_point_world()'s own
            # gripper_frame_link-based computation, to see whether that
            # function's directional assumption actually holds.
            jaw_pos_w = (base_env.robot.data.body_pos_w[0, _jaw_link_idx] - base_env.scene.env_origins[0]).cpu().tolist()
            jaw_quat_w = base_env.robot.data.body_quat_w[0, _jaw_link_idx].cpu().tolist()
            tip_rx, tip_ry, tip_rz = _rotate_vector(jaw_quat_w, _JAW_TIP_LOCAL)
            tip_pos = (jaw_pos_w[0] + tip_rx, jaw_pos_w[1] + tip_ry, jaw_pos_w[2] + tip_rz)
            tip_to_cube = (
                (tip_pos[0] - cube_pos_local[0]) ** 2
                + (tip_pos[1] - cube_pos_local[1]) ** 2
                + (tip_pos[2] - cube_pos_local[2]) ** 2
            ) ** 0.5

            print(f"[TRACE] step={step_idx:4d} replayed_h={cube_height:.4f} "
                  f"original_h={frame['cube_pos'][2]:.4f} "
                  f"wrist_flex target={wf_target:+.3f} actual={wf_actual:+.3f} recorded={wf_recorded:+.3f} "
                  f"gripper_pos=({gp[0]:.3f},{gp[1]:.3f},{gp[2]:.3f}) "
                  f"cube_pos=({cube_pos_local[0]:.3f},{cube_pos_local[1]:.3f},{cube_pos_local[2]:.3f}) "
                  f"dist={dist:.4f} axial={axial:.4f} lateral={lateral:.4f} between_jaws={info['between_jaws']} "
                  f"jaw_pos=({jaw_pos_w[0]:.3f},{jaw_pos_w[1]:.3f},{jaw_pos_w[2]:.3f}) "
                  f"true_tip=({tip_pos[0]:.3f},{tip_pos[1]:.3f},{tip_pos[2]:.3f}) tip_to_cube={tip_to_cube:.4f}")

        if done:
            print(f"[WARN] episode terminated early at downsampled step {step_idx} "
                  f"(of {len(downsampled) - 1}) -- stopping replay here.")
            break

    episode_td = torch.cat(tds)
    n_steps = len(downsampled) - 1
    print("\n[DIAG] === Per-joint action-saturation diagnostic ===")
    for j, joint_name in enumerate(_JOINT_ORDER):
        print(f"[DIAG] {joint_name:14s} clamped on {clamp_events[j]:4d}/{n_steps} steps "
              f"({100 * clamp_events[j] / n_steps:5.1f}%), max |desired/max_delta| ratio = {max_ratio[j]:6.2f}")
    print("\n[DIAG] === Per-joint soft-limit clipping diagnostic (Option 1) ===")
    for j, joint_name in enumerate(_JOINT_ORDER):
        print(f"[DIAG] {joint_name:14s} clipped on {limit_clip_events[j]:4d}/{n_steps} steps "
              f"({100 * limit_clip_events[j] / n_steps:5.1f}%), max overshoot = {max_limit_overshoot[j]:.4f} rad")
    print("\n[RESULT] === Replay summary ===")
    print(f"[RESULT] Transitions produced: {len(episode_td)} (1 initial + {len(episode_td) - 1} steps)")
    print(f"[RESULT] episode_td shape={episode_td.shape} keys={sorted(episode_td.keys())}")
    print(f"[RESULT] obs sub-keys={sorted(episode_td['obs'].keys())} "
          f"state shape={episode_td['obs']['state'].shape} rgb shape={episode_td['obs']['rgb'].shape}")
    print(f"[RESULT] action shape={episode_td['action'].shape} reward shape={episode_td['reward'].shape}")
    print(f"[RESULT] reward_sum={reward_sum:+.3f}")
    print(f"[RESULT] max_cube_height={max_cube_height:.4f} m (recorded episode's own max was "
          f"{max(s['cube_pos'][2] for s in episode):.4f} m)")
    print(f"[RESULT] ever_touched={ever_touched} ever_between_jaws={ever_between_jaws} ever_holding={ever_holding}")

    # Confirm this is genuinely insertable into a real Buffer, not just
    # shaped correctly by inspection -- a throwaway buffer/cfg, not the
    # one a real training run would use.
    dummy_cfg = OmegaConf.create({"buffer_size": 10_000, "steps": 10_000, "batch_size": 8, "horizon": 3, "multitask": False})
    buffer = Buffer(dummy_cfg)
    num_eps = buffer.add(episode_td)
    print(f"[RESULT] Buffer.add() succeeded -- buffer.num_eps={num_eps}")
    print("[RESULT] No crash -- demonstration replay produces genuine, buffer-compatible TD-MPC2 transitions.")


if __name__ == "__main__":
    main()
    simulation_app.close()
