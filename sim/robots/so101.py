# SPDX-License-Identifier: BSD-3-Clause
"""Isaac Lab articulation configuration for the SO-101 robot arm.

USD asset converted from TheRobotStudio/SO-ARM100's `so101_new_calib.urdf`
(see docs/so101_asset_notes.md for source and conversion details).

Joint torque/velocity limits are estimated from the Feetech STS3215 servo's
published specs (~19.5 kgf*cm peak torque =~ 1.9 N*m, ~45 RPM no-load speed
=~ 4.7 rad/s at 7.4V, 1/345 gearing on the follower arm) since the raw URDF's
placeholder limits (effort=10, velocity=10 for every joint) do not reflect
the real servo. These are estimates, not measured values -- validate against
the real arm (e.g. observed max joint velocity during teleop) once the
physical arm is connected, and tighten if sim behavior looks unrealistic.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

SO101_USD_PATH = "/home/keerthan/SO-101-WM/assets/usd/so101/so101.usd"

# Estimated from STS3215 datasheet figures -- see module docstring.
_STS3215_EFFORT_LIMIT = 1.9  # N*m, peak torque estimate
_STS3215_VELOCITY_LIMIT = 4.7  # rad/s, no-load speed estimate at 7.4V

SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=SO101_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            # Was 5.0 -- when the gripper commands "fully closed" against an
            # object too big to actually close on, position control keeps
            # pushing and drives deep interpenetration; PhysX then resolves
            # it by "ejecting" the object at up to this speed, seen as the
            # cube violently bouncing out of the gripper. Lower cap = gentler
            # separation instead of a launch.
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            # Was 8 -- community reports (Isaac Lab discussions/forums) note
            # grasping/contact-rich tasks specifically benefit from higher
            # position iteration counts (e.g. 10 vs. a default of 4) to
            # prevent penetration.
            solver_position_iteration_count=12,
            # Was 0 -- every single run's log carried a PhysX warning saying
            # exactly this causes poor contact accuracy and recommending 1-2.
            # Ignored it until teleop made the resulting clip-through
            # (gripper visually passing into the cube instead of colliding)
            # obvious and unmistakable.
            solver_velocity_iteration_count=2,
        ),
        # Tried adding a robot-side physics_material override here too, but
        # `physics_material` isn't a plain field on UsdFileCfg -- it only
        # exists on UsdFileWithCompliantContactCfg, which needs an explicit
        # per-prim path (not a simple whole-asset override) and uses a
        # different spawn function. Not worth the extra complexity: the
        # cube's own physics_material (see pickplace_scene.py) uses
        # friction_combine_mode="max", which should make the *effective*
        # contact friction high regardless of the gripper's own (default)
        # material, since PhysX uses the higher-priority combine mode's
        # value for the pair. Revisit only if that turns out insufficient.
        # No visual_material override here -- tried a uniform blue override
        # for debug visibility, but that replaces ALL materials including
        # the motor housings (originally black, distinct from the yellow-ish
        # link plastic), losing that contrast. Getting "blue links, black
        # motors" specifically needs per-mesh material assignment (targeting
        # only the link meshes, leaving motor sub-mesh materials alone),
        # which needs more USD-scripting effort than justified right now --
        # keeping the original per-mesh materials instead. Revisit if
        # per-mesh recoloring becomes worth the effort later.
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            effort_limit_sim=_STS3215_EFFORT_LIMIT,
            velocity_limit_sim=_STS3215_VELOCITY_LIMIT,
            stiffness=50.0,
            damping=2.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=_STS3215_EFFORT_LIMIT,
            velocity_limit_sim=_STS3215_VELOCITY_LIMIT,
            # Softer than the arm joints (was 50/2, same as arm) -- when
            # commanded fully closed against an object it physically can't
            # close past, a stiff PD gain keeps applying strong corrective
            # force instead of complying, building up the interpenetration
            # that causes the object to pop out (see max_depenetration_velocity
            # comment above). Lower gains let it behave more like a
            # torque-limited real servo yielding against the object.
            stiffness=15.0,
            damping=1.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration for the SO-101 robot arm (5 DOF + gripper)."""
