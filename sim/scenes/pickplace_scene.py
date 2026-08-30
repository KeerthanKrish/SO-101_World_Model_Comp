# SPDX-License-Identifier: BSD-3-Clause
"""Scene configuration for the SO-101 pick-and-place task.

Layout: SO-101 fixed to a small table, a cube within reach on the table
surface, ground plane + dome light, and a fixed viewpoint camera for
offscreen rendering (since we're running headless on the remote GPU box --
see docs/preferences.md on why: no practical GPU-accelerated GUI over X11
forwarding to view Isaac Sim's 3D viewport remotely).

Cube placement: moved from 0.2m to 0.28m in front of the robot base after
teleop feedback that 0.2m was uncomfortably close to the base (awkward
joint angles to reach down at that distance) -- informed by actually
teleoperating the real arm, not just guessing.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import configclass

import sys

sys.path.insert(0, "/home/keerthan/SO-101-WM/sim")
from robots.so101 import SO101_CFG  # isort:skip

# Table top surface sits at z=0; robot base is fixed there (URDF converted with --fix-base).
_TABLE_HEIGHT = 0.02
_CUBE_SIZE = 0.03
_CUBE_POS = (0.28, 0.0, _CUBE_SIZE / 2)


@configclass
class PickPlaceSceneBaseCfg(InteractiveSceneCfg):
    """Physical scene only (table, robot, cube) -- no cameras.

    Use this directly for interactive/teleop sessions where a human is
    watching the native Kit viewport (mouse-navigable) rather than any of
    our offscreen CameraCfg sensors -- those render every single frame
    regardless of whether anything reads their output, and having 3-4 of
    them active was a real, measurable source of lag during teleop.
    PickPlaceSceneCfg (below) adds the offscreen cameras back for headless
    scripted runs that need to save images/video.
    """

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
    )

    # table (static, robot and cube sit on its top surface at z=0)
    table = AssetBaseCfg(
        prim_path="/World/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, _TABLE_HEIGHT),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.35, 0.2)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -_TABLE_HEIGHT / 2)),
    )

    # robot
    robot: ArticulationCfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # cube
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(_CUBE_SIZE, _CUBE_SIZE, _CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=2,
                # Matches the robot's cap -- see so101.py's comment. Without
                # this the cube could still get ejected at a high default
                # speed when the gripper drives deep interpenetration.
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # Explicit higher friction (default is 0.5/0.5, "average" combine
            # mode) -- felt "slippery" in teleop; this is a first guess at a
            # grippier surface, not a measured value. Root cause of the worse
            # symptoms (clipping, popping out under full grip) was actually
            # the gripper's convex-hull collision approximation (see
            # reconvert_urdf_convex_decomp.py) -- a convex hull can't
            # represent the pincer's concave notch at all. Friction alone
            # wouldn't have fixed that.
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2, dynamic_friction=1.2, friction_combine_mode="max", restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_CUBE_POS),
    )


@configclass
class PickPlaceSceneCfg(PickPlaceSceneBaseCfg):
    """Pick-and-place scene with offscreen cameras, for headless scripted runs."""

    # fixed viewpoint camera for offscreen rendering (no live GUI available remotely).
    # Placeholder offset/rot below -- actual view direction is set at runtime via
    # camera.set_world_poses_from_view(eye, target), see sim/scripts/capture_scene_image.py.
    scene_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/SceneCamera",
        offset=CameraCfg.OffsetCfg(pos=(0.6, -0.6, 0.5), convention="world"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 5.0)),
        width=960,
        height=720,
        data_types=["rgb"],
    )

    # extra fixed viewpoints for diagnosing grasp attempts without relying
    # on the (currently unresolved, parked) wrist camera. Aimed via
    # set_world_poses_from_view in the calling script, same as scene_camera.
    side_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/SideCamera",
        offset=CameraCfg.OffsetCfg(pos=(0.2, -0.5, 0.1), convention="world"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 5.0)),
        width=960,
        height=720,
        data_types=["rgb"],
    )
    top_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/TopCamera",
        offset=CameraCfg.OffsetCfg(pos=(0.15, 0.0, 0.55), convention="world"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 5.0)),
        width=960,
        height=720,
        data_types=["rgb"],
    )

    # wrist/eye-in-hand camera, rigidly attached to gripper_link (the
    # static wrist/servo housing -- matches where the user's real webcam
    # is actually mounted; it does NOT move with the moving jaw's
    # open/close motion). See docs/real_camera_setup.md for the full
    # history: an earlier "round 2" attempt aimed the camera at the
    # "gripper" joint's URDF origin, which is only the moving jaw's pivot
    # point near the base -- that produced a foreshortened view of the
    # whole mechanism, not the tight fingertip-converging framing in the
    # user's reference photos. This config instead aims at
    # gripper_frame_link (gripper_frame_joint's URDF <origin>,
    # (-0.0079, -0.000218121, -0.0981274) in gripper_link's own local
    # frame) -- that's Isaac Lab's actual IK end-effector/fingertip frame,
    # so "down the fingers" is almost exactly local -Z. Rotation sends
    # local +Z (ROS "forward") onto that direction, tilted 20 degrees
    # toward -Y, with position backed off 7cm along -Y plus a small
    # forward push -- found by testing standoffs/tilts empirically (no
    # analytical shortcut for the perpendicular "which way is up on the
    # housing" axis). Confirmed working: renders show two fingertips
    # converging into the bottom of frame with the tabletop filling the
    # rest, matching the user's real camera's reference photo.
    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link/WristCamera",
        offset=CameraCfg.OffsetCfg(
            pos=(-0.00080248, -0.07002216, -0.00996772),
            rot=(0.03478796, 0.02019336, -0.98402225, -0.17344234),
            convention="ros",
        ),
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.005, 2.0)),
        width=480,
        height=480,
        data_types=["rgb"],
    )
