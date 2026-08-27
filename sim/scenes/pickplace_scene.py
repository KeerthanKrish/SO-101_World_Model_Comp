# SPDX-License-Identifier: BSD-3-Clause
"""Scene configuration for the SO-101 pick-and-place task.

Layout: SO-101 fixed to a small table, a cube within reach on the table
surface, ground plane + dome light, and a fixed viewpoint camera for
offscreen rendering (since we're running headless on the remote GPU box --
see docs/preferences.md on why: no practical GPU-accelerated GUI over X11
forwarding to view Isaac Sim's 3D viewport remotely).

Cube placement (0.2m in front of the robot base) is an approximate guess at
what's within the SO-101's reach -- not yet validated against the real
arm's actual workspace. Revisit once we can compare against real-arm reach.
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
_CUBE_POS = (0.2, 0.0, _CUBE_SIZE / 2)


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Configuration for the SO-101 pick-and-place scene."""

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
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_CUBE_POS),
    )

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
