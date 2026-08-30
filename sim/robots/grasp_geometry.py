# SPDX-License-Identifier: BSD-3-Clause
"""Shared grasp-point geometry, extracted out of run_pickplace_demo.py so
it can be imported without side effects.

run_pickplace_demo.py (like every script in this project that drives Isaac
Sim) calls AppLauncher/boots the Kit app as an import-time side effect --
importing that script from anywhere else would launch a full Isaac Sim
instance just to read one constant. This module has no Isaac Lab/torch/omni
dependency at all, so it's safe to import from the reward function, a unit
test, or anywhere else that just needs the geometry.

This is the single source of truth for the jaw offset now -- previously
duplicated as a local constant in run_pickplace_demo.py.
"""

import math

# Fixed local-frame offset from gripper_frame_link's origin to the gripper
# joint's pivot (where the jaws actually meet), derived from the URDF's
# fixed joint transforms (gripper_frame_joint and the gripper joint's
# origin, both relative to gripper_link) -- see docs/so101_asset_notes.md.
# This is constant regardless of arm pose (both frames are rigidly attached
# to gripper_link), unlike a world-frame offset which rotates with the
# wrist. Verified: its magnitude (~8.2cm) matches an empirical world-frame
# measurement taken at one specific arm pose via calibrate_grasp.py.
JAW_OFFSET_LOCAL = (-0.0281, 0.019018, -0.0747274)


def grasp_point_world(ee_pos_w, ee_quat_w, jaw_offset_local=JAW_OFFSET_LOCAL):
    """World-frame position of the actual jaw pivot (where the fingers
    meet), given the IK end-effector frame's (gripper_frame_link) world
    pose. This is the point that should be used as "the gripper's position"
    for any distance-based reward shaping -- gripper_frame_link's own
    origin is ~8cm away from where the fingers actually are.

    Args:
        ee_pos_w: world position of gripper_frame_link, (x, y, z).
        ee_quat_w: world orientation of gripper_frame_link, (w, x, y, z).
        jaw_offset_local: offset in gripper_frame_link's local frame.

    Returns:
        (x, y, z) tuple, the jaw pivot's position in world coordinates.
    """
    px, py, pz = ee_pos_w
    w, x, y, z = ee_quat_w
    ox, oy, oz = jaw_offset_local

    # Standard quaternion-vector rotation: v' = v + 2w(q_xyz x v) + 2(q_xyz x (q_xyz x v))
    # written out component-wise to avoid a numpy/torch dependency here.
    qv = (x, y, z)
    tx = 2.0 * (qv[1] * oz - qv[2] * oy)
    ty = 2.0 * (qv[2] * ox - qv[0] * oz)
    tz = 2.0 * (qv[0] * oy - qv[1] * ox)

    cx = qv[1] * tz - qv[2] * ty
    cy = qv[2] * tx - qv[0] * tz
    cz = qv[0] * ty - qv[1] * tx

    rx = ox + w * tx + cx
    ry = oy + w * ty + cy
    rz = oz + w * tz + cz

    return (px + rx, py + ry, pz + rz)


def _self_test():
    # Identity rotation: offset should pass through unrotated.
    result = grasp_point_world((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0), (0.1, 0.2, 0.3))
    expected = (1.1, 2.2, 3.3)
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(result, expected)), result

    # 90-degree rotation about world Z: (1,0,0) local offset -> (0,1,0) world direction.
    half = math.sqrt(0.5)
    result = grasp_point_world((0.0, 0.0, 0.0), (half, 0.0, 0.0, half), (1.0, 0.0, 0.0))
    assert math.isclose(result[0], 0.0, abs_tol=1e-9), result
    assert math.isclose(result[1], 1.0, abs_tol=1e-9), result
    assert math.isclose(result[2], 0.0, abs_tol=1e-9), result
    print("[OK] grasp_geometry self-test passed")


if __name__ == "__main__":
    _self_test()
