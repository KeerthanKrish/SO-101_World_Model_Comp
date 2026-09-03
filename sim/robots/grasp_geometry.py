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


def _rotate_vector(quat_wxyz, v):
    """Rotates a plain 3-vector by a quaternion, both in local/world frame
    terms as the caller intends -- shared by grasp_point_world() (rotating
    a POINT offset, then translated by the caller) and
    jaw_approach_axis_world() (rotating a pure DIRECTION, never
    translated). Standard quaternion-vector rotation:
    v' = v + 2w(q_xyz x v) + 2(q_xyz x (q_xyz x v)), written out
    component-wise to avoid a numpy/torch dependency here."""
    w, x, y, z = quat_wxyz
    ox, oy, oz = v
    qv = (x, y, z)
    tx = 2.0 * (qv[1] * oz - qv[2] * oy)
    ty = 2.0 * (qv[2] * ox - qv[0] * oz)
    tz = 2.0 * (qv[0] * oy - qv[1] * ox)

    cx = qv[1] * tz - qv[2] * ty
    cy = qv[2] * tx - qv[0] * tz
    cz = qv[0] * ty - qv[1] * tx

    return (ox + w * tx + cx, oy + w * ty + cy, oz + w * tz + cz)


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
    rx, ry, rz = _rotate_vector(ee_quat_w, jaw_offset_local)
    return (px + rx, py + ry, pz + rz)


# Pure direction (unit vector), not a point -- normalized JAW_OFFSET_LOCAL,
# reused as "the direction the fingers reach from the wrist" rather than
# deriving a second, separately-calibrated axis. Justified because
# JAW_OFFSET_LOCAL is BY DEFINITION the vector from gripper_frame_link's
# origin to where the fingers meet -- i.e. exactly the reach/approach
# direction already, just also carrying a magnitude that grasp_point_world()
# needs and this doesn't. Added 2026-09-03 for is_between_jaws() below, see
# that function's docstring for why a single approach-axis check was needed
# on top of the existing (grasp_point_world-based) distance-only checks.
_JAW_OFFSET_MAGNITUDE = sum(c * c for c in JAW_OFFSET_LOCAL) ** 0.5
JAW_AXIS_LOCAL = tuple(c / _JAW_OFFSET_MAGNITUDE for c in JAW_OFFSET_LOCAL)


def jaw_approach_axis_world(ee_quat_w, jaw_axis_local=JAW_AXIS_LOCAL):
    """World-frame unit vector pointing from the jaw pivot toward the
    fingertips -- i.e. "which way the gripper is currently reaching," given
    the IK end-effector frame's (gripper_frame_link) world orientation.
    Unlike grasp_point_world(), this has no position component at all (a
    direction doesn't translate, only rotate) -- callers wanting "is the
    cube positioned in front of the jaws, not beside them" combine this
    with grasp_point_world()'s point via a dot/cross-product decomposition
    (see pickplace_reward.is_between_jaws()).

    Args:
        ee_quat_w: world orientation of gripper_frame_link, (w, x, y, z).
        jaw_axis_local: unit direction in gripper_frame_link's local frame.

    Returns:
        (x, y, z) unit vector in world coordinates.
    """
    return _rotate_vector(ee_quat_w, jaw_axis_local)


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

    # JAW_AXIS_LOCAL must be a genuine unit vector (a direction, not a point).
    mag = sum(c * c for c in JAW_AXIS_LOCAL) ** 0.5
    assert math.isclose(mag, 1.0, abs_tol=1e-9), JAW_AXIS_LOCAL
    # It must point the same way as JAW_OFFSET_LOCAL, just rescaled --
    # otherwise "the direction the fingers reach" and "how far they are"
    # would silently disagree.
    for a, o in zip(JAW_AXIS_LOCAL, JAW_OFFSET_LOCAL):
        assert math.isclose(a * _JAW_OFFSET_MAGNITUDE, o, abs_tol=1e-9), (JAW_AXIS_LOCAL, JAW_OFFSET_LOCAL)

    # jaw_approach_axis_world() must rotate like grasp_point_world() does,
    # but never translate -- identity rotation passes the axis through
    # unchanged regardless of ee_pos_w (there is no ee_pos_w argument at
    # all, by design).
    result = jaw_approach_axis_world((1.0, 0.0, 0.0, 0.0))
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(result, JAW_AXIS_LOCAL)), result
    # Same 90-degree-about-Z check as grasp_point_world() above, applied to
    # a direction instead of a point -- must rotate identically.
    result = jaw_approach_axis_world((half, 0.0, 0.0, half), jaw_axis_local=(1.0, 0.0, 0.0))
    assert math.isclose(result[0], 0.0, abs_tol=1e-9), result
    assert math.isclose(result[1], 1.0, abs_tol=1e-9), result
    assert math.isclose(result[2], 0.0, abs_tol=1e-9), result

    print("[OK] grasp_geometry self-test passed")


if __name__ == "__main__":
    _self_test()
