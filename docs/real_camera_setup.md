# Real Camera Setup Reference

Documents the physical camera the user mounted on the real SO-101, and how
we're matching it in the sim scene. Created 2026-08-30.

## Physical mount

User attached a USB webcam (labeled "HD FULL WEBCAM" on the housing) to
the real arm via a 3D-printed/custom bracket, mounted on top of the
wrist/gripper housing (roughly at the `gripper_link` body, just before
the jaw pivot), angled forward and downward over the two-finger gripper.

Two reference images were shared in the conversation (not saved to disk --
no image-saving mechanism available for pasted chat attachments, described
here instead):

1. **The camera's actual view**: shows the two white gripper fingers
   entering the bottom of frame, converging inward, pointing down/forward
   over a wood-grain table surface that fills most of the frame. Confirms
   the real SO-101's actual color is white (matches what we were told
   early in the project, when we chose not to force a color override in
   sim -- see docs/so101_asset_notes.md).
2. **A photo of the mount itself**: shows the webcam clipped to a bracket
   sitting on top of the gripper/wrist servo housing, lens angled down
   toward the fingers, wires running back along the arm.

## Sim camera (wrist/eye-in-hand)

This is the SAME camera concept we attempted earlier this session and
parked (see docs/progress.md, "wrist camera" abandonment) after ~8 failed
mount configurations found by blind trial and error. This time we have an
actual reference photo to match against instead of guessing -- see
sim/scripts/test_wrist_camera_angles.py and pickplace_scene.py for the
current attachment approach (attached to `gripper_link`, angled
forward-and-down to reproduce the reference view: two fingers visible at
the bottom of frame, table filling the rest).

## Second (context) camera: top-down

User asked whether a top-down camera is a reasonable choice for the
second/context camera. Answer: yes -- gives a clean global view of the
whole workspace (cube position, gripper position, table layout) that
complements the close-up wrist view. This is exactly what `top_camera` in
`pickplace_scene.py` already is, so no new sim work was needed for this
part, just confirmation that it's a sensible choice.

**Open question for the user**: is there a second *physical* camera
planned to match this top-down sim view, or is `top_camera` purely a
sim-side visualization aid (like `scene_camera`/`side_camera`) with no
real-world counterpart? If a second real camera is planned, its actual
mounting position needs to be measured and matched in sim the same way
we're doing for the wrist camera -- if not, `top_camera` can stay as an
arbitrary convenient viewpoint since it never needs to match anything real.

## Status: wrist camera working (2026-08-30)

Second attempt succeeded after the first (parked) attempt failed ~8 times.
What changed:

1. **Attachment point**: switched from `gripper_frame_link` to
   `gripper_link` -- closer to where the real webcam bracket actually
   sits (on the wrist/servo housing, not out at the frame reference used
   for IK).
2. **Rotation computed analytically, not guessed**: the "gripper" joint's
   URDF `<origin>` (`0.0202, 0.0188, -0.0234`) is already the jaw pivot's
   position directly in `gripper_link`'s own local frame (URDF joint
   origins are relative to the parent link -- `gripper_link` is the
   parent of the `gripper` joint). No need to guess a direction at all:
   computed the shortest-arc rotation quaternion sending the camera's
   local +Z (ROS "forward") onto that normalized direction, same
   technique that worked for `_JAW_OFFSET_LOCAL` earlier. Two rounds of
   blind rotation guessing (round 1 and round 2 in
   `test_wrist_camera_angles2.py`'s git history) had failed before this.
3. **Standoff distance**: initial attempts were too close (camera saw
   only the robot's own body, filling the whole frame) -- backed off to a
   9cm standoff along the same computed direction, matching what the
   real bracket's visible height suggested. 5cm was still too close;
   7cm and 9cm both worked, 9cm chosen for a slightly wider view.

Confirmed working: renders now show both the gripper mechanism and the
cube on the table in frame -- see `sim/scenes/pickplace_scene.py`'s
`wrist_camera` for the final config.

**Lesson for next time a camera/frame orientation needs figuring out**:
before guessing candidate rotations, check whether the needed direction is
already sitting in a URDF `<origin>` tag relative to the body being
attached to -- URDF joint origins are relative to the *parent* link, so if
attaching to that same parent, the direction is already known exactly,
no trial and error required.

## Status: top-down (context) camera

`top_camera` in `pickplace_scene.py` -- already existed, confirmed as a
sensible choice, no changes needed. See "Open question" above re: whether
a second physical camera is planned to match it.
