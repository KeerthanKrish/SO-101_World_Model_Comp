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

Renders at the time showed both the gripper mechanism and the cube on
the table in frame -- looked plausible, but see the correction below:
this render did NOT actually match the user's reference photo framing.

**Lesson for next time a camera/frame orientation needs figuring out**:
before guessing candidate rotations, check whether the needed direction is
already sitting in a URDF `<origin>` tag relative to the body being
attached to -- URDF joint origins are relative to the *parent* link, so if
attaching to that same parent, the direction is already known exactly,
no trial and error required.

## Status: round-3 correction -- wrong target point (2026-08-30)

User sent 5 additional reference photos plus the actual camera-view still
frame (not saved as separate files -- described here). Two clarifications
that corrected assumptions above:

1. The camera is mounted on the **gripper assembly**, not the wrist --
   this was initially misread as "moves with the jaw," prompting a
   (wrong) attempt to attach the camera to `moving_jaw_so101_v1_link`.
   The user then clarified: it does **not** move with the gripper
   opening/closing -- it's solidly mounted on the static housing. So the
   attachment point (`gripper_link`) from the first "working" attempt was
   actually correct all along.
2. The real bug was the **aim point**. The first attempt's rotation aimed
   the camera at the "gripper" joint's URDF origin -- but that's only the
   position of the moving jaw's *pivot/hinge*, near the base of the
   mechanism. Aiming at a hinge point produces a foreshortened view of
   the whole gripper body (confirmed by re-inspecting the "working"
   render: it showed a blobby yellow mechanism, not two distinct fingers)
   -- not the tight, fingertip-converging framing in the reference
   photos.

   The fix: `gripper_link` has a second fixed child frame,
   `gripper_frame_link` (Isaac Lab's actual IK end-effector frame),
   attached via `gripper_frame_joint` with URDF `<origin>`
   `(-0.0079, -0.000218121, -0.0981274)` -- overwhelmingly along local
   -Z. That's the direction from the housing straight down the fingers
   toward the actual grasp point, not the pivot. Re-aiming the camera at
   that direction instead was the key correction.
3. The perpendicular "which way is up on the housing" axis still had no
   analytical shortcut (unlike the aim direction), so it was found by
   rendering candidates across all 4 lateral directions (`+X`, `-X`,
   `+Y`, `-Y`) at a few standoffs each (`test_wrist_camera_angles5.py`,
   `test_wrist_camera_angles6.py`). Short standoffs (1-5cm along the
   lateral axis alone) still landed inside the housing's own solid mesh
   (near-clip artifacts, flat color, no scene visible) -- same lesson as
   the very first attempt: the servo-box part of `gripper_link` is bigger
   than a few cm across. `-Y` at 7cm finally reproduced the reference
   framing: two fingertips entering the bottom corners, tabletop filling
   the rest.

Final config (`sim/scenes/pickplace_scene.py`'s `wrist_camera`):
attached to `gripper_link`, `pos=(-0.00080248, -0.07002216, -0.00996772)`,
`rot=(0.03478796, 0.02019336, -0.98402225, -0.17344234)` (ROS
convention), 20-degree tilt applied on top of the analytically-aimed
base rotation. Render matches the reference photo's framing.

**Updated lesson**: when a URDF has more than one child frame off the
attachment body, check which one actually represents the direction
needed (a fingertip/end-effector frame vs. a joint pivot are very
different directions even though both originate from the same parent
link) -- don't assume the first candidate origin found is the right one
without sanity-checking what it geometrically represents.

## Status: top-down (context) camera

`top_camera` in `pickplace_scene.py` -- already existed, confirmed as a
sensible choice, no changes needed. See "Open question" above re: whether
a second physical camera is planned to match it.
