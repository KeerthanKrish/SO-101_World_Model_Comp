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

## Status: top-down (context) camera -- matched to a real reference photo (2026-08-30)

User confirmed the top-down concept and sent an actual reference photo of
their real overhead setup: camera mounted well above the workspace,
looking straight down over a large cardboard sheet, with the arm entering
the frame from the top edge (mounted at the near edge of the surface) and
the rest of the frame filled by the reachable workspace.

Two things needed fixing in `top_camera` to match this:

1. **Feedback requested a wider wrist camera FOV too** -- lowered
   `focal_length` from 12.0 to 6.0 on `wrist_camera` (smaller focal length
   = wider field of view for a fixed sensor size). Confirmed direction/
   framing was already right, just needed to see more of the scene.
2. **Top-down rotation/position, computed + tuned**: `top_camera`
   previously had no fixed rotation of its own (aimed only at runtime via
   `set_world_poses_from_view` in test/capture scripts, never baked into
   the scene config for headless use). Computed a fixed straight-down
   rotation analytically -- local +Z (camera forward) onto world -Z
   (straight down), local -Y (image "up") onto world -X (back toward the
   robot base, so the arm appears near the top of frame like the
   reference) -- a 180-degree rotation about the world `(1,1,0)` axis,
   quaternion `(0, 1/sqrt2, 1/sqrt2, 0)`. Position tuned empirically
   (`test_wrist_fov_and_topdown.py`), initially settled on
   `pos=(0.35, 0.0, 0.85)`, `focal_length=16.0` -- see correction below,
   this position was wrong.

**Resolved**: a second real camera (Logitech C922 Pro Stream Webcam) was
in fact connected to the Ubuntu box. A live capture from it
(`ffmpeg -f v4l2 ... /dev/video2`) revealed the `pos=(0.35, ...)` config
above didn't actually match: the real photo shows the cardboard workspace
filling nearly the *entire* frame height, arm hugging just the top edge.

**Root cause**: `x=0.35` assumed the robot sits at the *near edge* of the
visible table, matching the real setup where the cardboard only extends
forward from the arm. But our sim table is centered ON the robot (spans
-0.3 to +0.3m in x), so only the 0 to +0.3m range ahead of the base is
actually usable/reachable table -- `x=0.35` aimed the camera's footprint
mostly past that edge, showing empty ground for most of the frame instead
of tabletop.

**Fix**: recentered to `x=0.15` (the midpoint of the 0-0.3m usable range)
and re-swept `focal_length` at that position (8/10/12/16) -- higher
focal_length = *more* zoomed in for this camera model, opposite of what
lower values did on the wrist camera (that one only ever tested a narrow
range where the relationship happened to look monotonic in one
direction -- worth remembering that "which way is wider" isn't always
intuitive and should be checked empirically per-camera, not assumed from
a previous camera's tuning). `focal_length=16.0` at `pos=(0.15, 0.0, 0.85)`
reproduced the reference proportions closely: tabletop filling the full
width and most of the height, arm near the top.

**Superseded (2026-08-30)**: even after this fix, the match was capped by
the sim table simply being much smaller than the real cardboard
workspace. User chose to enlarge the table (0.6m -> 1.2m, see
`pickplace_scene.py`) rather than reposition the real camera. The
top-down camera was re-tuned to match: x moved to 0.3 (the new usable
range's midpoint) and standoff height doubled to 1.7 (visible extent
scales linearly with height for fixed focal length, confirmed by
re-rendering rather than assumed). `pos=(0.3, 0.0, 1.7)`,
`focal_length=16.0` is now the final config.
