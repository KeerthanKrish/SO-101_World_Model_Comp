# Real Arm / Physical Machine Setup Reference

Practical reference for connecting to and operating the Ubuntu machine now
that it has a monitor, keyboard, and the leader arm physically attached.
Created 2026-08-29.

## Connecting to the machine

- The machine's LAN IP changes on reboot (DHCP) -- it has been
  `10.0.0.91`, `10.0.0.240` (seen again on 2026-09-02/03, from a
  different client machine), and others at different points. If a known
  IP times out, check with the user for the current one, or try the
  Tailscale IP (`100.71.12.16`) as a fallback -- Tailscale auto-starts on
  boot, but reconnection can occasionally be slow/flaky (seen a few times
  as spurious connection timeouts that resolved on retry), and on a fresh
  client machine Tailscale may not even be installed/active yet.
- **Rediscovering the current LAN IP without asking**, when reachable
  from the same local network: check `~/.ssh/known_hosts` on the client
  for previously-seen `10.0.0.x` entries -- several rows sharing the
  EXACT SAME host-key text are the same physical machine across past
  DHCP leases, even though the IP differs. Ping-sweep the subnet
  (`for i in $(seq 1 254); do ping -c1 -W200 10.0.0.$i ...; done`) for
  live hosts, then `ssh-keyscan -t ed25519 <candidate-ip>` each one and
  compare against the known host-key text -- a match, plus an Ubuntu SSH
  banner, confirms identity before ever connecting. This worked cleanly
  when re-establishing access from a brand new Mac on 2026-09-02 with no
  prior known_hosts entry of its own -- the technique only needs ONE
  prior client's known_hosts to have seen the machine before, not the
  current client.
- SSH access is passwordless (key-based) as `keerthan@<ip>` -- set up
  per-client via `ssh-keygen` + `ssh-copy-id` (the latter needs the
  account password once, interactively, so run it in the user's own
  terminal, not through a tool call).

## Getting a GUI window onto the physical monitor

The machine now has a real logged-in graphical (X11) session on the
physical monitor, not just the GDM login screen. To open windows on it
from an SSH session:

```bash
DISPLAY=:1 XAUTHORITY=/run/user/1001/gdm/Xauthority <command>
```

Verify with `DISPLAY=:1 XAUTHORITY=/run/user/1001/gdm/Xauthority xdpyinfo`.
The exact display number and Xauthority path could change across
reboots/logins -- if `:1` stops working, check `loginctl list-sessions`
and `who` for the current graphical session, and look for a fresh
`/run/user/<uid>/gdm/Xauthority` file.

Isaac Sim/Isaac Lab launched WITHOUT `--headless` (and with
`--enable_cameras` if the scene has any `CameraCfg`) will render its
actual GUI to that display -- confirmed working by capturing a screenshot
via `ffmpeg -f x11grab -i :1 -frames:v 1 out.png` (no dedicated screenshot
tool like `scrot`/`gnome-screenshot` is installed) and by direct user
confirmation watching their own monitor.

## Leader arm (teleoperation)

- Detected at `/dev/ttyUSB*` or `/dev/ttyACM0` (has been `/dev/ttyACM0`).
  User is in the `dialout` group already (set up earlier in the project).
- **Needs external power, separate from the USB data cable.** Feetech
  servos won't respond on the bus (calibration reports "found motor
  list: {}" for all IDs) if only USB is connected without power -- this
  was the first failure mode hit and the fix was simply plugging in power.
- Calibration (`lerobot-calibrate --teleop.type so101_leader --teleop.port
  /dev/ttyACM0 --teleop.id leader1`, in the `lerobot` conda env) is
  **interactive** -- it prompts to move the arm and press Enter, which
  requires a real TTY. Does NOT work via a non-interactive SSH command
  (Claude running a single `ssh host 'command'` has no TTY, gets
  `EOFError` on `input()`). Works fine from any normal interactive SSH
  session (the user's own terminal, MobaXterm, etc.) -- doesn't need to be
  run at the physical machine specifically, just needs a real terminal.
- Calibration file saved to
  `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/<id>.json`.
  Confirmed the calibrated joint names (`shoulder_pan`, `shoulder_lift`,
  `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`) match our sim's
  joint names exactly -- no name-mapping needed between LeRobot and the
  sim.

## LeRobot + Isaac Sim: a real Python version incompatibility

LeRobot's source now uses Python 3.12+ syntax (PEP 695 `type X = ...`
aliases, e.g. in `lerobot/motors/motors_bus.py`). Isaac Sim 5.1.0 requires
Python 3.11. **These cannot share a process** -- importing LeRobot's
modules into the `env_isaaclab` (3.11) environment fails with a
`SyntaxError` at import time, not a missing-package error.

Solution: run them as two separate processes bridged by a shared file:

- `sim/scripts/leader_reader.py` -- runs in the `lerobot` conda env
  (Python 3.12), connects to the leader arm, and continuously overwrites a
  small JSON file (`/tmp/leader_state.json` by default) with the latest
  `get_action()` reading (timestamp + per-joint values).
- `sim/scripts/teleop_bridge.py` -- runs in `env_isaaclab` (Python 3.11)
  under `isaaclab.sh`, polls that same JSON file every simulation step,
  converts the leader's units (degrees for the 5 arm joints, 0-100 range
  for the gripper) into the sim's radians, and commands the simulated
  robot's joints directly. Also records the full trajectory (+ cube pose)
  to a separate JSON file, periodically flushed so a hard kill doesn't
  lose data.

Run `leader_reader.py` first, then `teleop_bridge.py` (non-headless, with
the `DISPLAY`/`XAUTHORITY` env vars above) -- confirmed working, arm
movements visibly drive the simulated robot in real time.

Note: `feetech-servo-sdk`/`pyserial`/`draccus`/`deepdiff` were pip-installed
into `env_isaaclab` while first trying to import LeRobot directly into that
env, before discovering the Python version incompatibility above. Once the
two-process split was used instead, those packages became unnecessary in
`env_isaaclab` -- harmless residue, not cleaned up, but not needed for the
teleop bridge to work.

## Known issues found during first teleop session (2026-08-29)

- **Gripper visually clipping into the cube instead of colliding**: traced
  to `solver_velocity_iteration_count=0` in the robot's
  `ArticulationRootPropertiesCfg` -- every single sim run's log had
  carried a PhysX warning explicitly recommending 1-2 instead of 0 for
  contact accuracy, which had gone unaddressed until teleop-driven contact
  made the resulting clip-through obvious. Fixed: bumped to 2, and gave
  the cube's `RigidBodyPropertiesCfg` matching solver iteration counts.
- **Noticeable lag, both in leader-arm responsiveness and even native
  mouse-driven viewport navigation**: traced to the scene having 3-4
  `CameraCfg` sensors (`scene_camera`, `side_camera`, `top_camera`,
  `wrist_camera`) that all render every single frame regardless of
  whether their output is ever read -- a real, non-trivial GPU cost when
  nobody's using them. Fixed: split the scene into
  `PickPlaceSceneBaseCfg` (table/robot/cube only, no cameras -- used for
  interactive/teleop sessions where a human watches the native Kit
  viewport) and `PickPlaceSceneCfg` (adds the offscreen cameras back, for
  headless scripted runs that need to save images/video).
