# Working Preferences

Last updated: 2026-09-03

## Documentation

- Everything about this project should be documented. Docs live in
  `docs/` inside the project root, traveling with the code (not a separate
  top-level location).
- Keep: preferences (this file), project plan, a log of conceptual questions
  asked along the way, a progress log, and a decisions log for "chose X over Y
  because Z" reasoning.

## Machines / Sync

- Primary implementation machine: Ubuntu box, user `keerthan`, project root
  `~/SO-101-WM`. Robots (SO-101 leader + follower) and the NVIDIA GPU are
  physically attached there. X11 forwarding / MobaXterm setup is handled
  by the user, not Claude. All implementation work (code, running
  experiments) happens here.
- **Reaching it**: normally over the local LAN, not Tailscale -- the LAN
  IP is DHCP-assigned and changes on reboot (see docs/real_arm_setup.md
  for the known-IPs history and the known_hosts-matching technique used
  to rediscover it). Tailscale (`100.71.12.16`) is a fallback only, and
  has been flaky/slow to reconnect at times.
- Local machine: as of 2026-09-02, a MacBook Pro (previously a Windows
  laptop) -- project cloned at `~/Keerthan/Projects/SO-101-WM`.
- **Code sync**: git, not scp. Ubuntu is the sole `git push` origin;
  the local machine just `git pull`s. Never edit code on the local
  machine and push from there -- keeps one clear direction of flow and
  matches where the code actually gets tested (Ubuntu, via Isaac Lab).
- **Non-code sync** (things `.gitignore` excludes and always will --
  `sim/output/`: eval videos, checkpoints, recorded teleop episodes; and
  `assets/`: robot USD files): still `scp`/`rsync` from Ubuntu to the
  local machine, since these will never come through git. Done at natural
  checkpoints (e.g. after a training run finishes) or when the user asks,
  not on a timer.

## General Working Style

- User is doing this as a research-style project (not just "get it working"),
  interested in understanding *why* things work, not just implementing them.
- User is new to diffusion policies / world models conceptually — prefers
  intuitive, non-technical analogies before diving into technical detail when
  learning new concepts (see questions_log.md for the analogies used).
- Prefers focused scope over broad benchmark studies; open-ended timeline,
  doesn't want a calendar imposed.
- **Never trust a logged success/failure flag over the actual video.**
  Established the hard way, twice: run1's "promising" +97.9 eval reward
  and a run7 episode's `held=True` both turned out to be false positives
  (reward hacking and a grasp-detection bug, respectively) that were only
  caught by watching the video directly. Always watch before reporting a
  training result as good news, even when the numbers look right.
- **Debug at small scale before committing to a long run.** Established
  pattern: implement a fix, run it at a shorter step budget first, confirm
  no hacking/crashes/regressions by watching video, only then scale up.
- Comfortable with Claude launching the next run automatically once a
  prerequisite finishes, when explicitly told to do so in advance (e.g.
  "start an X-step run once this one's done, don't wait for me to
  confirm") -- this is opt-in per instance, not a standing default.
