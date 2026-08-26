# Working Preferences

Last updated: 2026-08-26

## Documentation

- Everything about this project should be documented. Docs live in
  `docs/` inside the project root, traveling with the code (not a separate
  top-level location).
- Keep: preferences (this file), project plan, a log of conceptual questions
  asked along the way, a progress log, and a decisions log for "chose X over Y
  because Z" reasoning.

## Machines / Sync

- Primary implementation machine: Ubuntu box at `keerthan@100.71.12.16`
  (reachable via Tailscale). Robots (SO-101 leader + follower) and the NVIDIA
  GPU are physically attached there. X11 forwarding / MobaXterm setup is
  handled by the user, not Claude.
- Local machine: Windows laptop, project mirrored at
  `c:\Keerthan\Projects\SO-101-WM`.
- All implementation work (code, running experiments) happens on the Ubuntu
  machine.
- Sync between local and Ubuntu copies happens via `scp`, not by rewriting
  files on both sides — avoids unnecessary token usage. Sync is **on-demand
  only**: only when the user explicitly says to sync (e.g. "sync now"), not
  automatically on a timer or after every change.

## General Working Style

- User is doing this as a research-style project (not just "get it working"),
  interested in understanding *why* things work, not just implementing them.
- User is new to diffusion policies / world models conceptually — prefers
  intuitive, non-technical analogies before diving into technical detail when
  learning new concepts (see questions_log.md for the analogies used).
- Prefers focused scope over broad benchmark studies; open-ended timeline,
  doesn't want a calendar imposed.
