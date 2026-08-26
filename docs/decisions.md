# Decisions Log

Record of choices made and the reasoning behind them, kept separate from the
project plan so the plan stays clean and the "why" doesn't get lost.

---

**Decision**: Compare a TD-MPC2-style world model against LeRobot's diffusion
policy, rather than comparing a diffusion world model against a diffusion
policy.

**Why**: The more interesting research axis is model-based (explicit
predicted consequences, replanning) vs. model-free (direct reflex policy),
not "diffusion vs. non-diffusion." A TD-MPC2-style world model is also
practical for online replanning during real-arm control (cheap per-step
imagination), whereas a diffusion world model is expensive per imagined step
(full denoising loop per step) and better suited to offline high-fidelity
simulation than live planning. A diffusion-based world model remains a
possible stretch goal, not core scope.

---

**Decision**: Use NVIDIA Isaac Lab as the simulator.

**Why**: GPU-parallelized, strong domain-randomization tooling for sim-to-real
work, and a good fit given the available NVIDIA GPU. Alternatives considered:
Genesis (faster/lighter but less mature tooling), ManiSkill3 (manipulation-
focused, good middle ground), MuJoCo MJX (lightweight but less
domain-randomization/manipulation tooling built in).

---

**Decision**: Keep project scope to one task (pick-and-place a single cube),
no fixed calendar timeline.

**Why**: User wants a research-quality comparison, not a multi-task benchmark
sweep. Avoids scope creep while still supporting a genuine sample-efficiency /
robustness comparison.

---

**Decision**: Documentation lives in `docs/` inside the project root (travels
with the code), not as a separate top-level location.

**Why**: User preference — docs and code should move together as one unit.

---

**Decision**: Local (Windows laptop) and remote (Ubuntu, `keerthan@100.71.12.16`)
copies of the project are kept in sync via `scp`, on-demand only (when the
user explicitly asks), not automatically.

**Why**: Avoids unnecessary token usage from rewriting files on both sides,
and avoids syncing on a timer for no reason. Implementation work (code,
running experiments, robots, GPU) all happens on the Ubuntu machine.
