# Questions Log

A running record of the conceptual questions asked during this project, and
the explanations given — a personal learning trail through world models and
diffusion policies.

---

### Q: What's the difference between a diffusion world model and a "normal" world model?

**Technical answer**: A "normal" world model (Dreamer's RSSM, TD-MPC2) encodes
observations into a compact latent state and predicts the next latent state
with a single cheap forward pass (recurrent/transition network), enabling fast
rollout of many imagined trajectories for online planning (CEM/MPPI). A
diffusion world model (DIAMOND, Genie, GameNGen) predicts the next
observation/frame via a full multi-step denoising process, producing much
higher visual fidelity but at far higher compute cost per imagined step —
better suited to offline high-fidelity simulation/data generation than live
replanning.

**Intuitive version**: A normal world model is like closing your eyes and
quickly doodling a rough mental sketch of what happens next — fast and cheap
enough to doodle hundreds of alternate futures and pick the best one in real
time. A diffusion world model is like actually painting a detailed, realistic
picture of what happens next, starting from noise and refining it stroke by
stroke — much more vivid and accurate, but too slow to paint hundreds of
alternatives fast enough for split-second decisions.

**Why it matters for this project**: real-time replanning on the physical arm
needs the fast doodler (TD-MPC2-style), which is why that's the core model,
not a diffusion world model (see decisions.md).

---

### Q: What does a diffusion policy look like in practice? (No prior hands-on experience with it.)

A diffusion policy is pure imitation learning — no dynamics prediction at all.
It takes a chunk of noise shaped like a future action sequence and denoises it
into a plausible sequence of actions, conditioned on the current observation.
Trained via standard DDPM-style training directly on (observation,
action-chunk) pairs from demonstrations — no environment interaction needed.
At inference time it's receding-horizon control: denoise a chunk (e.g. 16
future actions), execute the first few, re-observe, repeat.

Key reason diffusion is used here (vs. plain MSE regression): demonstrations
are often multimodal (multiple valid ways to grasp the same object), and a
regression policy averages those modes into a blurry, invalid action, while a
diffusion model represents the full distribution properly.

Practical wrinkle: multi-step denoising at inference can be slower per control
step than a plain feedforward policy — addressed in practice with fewer
denoising steps (DDIM) or distillation.

---

### Q: Is a world model defined by the data it's trained on (predicting environment vs. agent state)?

No — both a policy and a world model can train on the same recorded data
(observations + actions). The difference is what the model is trained to
*predict*:
- A **policy** learns observation → action directly (a reflex).
- A **world model** learns (observation, hypothetical action) → predicted
  next observation/state — i.e., it explicitly models cause-and-effect,
  which lets you imagine multiple hypothetical futures and compare them
  before acting.

This is the classic **model-based vs. model-free** distinction in
control/RL. World model = model-based (can plan/imagine ahead, like
TD-MPC2). Diffusion policy = model-free (direct reflex mapping, no explicit
"what happens if" step).

---

### Q: Concrete example with the arm and cube?

Same moment for both: the arm is reaching for a cube that's slightly off from
where expected.

- **Diffusion policy**: pattern-matches the current image to "what the
  demonstrator did in similar-looking situations" and outputs an action chunk
  directly — no explicit consideration of alternatives, just reflex.
- **World model + planner**: proposes several candidate action sequences,
  asks the world model to imagine the outcome of each (cube grasped cleanly /
  knocked over / missed), and picks whichever imagined outcome scores best
  before acting.

Where it shows up concretely: if the cube slips mid-grasp, the diffusion
policy only reacts once it re-observes at the next control step (same reflex
mechanism, new input). The world-model planner re-imagines from the new
state at every step, explicitly re-planning around the slip (e.g. "reopen and
re-center" vs. "keep closing") — this is the mechanism the project's
disturbance-robustness evaluation is designed to test.

---

### Q: Is there any other way to introduce a reward gradient without allowing reward hacking? (Asked after killing run2, which showed no directed movement.)

Yes — **potential-based reward shaping** (Ng, Harada & Russell, ICML
1999). The old design rewarded the ABSOLUTE value of a "how close am I"
score every step, which is exactly why a policy could hack it: just
occupy a decent-looking spot and collect that score repeatedly, with no
requirement to have ever tracked the actual target. The fix is to reward
only the CHANGE in that score between steps — `reward = Phi(new state) -
Phi(old state)`. This has a real mathematical guarantee behind it: adding
a term of exactly this form can NEVER change what the optimal policy is,
no matter how the potential function `Phi` is chosen — it can only
change how quickly/easily that policy is found. Practically, it means
holding still anywhere earns exactly zero reward every step, since
nothing is changing — the "sit in a good spot and collect reward"
exploit stops being possible by construction, not by tuning a threshold
tighter. This became the core fix for the run1 exploit (see
docs/reward_function.md and docs/decisions.md) and was later reused for
the gripper-closing shaping term too (docs/decisions.md, 2026-09-03).

---

### Q: Why did the arm just sit still instead of exploring, once reward hacking was fixed? (Asked after two consecutive 50k-step runs showed zero directed movement.)

Traced into TD-MPC2's own planning code rather than guessing. Two
compounding reasons:

1. **A "nothing rewarding has happened yet" bootstrapping problem.**
   TD-MPC2 only plans 3 steps directly ahead (`horizon: 3`); anything
   beyond that relies entirely on a learned VALUE function, which only
   knows what it's seen in the replay buffer. Once reward-hacking was
   fixed, holding still correctly earns zero reward — which also means
   there's no reward-based pressure pushing an untrained policy to move
   at all, unless the buffer already contains a genuine touch/grasp
   trajectory for the value function to learn "reaching pays off" from.
   Pure random exploration essentially never stumbles onto touching a
   small, randomly-placed target by chance, so that experience never
   showed up.
2. **A planning-algorithm pathology on top of that.** TD-MPC2's action
   selection (CEM/MPPI) samples 512 candidate action sequences every
   step and iteratively narrows toward the best-scoring ones over 6
   refinement rounds. CEM is well known to over-confidently narrow its
   own sampling spread even when the scores it's ranking by are pure
   noise (no real signal yet to distinguish good from bad) — producing a
   falsely-precise, repeatably-idle action instead of continued
   exploration. The actual exploration noise added during training
   (`a = a + std * randn(...)`) uses exactly that potentially-
   falsely-converged spread.

Fix: a curriculum lever (fix the cube to one known position instead of
randomizing it every episode, so the "find the target" half of the
problem is trivial while learning "reach and grasp" from scratch) plus
raising the floor under that planning noise (`min_std`) so it can't
collapse into false confidence quite so easily. Together these produced
the first reliable cube contact across any run (see docs/progress.md,
2026-09-02/03 entry).
