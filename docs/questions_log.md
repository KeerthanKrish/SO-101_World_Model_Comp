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
