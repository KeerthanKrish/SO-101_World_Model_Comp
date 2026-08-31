# TD-MPC2 and Diffusion Policy, Explained

Written 2026-08-30, before wiring either into `PickPlaceEnv`, because the
user asked for a complete technical-to-intuitive explanation of both
algorithms as one of the core aspects of the project, not a summary.
This is the reference version; the chat explanation covers the same
ground.

## The one-sentence version of each

- **TD-MPC2** learns a compressed internal model of how the world
  behaves, and at every single step, uses that model to imagine several
  possible near-futures, picks whichever looks best, and only acts on
  the very first step of that plan before immediately re-imagining from
  scratch.
- **Diffusion policy** learns to reproduce the kind of action sequences
  a human demonstrator performed, by learning to "sculpt" a clean action
  sequence out of random noise, conditioned on what it currently
  observes -- with no internal model of consequences at all.

Everything below unpacks exactly what that means and how it connects to
this project's actual code.

## TD-MPC2, component by component

TD-MPC2 (Hansen et al.) has five learned pieces working together, all
operating in a small **latent space** rather than directly on raw pixels
-- this is the key efficiency idea, explained after the components:

1. **Encoder** `h(observation) -> z`. Compresses the raw observation
   (our `wrist_rgb` + `top_rgb` images plus `proprio`) into a small
   vector `z`. This is the "what actually matters" filter -- exact
   lighting or table grain gets thrown away, whatever's actually
   predictive of the task (roughly: where's the cube, where's the
   gripper, is it closed) gets kept.
2. **Latent dynamics model** `d(z, action) -> z'`. Predicts the *next*
   compressed state from the current one and an action -- entirely in
   latent space, never predicting future pixels. This is the "world
   model" in the literal sense: an internal, learned simulator of how
   the task responds to actions, cheap enough to run thousands of times
   per real decision because it never has to render an image.
3. **Reward model** `R(z, action) -> reward`. Predicts the immediate
   reward for an action, trained to match `pickplace_reward.py`'s actual
   output. This is the literal connection point between our
   already-validated reward function and TD-MPC2 -- every real step
   through `PickPlaceEnv` gives a ground-truth reward that trains this
   model to predict reward from latent state alone, which is what lets
   *imagined* rollouts (that never touch the real simulator) still be
   scored.
4. **Value function** `Q(z, action) -> expected future return`.
   Estimates not just the immediate reward but the discounted sum of
   *all future* reward from acting optimally onward -- this is what lets
   the model reason about long-term consequences (e.g. "closing the
   gripper here gets zero immediate reward but sets up a big reward two
   seconds later") rather than being greedy. Trained via temporal-
   difference learning (the "TD" in TD-MPC2), the same family of
   technique behind DQN/SAC.
5. **Policy prior** `pi(z) -> action`. A fast, reactive "first guess" at
   a good action, used to seed the planning search below rather than
   searching the entire continuous 6-dimensional action space from
   scratch every time.

### How it actually decides what to do (the planner)

At every real control step:

1. Sample many candidate action *sequences* a few steps into the future
   (some from the policy prior, some randomly, for exploration).
2. "Imagine" each one: roll it forward through the *learned* dynamics
   model repeatedly, accumulating predicted rewards, plus a value-
   function estimate of everything beyond the imagined horizon.
3. Score every candidate sequence by its total imagined return.
4. Re-fit a distribution over action sequences weighted toward the
   best-scoring ones (this is MPPI -- Model Predictive Path Integral,
   a cousin of the cross-entropy method), and repeat steps 1-4 a handful
   of times to concentrate the search.
5. Execute *only the first action* of the best final sequence in the
   real environment.
6. Throw the rest of the plan away and repeat the entire process from
   the new real observation next step.

Step 6 is the important one: TD-MPC2 never commits to a multi-step plan.
It replans from scratch, from the real observed state, every single
step. That's the literal mechanism behind "the world model can notice
and correct for a surprise" from the earlier evaluation-plan discussion
-- if something unexpected happened, the very next replanning cycle
already sees it and adjusts, because planning always starts from *actual*
current state, not from what was expected.

### How it trains

Online, by actually interacting with `PickPlaceEnv` (using its own
planner, plus exploration noise, to choose real actions), storing every
real `(observation, action, reward, next_observation)` transition in a
replay buffer, and periodically updating all five components against
batches from that buffer. Critically, it does **not** need a
pre-existing demonstration dataset -- it generates its own training data
as it goes. It's also considerably more sample-efficient than typical
model-free RL (plain PPO/SAC with no model), because between real steps
it can "practice" extensively by replaying stored past transitions
through its *current* (improving) dynamics model -- getting many virtual
repetitions out of relatively few real environment steps. This sample
efficiency is TD-MPC2's headline strength, and is exactly what Test 1 in
`evaluation_plan.md` (sample efficiency curve) is designed to probe.

### The intuitive version

Imagine reaching for a cup of coffee while briefly closing your eyes
between each tiny motion: before every small movement, you picture a few
different ways the next instant could go ("a little more left... a
little slower...") using your own mental sense of physics, pick whichever
imagined option seems best, make just that one small movement, open your
eyes to see where you actually ended up, then immediately re-imagine from
there. If someone nudges the cup while your eyes are closed, you don't
need any special "notice the disturbance" logic -- the very next
re-imagining automatically starts from where the cup actually is now,
not where you expected it. That constant imagine-act-reimagine loop, not
any single clever recovery mechanism, is what makes this approach
naturally robust to surprises.

## Diffusion policy, component by component

Diffusion policy applies the same machinery behind image-generation
models (Stable Diffusion, etc.) to **action sequences** instead of
images.

### The core idea, and why it's framed as "denoising" at all

The straightforward alternative would be: train a network to directly
predict the right action given the current observation (plain
regression, minimizing e.g. mean-squared error against demonstrated
actions). This fails in a specific, important way for manipulation:
demonstrations are often **multi-modal** -- for a similar-looking
approach to the cube, a human might sometimes go from the left, sometimes
from the right, both equally valid. A regression network trained on both
doesn't learn "pick left or right" -- it learns to predict something
close to the *average* of left and right, which is neither, and can be a
physically invalid, useless motion that resembles neither demonstrated
style. This is the classic "mode averaging" failure of naive imitation
learning, and it's exactly what Test 8 in `evaluation_plan.md` is
designed to probe.

Diffusion models sidestep this by learning the full *distribution* over
plausible action sequences, not a single point estimate. Sampling from a
trained diffusion model repeatedly (different random noise each time)
produces either a clean left-approach sample or a clean right-approach
sample -- distinct, valid modes -- never an invalid blend of the two.

### Components

1. **Observation encoder**: compresses recent observation history (a
   short window of camera images + proprioception) into a conditioning
   vector. Same encoding role as TD-MPC2's encoder, but with no dynamics
   model attached to it at all -- there is no "imagining the future"
   anywhere in this architecture.
2. **Noise-prediction network** (typically a temporal U-Net or
   transformer operating over the action sequence as a signal): given a
   noisy action sequence, a noise level, and the observation conditioning
   vector, predicts the noise that was added.
3. **Training (denoising score matching)**: take a real demonstrated
   action chunk from the dataset, corrupt it with a random amount of
   Gaussian noise, and train the network (simple MSE loss) to predict
   exactly the noise that was added, across many (observation, action
   chunk) pairs and noise levels. This is ordinary supervised learning --
   no environment interaction at all during training.
4. **Inference**: start from pure random noise shaped like an action
   sequence, and repeatedly apply the trained denoiser (predict noise,
   remove a scheduled portion, repeat) following a diffusion sampling
   schedule until it converges to a clean, plausible action sequence
   conditioned on the current real observation. Execute a prefix of that
   sequence in the environment, then regenerate a fresh one from the new
   observation.

### How it trains

Entirely offline, from a **fixed, pre-collected dataset** of
(observation, action) demonstrations. This is a hard requirement, not a
detail -- and it's the actual blocker discussed earlier: the project
currently has 8 real teleop episodes, nowhere near enough for a diffusion
policy to learn robustly (typically hundreds to thousands are needed).
`PickPlaceEnv` is not needed at all during diffusion policy training --
only afterward, to evaluate the trained result.

### The intuitive version

Imagine a sculptor who has studied thousands of photographs of "a hand
reaching for a coffee cup," and has learned to shape a plausible
hand-motion out of a lump of random clay through many small refining
passes -- each pass looks at the current rough shape and the current
photograph, and nudges the clay a little closer to something that
resembles a real, valid reach. Enough passes and you get a clean,
realistic motion. But the sculptor never simulates physics or thinks
ahead about consequences -- they're pattern-matching against what similar
situations looked like in training, not reasoning about what happens
next. If the cup moves mid-motion, there's no mechanism to notice and
adjust mid-sculpt -- the current chunk of motion plays out on stale
information until the next photograph is taken and a fresh chunk gets
sculpted.

## The structural asymmetry that matters for this project right now

| | TD-MPC2 | Diffusion policy |
|---|---|---|
| Learns from | its own interaction with `PickPlaceEnv` | a fixed, pre-recorded demonstration dataset |
| Needs the env during training? | Yes -- generates its own data | No -- only for evaluation afterward |
| Needs demonstrations to start? | No | Yes, and a lot of them |
| Has an internal model of consequences? | Yes (the latent dynamics model) | No -- purely reactive |
| Can replan mid-episode? | Yes, every single step | No -- commits to a chunk, then regenerates |

This is why the two models don't have the same next blocker: TD-MPC2's
environment dependency is already satisfied (`PickPlaceEnv`, built and
measured this session), so it's unblocked purely on algorithm
implementation. Diffusion policy's data dependency is *not* satisfied --
this project doesn't yet have a good source of bulk demonstrations, which
is a real open decision (more teleop reps, a reliable scripted demo, or
using TD-MPC2's own successful rollouts once it exists -- each with
different implications for what the eventual comparison is actually
measuring, see the earlier discussion this doc doesn't repeat).

## Concrete wiring plan

For both models, the plan is to **adapt an existing, published
implementation** rather than write the algorithm from scratch -- both are
substantial, carefully-tuned pieces of ML engineering (ensembled
Q-functions and careful loss balancing for TD-MPC2; the diffusion
sampling schedule and U-Net/transformer architecture for diffusion
policy) where reimplementing from a paper description risks subtle,
hard-to-diagnose bugs for little benefit over adapting code the original
authors already got working.

- **TD-MPC2**: the main integration work is a thin adapter so the
  official codebase's training loop can drive `PickPlaceEnv`'s
  `reset()`/`step()` -- likely reformatting our dict observation
  (`wrist_rgb`/`top_rgb`/`proprio`) into whatever multi-modal input
  convention their encoder expects (TD-MPC2 already supports mixed
  state+pixel observations natively), plus scaling down planning
  hyperparameters (horizon, sample count, latent size) sized for this
  project's short manipulation episodes rather than TD-MPC2's usual
  longer benchmark tasks.
- **Diffusion policy**: the main integration work is a dataset loader
  converting this project's recorded episode format
  (`sim/output/teleop_episodes/episode_*.json`) into whatever
  observation-history-window + action-chunk batch format the adapted
  codebase expects. The bigger open item, as above, is *where the bulk
  demonstration data itself comes from* -- not the wiring.

## Open decisions before implementation starts

- Where diffusion policy's bulk training data comes from (see the
  structural asymmetry section above).
- Which specific public codebases to adapt for each.
- Training compute budget/duration expectations, now that `num_envs=16`
  is a measured ceiling for camera-based `PickPlaceEnv` rollouts (see
  docs/training_env.md).
