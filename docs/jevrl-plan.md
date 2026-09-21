# JevRL: classic environments and paper

**Goal:** Add four standard Gymnasium tasks, reproducible JEV-reward experiments,
checkpoint playback, and a measured, submission-ready research paper.

**Architecture:** Preserve Key Quest. Add a classic-environment suite using SB3
DQN with identical per-task hyperparameters across reward conditions. JEV grades
explicit, finite transition descriptions; the exact same description reuses its
stored answer. Native reward is retained only for evaluation/logging in JEV mode.
The rule control implements the same stated rubric, exposing how much of the
behavior is attributable to human reward engineering. No learned surrogate or
native-reward fallback is used. Every cache answer includes model identity.

**Tasks:** CartPole-v1, MountainCar-v0, Acrobot-v1, slippery FrozenLake-v1 (4x4).
Keep standard physics, termination and time limits. Use raw vector/discrete
observations for policy input, never the judge's features. Primary measurements
are native return and task-specific success, separately from training reward.

**Experiment protocol:** Three fixed training seeds (7, 19, 42), equal environment
step budgets within each task and three reward providers (native, rules, JEV).
Checkpoint evaluation uses 20 fixed development seeds; final evaluation uses
100 disjoint test seeds. Report all runs, mean and between-training-seed standard
deviation, fixed final checkpoints (no best-checkpoint selection). Freeze the
protocol before full runs. Evaluation cannot query or update JEV. Record steps,
seeds, versions, reward prompts, cache hash, model identities, API usage and cost.
Random policies are a separate reference. Keep any pilot runs separate.

## Implementation sequence

- [x] Generalize Score parsing without changing the original Key Quest defaults;
  test arbitrary criteria, invalid probabilities, and credential-safe failures.
- [x] Add `classic/envs.py` and `classic/rewards.py`; test official transitions,
  goal metrics, native reward isolation and exact-description cache persistence.
- [x] Add `classic/training.py`, classic CLI commands and a benchmark runner;
  test short actual training, checkpoint reload, evaluation RNG isolation,
  cancellation and final artifacts.
- [x] Run a separate smoke/pilot, freeze `experiments/protocol.json`, collect JEV
  rubric answers, then run every preregistered seed/condition and random baseline.
- [x] Add a classic browser page with task/provider/seed selectors, animated
  state replay, checkpoint slider and reward/native-return/success curves; test
  all four games, mobile layout, empty/error states and live start/stop controls.
- [x] Generate paper/README figures directly from the stored experiment records.
- [x] Verify primary references; write LaTeX with a first-page game/result teaser.
  Title: `JevRL: TypeSafe Jev Score as Reinforcement Learning Reward`.
  Team credit: BringAI Team. Individual author names and institutional affiliation omitted.
- [x] Audit claims against data and limitations, compile and visually inspect PDF,
  prepare a minimal arXiv source bundle and submission metadata.
- [x] Run tests/build/browser verification, review code, push only to the existing
  private repository. Inspect arXiv submission access; report real submission
  status without claiming an identifier or acceptance that has not occurred.

## Research boundaries

This is a small controlled systems study of a structured decision API as a reward
provider, not a new RL algorithm, calibration proof or general-game benchmark.
Human-written abstractions and reward scales are part of the method. Negative
results and rule/JEV agreement must appear in the paper. The user has authorized
an MIT source release; manuscript artifacts are kept outside the public repository.

## Delivery verification

- 36 completed runs, 3,240,000 environment steps and 360 saved checkpoints.
- 51 tests passed; Ruff checks and Python wheel/source builds passed.
- All four seed-7 JEV final checkpoints reproduce their 100 saved test outcomes.
- Browser checks cover all four games, providers/seeds, checkpoints, new offline
  JEV training, stop/save, downloads and mobile layout; original Key Quest works.
- Static export works under a project prefix. Manuscript downloads have been removed.
- Eight-page PDF visually inspected; the extracted arXiv source archive compiles
  independently with Tectonic. Team credit follows the request; personal names and affiliation are omitted.
- No Docker runtime was available; the Docker image was not built.
- The source release is authorized. arXiv submission is pending: the user has an author
  account, but this task has no tool controlling that authenticated author session.
  No arXiv identifier or completed submission is claimed.
