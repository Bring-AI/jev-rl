> The original Key Quest design is below. The four-game DQN suite is documented in [JevRL plan](jevrl-plan.md).

# JEV Reward Arcade

Build a small, reproducible experiment: a Q-learning agent plays Key Quest,
a grid game where it must collect a key, avoid lava and reach an exit.
JEV judges transitions; the agent chooses actions and updates its Q table.
The browser shows the game, training history, reward provenance and evaluation.

## Scope and choices

- Python + FastAPI + NumPy with a dependency-free browser canvas UI. Runs on CPU.
- Tabular Q-learning keeps the complete learning loop inspectable and fast.
  A neural DQN would add install cost without helping this tiny state space.
- Official JEV Score API (OpenRouter or TypeSafe direct) plus an explicitly labelled deterministic demo judge.
  The demo never claims to contain JEV outputs. No silent fallback on API errors.
- JEV consumes structured game state, not screenshots: the official interface is
  a typed state/question API. The UI renders that same state for humans.
- Judge output is the only training reward. Environment success is used for
  evaluation and episode termination; it is never added to the reward.
- The judge sees safe route distance as an engineered observation. That uses
  environment geometry and is a deliberate scaffold, not pixel-only learning.
- Cache repeated semantic transitions within a run. Cap HTTP attempts, including
  retries. Keep credentials on the server, outside artifacts and source control.
- Save policy, metrics, judging evidence and replay as JSON. Export a portable
  replay viewer, with a bundled demo for GitHub Pages.
- Save the policy, 60-game evaluation and replay every 25 episodes (configurable).
  Expose the saved stages through a browser timeline; evaluation makes no API calls.
- Plot real episode returns, checkpoint evaluation win rate, and training outcomes
  for the README using Matplotlib. Keep actual JEV and rules results distinguishable.
- MIT license, bilingual README, tests, CI, Docker and one-command quick start.

## Verification

Check terminal and key mechanics, Bellman updates, API contract and failure modes,
reward provenance, budget/cache behavior, deterministic training improvement,
server lifecycle and browser controls at desktop and mobile sizes. Publish only
after tests and a real browser run. Actual JEV quality requires an authenticated JEV endpoint;
test fixtures and deterministic demo results are not evidence of JEV performance.

## Delivery plan

1. Tests and implementation for game, judges, Q-learning and run artifacts.
2. Background training API, controls and local dashboard with a playable view.
3. Reproducible demo run, browser smoke test, docs, CI and packaging.
4. Independent code review, corrections, final verification and GitHub publish.

The user requested immediate implementation and GitHub publication; proceed with
these defaults while requesting only missing credentials/account information.

Update: the user requested keeping development local and the GitHub repository
private until the program is ready. Delivery goes to the designated private repo;
automation stays as inactive templates in docs/workflows because the authenticated
GitHub credential lacks workflow scope. The Pages template is manual and refuses
private repositories. Official JEV is now
verified through OpenRouter; the bundled run records model, provider, request IDs,
token usage and cost. The user's key is kept outside version control.
