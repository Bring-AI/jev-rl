# Homepage reward diagram and training gallery

User layout: reward-flow diagram, existing four-game result figure immediately
below it, then a matrix of animated games at successive training stages.
Apply the same sequence to both READMEs and the local classic homepage.

- [x] Add a repository-native SVG: agent acts in environment, observed transitions
  go to JEV, and the JEV-derived reward returns to the learning agent. Highlight
  “Fast, typesafe, and accurate” and explicitly identify JEV as the reward source.
- [x] Generate 16 real Gymnasium GIFs from saved seed-7 JEV checkpoints at 0,
  10,000, 30,000 and final training steps. Keep replay seed 10000 fixed; preserve
  every recorded action and outcome. Label time compression and actual budgets.
  Validate reproduced observations, termination and native return while rendering.
- [x] Display four game columns and four checkpoint rows in English/Chinese
  READMEs and the web page. Keep native results and existing interactive training
  below. Provide animation pause and reduced-motion support in the web gallery.
- [x] Check real GIF frame counts, browser animation, section order, mobile
  scrolling and the static export under a project prefix. Commit and push the
  reviewed change to the existing private repository.

The diagram and animations describe the existing experiment; no retraining,
new model requests, changed paper claims or publication is needed. DQN progress
uses actual environment steps rather than inventing optimizer epochs. Every
animation uses the preselected seed, including failed intermediate rollouts.

## Verification

- All 16 GIFs reproduce saved Gymnasium observations, termination and outcomes.
  Frame counts, durations and source SHA-256 hashes match the gallery manifest.
- Live and project-prefix static browser checks pass: section order, image loads,
  actual animation, pause/replay, reduced motion and 390px horizontal scrolling.
- Existing classic browser checks pass: four games, reward/seed selection,
  checkpoints, offline JEV training, stop/save, downloads and mobile layout.
- Ruff checks and formatting pass for the new generators and browser check.

## September 22 layout revision

The lead figure now combines the reward loop and the measured CartPole learning
curve. The separate four-game overview is removed from both homepages; the
16 replay animations remain. User-facing names are Human design and Total score
(the original cumulative game score). Recorded provider IDs and metric fields
stay unchanged. Regenerate the combined figure with `scripts/homepage_figure.py`.
Paper legends/tables and the manuscript download use the same display labels.
Live/static browser checks and the rebuilt eight-page manuscript pass.
