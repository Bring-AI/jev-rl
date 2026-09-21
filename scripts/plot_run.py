"""Plot measured rewards, held-out rollout seeds and outcomes for the README."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, default=Path("runs/jev-checkpoints/run.json"))
parser.add_argument("--output", type=Path, default=Path("assets/training.png"))
args = parser.parse_args()
run = json.loads(args.run.read_text())
history, points = run["history"], run["checkpoints"]
episodes = np.array([r["episode"] for r in history])
rewards = np.array([r["return"] for r in history])
window = 20
smoothed = np.array([rewards[max(0, i - window + 1) : i + 1].mean() for i in range(len(rewards))])
checkpoint_x = [p["episode"] for p in points]
checkpoint_y = [p["evaluation"]["success_rate"] for p in points]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#394440",
        "text.color": "#182720",
        "xtick.color": "#627169",
        "ytick.color": "#627169",
    }
)
fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9), gridspec_kw={"width_ratios": [1.15, 1, 1]})
fig.patch.set_facecolor("#f7faf6")
for ax in axes:
    ax.set_facecolor("#f7faf6")
    ax.grid(axis="y", color="#dae4d8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Training episode")
    ax.spines["left"].set_color("#bdcdbd")
    ax.spines["bottom"].set_color("#bdcdbd")

axes[0].plot(episodes, rewards, color="#93b4a0", alpha=0.4, linewidth=0.7, label="Per episode")
axes[0].plot(episodes, smoothed, color="#20784c", linewidth=2.3, label="20-episode mean")
axes[0].set_title("1  Reward supplied by the judge", loc="left", fontweight="bold", pad=14)
axes[0].set_ylabel("Episode return (sum of rewards)")
axes[0].legend(frameon=False, fontsize=8, loc="lower right")

axes[1].plot(checkpoint_x, checkpoint_y, marker="o", markersize=4, color="#226aab", linewidth=2.3)
axes[1].fill_between(checkpoint_x, checkpoint_y, alpha=0.08, color="#226aab")
axes[1].yaxis.set_major_formatter(PercentFormatter(1))
axes[1].set_ylim(-0.04, 1.1)
axes[1].set_ylabel("Evaluation win rate")
axes[1].set_title("2  Can the saved policy win?", loc="left", fontweight="bold", pad=14)
axes[1].text(
    0.98,
    0.06,
    "60 evaluation episodes / checkpoint\nSame map and seeds; no exploration",
    transform=axes[1].transAxes,
    ha="right",
    color="#627169",
    fontsize=8,
    linespacing=1.5,
)
axes[1].annotate(
    f"{100 * checkpoint_y[-1]:.0f}%",
    (checkpoint_x[-1], checkpoint_y[-1]),
    xytext=(-25, 10),
    textcoords="offset points",
    color="#226aab",
    weight="bold",
)

batch = run["config"]["checkpoint_every"]
ends, wins, lava, timeouts = [], [], [], []
for start in range(0, len(history), batch):
    group = history[start : start + batch]
    ends.append(group[-1]["episode"])
    wins.append(sum(row["success"] for row in group))
    lava.append(sum(row.get("outcome") == "lava" for row in group))
    timeouts.append(sum(row.get("outcome") == "timeout" for row in group))
axes[2].bar(ends, wins, width=batch * 0.7, color="#3e9364", label="Win")
axes[2].bar(ends, lava, bottom=wins, width=batch * 0.7, color="#d68166", label="Lava")
axes[2].bar(
    ends,
    timeouts,
    bottom=np.array(wins) + lava,
    width=batch * 0.7,
    color="#c8cdb9",
    label="Timeout",
)
axes[2].set_title("3  Wins and losses while learning", loc="left", fontweight="bold", pad=14)
axes[2].set_ylabel(f"Training games per {batch}-episode group")
axes[2].legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01))
axes[2].set_ylim(0, batch * 1.23)

judge = run["judge"]
model = ", ".join(judge.get("served_models", [])) or judge["model"]
gateway = {"openrouter": "OpenRouter", "typesafe": "TypeSafe direct"}.get(
    judge.get("gateway"), "the configured endpoint"
)
label = (
    f"Official TypeSafe JEV via {gateway}"
    if run["provider"] == "jev"
    else "Deterministic rules demo (not JEV)"
)
fig.suptitle(
    "JEV REWARD ARCADE  /  Learning to play Key Quest",
    x=0.055,
    ha="left",
    y=0.99,
    fontsize=18,
    fontweight="bold",
)
fig.text(
    0.055,
    0.892,
    f"{label}  |  {model}  |  seed {run['config']['seed']}",
    fontsize=10,
    color="#627169",
)
cost = judge.get("cost_usd")
cost_label = f"${cost:.5f}" if cost is not None else "not reported"
fig.text(
    0.055,
    0.02,
    f"Measured run: {len(history)} episodes  ·  {judge['api_calls']} API requests  ·  "
    f"API cost {cost_label}  ·  No environment reward added. Engineered route-distance feature; fixed-map experiment.",
    fontsize=9,
    color="#627169",
)
fig.subplots_adjust(left=0.055, right=0.985, top=0.77, bottom=0.17, wspace=0.34)
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=180, facecolor=fig.get_facecolor())
fig.savefig(args.output.with_suffix(".svg"), facecolor=fig.get_facecolor())
svg_path = args.output.with_suffix(".svg")
svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
print(f"Saved {args.output} and {args.output.with_suffix('.svg')}")
