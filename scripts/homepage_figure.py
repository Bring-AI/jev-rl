"""Combine the JEV reward loop with the measured CartPole learning curve."""

import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "src/jev_reward/static/classic/media"
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor("#f4f5ef")
    ax.set_facecolor("#f4f5ef")
    for provider, label, color in (
        ("jev", "JEV", "#267c56"),
        ("rules", "Human design", "#c48148"),
        ("native", "Native", "#5279bc"),
    ):
        runs = [
            json.loads(
                (ROOT / f"experiments/jevrl-v1/results/cartpole-{provider}-{seed}.json").read_text()
            )
            for seed in (7, 19, 42)
        ]
        x = np.array([c["steps"] for c in runs[0]["checkpoints"]]) / 1000
        y = np.array(
            [[c["evaluation"]["native_return_mean"] for c in r["checkpoints"]] for r in runs]
        )
        mean, sd = y.mean(axis=0), y.std(axis=0, ddof=1)
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.10)
        ax.plot(
            x,
            mean,
            color=color,
            label=label,
            linewidth=2.7,
            linestyle="--" if provider == "rules" else "-",
            zorder=4 if provider == "jev" else 3,
        )
    ax.set_title(
        "CartPole · learning to balance",
        loc="left",
        fontsize=19,
        fontweight="bold",
        color="#172820",
        pad=20,
    )
    ax.set_xlabel("Training steps (×1,000)", labelpad=12)
    ax.set_ylabel("Total score", labelpad=12)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 530)
    ax.set_yticks(range(0, 501, 100))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#bdc9b8")
    ax.tick_params(colors="#66766b")
    ax.grid(axis="y", color="#d9dfd5")
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.31), ncol=3, frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.85, bottom=0.26)
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg")
    plt.close(fig)
    root = ET.Element(
        f"{{{SVG}}}svg",
        {
            "width": "2040",
            "height": "620",
            "viewBox": "0 0 2040 620",
            "role": "img",
            "aria-labelledby": "figure-title",
        },
    )
    ET.SubElement(
        root, f"{{{SVG}}}title", {"id": "figure-title"}
    ).text = "JEV supplies the reward: learning loop and measured CartPole total score"
    ET.SubElement(
        root, f"{{{SVG}}}rect", {"width": "2040", "height": "620", "rx": "18", "fill": "#f4f5ef"}
    )
    loop = ET.parse(MEDIA / "reward-loop.svg").getroot()
    loop.attrib.update({"x": "0", "y": "45", "width": "1320", "height": "530"})
    root.append(loop)
    curve = ET.fromstring(buffer.getvalue())
    curve.attrib.update({"x": "1340", "y": "20", "width": "680", "height": "550"})
    root.append(curve)
    ET.SubElement(
        root,
        f"{{{SVG}}}text",
        {
            "x": "1670",
            "y": "580",
            "text-anchor": "middle",
            "font-family": "DejaVu Sans, Arial, sans-serif",
            "font-size": "15",
            "fill": "#66766b",
        },
    ).text = "Mean ± SD · 3 seeds · 20 evaluation episodes / checkpoint"
    path = MEDIA / "jevrl-cartpole.svg"
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=False)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True)
        page = browser.new_page(viewport={"width": 2040, "height": 620}, device_scale_factor=1.5)
        page.goto(path.as_uri())
        page.screenshot(path=str(MEDIA / "jevrl-cartpole.png"))
        browser.close()
    print("Generated homepage diagram + measured CartPole curve.")


if __name__ == "__main__":
    main()
