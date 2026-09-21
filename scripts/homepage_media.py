"""Render real saved JEV rollouts into a four-game checkpoint GIF gallery.

Replays every recorded action in Gymnasium and checks observations/outcomes before
exporting. This neither trains policies nor calls a judge API.
"""

import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jev_reward.classic.envs import TASKS, success

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "jev_reward" / "static" / "classic"
MEDIA = STATIC / "media"
STAGES = (("untrained", 0), ("10k", 10000), ("30k", 30000), ("final", None))
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(size, bold=False):
    path = FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size)


def card(screen, task, training_steps, frame_index, replay):
    image = Image.new("RGB", (320, 252), "#f4f5ef")
    draw = ImageDraw.Draw(image)
    draw.text((12, 11), task.name, font=font(13, True), fill="#172820")
    label = f"{training_steps:,} steps"
    draw.text((308, 12), label, anchor="ra", font=font(11), fill="#66766b")
    draw.rounded_rectangle((8, 36, 312, 214), radius=7, fill="white")
    visual = Image.fromarray(screen)
    visual.thumbnail((298, 172), Image.Resampling.LANCZOS)
    image.paste(visual, ((320 - visual.width) // 2, 39 + (172 - visual.height) // 2))
    draw.text(
        (12, 225),
        f"REPLAY {frame_index:03d} / {replay['steps']:03d}",
        font=font(10),
        fill="#66766b",
    )
    finished = frame_index == replay["steps"]
    outcome = "SUCCESS" if replay["success"] else "NOT SOLVED"
    draw.text(
        (308, 223),
        outcome if finished else "JEV REWARD",
        anchor="ra",
        font=font(11, True),
        fill="#267c56" if not finished or replay["success"] else "#a46b40",
    )
    draw.line((12, 247, 308, 247), fill="#d9dfd5", width=3)
    draw.line((12, 247, 12 + 296 * frame_index / replay["steps"], 247), fill="#267c56", width=3)
    return image


def render(task, point, stage):
    replay = point["evaluation"]["replay"]
    frames = replay["frames"]
    stride = max(1, math.ceil((len(frames) - 1) / 120))
    sampled = set(range(0, len(frames), stride)) | {len(frames) - 1}
    output = []
    env = gym.make(task.env_id, render_mode="rgb_array")
    try:
        observation, _ = env.reset(seed=replay["seed"])
        total_reward = 0
        terminated = truncated = False
        for index, frame in enumerate(frames):
            if index:
                if terminated or truncated:
                    raise ValueError("Saved replay continues beyond episode end")
                observation, reward, terminated, truncated, _ = env.step(frame["action"])
                total_reward += reward
                if terminated != frame["terminated"] or truncated != frame["truncated"]:
                    raise ValueError("Recorded episode outcome differs from Gymnasium")
            np.testing.assert_allclose(observation, frame["observation"], atol=1e-6, rtol=1e-6)
            if index in sampled:
                output.append(card(env.render(), task, point["steps"], index, replay))
        if not terminated and not truncated:
            raise ValueError("Incomplete saved episode")
        if not math.isclose(total_reward, replay["native_return"], abs_tol=1e-8):
            raise ValueError("Recorded native return differs from replay")
        if (
            success(task.slug, observation, terminated, truncated, replay["steps"])
            != replay["success"]
        ):
            raise ValueError("Recorded success differs from replay")
    finally:
        env.close()
    # Complete episodes are time-compressed, with a readable pause at the end.
    duration = round(max(60, min(180, 8000 / len(output))) / 10) * 10
    durations = [duration] * (len(output) - 1) + [1500]
    filename = f"{task.slug}-{stage}.gif"
    output[0].save(
        MEDIA / filename,
        save_all=True,
        append_images=output[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    output[0].save(MEDIA / f"{task.slug}-{stage}.png")
    return {
        "task": task.slug,
        "stage": stage,
        "training_steps": point["steps"],
        "gif": filename,
        "poster": f"{task.slug}-{stage}.png",
        "frames": len(output),
        "replay_steps": replay["steps"],
        "success": replay["success"],
        "native_return": replay["native_return"],
        "duration_ms": sum(durations),
    }


def readme_gallery(entries, chinese=False):
    paths = "src/jev_reward/static/classic/media"
    lines = [
        "<!-- checkpoint-gallery:start -->",
        "## 看见策略如何学会玩游戏" if chinese else "## Watch the policy learn",
        "",
        (
            "每列一个游戏，每行一个训练阶段。使用 **JEV reward、训练种子 7、回放种子 10000**；"
            "展示完整局并压缩播放时间，保留中间失败。DQN 用真实环境步数记录训练进度。"
            if chinese
            else "One game per column; one checkpoint per row. **JEV reward, training seed 7, replay seed 10000.** "
            "Complete episodes are time-compressed, including failed intermediate policies. DQN progress is measured in environment steps."
        ),
        "",
        "| "
        + ("训练阶段" if chinese else "Training stage")
        + " | CartPole | MountainCar | Acrobot | FrozenLake |",
        "|:--|:--:|:--:|:--:|:--:|",
    ]
    labels = (
        ["未训练 · 0 步", "10,000 步", "30,000 步", "最终策略"]
        if chinese
        else ["Untrained · 0 steps", "10,000 steps", "30,000 steps", "Final policy"]
    )
    for (stage, _), label in zip(STAGES, labels, strict=True):
        cells = []
        for slug, task in TASKS.items():
            item = next(e for e in entries if e["task"] == slug and e["stage"] == stage)
            outcome = (
                ("成功" if item["success"] else "未通关")
                if chinese
                else ("Success" if item["success"] else "Not solved")
            )
            cells.append(
                f'<img src="{paths}/{item["gif"]}" width="220" alt="{task.name}, {item["training_steps"]:,} training steps, {outcome}"><br>'
                f"{item['training_steps']:,} steps · {outcome}"
            )
        lines.append("| **" + label + "** | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            (
                "动图是固定单局回放；上图 CartPole 曲线为三个训练种子的 Total score（游戏原始累计得分）均值。"
                if chinese
                else "Each animation is one fixed rollout. The CartPole curve above shows mean total score (the game’s original cumulative score) across three training seeds."
            ),
            "<!-- checkpoint-gallery:end -->",
        ]
    )
    return "\n".join(lines)


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    entries = []
    sources = {}
    for slug, task in TASKS.items():
        path = STATIC / "data" / f"{slug}-jev-7.json"
        run = json.loads(path.read_text())
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for stage, step in STAGES:
            point = (
                run["checkpoints"][-1]
                if step is None
                else next(p for p in run["checkpoints"] if p["steps"] == step)
            )
            entries.append(render(task, point, stage))
            print(f"Rendered and verified {slug}, {point['steps']:,} training steps", flush=True)
    manifest = {
        "provider": "jev",
        "training_seed": 7,
        "replay_seed": 10000,
        "stages": [s for s, _ in STAGES],
        "tasks": [t.spec() for t in TASKS.values()],
        "note": "Complete fixed rollouts, time-compressed. Actual observations and outcomes are verified against Gymnasium.",
        "source_sha256": sources,
        "entries": entries,
    }
    (MEDIA / "gallery.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, chinese in (("README.md", False), ("README.zh-CN.md", True)):
        path = ROOT / name
        text = path.read_text()
        start, end = "<!-- checkpoint-gallery:start -->", "<!-- checkpoint-gallery:end -->"
        gallery = readme_gallery(entries, chinese)
        if start in text:
            before, rest = text.split(start, 1)
            _, after = rest.split(end, 1)
            text = before + gallery + after
        else:
            anchor = "## Four classic RL environments" if not chinese else "已接入 **CartPole"
            text = text.replace(anchor, gallery + "\n\n" + anchor, 1)
        path.write_text(text)


if __name__ == "__main__":
    main()
