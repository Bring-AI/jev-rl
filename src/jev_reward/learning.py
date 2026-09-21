"""Transparent tabular Q-learning and reproducible evaluation."""

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .game import Game, State
from .judges import QUESTION, JudgeCancelled, JudgeError


@dataclass
class TrainConfig:
    episodes: int = 350
    seed: int = 7
    max_steps: int = 80
    alpha: float = 0.25
    gamma: float = 0.95
    episode_delay: float = 0.0
    checkpoint_every: int = 25

    def __post_init__(self):
        if not 1 <= self.episodes <= 5000 or not 1 <= self.max_steps <= 200:
            raise ValueError("episodes must be 1..5000 and max_steps 1..200")
        if not 0 < self.alpha <= 1 or not 0 <= self.gamma < 1:
            raise ValueError("alpha must be (0,1], gamma must be [0,1)")
        if not 0 <= self.seed <= 2**32 - 1 or not 0 <= self.episode_delay <= 1:
            raise ValueError("Invalid seed or episode delay")
        if not 1 <= self.checkpoint_every <= 5000:
            raise ValueError("checkpoint_every must be 1..5000")


class QAgent:
    def __init__(self, seed=7, alpha=0.25, gamma=0.95):
        self.q = np.zeros((Game.n_states, 4), dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self.alpha = alpha
        self.gamma = gamma

    def act(self, state, epsilon=0.0, rng=None):
        rng = self.rng if rng is None else rng
        if rng.random() < epsilon:
            return int(rng.integers(4))
        values = self.q[Game().encode(state)]
        return int(rng.choice(np.flatnonzero(values == values.max())))

    def update(self, before, action, reward, after, terminal):
        row = self.q[Game().encode(before)]
        target = reward if terminal else reward + self.gamma * self.q[Game().encode(after)].max()
        row[action] += self.alpha * (target - row[action])


def rollout(agent, seed, max_steps=80, random_policy=False):
    rng, game = np.random.default_rng(seed), Game()
    state = game.start(rng)
    frames = [{**asdict(state), "event": "start", "action": None}]
    for _ in range(max_steps):
        action = int(rng.integers(4)) if random_policy else agent.act(state, rng=rng)
        transition = game.step(state, action)
        state = transition.after
        frames.append({**asdict(state), "event": transition.event, "action": action})
        if transition.terminated:
            break
    return {"frames": frames, "success": frames[-1]["event"] == "win", "steps": len(frames) - 1}


def evaluate(agent, seed=10000, episodes=60, max_steps=80):
    episodes_data = [rollout(agent, seed + i, max_steps) for i in range(episodes)]
    return {
        "success_rate": sum(e["success"] for e in episodes_data) / episodes,
        "mean_steps": sum(e["steps"] for e in episodes_data) / episodes,
        "episodes": episodes,
        "seed": seed,
    }


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def judged_replay(agent, judge, max_steps):
    """Replay an existing policy; attach evidence without making judge calls."""
    replay = rollout(agent, 10000, max_steps)
    game = Game()
    for i in range(1, len(replay["frames"])):
        previous, frame = replay["frames"][i - 1], replay["frames"][i]
        transition = game.step(
            State(previous["x"], previous["y"], previous["has_key"]), frame["action"]
        )
        key = json.dumps(transition.context(), sort_keys=True)
        if key in judge.cache:
            frame["verdict"] = {**judge.cache[key]["verdict"], "cached": True}
    return replay


def checkpoint(agent, judge, config, output, episode, partial=False):
    evaluation = evaluate(agent, max_steps=config.max_steps)
    replay = judged_replay(agent, judge, config.max_steps)
    suffix = "-partial" if partial else ""
    relative = f"checkpoints/episode-{episode:05d}{suffix}.json"
    saved = {
        "schema_version": 1,
        "algorithm": "tabular-q-learning",
        "provider": judge.provider,
        "episode": episode,
        "partial_episode": partial,
        "evaluation": evaluation,
        "replay": replay,
        "q": agent.q.tolist(),
        "config": asdict(config),
        "game": Game().spec(),
        "judge": judge.stats(),
    }
    (output / "checkpoints").mkdir(exist_ok=True)
    save_json(output / relative, saved)
    return {
        "episode": episode,
        "partial_episode": partial,
        "evaluation": evaluation,
        "replay": replay,
        "file": relative,
    }


def train(config, judge, output, on_progress: Callable | None = None, stop=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    game, agent = Game(), QAgent(config.seed, config.alpha, config.gamma)
    checkpoints = [checkpoint(agent, judge, config, output, 0)]
    before = checkpoints[0]["evaluation"]
    before_replay = checkpoints[0]["replay"]
    history, latest_frames = [], []
    status, error = "completed", None
    started = time.monotonic()
    judge.stop = stop
    try:
        for episode in range(config.episodes):
            if stop and stop.is_set():
                status = "stopped"
                break
            epsilon = max(0.04, 1 - episode / max(1, config.episodes * 0.8))
            state = game.start(agent.rng)
            frames = [{**asdict(state), "event": "start", "action": None}]
            total, won = 0.0, False
            for _ in range(config.max_steps):
                if stop and stop.is_set():
                    status = "stopped"
                    break
                action = agent.act(state, epsilon)
                transition = game.step(state, action)
                verdict = judge.score(transition)
                reward = verdict["reward"]
                # Time-limit truncation bootstraps. Only genuine terminal states do not.
                agent.update(state, action, reward, transition.after, transition.terminated)
                state = transition.after
                total += reward
                frames.append(
                    {
                        **asdict(state),
                        "event": transition.event,
                        "action": action,
                        "verdict": verdict,
                    }
                )
                if transition.terminated:
                    won = transition.event == "win"
                    break
            if status == "stopped":
                break
            record = {
                "episode": episode + 1,
                "return": round(total, 6),
                "steps": len(frames) - 1,
                "success": won,
                "epsilon": round(epsilon, 4),
                "outcome": "win" if won else "lava" if frames[-1]["event"] == "lava" else "timeout",
            }
            history.append(record)
            latest_frames = frames
            point = None
            if (episode + 1) % config.checkpoint_every == 0 or episode + 1 == config.episodes:
                point = checkpoint(agent, judge, config, output, episode + 1)
                checkpoints.append(point)
            if on_progress:
                on_progress(
                    {
                        "record": record,
                        "frames": frames,
                        "judge": judge.stats(),
                        "before": before,
                        "provider": judge.provider,
                        "checkpoint": point,
                        "initial_checkpoint": checkpoints[0],
                    }
                )
            if config.episode_delay:
                if stop:
                    stop.wait(config.episode_delay)
                else:
                    time.sleep(config.episode_delay)
    except (KeyboardInterrupt, JudgeCancelled):
        status = "stopped"
    except JudgeError as exc:
        status, error = "failed", str(exc)
    finally:
        judge.close()
    if status != "completed" or checkpoints[-1]["episode"] != len(history):
        checkpoints.append(
            checkpoint(agent, judge, config, output, len(history), partial=status != "completed")
        )
    after, replay = checkpoints[-1]["evaluation"], checkpoints[-1]["replay"]
    result = {
        "schema_version": 1,
        "status": status,
        "error": error,
        "provider": judge.provider,
        "config": asdict(config),
        "game": game.spec(),
        "history": history,
        "checkpoints": checkpoints,
        "before": before,
        "after": after,
        "before_replay": before_replay,
        "replay": replay,
        "last_training_frames": latest_frames,
        "judge": judge.stats(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "reward_formula": "sum(P(level) * [-1, -.15, -.12, .08, .6, 1][level])",
        "evaluation_note": "Same 60 seeds on the training map; no judge calls, no unseen-map claim.",
    }
    save_json(
        output / "policy.json",
        {
            "algorithm": "tabular-q-learning",
            "q": agent.q.tolist(),
            "config": asdict(config),
            "game": game.spec(),
        },
    )
    save_json(
        output / "judgments.json",
        {"provider": judge.provider, "question": QUESTION, "entries": list(judge.cache.values())},
    )
    save_json(output / "run.json", result)
    return result
