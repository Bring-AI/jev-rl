"""Matched DQN experiments with development checkpoints and held-out final tests."""

import importlib.metadata
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from ..judges import JudgeCancelled, JudgeError
from ..learning import save_json
from .envs import TASKS, make_env, observation_json
from .rewards import ScoreCache


@dataclass
class ClassicConfig:
    task: str = "cartpole"
    provider: str = "rules"
    seed: int = 7
    total_steps: int = 60000
    checkpoint_steps: int = 10000
    eval_episodes: int = 20
    test_episodes: int = 100
    max_calls: int = 2000
    cache_path: str | None = None
    frozen_cache: bool = False
    learning_rate: float = 0.001
    gamma: float = 0.99
    batch_size: int = 64
    learning_starts: int = 1000
    buffer_size: int = 50000
    train_freq: int = 4
    target_update_interval: int = 1000
    exploration_fraction: float = 0.5
    exploration_final_eps: float = 0.05

    def __post_init__(self):
        if self.task not in TASKS or self.provider not in ("native", "jev", "rules"):
            raise ValueError("Unknown classic task or provider")
        if not 1 <= self.total_steps <= 2000000 or not 1 <= self.checkpoint_steps <= 2000000:
            raise ValueError("Steps and checkpoint interval must be 1..2000000")
        if not 1 <= self.eval_episodes <= 1000 or not 1 <= self.test_episodes <= 1000:
            raise ValueError("Evaluation episodes must be 1..1000")
        if not 0 <= self.seed < 2**32 or not 1 <= self.max_calls <= 10000:
            raise ValueError("Invalid seed or API request budget")
        if not 0 < self.learning_rate <= 1 or not 0 <= self.gamma < 1:
            raise ValueError("Invalid optimizer parameters")


def evaluate_policy(model, task, episodes=20, seed_start=10000, record=True):
    """Use native rewards and fresh env RNGs, with no score cache or HTTP client."""
    numpy_state, python_state, torch_state = (
        np.random.get_state(),
        random.getstate(),
        torch.get_rng_state(),
    )
    env = make_env(task, "native")
    returns, successes, lengths = [], [], []
    replay = None
    try:
        for index in range(episodes):
            seed = seed_start + index
            obs, _ = env.reset(seed=seed)
            # Action randomness must not share Gymnasium's environment RNG stream.
            rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4A455652]))
            frames = [{"observation": observation_json(obs), "action": None}]
            total = 0.0
            for _step in range(TASKS[task].horizon):
                action = (
                    int(rng.integers(env.action_space.n))
                    if model is None
                    else int(model.predict(obs, deterministic=True)[0])
                )
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                if index == 0 and record:
                    frames.append(
                        {
                            "observation": observation_json(obs),
                            "action": action,
                            "native_reward": reward,
                            "terminated": terminated,
                            "truncated": truncated,
                        }
                    )
                if terminated or truncated:
                    break
            returns.append(total)
            successes.append(bool(info["is_success"]))
            lengths.append(env.steps)
            if index == 0 and record:
                replay = {
                    "seed": seed,
                    "frames": frames,
                    "native_return": total,
                    "success": successes[-1],
                    "steps": lengths[-1],
                }
    finally:
        env.close()
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
    return {
        "episodes": episodes,
        "seed_start": seed_start,
        "native_return_mean": float(np.mean(returns)),
        "native_return_std": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "mean_steps": float(np.mean(lengths)),
        "returns": returns,
        "successes": successes,
        "lengths": lengths,
        "replay": replay,
    }


def train_classic(config, output, on_progress=None, stop=None):
    output = Path(output)
    if (output / "run.json").exists():
        raise ValueError("Output already contains a run; choose a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    torch.set_num_threads(1)
    cache = None
    if config.provider != "native":
        cache = ScoreCache(
            config.task, config.provider, config.cache_path, config.frozen_cache, config.max_calls
        )
        cache.stop = stop
    env = make_env(config.task, config.provider, cache)
    model = DQN(
        "MlpPolicy",
        env,
        seed=config.seed,
        device="cpu",
        verbose=0,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        train_freq=config.train_freq,
        gradient_steps=1,
        target_update_interval=config.target_update_interval,
        exploration_fraction=config.exploration_fraction,
        exploration_final_eps=config.exploration_final_eps,
        policy_kwargs={"net_arch": [64, 64]},
    )
    history, checkpoints = [], []
    status, error = "completed", None
    started = time.monotonic()

    def checkpoint():
        steps = int(model.num_timesteps)
        relative = f"checkpoints/step-{steps:07d}.zip"
        model.save(output / relative)
        evaluation = evaluate_policy(model, config.task, config.eval_episodes)
        point = {
            "steps": steps,
            "episode": len(history),
            "evaluation": evaluation,
            "model_file": relative,
        }
        if checkpoints and checkpoints[-1]["steps"] == steps:
            checkpoints[-1] = point
        else:
            checkpoints.append(point)
        save_json(output / relative.replace(".zip", ".json"), point)
        return point

    class Callback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.native_return, self.score_return, self.length = 0.0, 0.0, 0
            self.next_checkpoint = config.checkpoint_steps

        def _on_step(self):
            info = self.locals["infos"][0]
            self.native_return += float(info["native_reward"])
            self.score_return += float(self.locals["rewards"][0])
            self.length += 1
            if self.locals["dones"][0]:
                history.append(
                    {
                        "episode": len(history) + 1,
                        "steps": self.num_timesteps,
                        "length": self.length,
                        "native_return": self.native_return,
                        "reward_return": self.score_return,
                        "success": bool(info["is_success"]),
                        "epsilon": float(model.exploration_rate),
                    }
                )
                self.native_return, self.score_return, self.length = 0.0, 0.0, 0
            point = None
            if (
                self.num_timesteps >= self.next_checkpoint
                and self.num_timesteps < config.total_steps
            ):
                point = checkpoint()
                self.next_checkpoint += config.checkpoint_steps
            if on_progress and (point or self.locals["dones"][0]):
                on_progress(
                    {
                        "steps": self.num_timesteps,
                        "episodes": len(history),
                        "record": history[-1] if history else None,
                        "checkpoint": point,
                        "reward": cache.stats() if cache else {"provider": "native"},
                    }
                )
            return not (stop and stop.is_set())

    initial = checkpoint()
    if on_progress:
        on_progress(
            {
                "steps": 0,
                "episodes": 0,
                "record": None,
                "checkpoint": initial,
                "reward": cache.stats() if cache else {"provider": "native"},
            }
        )
    try:
        if stop and stop.is_set():
            status = "stopped"
        else:
            model.learn(total_timesteps=config.total_steps, callback=Callback())
            if stop and stop.is_set():
                status = "stopped"
    except (KeyboardInterrupt, JudgeCancelled):
        status = "stopped"
    except JudgeError as exc:
        status, error = "failed", str(exc)
    finally:
        env.close()
        if cache:
            cache.save(output / "judgments.json")
            cache.close()
    checkpoint()
    test = evaluate_policy(model, config.task, config.test_episodes, seed_start=100000)
    result = {
        "schema_version": 1,
        "suite": "classic",
        "status": status,
        "error": error,
        "task": TASKS[config.task].spec(),
        "provider": config.provider,
        "config": asdict(config),
        "algorithm": "Stable-Baselines3 DQN, MLP(64,64)",
        "steps": int(model.num_timesteps),
        "history": history,
        "checkpoints": checkpoints,
        "test": test,
        "reward": cache.stats() if cache else {"provider": "native", "api_calls": 0},
        "duration_seconds": round(time.monotonic() - started, 3),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("gymnasium", "stable-baselines3", "torch", "numpy")
        },
        "python": platform.python_version(),
        "evaluation_note": "Development seeds 10000+; final held-out seeds 100000+. Native reward; no judge calls.",
    }
    save_json(output / "run.json", result)
    return result


def brief(result):
    return {
        "task": result["task"]["slug"],
        "provider": result["provider"],
        "seed": result["config"]["seed"],
        "status": result["status"],
        "steps": result["steps"],
        "native_return": result["test"]["native_return_mean"],
        "success_rate": result["test"]["success_rate"],
        "duration_seconds": result["duration_seconds"],
    }
