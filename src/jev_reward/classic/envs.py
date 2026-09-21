"""Official environment dynamics; task outcomes are independent of judge scores."""

from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class Task:
    slug: str
    name: str
    env_id: str
    horizon: int
    steps: int
    objective: str
    success_definition: str
    actions: tuple[str, ...]

    def spec(self):
        return asdict(self)


TASKS = {
    t.slug: t
    for t in (
        Task(
            "cartpole",
            "CartPole",
            "CartPole-v1",
            500,
            60000,
            "Balance the pole for 500 steps.",
            "Survive all 500 steps without termination.",
            ("Push left", "Push right"),
        ),
        Task(
            "mountaincar",
            "MountainCar",
            "MountainCar-v0",
            200,
            120000,
            "Build momentum and reach the flag on the right hill.",
            "Reach position >= 0.5 within 200 steps.",
            ("Left", "Coast", "Right"),
        ),
        Task(
            "acrobot",
            "Acrobot",
            "Acrobot-v1",
            500,
            120000,
            "Swing the two-link pendulum above the target height.",
            "Reach the Gymnasium terminal height within 500 steps.",
            ("Negative torque", "No torque", "Positive torque"),
        ),
        Task(
            "frozenlake",
            "FrozenLake",
            "FrozenLake-v1",
            100,
            60000,
            "Cross a slippery frozen lake without falling into a hole.",
            "Reach goal tile 15 on the standard slippery 4x4 map within 100 steps.",
            ("Left", "Down", "Right", "Up"),
        ),
    )
}


def success(task, observation, terminated, truncated, steps):
    if task == "cartpole":
        return bool(truncated and not terminated and steps >= 500)
    if task == "mountaincar":
        return bool(terminated and float(observation[0]) >= 0.5)
    if task == "acrobot":
        return bool(terminated)
    if task == "frozenlake":
        return bool(terminated and int(observation) == 15)
    raise ValueError("Unknown classic task")


def observation_json(observation):
    array = np.asarray(observation)
    return array.tolist()


class RewardEnv(gym.Wrapper):
    def __init__(self, task, provider, judge=None, render_mode=None):
        if task not in TASKS or provider not in ("native", "rules", "jev"):
            raise ValueError("Unknown task or reward provider")
        if provider != "native" and judge is None:
            raise ValueError("A score provider is required")
        super().__init__(gym.make(TASKS[task].env_id, render_mode=render_mode))
        self.task, self.provider, self.judge = task, provider, judge
        self.previous = None
        self.steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous = np.array(obs, copy=True)
        self.steps = 0
        return obs, info

    def step(self, action):
        from .rewards import describe_transition

        obs, native, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        info = dict(info)
        info["native_reward"] = float(native)
        info["is_success"] = success(self.task, obs, terminated, truncated, self.steps)
        reward = float(native)
        if self.provider != "native":
            context = describe_transition(self.task, self.previous, obs, terminated)
            verdict = self.judge.score(context)
            reward = float(verdict["reward"])
            info["judge"] = verdict
        self.previous = np.array(obs, copy=True)
        return obs, reward, terminated, truncated, info


def make_env(task, provider="native", judge=None, render_mode=None):
    return RewardEnv(task, provider, judge, render_mode)
