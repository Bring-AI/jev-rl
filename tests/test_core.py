import numpy as np
import pytest

from jev_reward.game import Game, State
from jev_reward.judges import DemoJudge
from jev_reward.learning import QAgent, TrainConfig, train


def test_wall_key_exit_and_lava():
    game = Game()
    assert game.step(State(1, 1), 0).after == State(1, 1)
    key = game.step(State(3, 1), 1)
    assert key.after.has_key and key.event == "key"
    assert game.step(State(6, 5), 1).event == "locked"
    win = game.step(State(6, 5, True), 1)
    assert win.terminated and win.event == "win"
    death = game.step(State(3, 2), 1)
    assert death.terminated and death.event == "lava"


def test_bellman_update_does_not_bootstrap_terminal():
    agent = QAgent(seed=1, alpha=0.5, gamma=0.9)
    game = Game()
    before, after = State(1, 1), State(2, 1)
    agent.q[game.encode(after)] = 100
    agent.update(before, 1, -1, after, terminal=True)
    assert agent.q[game.encode(before), 1] == -0.5
    agent.update(before, 2, 1, after, terminal=False)
    assert agent.q[game.encode(before), 2] == 45.5


def test_training_improves_and_exports_true_provenance(tmp_path):
    result = train(TrainConfig(episodes=350, seed=7), DemoJudge(), tmp_path)
    assert result["after"]["success_rate"] >= 0.9
    assert result["after"]["success_rate"] > result["before"]["success_rate"]
    assert result["provider"] == "demo"
    assert result["judge"]["api_calls"] == 0
    assert len(result["history"]) == 350
    assert (tmp_path / "policy.json").exists()
    assert (tmp_path / "run.json").exists()
    assert result["replay"]["frames"][-1]["event"] == "win"
    assert all(np.isfinite(h["return"]) for h in result["history"])


def test_training_uses_only_judge_reward(tmp_path):
    class ConstantJudge(DemoJudge):
        def score(self, transition):
            verdict = super().score(transition)
            verdict["reward"] = -0.123
            return verdict

    result = train(TrainConfig(episodes=2, seed=3), ConstantJudge(), tmp_path)
    for episode in result["history"]:
        assert episode["return"] == pytest.approx(-0.123 * episode["steps"], abs=1e-5)
