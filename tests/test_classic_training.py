import json
import threading

import numpy as np
import pytest


def test_short_training_saves_reloadable_checkpoints_and_heldout_metrics(tmp_path):
    from stable_baselines3 import DQN

    from jev_reward.classic.training import ClassicConfig, evaluate_policy, train_classic

    config = ClassicConfig(
        task="cartpole",
        provider="rules",
        total_steps=64,
        checkpoint_steps=32,
        eval_episodes=2,
        test_episodes=3,
    )
    result = train_classic(config, tmp_path)
    assert result["status"] == "completed"
    assert [c["steps"] for c in result["checkpoints"]] == [0, 32, 64]
    assert result["test"]["episodes"] == 3
    assert result["test"]["seed_start"] != result["checkpoints"][-1]["evaluation"]["seed_start"]
    loaded = DQN.load(tmp_path / result["checkpoints"][-1]["model_file"], device="cpu")
    again = evaluate_policy(loaded, "cartpole", episodes=3, seed_start=100000)
    assert again["returns"] == result["test"]["returns"]
    assert json.loads((tmp_path / "run.json").read_text())["status"] == "completed"
    assert result["reward"]["current_session"]["api_calls"] == 0


def test_evaluation_does_not_change_training_rng():
    from jev_reward.classic.training import evaluate_policy

    class ConstantPolicy:
        def predict(self, observation, deterministic=True):
            return np.array(0), None

    np.random.seed(777)
    state = np.random.get_state()
    result = evaluate_policy(ConstantPolicy(), "mountaincar", episodes=2)
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    assert result["returns"] == [-200, -200]
    assert result["success_rate"] == 0


def test_random_policy_is_independent_of_environment_rng(monkeypatch):
    import jev_reward.classic.training as training

    class CorrelationProbe:
        action_space = type("Actions", (), {"n": 4})()
        steps = 1

        def reset(self, seed):
            self.rng = np.random.default_rng(seed)
            return 0, {}

        def step(self, action):
            correlated = int(action == self.rng.integers(4))
            return 0, correlated, True, False, {"is_success": False}

        def close(self):
            pass

    monkeypatch.setattr(training, "make_env", lambda *_: CorrelationProbe())
    result = training.evaluate_policy(None, "frozenlake", episodes=1000, record=False)
    assert 0.20 < result["native_return_mean"] < 0.30


def test_cancelled_run_still_saves_policy_and_status(tmp_path):
    from jev_reward.classic.training import ClassicConfig, train_classic

    stop = threading.Event()
    stop.set()
    result = train_classic(
        ClassicConfig(total_steps=32, eval_episodes=1, test_episodes=1), tmp_path, stop=stop
    )
    assert result["status"] == "stopped"
    assert (tmp_path / result["checkpoints"][-1]["model_file"]).exists()


@pytest.mark.parametrize(
    "field,value",
    [("task", "not-a-game"), ("total_steps", 0), ("provider", "fake"), ("checkpoint_steps", 0)],
)
def test_invalid_training_configs_fail_before_training(field, value):
    from jev_reward.classic.training import ClassicConfig

    with pytest.raises(ValueError):
        ClassicConfig(**{field: value})
