import json

import numpy as np
import pytest


def test_classic_tasks_use_standard_gymnasium_dynamics():
    import gymnasium as gym

    from jev_reward.classic.envs import TASKS, make_env

    for task in TASKS.values():
        reference = gym.make(task.env_id)
        wrapped = make_env(task.slug, "native")
        a, _ = reference.reset(seed=123)
        b, _ = wrapped.reset(seed=123)
        np.testing.assert_equal(a, b)
        for action in [0, 1, 0, 1]:
            expected = reference.step(action)
            actual = wrapped.step(action)
            np.testing.assert_equal(expected[0], actual[0])
            assert expected[1:4] == actual[1:4]
        reference.close()
        wrapped.close()


def test_jev_wrapper_never_adds_native_reward():
    from jev_reward.classic.envs import make_env

    class FixedJudge:
        def score(self, context):
            assert "native_reward" not in context
            return {"reward": -0.375, "cached": True}

    env = make_env("cartpole", "jev", FixedJudge())
    env.reset(seed=7)
    _, reward, _, _, info = env.step(0)
    assert reward == -0.375
    assert info["native_reward"] == 1.0
    env.close()


def test_success_definitions_do_not_confuse_timeout_with_goal():
    from jev_reward.classic.envs import success

    assert success("cartpole", np.zeros(4), False, True, 500)
    assert not success("cartpole", np.zeros(4), True, False, 200)
    assert success("mountaincar", np.array([0.51, 0.01]), True, False, 199)
    assert not success("mountaincar", np.array([-0.5, 0.0]), False, True, 200)
    assert success("acrobot", np.array([-1, 0, 1, 0, 0, 0]), True, False, 50)
    assert success("frozenlake", 15, True, False, 20)
    assert not success("frozenlake", 5, True, False, 20)


def test_cache_is_bound_to_rubric_and_is_offline_only_when_frozen(tmp_path):
    from jev_reward.classic.rewards import ScoreCache, all_contexts
    from jev_reward.judges import JudgeError

    cache = ScoreCache("cartpole", "rules")
    context = all_contexts("cartpole")[0]
    first = cache.score(context)
    cache.save(tmp_path / "cache.json")
    loaded = ScoreCache("cartpole", "rules", cache_path=tmp_path / "cache.json", frozen=True)
    assert loaded.score(context)["reward"] == first["reward"]
    assert loaded.stats()["cache_hits"] == 1
    with pytest.raises(JudgeError, match="cache"):
        loaded.score({"unknown": True})
    data = json.loads((tmp_path / "cache.json").read_text())
    data["rubric_hash"] = "wrong"
    (tmp_path / "cache.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="rubric"):
        ScoreCache("cartpole", "rules", cache_path=tmp_path / "cache.json")


def test_unknown_api_cost_is_not_saved_as_free(tmp_path):
    from jev_reward.classic.rewards import ScoreCache

    class UnknownCost:
        def stats(self):
            return {
                "api_calls": 2,
                "input_tokens": 10,
                "output_tokens": 2,
                "cost_usd": None,
                "unpriced_calls": 2,
            }

    cache = ScoreCache("cartpole", "jev")
    cache.judge = UnknownCost()
    cache.save(tmp_path / "cache.json")
    usage = json.loads((tmp_path / "cache.json").read_text())["source_usage"]
    assert usage["cost_usd"] is None
    assert usage["unpriced_calls"] == 2
