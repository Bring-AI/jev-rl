import json

import pytest

from jev_reward.judges import DemoJudge
from jev_reward.learning import TrainConfig, train


def test_periodic_checkpoints_include_policy_evaluation_and_playable_replay(tmp_path):
    events = []
    result = train(
        TrainConfig(episodes=7, checkpoint_every=3),
        DemoJudge(),
        tmp_path,
        on_progress=events.append,
    )
    assert [c["episode"] for c in result["checkpoints"]] == [0, 3, 6, 7]
    for point in result["checkpoints"]:
        saved = json.loads((tmp_path / point["file"]).read_text())
        assert saved["episode"] == point["episode"]
        assert len(saved["q"]) == 126
        assert saved["replay"]["frames"][0]["event"] == "start"
        assert saved["evaluation"]["episodes"] == 60
    assert any((e.get("checkpoint") or {}).get("episode") == 3 for e in events)
    assert result["checkpoints"][-1]["evaluation"] == result["after"]


def test_checkpoint_evaluation_never_requests_more_judgments(tmp_path):
    a = train(TrainConfig(episodes=25, seed=7, checkpoint_every=1), DemoJudge(), tmp_path / "a")
    b = train(TrainConfig(episodes=25, seed=7, checkpoint_every=25), DemoJudge(), tmp_path / "b")
    assert a["history"] == b["history"]
    assert a["judge"]["unique_transitions"] == b["judge"]["unique_transitions"]
    assert a["judge"]["cache_hits"] == b["judge"]["cache_hits"]
    assert (
        json.loads((tmp_path / "a/policy.json").read_text())["q"]
        == json.loads((tmp_path / "b/policy.json").read_text())["q"]
    )


def test_invalid_checkpoint_frequency_rejected():
    with pytest.raises(ValueError, match="checkpoint"):
        TrainConfig(checkpoint_every=0)
