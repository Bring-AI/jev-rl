import threading

import httpx
import pytest

from jev_reward.judges import DemoJudge, JevJudge
from jev_reward.learning import TrainConfig, train


def test_keyboard_interrupt_preserves_policy_and_evidence(tmp_path):
    class Interrupted(DemoJudge):
        def _judge(self, context):
            raise KeyboardInterrupt

    result = train(TrainConfig(episodes=2), Interrupted(), tmp_path)
    assert result["status"] == "stopped"
    assert (tmp_path / "policy.json").exists()
    assert (tmp_path / "judgments.json").exists()


def test_cancellation_after_timeout_does_not_send_more_requests(tmp_path):
    stop = threading.Event()
    calls = []

    def handler(request):
        calls.append(request)
        stop.set()
        raise httpx.ReadTimeout("network timeout")

    judge = JevJudge(
        api_key="test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )
    result = train(TrainConfig(episodes=2), judge, tmp_path, stop=stop)
    assert len(calls) == 1
    assert result["status"] == "stopped"
    assert (tmp_path / "run.json").exists()


@pytest.mark.parametrize("usage", [None, [], "unexpected"])
def test_malformed_usage_fails_with_artifacts(tmp_path, usage):
    data = {
        "model": "fixture",
        "answers": {
            "quality": {
                "type": "score",
                "score": 3,
                "confidence": 1,
                "probabilities": {str(i): float(i == 3) for i in range(6)},
            }
        },
        "usage": usage,
    }
    judge = JevJudge(
        api_key="test",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=data))
        ),
    )
    result = train(TrainConfig(episodes=2), judge, tmp_path)
    assert result["status"] == "failed"
    assert (tmp_path / "run.json").exists()
