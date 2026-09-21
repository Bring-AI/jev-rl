import json

import httpx
import pytest

from jev_reward.game import Game, State
from jev_reward.judges import JevJudge, JudgeError, parse_answer


def answer(level=3):
    return {
        "model": "jev-test-fixture",
        "answers": {
            "quality": {
                "type": "score",
                "score": level,
                "confidence": 1,
                "probabilities": {str(i): float(i == level) for i in range(6)},
            }
        },
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }


def test_official_contract_cache_and_expected_reward():
    calls = []

    def handler(request):
        calls.append(request)
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["Authorization"] == "Bearer test-only"
        payload = json.loads(request.content)
        assert payload["model"] == "jev-latest"
        assert payload["questions"]["quality"]["type"] == "score"
        assert payload["state"]["before"]["x"] == 1
        return httpx.Response(200, json=answer())

    judge = JevJudge(
        api_key="test-only", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    transition = Game().step(State(1, 1), 1)
    assert judge.score(transition)["reward"] == pytest.approx(0.08)
    assert judge.score(transition)["cached"]
    assert len(calls) == 1 and judge.cache_hits == 1
    assert judge.stats()["input_tokens"] == 100
    assert "test-only" not in json.dumps(list(judge.cache.values()))


def test_retry_budget_counts_every_attempt():
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(429)))
    judge = JevJudge(api_key="test-only", client=client, max_calls=2, sleep=lambda _: None)
    with pytest.raises(JudgeError, match="budget"):
        judge.score(Game().step(State(1, 1), 1))
    assert judge.api_calls == 2
    assert not judge.cache


def test_auth_errors_fail_closed_and_do_not_echo_response():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(401, text="secret-value"))
    )
    judge = JevJudge(api_key="test-only", client=client)
    with pytest.raises(JudgeError, match="HTTP 401") as exc:
        judge.score(Game().step(State(1, 1), 1))
    assert "secret-value" not in str(exc.value)
    assert judge.api_calls == 1


@pytest.mark.parametrize("value", [float("nan"), -1, 2])
def test_invalid_probabilities_rejected(value):
    data = answer()
    data["answers"]["quality"]["probabilities"]["3"] = value
    with pytest.raises(ValueError):
        parse_answer(data)


def test_weighted_reward_not_argmax():
    data = answer()
    data["answers"]["quality"].update(
        score=4.5,
        confidence=0.5,
        probabilities={str(i): (0.5 if i in (4, 5) else 0) for i in range(6)},
    )
    assert parse_answer(data)["reward"] == pytest.approx(0.8)


def test_missing_key_and_insecure_remote_url(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JudgeError, match="TYPESAFE_API_KEY"):
        JevJudge()
    with pytest.raises(JudgeError, match="HTTPS"):
        JevJudge(api_key="test", base_url="http://example.com/v1")


@pytest.mark.parametrize("model", [[], {}, None, ""])
def test_invalid_model_identity_is_rejected(model):
    data = answer()
    data["model"] = model
    with pytest.raises(ValueError):
        parse_answer(data)
