import json

import httpx
import pytest

from jev_reward.game import Game, State
from jev_reward.judges import JevJudge, JudgeError, judge_settings


def test_official_jev_through_openrouter_with_file_credential(tmp_path, monkeypatch):
    secret = tmp_path / "key"
    secret.write_text("test-openrouter-key\n")
    monkeypatch.setenv("OPENROUTER_API_KEY_FILE", str(secret))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("JEV_GATEWAY", "openrouter")

    def handle(request):
        assert str(request.url) == "https://openrouter.ai/api/v1/systemone"
        assert request.headers["Authorization"] == "Bearer test-openrouter-key"
        assert json.loads(request.content)["model"] == "typesafe/jev-1.13"
        return httpx.Response(
            200,
            json={
                "model": "typesafe/jev-1.13-20260917",
                "provider": "TypeSafe",
                "id": "fixture-not-a-real-call",
                "answers": {
                    "quality": {
                        "type": "score",
                        "score": 3,
                        "confidence": 1,
                        "probabilities": {str(i): float(i == 3) for i in range(6)},
                    }
                },
                "usage": {"input_tokens": 100, "output_tokens": 10, "cost": 0.0000042},
            },
        )

    judge = JevJudge(client=httpx.Client(transport=httpx.MockTransport(handle)))
    verdict = judge.score(Game().step(State(1, 1), 1))
    assert verdict["provider"] == "jev"
    assert verdict["gateway"] == "openrouter"
    assert verdict["upstream_provider"] == "TypeSafe"
    assert verdict["request_id"] == "fixture-not-a-real-call"
    assert judge.stats()["cost_usd"] == pytest.approx(0.0000042)
    assert judge.stats()["served_models"] == ["typesafe/jev-1.13-20260917"]
    assert "test-openrouter-key" not in json.dumps(list(judge.cache.values()))
    assert judge_settings()["gateway"] == "openrouter"
    assert judge_settings()["available"] is True


def test_gateway_keeps_credentials_separate(monkeypatch):
    monkeypatch.setenv("JEV_GATEWAY", "typesafe")
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-send-to-typesafe")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JudgeError, match="TYPESAFE_API_KEY"):
        JevJudge()


def test_file_errors_are_safe(monkeypatch):
    monkeypatch.setenv("JEV_GATEWAY", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY_FILE", "/missing/key")
    with pytest.raises(JudgeError, match="key file"):
        JevJudge()
