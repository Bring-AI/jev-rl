import httpx
import pytest

from jev_reward.judges import JevJudge, JudgeError, parse_answer


def score_response():
    return {
        "model": "typesafe/jev-test",
        "answers": {
            "quality": {
                "type": "score",
                "score": 0.75,
                "confidence": 0.8,
                "probabilities": {"0": 0.25, "1": 0.75},
            }
        },
    }


def test_custom_score_rubric_uses_its_own_rewards():
    result = parse_answer(score_response(), criteria=["bad", "good"], rewards=[-2, 2])
    assert result["reward"] == 1.0
    assert result["label"] == "good"
    with pytest.raises(ValueError):
        parse_answer(score_response(), criteria=["bad", "good"], rewards=[1])


def test_custom_score_rubric_sent_to_real_transport_boundary(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-not-a-key")
    question = {"type": "score", "instructions": "Grade balance.", "criteria": ["bad", "good"]}
    payloads = []

    def handle(request):
        import json

        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=score_response())

    judge = JevJudge(
        gateway="openrouter",
        question=question,
        rewards=[-2, 2],
        client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    result = judge._judge({"angle": 0.1})
    assert payloads[0]["questions"]["quality"] == question
    assert result["reward"] == 1
    judge.close()


def test_live_score_rounding_is_accepted_and_probabilities_normalized():
    # Observed official JEV response: score is computed before two-decimal rounding.
    data = score_response()
    data["answers"]["quality"].update(
        score=0.11, probabilities={"0": 0.97, "1": 0, "2": 0, "3": 0.03, "4": 0, "5": 0}
    )
    verdict = parse_answer(data)
    assert verdict["reward"] == pytest.approx(-0.97 + 0.03 * 0.08)
    data["answers"]["quality"].update(
        score=2.0, probabilities={str(i): (0.33 if i in (0, 2, 4) else 0) for i in range(6)}
    )
    verdict = parse_answer(data)
    assert sum(verdict["probabilities"].values()) == pytest.approx(1)
    data["answers"]["quality"]["score"] = 4
    with pytest.raises(ValueError):
        parse_answer(data)


@pytest.mark.parametrize("data", [[], 42, None])
def test_non_object_api_response_is_a_controlled_judge_error(data):
    judge = JevJudge(
        api_key="test-only",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=data))
        ),
    )
    with pytest.raises(JudgeError, match="invalid"):
        judge._judge({})
    judge.close()
