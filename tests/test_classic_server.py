import time

from fastapi.testclient import TestClient

from jev_reward.server import create_app


def test_classic_routes_and_real_short_training(tmp_path):
    with TestClient(create_app(tmp_path), base_url="http://localhost") as client:
        config = client.get("/api/classic/config")
        assert config.status_code == 200
        assert len(config.json()["tasks"]) == 4
        response = client.post(
            "/api/classic/train",
            json={
                "task": "frozenlake",
                "provider": "rules",
                "total_steps": 32,
                "checkpoint_steps": 16,
                "eval_episodes": 1,
                "test_episodes": 2,
            },
        )
        assert response.status_code == 200
        for _ in range(500):
            state = client.get("/api/classic/state").json()
            if state["status"] != "running":
                break
            time.sleep(0.02)
        assert state["status"] == "completed"
        assert state["result"]["test"]["episodes"] == 2
        assert client.get("/api/classic/export").json()["task"]["slug"] == "frozenlake"
        assert client.post("/api/classic/train", json={"task": "fake"}).status_code == 422


def test_classic_recorded_mode_fails_clearly_without_cache(tmp_path, monkeypatch):
    import jev_reward.classic.api as api

    monkeypatch.setattr(api, "SCORE_DIR", tmp_path / "missing")
    with TestClient(create_app(tmp_path), base_url="http://localhost") as client:
        result = client.post(
            "/api/classic/train", json={"provider": "jev", "recorded_scores": True}
        )
        assert result.status_code == 400
        assert "recorded" in result.json()["detail"].lower()
