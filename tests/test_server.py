import time

from fastapi.testclient import TestClient

from jev_reward.server import create_app


def test_training_lifecycle_and_export(tmp_path):
    with TestClient(create_app(tmp_path), base_url="http://localhost") as client:
        assert client.get("/").status_code == 200
        started = client.post("/api/train", json={"episodes": 3, "provider": "demo"})
        assert started.status_code == 200
        for _ in range(200):
            state = client.get("/api/state").json()
            if state["status"] != "running":
                break
            time.sleep(0.01)
        assert state["status"] == "completed"
        assert len(state["result"]["history"]) == 3
        assert client.get("/api/export").json()["provider"] == "demo"
        assert client.post("/api/train", json={"episodes": -1}).status_code == 422


def test_missing_key_and_cross_origin_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with TestClient(create_app(tmp_path), base_url="http://localhost") as client:
        assert client.post("/api/train", json={"provider": "jev"}).status_code == 400
        assert (
            client.post(
                "/api/train", headers={"Origin": "https://evil.example"}, json={}
            ).status_code
            == 403
        )
        assert "TYPESAFE_API_KEY" not in client.get("/api/config").text


def test_untrusted_host_cannot_start_training(tmp_path):
    with TestClient(create_app(tmp_path), base_url="http://localhost") as client:
        response = client.post(
            "/api/train",
            json={},
            headers={"Host": "rebound.example:8000", "Origin": "http://rebound.example:8000"},
        )
        assert response.status_code == 400
