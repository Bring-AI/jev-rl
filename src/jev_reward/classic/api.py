"""Local browser controls for one classic training run."""

import copy
import importlib.util
import logging
import threading
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..judges import judge_settings

SCORE_DIR = Path(__file__).resolve().parents[1] / "static" / "classic" / "scores"


class ClassicRequest(BaseModel):
    task: Literal["cartpole", "mountaincar", "acrobot", "frozenlake"] = "cartpole"
    provider: Literal["native", "rules", "jev"] = "rules"
    recorded_scores: bool = True
    total_steps: int = Field(60000, ge=1, le=2000000)
    checkpoint_steps: int = Field(10000, ge=1, le=2000000)
    seed: int = Field(7, ge=0, le=2**32 - 1)
    eval_episodes: int = Field(20, ge=1, le=100)
    test_episodes: int = Field(100, ge=1, le=1000)
    max_calls: int = Field(300, ge=1, le=2000)


def available():
    return all(importlib.util.find_spec(name) for name in ("gymnasium", "stable_baselines3"))


class ClassicManager:
    def __init__(self, output):
        self.output = Path(output) / "classic"
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        self.state = {"status": "idle", "steps": 0, "checkpoints": [], "result": None}

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def start(self, request):
        if not available():
            raise HTTPException(400, "Install classic games: uv sync --extra classic")
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise HTTPException(409, "A classic run is already active")
            cache_path = SCORE_DIR / f"{request.task}.json"
            if request.provider == "jev":
                if request.recorded_scores and not cache_path.is_file():
                    raise HTTPException(400, "No recorded JEV scores are bundled for this game")
                if not request.recorded_scores and not judge_settings()["available"]:
                    raise HTTPException(400, "Configure a JEV API key before using live scoring")
            self.stop = threading.Event()
            run_id = uuid.uuid4().hex[:12]
            self.state = {
                "id": run_id,
                "status": "running",
                "steps": 0,
                "episodes": 0,
                "task": request.task,
                "provider": request.provider,
                "total_steps": request.total_steps,
                "seed": request.seed,
                "history": [],
                "checkpoints": [],
                "result": None,
                "error": None,
            }
            self.thread = threading.Thread(
                target=self._work, args=(request, run_id, str(cache_path)), daemon=False
            )
            self.thread.start()
        return {"status": "running", "id": run_id}

    def _work(self, request, run_id, cache_path):
        from .training import ClassicConfig, train_classic

        def progress(event):
            with self.lock:
                self.state.update(steps=event["steps"], episodes=event["episodes"])
                if event["record"] and (
                    not self.state["history"]
                    or self.state["history"][-1]["episode"] != event["record"]["episode"]
                ):
                    self.state["history"].append(event["record"])
                if event["checkpoint"]:
                    self.state["checkpoints"].append(event["checkpoint"])

        try:
            values = request.model_dump(exclude={"recorded_scores"})
            if request.provider == "jev" and request.recorded_scores:
                values.update(cache_path=cache_path, frozen_cache=True)
            result = train_classic(
                ClassicConfig(**values), self.output / run_id, progress, self.stop
            )
            with self.lock:
                self.state.update(
                    status=result["status"],
                    result=result,
                    steps=result["steps"],
                    checkpoints=result["checkpoints"],
                    history=result["history"],
                    error=result["error"],
                )
        except Exception as exc:
            logging.getLogger(__name__).error("Classic run failed: %s", type(exc).__name__)
            with self.lock:
                self.state.update(
                    status="failed", error="Training failed; inspect local server logs."
                )

    def shutdown(self):
        self.stop.set()
        if self.thread:
            self.thread.join()


def classic_router(manager):
    router = APIRouter(prefix="/api/classic")

    @router.get("/config")
    def config():
        tasks = []
        if available():
            from .envs import TASKS

            tasks = [task.spec() for task in TASKS.values()]
        return {
            "available": bool(available()),
            "tasks": tasks,
            "jev_available": judge_settings()["available"],
            "recorded_scores": [p.stem for p in SCORE_DIR.glob("*.json")],
        }

    @router.get("/state")
    def state():
        return manager.snapshot()

    @router.post("/train")
    def train(request: ClassicRequest):
        return manager.start(request)

    @router.post("/stop")
    def stop():
        manager.stop.set()
        return {"status": "stop_requested"}

    @router.get("/export")
    def export():
        result = manager.snapshot().get("result")
        if result is None:
            raise HTTPException(409, "No finished classic run to export")
        return JSONResponse(
            result, headers={"Content-Disposition": "attachment; filename=jevrl-run.json"}
        )

    return router
