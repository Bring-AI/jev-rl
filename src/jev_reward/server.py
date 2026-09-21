"""Local training dashboard. One active run, with explicit cancellation."""

import copy
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .classic.api import ClassicManager, classic_router
from .game import Game
from .judges import JudgeError, judge_settings, make_judge
from .learning import TrainConfig, train

STATIC = Path(__file__).parent / "static"


class StartRequest(BaseModel):
    provider: Literal["demo", "jev"] = "demo"
    episodes: int = Field(350, ge=1, le=5000)
    seed: int = Field(7, ge=0, le=2**32 - 1)
    max_calls: int = Field(300, ge=1, le=2000)
    checkpoint_every: int = Field(25, ge=1, le=5000)


class RunManager:
    def __init__(self, output):
        self.output = Path(output)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        self.state = {"status": "idle", "history": [], "result": None, "version": 0}

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def start(self, request):
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise HTTPException(
                    409, "A run is already active. Stop it before starting another."
                )
            try:
                judge = make_judge(request.provider, request.max_calls)
            except JudgeError as exc:
                raise HTTPException(400, str(exc)) from None
            self.stop = threading.Event()
            run_id = uuid.uuid4().hex[:12]
            self.state = {
                "id": run_id,
                "status": "running",
                "provider": request.provider,
                "history": [],
                "frames": [],
                "result": None,
                "version": 0,
                "total_episodes": request.episodes,
                "error": None,
                "checkpoints": [],
            }
            config = TrainConfig(
                episodes=request.episodes,
                seed=request.seed,
                episode_delay=0.015,
                checkpoint_every=request.checkpoint_every,
            )
            self.thread = threading.Thread(
                target=self._work, args=(config, judge, run_id), daemon=False
            )
            self.thread.start()
            return {"id": run_id, "status": "running"}

    def _progress(self, event):
        with self.lock:
            self.state["history"].append(event["record"])
            self.state.update({key: event[key] for key in ("frames", "judge", "before")})
            if not self.state["checkpoints"]:
                self.state["checkpoints"].append(event["initial_checkpoint"])
            if event["checkpoint"]:
                self.state["checkpoints"].append(event["checkpoint"])
            self.state["version"] += 1

    def _work(self, config, judge, run_id):
        try:
            result = train(config, judge, self.output / run_id, self._progress, self.stop)
            with self.lock:
                self.state.update(status=result["status"], result=result, error=result["error"])
                self.state["version"] += 1
        except Exception as exc:
            # Do not return exception bodies that could contain upstream secrets.
            logging.getLogger(__name__).error("Run %s failed: %s", run_id, type(exc).__name__)
            with self.lock:
                self.state.update(
                    status="failed", error="Internal training error; inspect server logs."
                )

    def shutdown(self):
        self.stop.set()
        if self.thread:
            # Finish the in-flight bounded HTTP call, then persist the stopped run.
            self.thread.join()


def create_app(output="runs"):
    manager = RunManager(output)
    classic_manager = ClassicManager(output)

    @asynccontextmanager
    async def lifespan(app):
        yield
        manager.shutdown()
        classic_manager.shutdown()

    app = FastAPI(title="JEV Reward Arcade", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"])
    app.state.manager = manager
    app.state.classic_manager = classic_manager
    app.include_router(classic_router(classic_manager))

    @app.middleware("http")
    async def same_origin_mutations(request: Request, call_next):
        origin = request.headers.get("origin")
        if (
            request.method == "POST"
            and origin
            and urlparse(origin).netloc != request.headers.get("host")
        ):
            return JSONResponse(
                {"detail": "Cross-origin training requests are disabled."}, status_code=403
            )
        return await call_next(request)

    @app.get("/api/config")
    def config():
        settings = judge_settings()
        return {
            "jev_available": settings["available"],
            "game": Game().spec(),
            "model": settings["model"],
            "gateway": settings["gateway"],
        }

    @app.get("/api/state")
    def state():
        return manager.snapshot()

    @app.post("/api/train")
    def start(request: StartRequest):
        return manager.start(request)

    @app.post("/api/stop")
    def stop():
        manager.stop.set()
        return {"status": "stop_requested"}

    @app.get("/api/export")
    def export():
        result = manager.snapshot().get("result")
        if result is None:
            raise HTTPException(409, "No finished run to export yet.")
        return JSONResponse(
            result, headers={"Content-Disposition": 'attachment; filename="jev-run.json"'}
        )

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "classic" / "index.html")

    @app.get("/key-quest")
    def key_quest():
        return FileResponse(STATIC / "index.html")

    return app
