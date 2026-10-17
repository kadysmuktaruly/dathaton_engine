"""The engine's HTTP API (api/main.py: same endpoints, key check and job store) running
orig_and_vlm.analyze instead of engine.analyze.

  ENGINE_API_KEY=<secret> HF_HUB_OFFLINE=1 .venv/bin/uvicorn orig_and_vlm.api:app --port 8000

api/main.py's worker calls `engine.analyze`; at startup that name is pointed at the wrapper
below, so api/main.py itself stays untouched.
"""
from __future__ import annotations

import secrets
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.routing import APIRoute

import api.main as base
import orig_and_vlm


def analyze_shared(path, job_id=None, on_partial=None, deadline=None, **kw):
    """Jobs still waiting in the queue share the time left before their deadline, so a burst
    of uploads all finish (with smaller VLM budgets) instead of the last ones timing out."""
    if deadline is not None:
        now = time.time()
        deadline = now + (deadline - now) / (1 + base._queue.qsize())
    return orig_and_vlm.analyze(path, job_id=job_id, on_partial=on_partial, deadline=deadline, **kw)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not base.API_KEY:
        base.API_KEY = secrets.token_urlsafe(24)
        print(f"[api] ENGINE_API_KEY not set; generated key for this run: {base.API_KEY}", flush=True)
    import engine
    from engine import perception, vlm
    from orig_and_vlm import eyes
    engine._models = perception.Models()  # load once
    vlm.load()                            # the engine's Qwen3-VL verifier
    eyes.load()                           # Qwen3.5-4B scan + close-out
    engine.analyze = analyze_shared       # base._worker imports this name when it starts
    threading.Thread(target=base._worker, daemon=True).start()
    yield


app = FastAPI(title="Cash-desk discipline engine (orig_and_vlm)", lifespan=lifespan)
for route in base.app.routes:
    if isinstance(route, APIRoute):
        app.router.routes.append(route)
