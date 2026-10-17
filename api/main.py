"""Thin HTTP wrapper around engine.analyze.

POST /api/v1/analyze       multipart `file` -> {"job_id": ..., "status": "processing"}
GET  /api/v1/jobs/{job_id} -> {"job_id", "status", "verdict", "violations"}
Every request needs header X-API-Key (env ENGINE_API_KEY), else 401.

One sequential worker thread; models are loaded once at startup. If a job runs longer than
RESULT_DEADLINE_S, GET returns the best result so far (the rules-only result produced before
the VLM stage) with status "done".

Run: ENGINE_API_KEY=... HF_HUB_OFFLINE=1 uv run uvicorn api.main:app --port 8000
"""
from __future__ import annotations

import os
import queue
import secrets
import shutil
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile

from engine import config

API_KEY = os.environ.get("ENGINE_API_KEY") or ""
RESULT_DEADLINE_S = float(os.environ.get("ENGINE_DEADLINE_S", 240))
UPLOADS = config.RUNS_DIR / "uploads"
ALLOWED = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_queue: "queue.Queue[str]" = queue.Queue()


def _empty(job_id: str, status: str) -> dict:
    return {"job_id": job_id, "status": status, "verdict": None, "violations": []}


def _worker():
    from engine import analyze
    while True:
        job_id = _queue.get()
        with _lock:
            job = _jobs[job_id]
            job["started"] = time.time()

        def on_partial(res, job=job):
            with _lock:
                job["partial"] = res

        try:
            res = analyze(job["path"], job_id=job_id, on_partial=on_partial,
                          deadline=job["created"] + RESULT_DEADLINE_S)
            with _lock:
                job["result"], job["status"] = res, "done"
        except Exception as e:  # report, keep serving
            with _lock:
                job["status"], job["error"] = "failed", f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            _queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global API_KEY
    if not API_KEY:
        API_KEY = secrets.token_urlsafe(24)
        print(f"[api] ENGINE_API_KEY not set; generated key for this run: {API_KEY}", flush=True)
    import engine
    from engine import perception, vlm
    engine._models = perception.Models()   # load once
    vlm.load()
    threading.Thread(target=_worker, daemon=True).start()
    yield


app = FastAPI(title="Cash-desk discipline engine", lifespan=lifespan)


def require_key(x_api_key: str | None = Header(default=None)):
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.post("/api/v1/analyze", dependencies=[Depends(require_key)])
async def submit(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower() or ".mov"
    if ext not in ALLOWED:
        raise HTTPException(status_code=415, detail=f"unsupported file type {ext}")
    job_id = uuid.uuid4().hex[:12]
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / f"{job_id}{ext}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    with _lock:
        _jobs[job_id] = {"path": str(path), "status": "processing", "created": time.time()}
    _queue.put(job_id)
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
async def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job_id")
        if job["status"] == "done":
            return job["result"]
        if job["status"] == "failed":
            return _empty(job_id, "failed")
        if time.time() - job["created"] >= RESULT_DEADLINE_S and job.get("partial"):
            job["status"], job["result"] = "done", job["partial"]  # best result so far
            return job["result"]
        return _empty(job_id, "processing")


@app.get("/health")
async def health():
    return {"ok": True, "queue": _queue.qsize()}
