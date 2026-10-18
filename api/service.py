"""One warm inference process, supervised independently of HTTP polling.

The backend owns the queue. A hard deadline/cancel kills and joins inference before
publishing a terminal status, then loads a fresh worker. No orphan inference threads.
"""

from __future__ import annotations

import fcntl
import json
import logging
import multiprocessing as mp
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServiceConfig:
    root: Path
    deadline_s: float = 210
    startup_timeout_s: float = 600
    max_upload_bytes: int = 100 * 1024 * 1024
    retained_jobs: int = 200

    def __post_init__(self):
        if (
            min(
                self.deadline_s,
                self.startup_timeout_s,
                self.max_upload_bytes,
                self.retained_jobs,
            )
            <= 0
        ):
            raise ValueError("Engine limits must be positive")


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def inference_worker(connection, _settings):
    """Spawn target: heavy imports and all model state stay in this process."""
    try:
        # A killed Linux API process must not leave CPU inference running orphaned.
        import sys

        from engine import config

        if sys.platform == "linux":
            import ctypes
            import signal

            parent = os.getppid()
            if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL) != 0:
                raise OSError(ctypes.get_errno(), "Could not set parent-death signal")
            # PID 1 is a valid parent when Uvicorn is the container entrypoint.
            if os.getppid() != parent:
                return
        os.environ["OMP_NUM_THREADS"] = str(config.CPU_THREADS)
        os.environ["OPENBLAS_NUM_THREADS"] = str(config.CPU_THREADS)
        import cv2
        import torch

        torch.set_num_threads(config.CPU_THREADS)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        import engine
        from engine import perception, vlm

        engine._models = perception.Models()
        if config.USE_VLM:
            vlm.load()
        connection.send(("ready", None))
        while True:
            request = connection.recv()
            job_id = request["job_id"]

            def report(result, analysis, partial):
                connection.send(
                    (
                        "partial" if partial else "done",
                        {
                            "result": result,
                            "analysis": analysis,
                        },
                    )
                )

            try:
                from engine.video import probe

                info = probe(request["path"])
                if info.width <= 0 or info.height <= 0 or info.duration <= 0:
                    raise ValueError("Video has no decodable frames or duration")
                engine.analyze(
                    request["path"],
                    job_id=job_id,
                    use_vlm=config.USE_VLM,
                    deadline=request["deadline"],
                    on_analysis=report,
                )
            except Exception:
                logger.exception("Inference failed for %s", job_id)
                connection.send(
                    (
                        "failed",
                        {
                            "code": "ANALYSIS_FAILED",
                            "message": "Не удалось выполнить анализ видео.",
                        },
                    )
                )
    except EOFError:
        pass
    except Exception:
        logger.exception("Inference worker could not start")
        connection.send(("startup_failed", None))
    finally:
        connection.close()


class EngineService:
    def __init__(self, settings: ServiceConfig, worker_target=inference_worker):
        self.settings = settings
        self.worker_target = worker_target
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.jobs: dict[str, dict] = {}
        self.active: str | None = None
        self.ready = False
        self.startup_error = None
        self.process = None
        self.connection = None
        self.thread = None
        self.instance_lock = None

    def start(self):
        self.settings.root.mkdir(parents=True, exist_ok=True)
        self.instance_lock = (self.settings.root / ".lock").open("a")
        try:
            fcntl.flock(self.instance_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.instance_lock.close()
            self.instance_lock = None
            raise RuntimeError(
                "One engine instance per runs directory; use --workers 1"
            ) from None
        # Engine jobs are ephemeral; the backend database is the source of truth.
        for pattern in ("*.json", "*.tmp", "uploads/*"):
            for stale in self.settings.root.glob(pattern):
                if stale.is_file():
                    stale.unlink()
        self._spawn()
        self.thread = threading.Thread(
            target=self._watch, name="engine-supervisor", daemon=True
        )
        self.thread.start()

    def _spawn(self):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=self.worker_target, args=(child, self.settings), daemon=True
        )
        self.ready = False
        self.startup_error = None
        self.spawned = time.monotonic()
        self.process.start()
        child.close()

    def _kill(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.ready = False

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        with self.lock:
            self._kill()
            if self.instance_lock is not None:
                self.instance_lock.close()
                self.instance_lock = None

    def health(self):
        with self.lock:
            return {
                "ok": True,
                "ready": self.ready,
                "busy": self.active is not None,
                "accepting": self.ready and self.active is None,
                "active_job_id": self.active,
                "queue": 0,
                "worker_pid": self.process.pid if self.process else None,
                "startup_error": self.startup_error,
            }

    def submit(self, job_id: str, path: Path):
        with self.lock:
            # Recheck after streaming the upload: concurrent retries can both have
            # passed the route's early lookup before either registered the UUID.
            if job_id in self.jobs:
                path.unlink(missing_ok=True)
                return {"job_id": job_id, "status": "processing"}
            if self.active:
                raise ServiceError(
                    409, "ENGINE_BUSY", "Engine is analysing another video"
                )
            if not self.ready:
                raise ServiceError(
                    503, "ENGINE_NOT_READY", "Engine models are loading or unavailable"
                )
            self.jobs[job_id] = {
                "job_id": job_id,
                "status": "processing",
                "path": path,
                "expires": time.monotonic() + self.settings.deadline_s,
                "partial": False,
            }
            self.active = job_id
            try:
                self.connection.send(
                    {
                        "job_id": job_id,
                        "path": str(path),
                        "deadline": time.time() + self.settings.deadline_s,
                    }
                )
            except (EOFError, BrokenPipeError, OSError):
                self._finish(
                    "failed",
                    error={"code": "WORKER_EXITED", "message": "Worker exited"},
                )
            self._prune()
            return {"job_id": job_id, "status": "processing"}

    def get(self, job_id: str, analysis=False):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise ServiceError(404, "JOB_NOT_FOUND", "Unknown engine job")
            if analysis:
                if job["status"] != "done":
                    raise ServiceError(
                        409, "JOB_NOT_READY", "Job has no completed analysis"
                    )
                return self._read(job_id)["analysis"]
            if job["status"] == "done":
                return self._read(job_id)["result"]
            return {
                "job_id": job_id,
                "status": job["status"],
                "verdict": None,
                "violations": [],
                **({"error": job["error"]} if "error" in job else {}),
            }

    def details(self, job_id: str):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise ServiceError(404, "JOB_NOT_FOUND", "Unknown engine job")
            return {"partial": job["partial"], "status": job["status"]}

    def cancel(self, job_id: str):
        with self.lock:
            self.get(job_id)
            if self.active == job_id:
                self._kill()
                self._finish(
                    "failed",
                    error={"code": "CANCELLED", "message": "Analysis cancelled"},
                )
                if not self.stop_event.is_set():
                    self._spawn()
            return self.get(job_id)

    def _file(self, job_id):
        return self.settings.root / f"{job_id}.json"

    def _read(self, job_id):
        return json.loads(self._file(job_id).read_text(encoding="utf-8"))

    def _store(self, job_id, payload):
        path = self._file(job_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def _finish(self, status, error=None):
        job = self.jobs[self.active]
        job["status"] = status
        if error:
            job["error"] = error
            self._file(self.active).unlink(missing_ok=True)
            job["partial"] = False
        job["path"].unlink(missing_ok=True)
        self.active = None

    def _prune(self):
        for job_id in list(self.jobs):
            if len(self.jobs) <= self.settings.retained_jobs:
                break
            if job_id != self.active:
                self._file(job_id).unlink(missing_ok=True)
                del self.jobs[job_id]

    def _watch(self):
        while not self.stop_event.wait(0.05):
            with self.lock:
                try:
                    self._tick()
                except Exception:
                    logger.exception("Engine supervisor failed")
                    self._kill()
                    if self.active:
                        self._finish(
                            "failed",
                            error={
                                "code": "WORKER_EXITED",
                                "message": "Worker unavailable",
                            },
                        )
                    self.startup_error = (
                        "Worker unavailable; restart the engine service"
                    )

    def _tick(self):
        if self.process is None:
            return
        # Deadline wins over a late message. Stop actual inference before terminal status.
        if self.active and time.monotonic() >= self.jobs[self.active]["expires"]:
            partial = self.jobs[self.active]["partial"]
            self._kill()
            if partial:
                self._finish("done")
            else:
                self._finish(
                    "failed",
                    error={"code": "ANALYSIS_TIMEOUT", "message": "Analysis timed out"},
                )
            self._spawn()
            return
        if self.connection.poll():
            try:
                kind, payload = self.connection.recv()
            except (EOFError, OSError):
                kind, payload = "exited", None
            if kind == "ready":
                self.ready = True
            elif kind in {"partial", "done"} and self.active:
                self._store(self.active, payload)
                self.jobs[self.active]["partial"] = kind == "partial"
                if kind == "done":
                    self._finish("done")
            elif kind == "failed" and self.active:
                self._finish("failed", error=payload)
            elif kind in {"startup_failed", "exited"}:
                self._worker_exited()
                return
        if self.process is not None and not self.process.is_alive():
            self._worker_exited()
        elif (
            not self.ready
            and time.monotonic() - self.spawned > self.settings.startup_timeout_s
        ):
            self._kill()
            self.startup_error = (
                "Model loading timed out; check weights and logs, then restart"
            )

    def _worker_exited(self):
        was_ready = self.ready
        self._kill()
        if self.active:
            self._finish(
                "failed",
                error={"code": "WORKER_EXITED", "message": "Inference process exited"},
            )
        if was_ready:
            self._spawn()
        else:
            self.startup_error = (
                "Model loading failed; check weights and logs, then restart"
            )
