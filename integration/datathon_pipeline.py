"""Legacy blocking adapter for the earlier thread-based backend.

The updated datathon backend ships an async adapter; do not overwrite it with this file.

Runs in the existing APScheduler task. All waits, uploads and polling share one
deadline, shorter than JOB_TIMEOUT_SEC. Cancels unfinished engine work on failure.
"""

import time
import uuid
from pathlib import Path

import httpx
from app.config import settings
from app.schemas import Analysis

MODELS = {"kadys-model": "Kadys model"}


def analyze_video(video_path: Path, model_id: str) -> Analysis:
    if model_id not in MODELS:
        raise ValueError(f"Unknown model: {model_id}")
    if not settings.engine_api_key:
        raise ValueError("Set ENGINE_API_KEY in the backend environment")
    if not 0 < settings.engine_timeout_sec <= settings.job_timeout_sec - 10:
        raise ValueError(
            "ENGINE_TIMEOUT_SEC must leave at least 10 seconds before JOB_TIMEOUT_SEC"
        )
    deadline = time.monotonic() + settings.engine_timeout_sec
    job_id = uuid.uuid4().hex
    submitted, completed = False, False

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("Engine analysis deadline exceeded")
        return min(10.0, seconds)

    def pause():
        time.sleep(min(1.0, remaining()))

    with httpx.Client(
        base_url=settings.engine_url.rstrip("/"),
        headers={"X-API-Key": settings.engine_api_key},
    ) as client:
        try:
            while not submitted:
                response = client.get("/health", timeout=remaining())
                response.raise_for_status()
                health = response.json()
                if health.get("startup_error"):
                    raise RuntimeError(health["startup_error"])
                if not health.get("accepting"):
                    pause()
                    continue
                with video_path.open("rb") as file:
                    response = client.post(
                        "/api/v1/analyze",
                        timeout=remaining(),
                        data={"model_id": model_id, "client_job_id": job_id},
                        files={
                            "file": (video_path.name, file, "application/octet-stream")
                        },
                    )
                if response.status_code in {409, 503}:
                    pause()
                    continue
                response.raise_for_status()
                submitted = True

            while True:
                response = client.get(f"/api/v1/jobs/{job_id}", timeout=remaining())
                response.raise_for_status()
                result = response.json()
                if result["status"] == "done":
                    response = client.get(
                        f"/api/v1/jobs/{job_id}/analysis", timeout=remaining()
                    )
                    response.raise_for_status()
                    analysis = Analysis.model_validate(response.json())
                    completed = True
                    return analysis
                if result["status"] == "failed":
                    error = result.get("error", {})
                    if error.get("code") == "ANALYSIS_TIMEOUT":
                        raise TimeoutError(error.get("message", "Engine timed out"))
                    raise RuntimeError(
                        f"{error.get('code', 'ANALYSIS_FAILED')}: {error.get('message', '')}"
                    )
                pause()
        finally:
            if not completed:
                # Also covers a lost POST response: the client supplied the job UUID.
                # A network failure can prevent cancellation; the engine's hard deadline
                # still terminates inference independently of backend polling.
                try:
                    client.delete(f"/api/v1/jobs/{job_id}", timeout=3)
                except httpx.HTTPError:
                    pass
