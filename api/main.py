"""Private engine API. The datathon backend owns durable jobs, queueing and playback."""

from __future__ import annotations

import os
import re
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from api.service import EngineService, ServiceConfig, ServiceError
from engine import config

MODEL_ID = "kadys-model"
ALLOWED = {".mov", ".mp4", ".webm", ".m4v", ".avi", ".mkv"}


def create_app(service: EngineService | None = None, api_key: str | None = None):
    key = api_key if api_key is not None else os.environ.get("ENGINE_API_KEY", "")
    service = service or EngineService(
        ServiceConfig(
            root=config.RUNS_DIR / "service",
            deadline_s=float(os.environ.get("ENGINE_DEADLINE_S", "210")),
            startup_timeout_s=float(os.environ.get("ENGINE_STARTUP_TIMEOUT_S", "600")),
            max_upload_bytes=int(os.environ.get("MAX_UPLOAD_MB", "100")) * 1024 * 1024,
            retained_jobs=int(os.environ.get("ENGINE_RETAINED_JOBS", "200")),
        )
    )

    @asynccontextmanager
    async def lifespan(app):
        if not key:
            raise RuntimeError(
                "Set ENGINE_API_KEY explicitly before starting the engine"
            )
        service.start()
        try:
            yield
        finally:
            await run_in_threadpool(service.close)

    app = FastAPI(title="Kadys model engine", lifespan=lifespan)
    app.state.service = service

    @app.exception_handler(ServiceError)
    async def service_error(_request, exc):
        headers = {"Retry-After": "2"} if exc.status in {409, 503} else None
        return JSONResponse(
            {"error": {"code": exc.code, "message": exc.message}},
            status_code=exc.status,
            headers=headers,
        )

    @app.exception_handler(HTTPException)
    async def http_error(_request, exc):
        return JSONResponse(
            {"error": {"code": "HTTP_ERROR", "message": str(exc.detail)}},
            status_code=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, _exc):
        return JSONResponse(
            {"error": {"code": "INVALID_REQUEST", "message": "Invalid form fields"}},
            status_code=422,
        )

    def require_key(x_api_key: str | None = Header(default=None)):
        if not x_api_key or not secrets.compare_digest(x_api_key, key):
            raise ServiceError(401, "UNAUTHORIZED", "Invalid or missing X-API-Key")

    auth = [Depends(require_key)]

    @app.get("/health")
    def health():
        state = service.health()
        return {
            **state,
            "model_id": MODEL_ID,
            "vlm_enabled": config.USE_VLM,
            "vlm_backend": config.VLM_BACKEND,
            "vlm_repo": config.VLM_REPO,
        }

    @app.get("/api/v1/models", dependencies=auth)
    def models():
        return {"models": [{"id": MODEL_ID, "name": "Kadys model"}]}

    @app.post("/api/v1/analyze", dependencies=auth)
    async def submit(
        file: Annotated[UploadFile, File()],
        model_id: Annotated[str, Form()] = MODEL_ID,
        client_job_id: Annotated[str | None, Form()] = None,
    ):
        path = None
        accepted = False
        try:
            if model_id != MODEL_ID:
                raise ServiceError(400, "UNKNOWN_MODEL", "Unknown model_id")
            if client_job_id is not None:
                if not re.fullmatch(r"[a-f0-9]{32}", client_job_id):
                    raise ServiceError(
                        400,
                        "INVALID_JOB_ID",
                        "client_job_id must be a 32-character hex UUID",
                    )
                try:
                    service.get(client_job_id)
                except ServiceError as exc:
                    if exc.status != 404:
                        raise
                else:
                    return {"job_id": client_job_id, "status": "processing"}
            if not file.filename:
                raise ServiceError(400, "FILE_REQUIRED", "A filename is required")
            ext = Path(file.filename).suffix.lower()
            if ext not in ALLOWED:
                raise ServiceError(
                    415, "UNSUPPORTED_FORMAT", "Unsupported video format"
                )
            state = service.health()
            if state["busy"]:
                raise ServiceError(
                    409, "ENGINE_BUSY", "Engine is analysing another video"
                )
            if not state["ready"]:
                raise ServiceError(
                    503, "ENGINE_NOT_READY", "Engine models are loading or unavailable"
                )
            job_id = client_job_id or uuid.uuid4().hex
            upload_dir = service.settings.root / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            # Separate upload attempts must never overwrite a video already in use,
            # including concurrent retries with the same client_job_id.
            path = upload_dir / f"{uuid.uuid4().hex}{ext}"
            size = 0
            with path.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > service.settings.max_upload_bytes:
                        raise ServiceError(
                            413, "FILE_TOO_LARGE", "Video exceeds upload size limit"
                        )
                    await run_in_threadpool(output.write, chunk)
            if not size:
                raise ServiceError(400, "EMPTY_FILE", "Video file is empty")
            result = service.submit(job_id, path)
            accepted = True
            return result
        finally:
            await file.close()
            if path is not None and not accepted:
                path.unlink(missing_ok=True)

    @app.get("/api/v1/jobs/{job_id}", dependencies=auth)
    def get_job(job_id: str):
        return service.get(job_id)

    @app.get("/api/v1/jobs/{job_id}/analysis", dependencies=auth)
    def get_analysis(job_id: str):
        return service.get(job_id, analysis=True)

    @app.get("/api/v1/jobs/{job_id}/metadata", dependencies=auth)
    def get_metadata(job_id: str):
        return service.details(job_id)

    @app.delete("/api/v1/jobs/{job_id}", dependencies=auth)
    def cancel(job_id: str):
        return service.cancel(job_id)

    return app


app = create_app()
