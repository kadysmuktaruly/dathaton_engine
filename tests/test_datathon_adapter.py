"""Exercise the supplied backend adapter against the documented Analysis boundary."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel, ConfigDict


class AnalysisContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    duration_sec: float
    violations: list[dict] = []
    events: list[dict] = []


@pytest.fixture
def adapter(monkeypatch):
    app = types.ModuleType("app")
    config = types.ModuleType("app.config")
    config.settings = SimpleNamespace(
        engine_url="http://engine:8001",
        engine_api_key="secret",
        engine_timeout_sec=220,
        job_timeout_sec=240,
    )
    schemas = types.ModuleType("app.schemas")
    schemas.Analysis = AnalysisContract
    for name, module in [
        ("app", app),
        ("app.config", config),
        ("app.schemas", schemas),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1] / "integration" / "datathon_pipeline.py"
    spec = importlib.util.spec_from_file_location("datathon_adapter_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    return module


def mock_client(adapter, monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(
        adapter.httpx,
        "Client",
        lambda **kw: original(**kw, transport=httpx.MockTransport(handler)),
    )


def test_adapter_returns_analysis_and_passes_key_and_model(
    adapter, monkeypatch, tmp_path
):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    calls = []
    payload = {"duration_sec": 8.0, "violations": [{"code": "B5"}], "events": []}

    def handler(request):
        calls.append(request.method)
        assert request.headers["X-API-Key"] == "secret"
        if request.url.path == "/health":
            return httpx.Response(200, json={"accepting": True})
        if request.method == "POST":
            body = request.read()
            assert (
                b"kadys-model" in body and b"client_job_id" in body and b"video" in body
            )
            return httpx.Response(
                200, json={"job_id": "unused", "status": "processing"}
            )
        if request.url.path.endswith("/analysis"):
            return httpx.Response(200, json=payload)
        return httpx.Response(200, json={"status": "done"})

    mock_client(adapter, monkeypatch, handler)
    assert adapter.analyze_video(video, "kadys-model").model_dump() == payload
    assert "DELETE" not in calls


@pytest.mark.parametrize("failure", ["failed", "timeout", "lost_post", "bad_analysis"])
def test_adapter_cancels_on_failures(adapter, monkeypatch, tmp_path, failure):
    video = tmp_path / "clip.mov"
    video.write_bytes(b"video")
    cancelled = []

    def handler(request):
        if request.method == "DELETE":
            cancelled.append(request.url.path)
            return httpx.Response(200, json={"status": "failed"})
        if request.url.path == "/health":
            return httpx.Response(200, json={"accepting": True})
        if request.method == "POST":
            if failure == "lost_post":
                raise httpx.ReadTimeout("lost response", request=request)
            return httpx.Response(200, json={"status": "processing"})
        if request.url.path.endswith("/analysis"):
            return httpx.Response(200, json={"wrong_schema": True})
        code = "ANALYSIS_TIMEOUT" if failure == "timeout" else "ANALYSIS_FAILED"
        return httpx.Response(
            200,
            json={
                "status": "done" if failure == "bad_analysis" else "failed",
                "error": {"code": code, "message": "failure"},
            },
        )

    mock_client(adapter, monkeypatch, handler)
    expected = (
        httpx.ReadTimeout
        if failure == "lost_post"
        else TimeoutError
        if failure == "timeout"
        else ValueError
        if failure == "bad_analysis"
        else RuntimeError
    )
    with pytest.raises(expected):
        adapter.analyze_video(video, "kadys-model")
    assert len(cancelled) == 1


def test_busy_retry_reuses_uuid(adapter, monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"data")
    attempts = []

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"accepting": True})
        if request.method == "POST":
            import re

            attempts.append(
                re.search(
                    b'name="client_job_id"\r\n\r\n([a-f0-9]{32})', request.read()
                )[1]
            )
            return httpx.Response(
                409 if len(attempts) == 1 else 200, json={"status": "processing"}
            )
        if request.url.path.endswith("/analysis"):
            return httpx.Response(
                200, json={"duration_sec": 1, "violations": [], "events": []}
            )
        return httpx.Response(200, json={"status": "done"})

    mock_client(adapter, monkeypatch, handler)
    adapter.analyze_video(video, "kadys-model")
    assert len(attempts) == 2 and attempts[0] == attempts[1]


def test_adapter_bounds_total_wait_and_validates_timeout_margin(
    adapter, monkeypatch, tmp_path
):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"data")
    cancelled = []
    clock = [0]

    def monotonic():
        clock[0] += 100
        return clock[0]

    def handler(request):
        if request.method == "DELETE":
            cancelled.append(True)
        return httpx.Response(200, json={"accepting": False})

    mock_client(adapter, monkeypatch, handler)
    monkeypatch.setattr(adapter.time, "monotonic", monotonic)
    with pytest.raises(TimeoutError):
        adapter.analyze_video(video, "kadys-model")
    assert cancelled
    adapter.settings.engine_timeout_sec = 240
    with pytest.raises(ValueError, match="10 seconds"):
        adapter.analyze_video(video, "kadys-model")
