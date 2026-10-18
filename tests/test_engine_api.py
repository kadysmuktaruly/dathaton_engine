import time
import uuid

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.service import EngineService, ServiceConfig
from tests.worker_fakes import startup_failure, startup_hang, worker

AUTH = {"X-API-Key": "test-secret"}


def wait_for(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for worker state")


@pytest.fixture
def client(tmp_path):
    service = EngineService(
        ServiceConfig(tmp_path, deadline_s=1.2, max_upload_bytes=64), worker
    )
    with TestClient(create_app(service, "test-secret")) as client:
        wait_for(lambda: client.get("/health").json()["accepting"])
        yield client
    assert service.process is None


def post(client, content=b"ok", name="clip.mp4", **data):
    return client.post(
        "/api/v1/analyze", files={"file": (name, content)}, data=data, headers=AUTH
    )


def finished(client, job_id):
    result = client.get(f"/api/v1/jobs/{job_id}", headers=AUTH).json()
    return result if result["status"] != "processing" else None


def test_authentication_and_model_listing(client):
    assert client.get("/api/v1/models").status_code == 401
    assert (
        client.get("/api/v1/models", headers={"X-API-Key": "wrong"}).status_code == 401
    )
    assert (
        client.get("/api/v1/models", headers=AUTH).json()["models"][0]["id"]
        == "kadys-model"
    )
    assert (
        client.get("/api/v1/jobs/missing", headers=AUTH).json()["error"]["code"]
        == "JOB_NOT_FOUND"
    )


@pytest.mark.parametrize(
    ("content", "filename", "data", "status", "code"),
    [
        (b"", "clip.mp4", {}, 400, "EMPTY_FILE"),
        (b"a" * 65, "clip.mp4", {}, 413, "FILE_TOO_LARGE"),
        (b"ok", "clip.exe", {}, 415, "UNSUPPORTED_FORMAT"),
        (b"ok", "clip.mp4", {"model_id": "wrong"}, 400, "UNKNOWN_MODEL"),
        (b"ok", "clip.mp4", {"client_job_id": "../oops"}, 400, "INVALID_JOB_ID"),
    ],
)
def test_invalid_uploads_leave_no_files(client, content, filename, data, status, code):
    response = post(client, content, filename, **data)
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert not list(client.app.state.service.settings.root.glob("uploads/*"))


def test_webm_and_analysis_contract(client):
    response = post(client, name="clip.webm")
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    result = wait_for(lambda: finished(client, job_id))
    assert set(result) == {"job_id", "status", "verdict", "violations"}
    assert result["status"] == "done"
    assert client.get(f"/api/v1/jobs/{job_id}/analysis", headers=AUTH).json() == {
        "duration_sec": 12.5,
        "violations": [],
        "events": [],
    }
    assert (
        client.get(f"/api/v1/jobs/{job_id}/metadata", headers=AUTH).json()["partial"]
        is False
    )
    assert not list(client.app.state.service.settings.root.glob("uploads/*"))


def test_one_video_at_a_time_and_uuid_retry(client):
    job_id = uuid.uuid4().hex
    assert post(client, b"slow", client_job_id=job_id).json()["job_id"] == job_id
    assert (
        client.get(f"/api/v1/jobs/{job_id}/analysis", headers=AUTH).status_code == 409
    )
    retry = post(client, b"slow", client_job_id=job_id)
    assert retry.status_code == 200 and retry.json()["job_id"] == job_id
    busy = post(client)
    assert busy.status_code == 409 and busy.json()["error"]["code"] == "ENGINE_BUSY"
    assert len(client.app.state.service.jobs) == 1
    wait_for(lambda: finished(client, job_id))
    assert post(client).status_code == 200


@pytest.mark.parametrize("mode", [b"hang", b"partial"])
def test_deadline_kills_process_before_terminal_and_reloads(client, mode):
    service = client.app.state.service
    old_process = service.process
    old_pid = old_process.pid
    job_id = post(client, mode).json()["job_id"]
    result = wait_for(lambda: finished(client, job_id))
    assert result["status"] == ("done" if mode == b"partial" else "failed")
    if mode == b"hang":
        assert result["error"]["code"] == "ANALYSIS_TIMEOUT"
    else:
        assert client.get(f"/api/v1/jobs/{job_id}/metadata", headers=AUTH).json()[
            "partial"
        ]
        assert (
            client.get(f"/api/v1/jobs/{job_id}/analysis", headers=AUTH).status_code
            == 200
        )
    # _kill closes the old Process only after join, proving the old child has exited.
    with pytest.raises(ValueError, match="closed"):
        old_process.is_alive()
    wait_for(lambda: client.get("/health").json()["accepting"])
    assert service.process.pid != old_pid
    next_id = post(client).json()["job_id"]
    assert wait_for(lambda: finished(client, next_id))["status"] == "done"


def test_cancel_terminates_and_is_idempotent(client):
    old_process = client.app.state.service.process
    job_id = post(client, b"hang").json()["job_id"]
    response = client.delete(f"/api/v1/jobs/{job_id}", headers=AUTH)
    assert response.json()["error"]["code"] == "CANCELLED"
    with pytest.raises(ValueError, match="closed"):
        old_process.is_alive()
    assert (
        client.delete(f"/api/v1/jobs/{job_id}", headers=AUTH).json() == response.json()
    )
    wait_for(lambda: client.get("/health").json()["accepting"])


def test_concurrent_uuid_retry_cannot_replace_active_video(client):
    service = client.app.state.service
    job_id = post(client, b"slow").json()["job_id"]
    original = service.jobs[job_id]["path"]
    duplicate = service.settings.root / "retry.mp4"
    duplicate.write_bytes(b"other data")
    assert service.submit(job_id, duplicate)["job_id"] == job_id
    assert service.jobs[job_id]["path"] == original
    assert original.read_bytes() == b"slow"
    assert not duplicate.exists()
    wait_for(lambda: finished(client, job_id))


def test_worker_crash_and_analysis_failure(client):
    crashed = post(client, b"crash").json()["job_id"]
    assert (
        wait_for(lambda: finished(client, crashed))["error"]["code"] == "WORKER_EXITED"
    )
    wait_for(lambda: client.get("/health").json()["accepting"])
    failed = post(client, b"fail").json()["job_id"]
    assert (
        wait_for(lambda: finished(client, failed))["error"]["code"] == "ANALYSIS_FAILED"
    )


@pytest.mark.parametrize("target", [startup_failure, startup_hang])
def test_startup_failure_is_visible_and_rejects_uploads(tmp_path, target):
    service = EngineService(ServiceConfig(tmp_path, startup_timeout_s=0.5), target)
    with TestClient(create_app(service, "test-secret")) as client:
        wait_for(lambda: client.get("/health").json()["startup_error"])
        assert not client.get("/health").json()["ready"]
        assert post(client).status_code == 503


def test_explicit_key_required(tmp_path):
    service = EngineService(ServiceConfig(tmp_path), worker)
    with (
        pytest.raises(RuntimeError, match="ENGINE_API_KEY"),
        TestClient(create_app(service, "")),
    ):
        pass


def test_only_one_service_owns_directory(tmp_path):
    service = EngineService(ServiceConfig(tmp_path), worker)
    other = EngineService(ServiceConfig(tmp_path), worker)
    service.start()
    try:
        with pytest.raises(RuntimeError, match="One engine instance"):
            other.start()
    finally:
        service.close()


def test_retention_prunes_results_and_restart_discards_jobs(tmp_path):
    service = EngineService(ServiceConfig(tmp_path, retained_jobs=1), worker)
    with TestClient(create_app(service, "test-secret")) as client:
        wait_for(lambda: client.get("/health").json()["accepting"])
        first = post(client).json()["job_id"]
        wait_for(lambda: finished(client, first))
        second = post(client).json()["job_id"]
        wait_for(lambda: finished(client, second))
        assert client.get(f"/api/v1/jobs/{first}", headers=AUTH).status_code == 404
        assert not service._file(first).exists()
    fresh = EngineService(ServiceConfig(tmp_path), worker)
    with TestClient(create_app(fresh, "test-secret")) as client:
        assert client.get(f"/api/v1/jobs/{second}", headers=AUTH).status_code == 404
        assert not fresh._file(second).exists()
