# Deploy kadys_model_deploy with datathon

Architecture: frontend → datathon API/database/APScheduler → private engine API.
Keep datathon's existing worker and job history. The engine accepts exactly one active
video; HTTP 409 ENGINE_BUSY means keep waiting in the backend. There is no engine queue.
Use one Uvicorn process for each service and WORKER_CONCURRENCY=1 in datathon.

## Ubuntu 24.04 ARM / 12 cores / 96 GB RAM

Linux defaults to PyTorch Transformers and Qwen/Qwen3-VL-4B-Instruct, float32 on CPU.
Mac defaults to the existing MLX 4-bit checkpoint. Both use the same engine prompts and
rules, but precision/runtime differences can change outputs: re-evaluate on the server.
The Linux runtime does not install or import MLX. Full VLM analysis is enabled by default.
ENGINE_USE_VLM=0 is an explicit reduced-feature diagnostic mode; it omits VLM-dependent
signals and must not be treated as equivalent accuracy.

Start with 8 PyTorch threads, 2 ONNX threads, one analysis process. These are starting
values, not benchmark results. There is no claim that CPU inference meets 210 seconds.
Cold model loading has a separate 600-second limit, outside the per-job budget.

## Build and download weights

Requires Docker Engine and Compose on the server. Run in this repository:

```bash
cp .env.deploy.example .env.deploy
# Edit ENGINE_API_KEY to a random secret; reuse it in datathon's .env.
docker compose build
docker compose run --rm -e HF_HUB_OFFLINE=0 engine python -m tools.download_weights
```

This downloads the stock detector, the Linux Qwen checkpoint and RTMPose to persistent
volumes. OWLv2 is not needed for inference. The fine-tuned cash-room detector is
recommended: download detector_ft.zip from this repository's detector-ft-v1 release,
verify SHA-256 de130cca3f6f9ced6ef3277a05f4a602a05e84605d361a8d2ee28a61c17ee600,
and unpack locally so `weights/detector_ft/config.json` and the weights exist. Copy it:

```bash
docker compose run --rm -v "$PWD/weights/detector_ft:/checkpoint:ro" engine \
  python -c "import shutil; shutil.copytree('/checkpoint','/app/weights/detector_ft',dirs_exist_ok=True)"
```

Uncomment ENGINE_DETECTOR_PATH in .env.deploy. There is no silent fallback when an
explicit checkpoint path is invalid. Start the engine after downloads:

```bash
docker compose up -d
docker compose logs -f engine
curl http://127.0.0.1:8001/health
```

Wait for `ready: true` and `accepting: true` before uploading. `ok: true` alone means
the HTTP service is alive. Model loading failures appear as startup_error and in logs;
fix the cause and restart the service. The port is bound only to host loopback.
Docker health checks report readiness; Docker restart policy alone does not restart an
unhealthy-but-running container after a startup failure.

For native Python, use Python 3.11, `uv sync --locked`, download the same weights, set
environment variables explicitly (uvicorn does not read .env.deploy automatically),
then `uv run uvicorn api.main:app --host 127.0.0.1 --port 8001 --workers 1`.

## Connect the updated datathon backend

The datathon backend now contains its own async adapter in app/pipeline.py and a FIFO
APScheduler queue with one active slot. Its main docker-compose.yml builds this engine
checkout as a private service and injects the same ENGINE_API_KEY into both services.
Follow datathon/deploy/README.md for the combined deployment. Place this branch at
../dathaton_engine_deploy relative to datathon, or set ENGINE_SOURCE_DIR there.

Use the combined backend Compose stack instead of starting a second standalone engine
Compose stack. The backend uses http://engine:8001 in Docker, or ENGINE_URL pointing at
this engine for native development. It retrieves /analysis results, reuses its backend
UUID for engine jobs, and cancels unfinished inference during timeout/shutdown.

The integration/datathon_pipeline.py blocking adapter and datathon.compose.yaml override
are retained as legacy examples for the earlier thread-based backend. Do not copy them
into the updated async backend; its adapter and main Compose stack are already wired.

## Private API contract

All /api/v1 requests require X-API-Key. /health is unauthenticated on the private port.
Errors have `{ "error": { "code": "...", "message": "..." } }` bodies.

| Request | Response |
| --- | --- |
| GET /health | ready, busy, accepting, worker_pid, startup_error, VLM configuration |
| GET /api/v1/models | models: [{id: "kadys-model", name: "Kadys model"}] |
| POST /api/v1/analyze | multipart file, optional model_id and client_job_id; returns job_id, status |
| GET /api/v1/jobs/{id} | strict graded result when done; error object on failure |
| GET /api/v1/jobs/{id}/analysis | datathon Analysis: duration_sec, violations with codes, events |
| GET /api/v1/jobs/{id}/metadata | status and partial flag |
| DELETE /api/v1/jobs/{id} | stop/join active inference, mark failed/CANCELLED, reload models |

client_job_id is an optional lowercase 32-character hex UUID. Use a fresh one per analysis;
repeating an accepted UUID returns its existing job without submitting inference again.
Keys are scoped to this engine process lifetime and retained-job limit. The backend
adapter supplies a UUID before uploading, so it can cancel even if the POST response is lost.

MOV, MP4, WebM, M4V, AVI and MKV are accepted. Size defaults to 100 MB. Empty files,
unknown models and unsupported extensions are rejected. Invalid video contents fail the
job. Normalized annotation boxes are shifted out of the engine's cropped image back
into the full original frame. Sparse boxes appear only near their observation time;
missing boxes are omitted. Legal links come only from the existing rule catalog;
internal-only references have rule=null and a note. The strict graded JSON is unchanged.

## Timeouts, restarts and storage

The 210-second engine limit runs from engine acceptance and is enforced independently
of polling. At the deadline the supervisor stops and joins the actual model process.
If a complete rules-only snapshot exists, that snapshot becomes done and metadata.partial
is true; otherwise the job fails with ANALYSIS_TIMEOUT. The next worker reloads models
before accepting another video. Cancellation discards partial results. A crashed worker
fails the active job and reloads. The API stays responsive during model loading.

The adapter uses a 220-second budget covering engine-readiness waits, upload, polling
and result retrieval, with up to 3 extra seconds for cancellation. Set the backend's
JOB_TIMEOUT_SEC to at least 240. Backend queue waiting still happens before that budget:
a five-minute end-to-end requirement needs an admission limit/upload-based deadline in
datathon. No change in this branch claims to solve that backend policy.

Completed engine results are stored on disk, not retained as base64 blobs in RAM.
Uploads are deleted at completion/cancel/timeout. Only the latest 200 engine jobs are
retained by default. Engine job IDs are ephemeral: restart discards them and cleans
service result/upload files. Datathon's database and videos are durable and its startup
requeues unfinished backend jobs, which submit fresh engine jobs. A standalone engine
restart may fail an in-flight backend job; resubmit it through datathon. Do not run
multiple engine instances against the same runs directory (a file lock enforces this).

Keep all Docker volumes and datathon database/upload volumes on the persistent disk;
`docker compose down -v` deletes them. With the server stopping after 72 hours, restart
the VM and services before the next demo; there is no service availability while stopped.
Watch disk usage: model caches and perception/VLM caches need periodic cleanup, and
datathon also retains original videos, playback copies and evidence in its database.

## Verification before the demo

Run `uv run pytest -q` locally. API tests use real spawned dummy workers, not downloaded
models. On the server, benchmark uncached representative clips with ENGINE_NO_CACHE=1
and full VLM enabled; inspect durations, partial status, accuracy and peak memory.
Also submit two videos through datathon to confirm one stays queued, then test a restart
and a timeout. ARM Docker builds and actual inference must be verified on the target VM.
