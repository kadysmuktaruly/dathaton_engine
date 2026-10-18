# Cash-desk discipline engine (ForteBank Datathon 26)

**Deployment branch:** see [docs/deployment.md](docs/deployment.md) for Ubuntu ARM,
Docker, the backend adapter, API mapping and one-video-at-a-time scheduling.

Local, offline analysis of cash-desk videos: `engine.analyze(video_path) -> dict` returns possible cash-discipline violations («признаки возможного нарушения») with timecodes, evidence frames, a Russian description and a legal reference taken **only** from `rules/rules_catalog.json`.

**Reference rules:** [`docs/master-document.md`](docs/master-document.md) (the cashier routine, which violations are reported, what is tracked). It takes priority over older notes such as `reports/dataset_review.md`. Only catalog codes marked `"active": true` can be reported: A1, A2, A3, A4, A6, A7, B1, B3, B5, B8, B9, C2, D6, D7. The other codes stay in the catalog as reference.

No cloud calls. Models (all Apache-2.0/MIT) are listed in `MODELS.md`.

## Setup (once, needs network)
```bash
uv sync                                               # Python 3.11 env
uv run python -m tools.download_weights  # RT-DETRv2, platform-specific Qwen3-VL-4B, RTMPose
```
Then run offline:
```bash
export HF_HUB_OFFLINE=1
```
### Fine-tuned detector (optional, recommended)
The stock RT-DETRv2 doesn't know the cash-room objects. A fine-tuned checkpoint (`cash`, `safe`, `safe_open`,
`counting_machine`, `collection_bag`, `desk` + the COCO classes we use) is published as a GitHub Release asset,
because `weights/` is git-ignored.

1. Download `detector_ft.zip` from the repo's **Releases** page (tag `detector-ft-v1`).
2. Check it and unzip into `weights/`:
   ```bash
   shasum -a 256 detector_ft.zip   # de130cca3f6f9ced6ef3277a05f4a602a05e84605d361a8d2ee28a61c17ee600
   unzip detector_ft.zip -d weights/   # -> weights/detector_ft/
   ```
3. Turn it on (without this variable the engine uses the stock detector):
   ```bash
   export ENGINE_DETECTOR_PATH=weights/detector_ft
   ```

Val AP@0.5 (82 frames from 8 held-out clips): 0.75 on the 6 new classes (stock: 0), 0.72 over all classes
(stock: 0.28). Weak: `cash` 0.55, `id_card` 0.29, `cup` 0.07 (stock 1.00, only 1 training box).
Per-class numbers: `weights/detector_ft/metrics.json`. How it was trained: `docs/labeling.md`.

Videos are read from `videos/` (override with `DATA_DIR`), labels from `open_labels.xlsx` (`LABELS_XLSX`).

## Run
```python
from engine import analyze
res = analyze("videos/X-1.mov")            # graded JSON (exact schema)
# debug info (codes, scores, refs, timings): runs/jobs/<job_id>/debug.json
```

## Pipeline
```
video → bar crop → 3 fps sampling with exact timestamps
      → RT-DETRv2 + ByteTrack + RTMPose → per-frame records (cache/perception/*.parquet)
      → scene anchors (desk, counting machine, laptop, safe, back line) → in-room people (glass filter)
      → roles: cashier (nearest to the seat, per frame) / visitor sessions (client 1, 2, … or the collector)
      → hands: cashier's wrists at the counting machine / laptop / safe
      → VLM desk scan (Qwen3-VL-4B, zoomed desk crop) + targeted VLM checks (after each departure; UV detector)
      → rules → events (+ bisection timecode refinement, VLM verifier) → catalog type + refs → evidence JPEGs → JSON
```
| Code | How it is detected | Training clips |
|---|---|---|
| C2 | door-sized persistent background change on the back wall after someone enters an empty room; timecode = first visitor through the open door, else the door opening | X-30, X-32, X-33 |
| B5 | safe seen open ≥ 8 s while the cashier works; or safe seen open after the cashier left the room | X-20, X-22 |
| B1 | cash / collection bag on the desk when the cashier leaves (scan or departure check); cash already on the desk when a new client arrives, incl. the previous client's bundle still lying at the same place | X-4, X-15, X-26, X-38 |
| B8 | departure check (VLM, tight desk crop): stamps left on the desk, answered or listed among the desk items | X-24 |
| B9 | laptop screen lit after the cashier left (dark-pixel share of the screen); not flagged when the VLM sees the lock screen | X-11, X-28 |
| B3 | scan: bag or coat on the desk; a drink brought in mid-clip next to cash (scan or detector `cup` + `cash`) | X-40, X-46 |
| A7 | two client sessions overlap and the newcomer reaches the cashier | X-18 |
| A1 | the client leaves while the cashier keeps counting at the machine (≥ 1.5 s after) | X-34 |
| A2 | VLM: a UV detector is on the desk, the cashier handles cash, the notes never go to the detector | X-44 |
| D7 | collector session with no ID card seen; timecode = bag pulled in | X-17 |
| D6 | the last lasting "safe open" after a collection bag, with < 3 s at the counting machine since the bag appeared | X-13 |
| A4 | cashier leaves the room while a client stays and cash lies on the desk (optional; no training clip) | — |
| A6, A3 | not reported: A6 is a debug-only candidate (ID cards are not reliably visible); A3 has no signal yet | X-2, X-1 |

Known limits: a key left in the safe lock (X-22) and the ID handover (X-2) are a few pixels at 464×848, so neither the detector nor the 4B VLM sees them reliably. The B9 lock-screen check relies on this dataset's lock screen being a landscape photo.

## Tools
- `python -m tools.labels`: parsed labels (duplicate files merged by md5)
- `python -m tools.contact_sheets`: review sheets (`reports/sheets/`)
- `python -m tools.cache_all` / `python -m tools.scan_all`: precompute perception and the VLM scan
- `python -m tools.overlay X-1 ...`: debug overlay videos (`runs/overlays/`)
- `python -m tools.eval [--no-vlm] [--augment] --tag NAME`: metrics → `reports/eval_NAME.md`
- `python -m tools.vlm_probe "<question>" X-31@10 ...`: ask the VLM about frames
- `pytest`: tests (glass filter, schema, wording, catalog, roles, departure rules, events, labels)

## Reports
`docs/master-document.md` (reference rules), `reports/dataset_review.md` (types per clip; see its master-document section), `reports/perception_baseline.md`, `reports/step3_gate.md`, `reports/eval_*.md`. Current numbers: `reports/eval_master_v3.md` (fine-tuned detector; before the rule change: `reports/eval_ft_baseline.md`).

## HTTP API (Step 7)
```bash
ENGINE_API_KEY=your-secret HF_HUB_OFFLINE=1 uv run uvicorn api.main:app --host 127.0.0.1 --port 8001 --workers 1
curl localhost:8001/health  # wait until ready=true
curl -H "X-API-Key: your-secret" -F "file=@videos/X-1.mov" localhost:8001/api/v1/analyze
curl -H "X-API-Key: your-secret" localhost:8001/api/v1/jobs/JOB_ID
curl -H "X-API-Key: your-secret" localhost:8001/api/v1/jobs/JOB_ID/analysis  # datathon Analysis
```
- Set `ENGINE_API_KEY` explicitly; missing configuration stops startup. Wrong request keys → 401.
- One warm inference process, no internal queue. Busy submissions → 409; loading → 503.
- Jobs answer `processing` → `done` | `failed`. `ENGINE_DEADLINE_S` defaults to 210 s.
  At the deadline actual inference is terminated before returning a partial snapshot or failure.
- `/analysis` adds catalog codes, duration and normalized player boxes while `/jobs/JOB_ID`
  preserves the strict graded result. `DELETE /api/v1/jobs/JOB_ID` cancels and reloads the worker.
- Linux uses the CPU PyTorch Qwen checkpoint; Mac retains MLX. Benchmark CPU latency and
  accuracy before deployment. Job IDs are ephemeral; datathon owns durable history and queueing.
- `ENGINE_NO_CACHE=1` disables the perception/scan caches (for cold timings).
