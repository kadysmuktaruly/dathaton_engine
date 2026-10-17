# Cash-desk discipline engine (ForteBank Datathon 26)

Local, offline analysis of cash-desk videos: `engine.analyze(video_path) -> dict` returns possible cash-discipline violations («признаки возможного нарушения») with timecodes, evidence frames, a Russian description and a legal reference taken **only** from `rules/rules_catalog.json`.

No cloud calls. Models (all Apache-2.0/MIT) are listed in `MODELS.md`.

## Setup (once, needs network)
```bash
uv sync                                               # Python 3.11 env
uv run python -m tools.download_weights --with-owlv2  # RT-DETRv2, Qwen3-VL-4B (MLX 4-bit), RTMPose, OWLv2
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
      → scene anchors (desk = devices + table, safe, back line) → in-room people (glass filter)
      → per-frame facts (head count, cashier seat, departures, entries)
      → VLM desk scan (Qwen3-VL-4B, zoomed desk crop, ≤18 s) → object/action flags
      → rules per confirmed type → events (+ bisection timecode refinement)
      → catalog type + refs → evidence JPEGs (full res, thin boxes) → JSON
```
| Code | How it is detected |
|---|---|
| C1 | newcomer stands beside the cashier with no desk between them, or a 3rd person joins |
| C2 | door-sized persistent background change on the back wall after someone enters an empty room |
| B5 | cashier leaves the room (candidate), or the scan sees the safe open |
| B1 | scan: cash on the desk when the cashier leaves or a new client comes in |
| B3 / D1 / B7 / X0 | scan flags (bag or coat / green collection bag + second person / counting alone / drink near cash) |

## Tools
- `python -m tools.labels`: parsed labels (duplicate files merged by md5)
- `python -m tools.contact_sheets`: review sheets (`reports/sheets/`)
- `python -m tools.cache_all` / `python -m tools.scan_all`: precompute perception and the VLM scan
- `python -m tools.overlay X-1 ...`: debug overlay videos (`runs/overlays/`)
- `python -m tools.eval [--no-vlm] [--augment] --tag NAME`: metrics → `reports/eval_NAME.md`
- `python -m tools.vlm_probe "<question>" X-31@10 ...`: ask the VLM about frames
- `pytest`: tests (glass filter, schema, wording, events, labels)

## Reports
`reports/dataset_review.md` (types per clip, confirmed by the user), `reports/perception_baseline.md`, `reports/step3_gate.md`, `reports/eval_*.md`.

## HTTP API (Step 7)
```bash
ENGINE_API_KEY=<secret> HF_HUB_OFFLINE=1 uv run uvicorn api.main:app --port 8000
curl -H "X-API-Key: <secret>" -F "file=@videos/X-1.mov" localhost:8000/api/v1/analyze   # -> {"job_id": ...}
curl -H "X-API-Key: <secret>" localhost:8000/api/v1/jobs/<job_id>                      # -> status + result
```
- No or wrong `X-API-Key` → 401. If `ENGINE_API_KEY` is unset, a random key is generated and printed at startup.
- One sequential worker; models load once at startup (~5 s).
- Jobs answer `processing` → `done` | `failed`. If a job runs past `ENGINE_DEADLINE_S` (default 240 s), the best result so far is returned: the rules-only result built before the VLM stage. The VLM stage is also skipped when the deadline is near.
- `ENGINE_NO_CACHE=1` disables the perception/scan caches (for cold timings).
