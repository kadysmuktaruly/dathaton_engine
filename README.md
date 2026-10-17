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
