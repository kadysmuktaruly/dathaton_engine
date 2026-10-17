# orig_and_vlm: rules engine + a newer VLM as better eyes

The teammate's rules engine (`engine/`, from `main`, **not modified**) still makes every decision.
This folder adds a second, newer VLM (Qwen3.5-4B) where the engine was weakest:

1. **Better desk scan** (`scan.py`). The engine's rules for B1 / B3 / D1 / X0 read a per-frame desk scan
   (cash, bag, collection bag, drink). Here it is asked by Qwen3.5-4B in one JSON call per frame on
   the same zoomed desk crop and frame grid, and first at the moments the rules care about: just before
   a new person comes in, and when the cashier leaves. Same table format, so the rules run unchanged.
2. **Close-out check** (`closeout.py`). 5 of the engine's 6 false alarms were B5 "cashier left the room".
   One second after the room empties the check asks:
   - **Is anyone still in the room?** If yes, the departure was a perception miss and the alarm is dropped.
   - **Is anything left exposed?** That means a lit laptop screen, items on the desk, or an open safe.
     If nothing is, the workplace was closed out properly and the alarm is dropped.

   Checked on frames verified by eye: presence 7/7, screen 6/6, items 5/5.

Everything else is the engine's code, called as is: perception, rules, merging, the Qwen3-VL
verifier, evidence frames, catalog wording, the output schema, and the partial result plus deadline.
`orig_and_vlm.analyze` has the same signature and output as `engine.analyze`.

Why not "video → text → LLM decides": see `TEXTFLOW_REPORT.md`.

## Results (40 training clips)
Baseline: `main`'s `reports/eval_B0_uncached.md`. This version: `reports/eval_ov_full2.md`. Times are from
the cold run, `reports/eval_ov_full.md`; the second run reused cached scans.

| | pure vision (no VLM) | main (engine + Qwen3-VL) | orig_and_vlm |
|---|---|---|---|
| violation videos found (of 21) | 11 | 13 | **16** |
| false alarms (of 19 clean) | 5 | 6 | **4** |
| precision / recall / F1 | 0.688 / 0.524 / 0.595 | 0.684 / 0.619 / 0.65 | **0.80 / 0.762 / 0.78** |
| right type (matched timecodes)* | 0.727 | 0.923 | 0.824 |
| timecode within 1 s / median error | 0.55 / 1.0 s | 0.54 / 1.0 s | 0.53 / 1.0 s |
| seconds per clip, mean / max (cold) | ~3–14 (perception only) | 24 / 48 | 39 / 81 |

\* Type labels are being redone by hand (see PROGRESS.md), so this row is not reliable yet. Pure vision =
`tools.eval --no-vlm` (`reports/eval_pure_vision.md`): perception + rules only, so B1 / B3 / D1 / X0 can't
fire. It misses X-40 and X-46 on top of main's misses.

- **Newly found:** X-4 (B1 at 55.3 s vs 57), X-17 (D1, exact), plus X-2 and X-34, flagged as B1 although the
  labeled types are unclear (A5 / A1).
- **False alarms removed by the close-out check:** X-12, X-23, X-25, X-27.
- **New false alarms from the new scan:** X-19 (drink) and X-39 (cash as a client arrives).
- **Lost:** X-13. `main` caught it only through a "cashier left" alarm at 63 s, while the labeled
  violation is hand counting at 47 s; the close-out check correctly sees a closed-out desk at 63 s.
- **Still missed:** X-1 (unclear type), X-20 (safe left open while the cashier sits), X-38 (perception
  misses the client), X-44 (B7 hand counting; no rule for it in the engine).

## Run
```bash
cd dathaton_engine
export DATA_DIR=~/Downloads/open/videos HF_HUB_OFFLINE=1
.venv/bin/python -m orig_and_vlm.eval --tag full             # 40 clips -> orig_and_vlm/reports/eval_ov_full.md
OV_SCAN=0 .venv/bin/python -m orig_and_vlm.eval --tag co     # close-out only (idea 2)
OV_CLOSEOUT=0 .venv/bin/python -m orig_and_vlm.eval --tag sc # scan only (idea 1)
.venv/bin/python -m pytest orig_and_vlm/tests
ENGINE_API_KEY=<secret> .venv/bin/uvicorn orig_and_vlm.api:app --port 8000   # same endpoints as api/main.py
.venv/bin/python -m orig_and_vlm.loadtest --key <secret> X-26 X-20 X-34 X-12 X-30   # 5 uploads at once
```
Settings are in `config.py` (`OV_EYES_REPO`, `OV_SCAN`, `OV_CLOSEOUT`, budgets). The API splits the time
left between queued jobs, so a burst of uploads finishes with smaller VLM budgets instead of timing out.

## Models
| Role | Model | License |
|---|---|---|
| Desk scan + close-out | `mlx-community/Qwen3.5-4B-MLX-4bit` (base `Qwen/Qwen3.5-4B`), 3.1 GB | Apache-2.0 (HF card, checked 2026-10-17) |
| Verifier, perception | the engine's (see `MODELS.md`) | Apache-2.0 / MIT |

Download once: `.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen3.5-4B-MLX-4bit')"`.

## Known limits
- **X-29** is labeled clean but looks like a violation: the cashier leaves with the laptop screen lit,
  so it stays a false alarm.
- **X-38**: perception never counts the client walking in at 31–35 s, so no rule fires. That is a
  perception issue, not the scan.
- **Cost:** two VLMs are loaded (~4 GB each at peak). The scan takes ~2 s per frame, about 30–40 s per
  video on top of the engine.
