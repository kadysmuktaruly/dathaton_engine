# orig_and_vlm — progress checklist (handoff file)

Branch `orig-and-vlm` on GitHub, cut from `origin/main` at `95d35a6`. Everything new is in this folder; no file
outside it was edited.

**Labels caveat (from the team, 2026-10-18).** The per-clip violation *types* used for "right type" are wrong
for most violation clips: `CONFIRMED` in `tools/eval.py` and `reports/dataset_review.md` are early guesses. The
team is relabeling every clip by hand with context: what happened, what was missed, what must have been done.
So "type accuracy" numbers here are not reliable yet. Verdict (TP/FP/FN/TN) and timecode errors come from the
organizers' `open_labels.xlsx` and stay valid. The team's view: the vision side is roughly ready, and the
reasoning side (facts → type, description, rule) is what needs tuning against the new labels.

## Done (2026-10-18)
- [x] `scan.py`, idea 1: Qwen3.5-4B desk scan in the engine's scan format. It asks first at key moments
      (just before entries, at departures), then the engine's grid coarse-to-fine; bisection refine.
- [x] `closeout.py`, idea 2: on B5 `cashier_left_room` it asks whether anyone is still in the room, then
      whether a screen is on, items are left, or the safe is open. Verified on frames: presence 7/7,
      screen 6/6 (after rewording for X-22's dark-blue lock screen), items 5/5.
- [x] `__init__.py`: `analyze()` = `engine.analyze` with the two switches (`OV_SCAN`, `OV_CLOSEOUT`).
- [x] `eval.py`, `api.py` (wraps api/main.py, splits time between queued jobs), `loadtest.py`, 4 unit tests.
- [x] 40-clip eval: F1 0.78 vs 0.65 (`reports/eval_ov_full2.md`; table in README).
- [x] `TEXTFLOW_REPORT.md`: why the VLM→text→LLM pipeline was dropped. That code is on local branch `textflow`.

## Fine-tuned detector (2026-10-18, `main` b88602c merged)
- Weights: Release `detector-ft-v1` → `detector_ft.zip` (SHA-256 checked) → `weights/detector_ft/`, which git ignores.
  Turn on with `ENGINE_DETECTOR_PATH=weights/detector_ft`.
- With it, `engine/rules/b5.py` gates "cashier left the room": the alarm stays below threshold unless the
  detector also sees `safe_open`. `closeout.filter_events` now lifts a gated alarm back to the rule's normal
  score when the close-out check passes (screen on / items left / safe open), because that is evidence too.
- Test on the 8 clips held out from the detector's training (X-13 X-18 X-19 X-24 X-29 X-33 X-38 X-45; 5 violation,
  3 clean). Reports: `reports/eval_ft_held8_pure.md`, `eval_ft_held8_main.md`, `eval_ov_ft_held8.md`.

  | 8 held-out clips | found (of 5) | false alarms (of 3) | F1 |
  |---|---|---|---|
  | pure vision, stock / fine-tuned | 4 / 2 | 1 / 0 | 0.80 / 0.57 |
  | main, stock / fine-tuned | 4 / 3 | 1 / 0 | 0.80 / 0.75 |
  | orig_and_vlm, stock / fine-tuned | 3 / **4** | 2 / 2 | 0.60 / 0.73 |

  With the fine-tuned detector, orig_and_vlm found X-38 (B1 at 36.3 s vs 33; the stock detector never saw the
  client come in) and X-24 (B5 at 9.0 s vs 8, lifted by the close-out check). It still misses X-13 (B7, no rule).
  Its false alarms are X-19 (B1, cash on the desk when a person enters) and X-29 (the clean walk-out that looks
  like a violation). Only 8 clips, so one clip moves F1 a lot.

## Not done / next
- [ ] API load test: 5 uploads at once (`loadtest.py`). Two VLMs are loaded; check that every job ends < 240 s.
- [ ] Ablations: `OV_SCAN=0` (close-out only) and `OV_CLOSEOUT=0` (scan only). The engine's own Qwen3-VL scan is
      not cached on this machine, so ~15 min each.
- [ ] Angle robustness: `eval.py --augment`.
- [ ] New false alarms: X-39 and X-10 (cash when a client arrives), X-19 (drink). Maybe require cash in 2
      scan frames before an arrival, or drop X0 (it has no confirmed rule source).
- [ ] X-20 (safe left open while the cashier sits) and X-44 (B7): the engine has no rule. The eyes could
      answer "safe open" reliably on the full frame (4/4 in the benchmark).
