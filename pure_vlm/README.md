# pure_vlm: one VLM as the only eyes, master-document rules as the judge

No detector, tracker or pose model. Qwen3.5-9B (MLX 4-bit) answers short, targeted questions on zoomed crops;
a deterministic timeline engine turns `timestamp + answers` into violations, following the team's master
document («Cashier routine, violations, and tracking list», 18.10.2026) code by code.

```
video ─► frame grid (0.5 s)
      ─► layout: the VLM grounds door / desk / safe / laptop / counting machine (once per clip, any camera angle)
      ─► state questions per zone            people: where is the cashier, how many visitors   (every 2 s)
         (coarse grid, then bisection        door:   open / closed                              (every 2 s)
          wherever an answer changes,        safe:   open? key in the lock?                     (every 4 s + key moments)
          down to 0.5 s; an unchanged        desk:   cash, collection bag, stamps, bag, clothing, drink, phone
          crop reuses its answer)                                                               (every 4 s + key moments)
                                             laptop: active / locked / dark / closed            (only when the cashier leaves)
      ─► action windows (4 frames, 1.5 s)    card / cash / bag handed over, counting machine, hand count, UV, typing
         while a visitor is in the room or cash lies on the desk
      ─► timeline.py (no model): departures, door runs, safe runs, desk items, visitor sessions → events
      ─► graded JSON: catalog type, timecode, «признаки возможного нарушения» description, evidence JPEGs
```

Why not "VLM → text → LLM judge": see `orig_and_vlm/TEXTFLOW_REPORT.md`. The judge was the weak link; here the
judge is code, and the VLM only answers questions it can see.

## Rules (timeline.py)
| Code | Master doc | What the timeline checks |
|---|---|---|
| C2 | door left open | door answered "open" for ≥ 5 s (≥ 3 s if it stays open to the end) |
| B5 | safe open while working | safe open while the cashier is at the desk for ≥ 3 s |
| B1 / B8 / B5 / B9 | cashier leaves (phase 6) | 1 s after the room empties: cash or sealed bag on the desk (B1), stamps (B8), safe open or key in lock (B5), laptop showing an open program (B9). Timecode: start of the walk out |
| B1 | next client while old cash on the desk | cash on the desk ≤ 2.5 s before the 2nd, 3rd … visitor comes in |
| B3 | personal items | handbag / clothing / drink on the desk at ≥ 2 asked times ≥ 1 s apart; a phone only with no visitor present |
| A6 | no ID card | the client's cash lies on the desk (2 samples in a row) before any ID card did |
| A7 | two clients at once | two visitors for ≥ 1.5 s and a handover while both are there |
| A1 | client leaves mid-count | the counting machine still runs in the window around the client's exit |
| D7 | bag taken without ID / power of attorney | a bag handover with no papers shown before it |
| D6 | bag contents to the safe unregistered | collection bag seen, then the safe opens with no machine count and no typing |
| A4 | (optional) cashier leaves with a visitor in the room | departure with visitors > 0 |

A2 (UV detector) and A3 (issued part not recounted) have no reliable signal yet; see Known limits.

**Second look (verify.py).** Every candidate gets one or two narrow yes/no questions on a high-resolution crop at its
own frames (B3 item, B1 cash or bag, B5 safe open *and* cashier sitting, C2 door still open 2 s and 4 s later), and
is dropped on a "no". At each departure the empty desk is asked "is anything else left lying on it?": cash → B1,
a bag → B1, anything else → B8 (X-24's teal stamps, which the list question calls "drink"/"phone").

## Results (40 training clips, master-document types)
`reports/eval_verified.md` (with the second look), `reports/eval_rules_only.md` (rules on the state answers only).

| | rules only | + second look | orig_and_vlm (engine + Qwen3.5-4B eyes) |
|---|---|---|---|
| violation videos found (of 21) | 18 | **19** | 16 |
| false alarms (of 19 clean) | 5 | **2** | 4 |
| precision / recall / F1 | 0.78 / 0.86 / 0.82 | **0.905 / 0.905 / 0.905** | 0.80 / 0.76 / 0.78 |
| right type (of 22 labeled moments) | 18 | 19 | not comparable (old type labels) |
| right type within 2 s | 14 | 14 | |

These are the clips the rules and prompts were tuned on, so expect less on new clips.

- **Missed:** X-22 (B5, key left in the safe lock: a few pixels; every key question said no) and X-44
  (A2, UV detector not used: the VLM does not locate a UV detector, and the check is not visible to it).
- **False alarms:** X-19 and X-35 (A6). The client's ID card is handed over hand to hand and never lies on the desk,
  so "cash before an ID card" fires. A6 is kept because it is the only rule that finds X-1 and X-2.
- **Types:** X-1 is reported as A6 (label A3). X-30's C2 is at 5 s (the cashier walks in and leaves the door open);
  the label marks the outsider walking through at 13 s.
- **Time:** ~5x real time, capped by the budget. Cold run of X-4 (73 s) through `analyze()` with the API's default
  240 s deadline: 205 s wall, peak 7.0 GB. On long clips the state questions take most of the budget and the action
  windows (A1, A7, D6, D7) get little; run the API with `ENGINE_DEADLINE_S=285`.

## Run
```bash
cd dathaton_engine            # (or the pure-vlm worktree)
export DATA_DIR=~/Downloads/open/videos HF_HUB_OFFLINE=1
.venv/bin/python -m pure_vlm.observe_all             # VLM answers for all labeled clips -> cache/pure_vlm/obs_<clip>.json
.venv/bin/python -m pure_vlm.eval --tag latest       # rules on the cached answers (seconds) -> pure_vlm/reports/
.venv/bin/python -m pure_vlm.show X-20               # one clip's answers as a timeline, with the events
.venv/bin/python -m pure_vlm.probe door X-32@8       # ask one question on a frame (prompt work)
.venv/bin/python -m pytest pure_vlm/tests
ENGINE_API_KEY=<secret> ENGINE_DEADLINE_S=285 .venv/bin/uvicorn pure_vlm.api:app --port 8000   # same endpoints as api/main.py
```
`pure_vlm.analyze(path)` has the same signature and output as `engine.analyze`. Settings: `config.py`
(`PV_VLM_REPO`, `PV_BUDGET_S`, sampling steps). Prompts: `prompts.py` (bump `VERSION` after an edit: it keys the cache).

## Model
| Role | Model | License |
|---|---|---|
| all perception | `mlx-community/Qwen3.5-9B-MLX-4bit` (base `Qwen/Qwen3.5-9B`), ~6 GB | Apache-2.0 |

The 4B was tried first: it could not tell an open glass door from a closed one (0/3 open doors, also with a
reference frame); the 9B got 8/8 on the same frames at the same speed per question.

## Catalog
A6, A7, B8, B9, D6, D7 were added to `rules/rules_catalog.json` from the master document and the extended
summary table. Their quotes are paraphrases marked with a `note` field: they were not checked against the act text.
