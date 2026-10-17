# Step 3 gate: B5 rule ("cashier leaves; safe/keys unattended")

The full pipeline runs end to end: perception → scene → facts → rules → evidence JPEGs → strict JSON (schema-validated). Debug output goes to `runs/jobs/<id>/debug.json`. Rules + evidence take under 0.5 s per clip on cached perception. Eval: `reports/eval_step3.md` (rules only, no VLM).

## Result: gate not met
| | Value |
|---|---|
| B5 clips found | 4 / 5 (X-20, "safe open while the cashier sits", needs a safe_open detector) |
| B5 timecode errors | 1.7, 4.7, 4.3, 1.3 s (median ≈ 3 s; gate is ≤ 2 s) |
| Clean clips falsely flagged by B5 | 5 (X-12, X-23, X-25, X-27, X-29) |

## Why
1. **The violation is not the walk-out itself.** Five clean clips show the same walk-out after a proper close-out. What differs is a key left in the lock or something left open/on. At 464×848 the key is a few pixels, and the stock detector has no safe class. In 4 of 9 clips the automatic safe guess picked the wrong object. No geometric rule can separate these, so this is the case the brief's VLM fallback is for.
2. **Timing:** the labels mark the moment the cashier turns away from the workplace. With this camera the door is directly behind the desk, so walking out barely changes image position. After passing the door, the cashier stays visible behind the glass near the door for 2–4 s and still passes the depth test. Exit time therefore lags the label, and the depth heuristic only partly fixes it.
3. **Desk scale in layout (b):** the desk is found only as the laptop box (desk width 40–70 px instead of about 270 px), which distorts every relative distance in those clips (X-13, X-22, X-23, X-29). A fine-tuned `desk`/`safe` class fixes the anchors.

## Other observations from this run
- C1 also fires on normal client arrivals (`to_staff_side`). The "cashier seat" estimate is unreliable when a client sits for long. Fixable with role logic (the cashier is the person on the laptop/chair side).
- Recall 0.76 / precision 0.57 overall; the misses are the types that need cash/bag/safe classes or the VLM (A3, D1, B5-open, B7, X0).

## Options
- **A. VLM-led verification and timing (recommended next).** Perception proposes candidate moments (departures, entries, safe visits). Qwen3-VL then checks 2–4 frames per candidate ("is a key left in the safe / is the safe open / is the door left open?") and refines the timecode by scanning at 0.5 s steps around the candidate ("is the cashier still at the workplace?"). This uses the brief's fallback for B5-leave and C2.
- **B. Fine-tune the detector first** (`safe`, `safe_open`, `cash`, `counting_machine`, `collection_bag`, `desk`). OWLv2 pre-labels are weak, so this needs roughly 1–2 hours of your time correcting boxes in local Label Studio (about 300 frames).
- Both are needed in the end. A can start immediately; B blocks on your labeling time.
