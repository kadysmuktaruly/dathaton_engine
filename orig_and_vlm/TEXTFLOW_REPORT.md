# Textflow (VLM → text → LLM judge): what we tried and why it was dropped

**Idea (2026-10-17).** Turn the video into text with a small VLM, then let a local LLM read the
text next to the rules catalog and decide on violations. The hope was that it would generalize
to unseen violation types and to the new camera angle better than hand-written rules.

**What we built** (branch `textflow`, local only, commits `53e3da1`, `d67b661`, `432233e`;
removed from this branch):
- **Eyes:** Qwen3.5-4B (MLX 4-bit) answers targeted yes/no questions on crops: cash / bag / laptop on the desk crop, safe open, a person beside the cashier, the door open, counting by hand. Plus a few free-text clip notes.
- **Diary:** perception's people events (entries, exits, the cashier standing up) plus the smoothed answers, as a timestamped text log.
- **Judge:** Qwen3.5-4B (later 9B) reads the diary plus the catalog and returns `{code, t, seen}`. Guardrails keep only answers the diary supports; legal references always come from the catalog.

**Results on the 40 training clips** (same metrics as `tools/eval.py`):

| | rules engine (main) | textflow v2 (4B judge) | v4 (stricter guards) | v5 (9B judge) |
|---|---|---|---|---|
| violations found / 21 | 13 | 14 | 9 | 3 |
| false alarms / 19 clean | **6** | 11 | 7 | 2 |
| F1 | **0.65** | 0.61 | 0.49 | 0.23 |
| type correct | **92%** | 20% | 40% | 67% (of 3) |
| s / clip | 24 | 65 | — | — |

**Why it did not work**
1. **Free captions lose the deciding details.** Generic "describe the frame" text missed cash on the desk, an open safe and a client leaving. Targeted questions on crops fixed the eyes: 15/16 on zoomed crops in a 44-question verified benchmark.
2. **The LLM judge was the weak link, not the eyes.** Even with correct facts in the diary, the 4B judge picked wrong types and times. On X-30 the diary showed a person entering and standing beside the cashier, and the judge said B3. Small prompt edits swung its answers a lot.
3. **Guardrails could not fix that.** Requiring each type's fact (cash for B1, a person beside the cashier for C1, ...) cut false alarms but dropped real hits about equally.
4. **A bigger judge became too cautious.** Qwen3.5-9B, the largest that fits in 18 GB next to the eyes, flagged 10 possible violations in 40 clips and found 3. It also pushed swap to 11/12 GB. The 27B models need ~16 GB even at 4-bit.
5. **It was slow.** It took 45–115 s per clip, against 24 s for the engine.

**What we kept.** Targeted questions on crops are reliable. `orig_and_vlm/` uses them where the
rules engine is weak (its desk scan and its "cashier left the room" false alarms), and the rules
keep making the decisions.
