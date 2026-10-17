# Perception baseline (Step 2)

Stock RT-DETRv2-R18 (COCO, threshold 0.25) + ByteTrack (supervision 0.26.1) + RTMPose-m (wrists, ankles), at 3 fps.
Clips: X-1, X-17 (layout a), X-20, X-40, X-44 (b/c), X-30 (c). Overlays: `runs/overlays/<clip>.mp4` and `<clip>_sheet.jpg` (`uv run python -m tools.overlay <clips>`).

## Speed (M5, MPS)
| Clip | Length | Frames | Decode | Detect | Pose | Total |
|---|---|---|---|---|---|---|
| X-1 | 26 s | 78 | 0.2 s | 4.1 s | 2.5 s | ~7 s |
| X-17 | 15 s | 45 | 0.1 s | 1.9 s | 1.0 s | ~3 s |
| X-40 | 25 s | 76 | 0.2 s | 3.0 s | 1.3 s | ~5 s |
The longest clip (143 s) is about 35 s of perception, well inside the 60 s target.

## What is and isn't detected
| Object | Stock result | Needed for |
|---|---|---|
| Person | Good (0.85–0.9). Pose wrists and ankles look plausible. | everything |
| People behind the glass wall | **Detected as people** (e.g. X-1, a small box in the upper-left glass). Filter needed. | C1, A4 |
| Desk | `diningtable` 0.3–0.5. The large white desk in the foreground is also a `diningtable`, so pick the table that people and devices sit around. | anchors |
| Laptop | `laptop` 0.35–0.4, stable | anchor (staff side) |
| Counting machine | Box is stable, **class flips** between suitcase/handbag/backpack/tvmonitor (0.3) | anchor, A1 |
| Safe | Found only in layout (a), as `tvmonitor`. **Missing** in layouts (b) and (c). | B5, anchors |
| Open safe door | not available | B5 |
| Cash | **never** (no COCO class) | B1, B7, A1 |
| Collection bag | as `handbag`/`suitcase` sometimes, so it gets confused with B3 | D1 |
| Handbag / coat (X-40) | `handbag`/`backpack` near the counting machine; the coat is not detected | B3 |
| Phone | Desk phone detected as `cell phone`. Not needed (B4 wasn't confirmed). | — |

OWLv2 zero-shot check (6 key frames): finds loose cash weakly (X-4 0.36, X-26 0.19), the safe at about 0.17, the bag as "handbag". Many false "desk telephone" boxes. It's useful as a pre-labeler, but every frame needs human correction.

## Conclusions
1. Rules based only on people (C1, the B5 leave-the-room variant, A1 timing) can start now on the stock detector.
2. **Fine-tuning is needed** for `cash`, `safe`, `safe_open`, `counting_machine`, `collection_bag` (Step 5). The plan: OWLv2 pre-labels → human correction in local Label Studio → train.
3. Until then, scene anchors fall back to a stable-box clustering of the stock classes: the counting machine is the stable "suitcase/handbag/tvmonitor" box on the desk; the safe is unknown.
