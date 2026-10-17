# Labeling and detector fine-tuning

Goal: one fine-tuned RT-DETRv2-R18 that detects `cash`, `safe`, `safe_open`, `counting_machine`,
`collection_bag`, `desk` and keeps the COCO classes we use (`person`, `laptop`, `handbag`, `backpack`,
`id_card`, `cup`). Everything runs locally. Expected human time: **about 3–4 hours** of box correction
(335 frames at 30–45 s each), plus about 15 minutes of setup.

**Why all 12 classes are labeled.** The fine-tuned model replaces the stock one, so it is trained on all
12 classes. The COCO classes are pre-filled by the stock RT-DETRv2 and mostly need a quick check.
`train_detector.py` copies their classifier weights from the stock COCO head, so training starts from the
stock model's skill on these classes. Merging pseudo-labels from two models at inference would mean
running two detectors, and boxes nobody checked would end up in training.

## 1. Extract frames (≈30 s)
```bash
uv run python -m tools.extract_frames
```
→ `data/labeling/images/<clip>_<t>.jpg` and `data/labeling/frames.csv` (`clip_id, t, path, split, layout, source`).

- Per clip: 6–10 frames spread evenly over the clip, plus frames at t−3, −1.5, −0.5, 0, +0.5, +1.5, +3 s around
  each labeled violation timecode.
- Near-duplicates are dropped: a frame is skipped when the mean absolute grey difference to a frame already kept
  from the same clip is below 2.0 (on a 116×212 thumbnail).
- **Train/val split by clip** (seed 0): 32 train clips / 253 frames, 8 val clips / 82 frames. Val clips are drawn
  from each of the three furniture layouts (hand-made table `LAYOUTS` in the script, built from the first frame
  of each clip; clustering by median frame grouped clips by recording session instead). The draw alternates
  violation and clean clips. X-17 (the only clip where a collection bag is handed over) and X-20 (safe left open)
  always stay in train.
  Val: layout a X-18 X-19 X-38, layout b X-13 X-24 X-29, layout c X-33 X-45.

## 2. Pre-label (≈3 min on the M5)
```bash
HF_HUB_OFFLINE=1 uv run python -m tools.prelabel_owlv2 --preview 12
```
→ `data/labeling/ls_tasks.json` (Label Studio tasks with `predictions`), `label_config.xml`, `prelabels.json`;
overlays in `runs/prelabel_preview/`.

- `safe`, `safe_open`, `counting_machine`, `desk`, `collection_bag`: OWLv2 **one-shot visual exemplars**. These are
  boxes drawn by hand on 7 frames covering the 3 layouts (`EXEMPLARS` in the script); each box is scored by the
  cosine similarity of its OWLv2 embedding to the exemplars. Text prompts alone put "safe" on the white stand or
  the white desk corner. `safe` vs `safe_open` is chosen by which exemplar set is closer, so treat `safe_open`
  as a hint only.
- `cash`, `collection_bag`: also text prompts ("banknotes", "paper money", "money bills in hand", "a wad of money";
  "a green bag"). Cash pre-labels are weak: expect misses and false boxes.
- COCO classes: the stock RT-DETRv2 (person ≥ 0.5, others ≥ 0.4). Stock `handbag`/`backpack` boxes that overlap
  the counting machine, safe or bag are dropped, because the stock model confuses these.

## 3. Start Label Studio locally
Label Studio (Apache-2.0) runs in its own env through `uvx`. It is not installed into the project env, so it
can't break torch/transformers. Run from the repo root:
```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$PWD"       # the repo root; image URLs are relative to it
export LABEL_STUDIO_BASE_DATA_DIR="$PWD/label_studio_data" # its sqlite DB (git-ignored)
export LATEST_VERSION_CHECK=false COLLECT_ANALYTICS=false  # no calls home
uvx --python 3.11 label-studio start --port 8080
```
Open http://localhost:8080 and sign up (a local account only).

1. **Create Project** → *Labeling Setup* → *Custom template* → *Code*: paste `data/labeling/label_config.xml` → Save.
2. *Settings → Cloud Storage → Add Source Storage → Local files*. Absolute local path:
   `<repo>/data/labeling/images`. Leave "Treat every bucket object as a source file" **off**. Click **Add**, but
   **do not Sync**: syncing would create a second, prediction-less task for every image. This storage only gives
   Label Studio permission to serve the files.
3. *Import* → upload `data/labeling/ls_tasks.json`. You should see 335 tasks with predictions. Images load from
   `/data/local-files/?d=data/labeling/images/<file>.jpg`.
4. *Settings → Annotation*: turn on "Use predictions to prelabel tasks" so each task opens with the
   pre-labels as an editable draft.

## 4. Correct the boxes: guidelines
Box **every visible instance** of every class, including partly hidden ones, with tight boxes on the visible
part. When unsure about a tiny object, box it when a person could tell what it is from this frame alone.

| Class | What counts | What doesn't |
|---|---|---|
| `cash` | Any visible banknotes: a single note, a fan, a stack or bundle, notes in a hand, in the counting machine's tray, on the desk. One box per separate pile or hand. | Cards, receipts, white paper or documents, a phone. If you can't tell notes from paper, leave it unboxed. |
| `safe` | The grey metal safe with its door **closed** (or only slightly ajar): the safe body only. | The white stand under it; the white drawer unit. |
| `safe_open` | The safe with its door **visibly swung open** (you can see the door panel standing out or the inside). Box the body plus the open door. Use **either** `safe` or `safe_open` for the safe, never both. | |
| `counting_machine` | The black banknote counting machine, including its trays. | The desk phone, the laptop, the black card/receipt device next to the machine (leave it unlabeled). |
| `collection_bag` | The green cash-in-transit bag or pouch (with a seal/tag), whether on the desk, in a hand or at the safe. | Personal handbags (`handbag`). |
| `desk` | The **top surface of the cashier's wooden working desk** (box around the visible desktop). | The front panel and legs down to the floor; the large white desk in the foreground; tables outside the glass. |
| `person` | Every person in the room, including people only partly visible. Also box people clearly visible through the glass wall (the engine filters them by position). | Reflections. |
| `laptop` | Laptop or tablet. | Monitor-like reflections in the glass. |
| `handbag` / `backpack` | Personal bags and coats on the desk or chair (B3). | The counting machine, the collection bag. |
| `id_card` | A client's ID document (government ID card or passport) on the desk or in a hand. A mobile phone, which is rare here, also counts as `id_card`. | The desk phone (delete those boxes), bank cards, receipts. |
| `cup` | Cups, mugs, bottles at the desk. | |

Workflow per task: delete wrong boxes (select, Backspace), fix boxes by dragging their corners, pick a label key
and drag to add a box, then **Submit**. If a frame is useless (motion blur, everything hidden), tick
`skip_frame` and submit: it will be excluded. Hotkeys: the number keys select labels; Ctrl+Enter submits.

Cost: the furniture boxes are usually right and only need a glance. The time goes on `cash` (check every
hand and the desk at full zoom) and on deleting false `id_card` boxes on the desk phone.

## 5. Export → COCO
*Export* → **JSON** (not "JSON-MIN"). Save it, e.g. as `data/labeling/export.json`, then:
```bash
uv run python -m tools.ls_export_to_coco data/labeling/export.json
```
→ `data/labeling/coco_train.json`, `coco_val.json`, with the split taken from `frames.csv`. Tasks that weren't
submitted are left out; add `--include-unlabeled` to treat them as images with no objects instead.

## 6. Train
Mac (MPS, ~1 min per epoch at batch 4 → ~40 min for 40 epochs):
```bash
HF_HUB_OFFLINE=1 uv run python -m tools.train_detector --epochs 40 --batch 4
uv run python -m tools.train_detector --smoke \
    --train data/labeling/coco_prelabels_train.json --val data/labeling/coco_prelabels_val.json  # 10-second self-test
```
(`coco_prelabels_*.json` come from `uv run python -m tools.ls_export_to_coco --from-prelabels`. They are unchecked
pre-labels, so use them only to test the pipeline.)

Colab (CUDA): upload `tools/train_detector.py`, `tools/det_classes.py`, `data/labeling/` (images + coco json):
```bash
pip install -q "transformers>=4.47" torchvision
python train_detector.py --image-root . --train data/labeling/coco_train.json \
    --val data/labeling/coco_val.json --out weights/detector_ft --epochs 60 --batch 8 --workers 2
```
Augmentation: colour jitter, random perspective, random affine (±10°, ±5 % shift, 0.7–1.3 scale), horizontal flip.
Then resize to 640×640, as the stock model does at inference. Learning rates: AdamW 1e-4 for the head/decoder,
1e-5 for the backbone, linear warm-up then cosine decay. Gradient-norm clip 0.1.

## 7. Output
The best epoch by mean AP@0.5 over the 6 new classes on val goes to **`weights/detector_ft/`**
(`model.safetensors`, `config.json` with the new `id2label`, `preprocessor_config.json`, `metrics.json` with the
per-class AP, and `history.json`). Load it with
`RTDetrV2ForObjectDetection.from_pretrained("weights/detector_ft")` and `RTDetrImageProcessor.from_pretrained(...)`.
`weights/` and `data/` are git-ignored.
