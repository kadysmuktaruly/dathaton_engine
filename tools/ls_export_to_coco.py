"""Convert a Label Studio JSON export (corrected boxes) to COCO train/val files.

The split comes from data/labeling/frames.csv (by clip). Images are matched by the `/data/local-files/?d=...`
URL (or, failing that, by file name). For each task the most recent non-cancelled annotation is used;
tasks without an annotation are skipped (not treated as empty images), unless --include-unlabeled.
A task whose annotation sets the `skip_frame` choice is dropped.

COCO `file_name` is the image path relative to the repo root (train_detector.py --image-root).
Category ids are 1..K in the order of tools/det_classes.CLASSES.

Usage:
  uv run python -m tools.ls_export_to_coco path/to/export.json      # -> data/labeling/coco_{train,val}.json
  uv run python -m tools.ls_export_to_coco --from-prelabels         # fake COCO from the pre-labels (smoke tests only)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from tools.det_classes import CLASSES

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "data" / "labeling"


def image_key(url: str) -> str:
    q = parse_qs(urlparse(url).query)
    return unquote(q["d"][0]) if "d" in q else unquote(url)


def boxes_from_result(result: list[dict]) -> tuple[list[tuple[str, list[float]]], bool]:
    boxes, skip = [], False
    for r in result:
        if r.get("type") == "choices" and any("skip_frame" in c for c in r["value"].get("choices", [])):
            skip = True
        if r.get("type") != "rectanglelabels":
            continue
        v, W, H = r["value"], r["original_width"], r["original_height"]
        x, y = v["x"] * W / 100, v["y"] * H / 100
        w, h = v["width"] * W / 100, v["height"] * H / 100
        if v.get("rotation"):
            print(f"warning: rotated box ignored rotation ({v['rotation']} deg)")
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(W), x + w), min(float(H), y + h)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        for lab in v["rectanglelabels"]:
            boxes.append((lab, [x0, y0, x1 - x0, y1 - y0]))
    return boxes, skip


def load_ls_export(path: Path, include_unlabeled: bool) -> dict[str, list]:
    tasks = json.load(open(path))
    out = {}
    for t in tasks:
        anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            if include_unlabeled:
                out[image_key(t["data"]["image"])] = []
            continue
        ann = max(anns, key=lambda a: a.get("updated_at") or a.get("created_at") or "")
        boxes, skip = boxes_from_result(ann["result"])
        if not skip:
            out[image_key(t["data"]["image"])] = boxes
    return out


def load_prelabels(path: Path, min_score: float) -> dict[str, list]:
    pre = json.load(open(path))
    return {p: [(d["label"], [d["box"][0], d["box"][1], d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]])
                for d in dets if d["score"] >= min_score] for p, dets in pre.items()}


def build(frames: list[dict], boxes_by_key: dict[str, list], split: str) -> dict:
    by_name = {Path(k).name: v for k, v in boxes_by_key.items()}
    cats = [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]
    cid = {c: i + 1 for i, c in enumerate(CLASSES)}
    images, anns, unknown = [], [], set()
    for f in frames:
        if f["split"] != split:
            continue
        boxes = boxes_by_key.get(f["path"], by_name.get(Path(f["path"]).name))
        if boxes is None:
            continue
        img_id = len(images) + 1
        images.append({"id": img_id, "file_name": f["path"], "width": int(f["width"]),
                       "height": int(f["height"]), "clip_id": f["clip_id"], "t": float(f["t"])})
        for lab, (x, y, w, h) in boxes:
            if lab not in cid:
                unknown.add(lab)
                continue
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": cid[lab],
                         "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                         "area": round(w * h, 2), "iscrowd": 0})
    if unknown:
        print(f"warning: labels not in CLASSES were dropped: {sorted(unknown)}")
    return {"images": images, "annotations": anns, "categories": cats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", type=Path, nargs="?", help="Label Studio JSON export")
    ap.add_argument("--frames", type=Path, default=LAB / "frames.csv")
    ap.add_argument("--out", type=Path, default=LAB)
    ap.add_argument("--prefix", default="coco")
    ap.add_argument("--include-unlabeled", action="store_true",
                    help="treat tasks with no annotation as images with no objects")
    ap.add_argument("--from-prelabels", action="store_true",
                    help="use data/labeling/prelabels.json instead of an export (for smoke tests; NOT ground truth)")
    ap.add_argument("--min-score", type=float, default=0.0, help="with --from-prelabels")
    args = ap.parse_args()

    if args.from_prelabels:
        boxes = load_prelabels(LAB / "prelabels.json", args.min_score)
        prefix = args.prefix if args.prefix != "coco" else "coco_prelabels"
    else:
        if not args.export:
            ap.error("give an export file or --from-prelabels")
        boxes = load_ls_export(args.export, args.include_unlabeled)
        prefix = args.prefix
    frames = list(csv.DictReader(open(args.frames)))
    for split in ("train", "val"):
        coco = build(frames, boxes, split)
        p = args.out / f"{prefix}_{split}.json"
        json.dump(coco, open(p, "w"), indent=1)
        per = {c["name"]: 0 for c in coco["categories"]}
        for a in coco["annotations"]:
            per[CLASSES[a["category_id"] - 1]] += 1
        print(f"{p}: {len(coco['images'])} images, {len(coco['annotations'])} boxes  "
              + " ".join(f"{k}={v}" for k, v in per.items()))


if __name__ == "__main__":
    main()
