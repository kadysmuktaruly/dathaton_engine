"""Pre-label the extracted frames for Label Studio.

- New classes: OWLv2 (google/owlv2-base-patch16-ensemble). Furniture-like classes (safe, safe_open,
  counting_machine, desk, collection_bag) use one-shot visual exemplars (hand-drawn boxes on a few frames,
  EXEMPLARS) scored by cosine similarity of the OWLv2 box embeddings; cash and collection_bag also use
  text prompts (sigmoid score). Then a per-class threshold, per-class NMS and a per-class cap on boxes.
  safe vs safe_open is decided by which exemplar set is closer. Thresholds were picked by looking at
  overlays of every 6th/28th frame; they lean towards recall: the human deletes more than they draw.
- COCO classes (person, laptop, handbag, backpack, id_card, cup): the stock RT-DETRv2 detector,
  which is already good on these, so the human only checks them.

Outputs (in data/labeling/):
  prelabels.json     {image path: [{label, score, box [x0,y0,x1,y1] px, source}]}
  ls_tasks.json      Label Studio import file (tasks with `predictions`)
  label_config.xml   Label Studio labeling config
and, with --preview N, N overlay images in runs/prelabel_preview/.

Usage:
  HF_HUB_OFFLINE=1 uv run python -m tools.prelabel_owlv2 [--preview 12] [--limit 20]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import uuid
from pathlib import Path
from urllib.parse import quote

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.ops import box_iou, nms  # noqa: E402

from engine.config import DETECTOR_REPO, ROOT  # noqa: E402
from tools.det_classes import CLASSES, COCO_CLASSES, COLORS, STOCK_NAME  # noqa: E402

LAB = ROOT / "data" / "labeling"
OWL_REPO = "google/owlv2-base-patch16-ensemble"

# class -> (text prompts, min text sigmoid score, min exemplar cosine, max boxes per image).
# A threshold > 1 disables that source. "safe" and "safe_open" are one object in two states: the best
# safe-like box is labeled safe_open when its safe_open exemplar cosine beats the safe one by a margin.
OWL_SPEC: dict[str, tuple[list[str], float, float, int]] = {
    "cash": (["banknotes", "paper money", "money bills in hand", "a wad of money"], 0.15, 9.0, 6),
    "safe": ([], 9.0, 0.85, 1),
    "safe_open": ([], 9.0, 0.85, 1),
    "counting_machine": ([], 9.0, 0.85, 1),
    "collection_bag": (["a green bag"], 0.25, 0.80, 2),
    "desk": ([], 9.0, 0.90, 1),
}
SAFE_OPEN_MARGIN = 0.015  # tuned on X-20/X-13 (open) vs X-12/X-35/X-4/X-30 (closed)
# One-shot visual exemplars (OWLv2 image-conditioned queries): (clip, time s, class, box x0,y0,x1,y1 px).
# Text prompts alone put "safe" on the white stand / white desk corner, so the furniture classes use
# exemplars drawn by hand on one frame of each layout. A box's exemplar score is the max over that
# class's exemplars.
EXEMPLARS: list[tuple[str, float, str, tuple[float, float, float, float]]] = [
    ("X-4", 55.5, "safe", (307, 358, 412, 498)),
    ("X-4", 55.5, "counting_machine", (37, 403, 113, 487)),
    ("X-4", 55.5, "desk", (20, 393, 297, 497)),
    ("X-4", 55.5, "cash", (148, 440, 175, 450)),
    ("X-17", 10.0, "safe", (322, 323, 428, 462)),
    ("X-17", 10.0, "collection_bag", (130, 363, 198, 422)),
    ("X-14", 21.33, "collection_bag", (170, 343, 227, 410)),
    ("X-14", 21.33, "safe", (230, 303, 300, 403)),
    ("X-14", 21.33, "counting_machine", (50, 373, 113, 453)),
    ("X-20", 3.0, "safe_open", (190, 251, 261, 353)),
    ("X-13", 49.0, "safe_open", (193, 265, 291, 357)),
    ("X-20", 5.5, "counting_machine", (52, 330, 120, 407)),
    ("X-20", 5.5, "desk", (70, 350, 327, 497)),
    ("X-30", 1.53, "counting_machine", (209, 320, 268, 377)),
    ("X-30", 1.53, "safe", (212, 273, 279, 367)),
    ("X-12", 20.0, "safe", (225, 304, 298, 397)),
    ("X-30", 1.53, "desk", (37, 363, 293, 453)),
]
COCO_THRESH = {"person": 0.5, "laptop": 0.4, "handbag": 0.4, "backpack": 0.4, "id_card": 0.4, "cup": 0.4}
NMS_IOU = 0.5


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class Owl:
    def __init__(self, device: str):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.device = device
        self.proc = Owlv2Processor.from_pretrained(OWL_REPO)
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL_REPO).to(device).eval()
        self.queries, self.qidx = [], {}
        for c, (qs, _, _, _) in OWL_SPEC.items():
            self.qidx[c] = list(range(len(self.queries), len(self.queries) + len(qs)))
            self.queries += qs
        self.ex_cls, self.ex_emb = self._exemplar_embeds()

    def _forward(self, im: Image.Image):
        inp = self.proc(text=[self.queries], images=im, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        fmap = out.image_embeds  # (1, h, w, D)
        feats = fmap.reshape(1, -1, fmap.shape[-1])
        return out, feats

    @torch.no_grad()
    def _exemplar_embeds(self):
        from engine.video import iter_frames_at
        from tools.labels import video_index
        vids = video_index()
        cls, embs = [], []
        for cid, t, c, box in EXEMPLARS:
            frame = next(iter_frames_at(vids[cid], [t]))[2]
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            S = max(im.size)
            out, feats = self._forward(im)
            _, class_embeds = self.model.class_predictor(feats)
            pb = out.pred_boxes[0].float() * S
            pb = torch.cat([pb[:, :2] - pb[:, 2:] / 2, pb[:, :2] + pb[:, 2:] / 2], 1)
            iou = box_iou(torch.tensor([box], dtype=torch.float32, device=pb.device), pb)[0]
            # among boxes overlapping the exemplar well, take the most "object-like" one
            obj = self.model.objectness_predictor(feats)[0].float()
            cand = iou >= iou.max() * 0.8
            j = int(torch.where(cand, obj, torch.full_like(obj, -1e9)).argmax())
            cls.append(c)
            embs.append(class_embeds[0, j])
        return cls, torch.stack(embs)[None]  # (1, E, D)

    @torch.no_grad()
    def scores(self, im: Image.Image):
        """Per OWLv2 box: xyxy px, text sigmoid scores (boxes, text queries), exemplar cosine (boxes, exemplars)."""
        W, H = im.size
        S = max(W, H)  # OWLv2 pads to a square at the bottom/right, so boxes are relative to S
        out, feats = self._forward(im)
        prob_t = torch.sigmoid(out.logits[0].float().cpu())
        # Image-conditioned logits saturate (sigmoid ~1 everywhere), so exemplar matching uses the
        # cosine similarity between the box class embeddings and the exemplar embeddings instead.
        _, ce = self.model.class_predictor(feats)
        ce = torch.nn.functional.normalize(ce[0].float(), dim=-1)
        cos_e = (ce @ torch.nn.functional.normalize(self.ex_emb[0].float(), dim=-1).T).cpu()
        cx, cy, w, h = (out.pred_boxes[0].float().cpu() * S).unbind(-1)
        xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
        xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, W)
        xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, H)
        return xyxy, prob_t, cos_e

    def _class_score(self, c, prob_t, cos_e):
        st = prob_t[:, self.qidx[c]].max(dim=1).values if self.qidx[c] else None
        ei = [k for k, e in enumerate(self.ex_cls) if e == c]
        se = cos_e[:, ei].max(dim=1).values if ei else None
        return st, se

    @torch.no_grad()
    def __call__(self, im: Image.Image) -> list[dict]:
        xyxy, prob_t, cos_e = self.scores(im)
        valid = ((xyxy[:, 2] - xyxy[:, 0]) > 2) & ((xyxy[:, 3] - xyxy[:, 1]) > 2)
        sc_safe = self._class_score("safe", prob_t, cos_e)[1]
        sc_open = self._class_score("safe_open", prob_t, cos_e)[1]
        dets = []
        for c, (_, thr_t, thr_e, cap) in OWL_SPEC.items():
            if c == "safe_open":
                continue  # handled together with "safe" (one object, two states)
            st, se = self._class_score(c, prob_t, cos_e)
            if c == "safe":
                se = torch.maximum(sc_safe, sc_open)
            parts = []
            if st is not None:
                parts.append((st, st >= thr_t, "owlv2-text", torch.arange(len(st))))
            if se is not None:
                parts.append((se, se >= thr_e, "owlv2-exemplar", torch.arange(len(se))))
            bs, ss, srcs, idx = [], [], [], []
            for sc, keep, src, ar in parts:
                keep = keep & valid
                bs.append(xyxy[keep]); ss.append(sc[keep]); idx.append(ar[keep]); srcs += [src] * int(keep.sum())
            if not srcs:
                continue
            b, sc, idx = torch.cat(bs), torch.cat(ss), torch.cat(idx)
            for j in nms(b, sc, NMS_IOU)[:cap].tolist():
                lab = c
                if c == "safe" and srcs[j] == "owlv2-exemplar" and \
                        sc_open[idx[j]] > sc_safe[idx[j]] + SAFE_OPEN_MARGIN:
                    lab = "safe_open"
                dets.append(dict(label=lab, score=round(float(sc[j]), 4),
                                 box=[round(float(v), 1) for v in b[j]], source=srcs[j]))
        return dets


class Stock:
    def __init__(self, device: str):
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
        self.device = device
        self.proc = RTDetrImageProcessor.from_pretrained(DETECTOR_REPO)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(DETECTOR_REPO).to(device).eval()
        id2label = self.model.config.id2label
        self.want = {i: c for c in COCO_CLASSES for i, n in id2label.items() if n == STOCK_NAME[c]}

    @torch.no_grad()
    def __call__(self, im: Image.Image) -> list[dict]:
        W, H = im.size
        inp = self.proc(images=im, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        res = self.proc.post_process_object_detection(out, threshold=min(COCO_THRESH.values()),
                                                      target_sizes=torch.tensor([[H, W]]))[0]
        dets = []
        for s, l, b in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
            c = self.want.get(int(l))
            if c is None or s < COCO_THRESH[c]:
                continue
            b = [min(max(b[0], 0), W), min(max(b[1], 0), H), min(max(b[2], 0), W), min(max(b[3], 0), H)]
            dets.append(dict(label=c, score=round(s, 4), box=[round(v, 1) for v in b], source="rtdetrv2"))
        return dets


def drop_confused(coco: list[dict], owl: list[dict], iou_thr: float = 0.4) -> list[dict]:
    """The stock model calls the counting machine / safe / collection bag 'handbag' or 'backpack';
    drop those stock boxes when they overlap an OWLv2 box of one of those classes."""
    anchors = [d["box"] for d in owl if d["label"] in ("counting_machine", "safe", "safe_open", "collection_bag")]
    if not anchors:
        return coco
    A = torch.tensor(anchors)
    keep = []
    for d in coco:
        if d["label"] in ("handbag", "backpack") and float(box_iou(torch.tensor([d["box"]]), A).max()) > iou_thr:
            continue
        keep.append(d)
    return keep


def label_config() -> str:
    labels = "\n".join(f'    <Label value="{c}" background="{COLORS[c]}"/>' for c in CLASSES)
    return f"""<View>
  <Header value="Box every visible instance. Guidelines: docs/labeling.md"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="label" toName="image" strokeWidth="2" opacity="0.25">
{labels}
  </RectangleLabels>
  <Choices name="frame_flags" toName="image" choice="multiple" showInline="true">
    <Choice value="skip_frame (unusable, exclude from training)"/>
  </Choices>
</View>
"""


def ls_task(row: dict, dets: list[dict], doc_root: Path, W: int, H: int) -> dict:
    rel = (ROOT / row["path"]).resolve().relative_to(doc_root.resolve()).as_posix()
    result = []
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        result.append({
            "id": uuid.uuid4().hex[:10], "from_name": "label", "to_name": "image",
            "type": "rectanglelabels", "original_width": W, "original_height": H, "image_rotation": 0,
            "score": d["score"],
            "value": {"x": 100 * x0 / W, "y": 100 * y0 / H, "width": 100 * (x1 - x0) / W,
                      "height": 100 * (y1 - y0) / H, "rotation": 0, "rectanglelabels": [d["label"]]},
        })
    return {
        "data": {"image": f"/data/local-files/?d={quote(rel)}", "clip_id": row["clip_id"],
                 "t": float(row["t"]), "split": row["split"], "path": row["path"]},
        "predictions": [{"model_version": "owlv2+rtdetrv2-stock",
                         "score": float(np.mean([d["score"] for d in dets])) if dets else 0.0,
                         "result": result}],
    }


def draw(path: Path, dets: list[dict]) -> np.ndarray:
    img = cv2.imread(str(path))
    for d in dets:
        hx = COLORS[d["label"]].lstrip("#")
        col = (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))
        x0, y0, x1, y1 = map(int, d["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
        cv2.putText(img, f"{d['label']} {d['score']:.2f}", (x0 + 2, max(12, y0 - 3)), 0, 0.4, col, 1,
                    cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, default=LAB / "frames.csv")
    ap.add_argument("--out", type=Path, default=LAB)
    ap.add_argument("--doc-root", type=Path, default=ROOT,
                    help="must equal LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT (image URLs are relative to it)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N frames (debug)")
    ap.add_argument("--preview", type=int, default=12, help="write N overlays to runs/prelabel_preview/")
    ap.add_argument("--no-coco", action="store_true", help="skip the stock RT-DETRv2 COCO pre-labels")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.frames)))
    if args.limit:
        rows = rows[:args.limit]
    dev = pick_device()
    print(f"device={dev}, {len(rows)} frames")
    owl = Owl(dev)
    stock = None if args.no_coco else Stock(dev)

    pre, tasks = {}, []
    for i, r in enumerate(rows):
        im = Image.open(ROOT / r["path"]).convert("RGB")
        dets = owl(im)
        if stock:
            dets += drop_confused(stock(im), dets)
        pre[r["path"]] = dets
        tasks.append(ls_task(r, dets, args.doc_root, *im.size))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}")

    args.out.mkdir(parents=True, exist_ok=True)
    json.dump(pre, open(args.out / "prelabels.json", "w"), indent=1)
    json.dump(tasks, open(args.out / "ls_tasks.json", "w"))
    (args.out / "label_config.xml").write_text(label_config())

    counts = {c: 0 for c in CLASSES}
    imgs = {c: 0 for c in CLASSES}
    for dets in pre.values():
        for c in {d["label"] for d in dets}:
            imgs[c] += 1
        for d in dets:
            counts[d["label"]] += 1
    print(f"{'class':18s} boxes images")
    for c in CLASSES:
        print(f"{c:18s} {counts[c]:5d} {imgs[c]:6d}")
    print(f"wrote {args.out / 'ls_tasks.json'} ({len(tasks)} tasks), prelabels.json, label_config.xml")

    if args.preview:
        pdir = ROOT / "runs" / "prelabel_preview"
        pdir.mkdir(parents=True, exist_ok=True)
        idx = np.linspace(0, len(rows) - 1, min(args.preview, len(rows))).astype(int)
        for j in idx:
            p = rows[j]["path"]
            cv2.imwrite(str(pdir / Path(p).name), draw(ROOT / p, pre[p]))
        print(f"previews -> {pdir}")


if __name__ == "__main__":
    main()
