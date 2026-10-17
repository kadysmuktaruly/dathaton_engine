"""Fine-tune RT-DETRv2-R18 (PekingU/rtdetr_v2_r18vd, Apache-2.0) on the labeled cash-room frames.

Classes: tools/det_classes.CLASSES = 6 new classes (cash, safe, safe_open, counting_machine,
collection_bag, desk) + 6 COCO classes we still need (person, laptop, handbag, backpack, id_card, cup).
All 12 are annotated in Label Studio (the COCO ones are pre-filled by the stock model and only checked),
so a single fine-tuned model replaces the stock one. The classifier rows of the COCO classes are
copied from the stock COCO head, so they start from the stock detector's ability; the new rows start
from the stock prior bias.

Augmentation (torchvision.transforms.v2, BSD-3): colour jitter, random perspective, random affine
(rotation +-10 deg, translate 5%, scale 0.7-1.3 = zoom in/out/crop), horizontal flip. Images are then
resized to 640x640 by RTDetrImageProcessor (same as at inference).

Eval: COCO-style AP per class (101-point, IoU 0.50:0.95 and 0.50), implemented here (no pycocotools /
torchmetrics in the env). The best epoch by mean AP@0.5 over the 6 NEW classes is saved (the COCO
classes have only a handful of val instances, e.g. 1 handbag, so they would make the choice noisy) with save_pretrained to --out
(model + image processor + metrics.json).

Device: MPS -> CUDA -> CPU (PYTORCH_ENABLE_MPS_FALLBACK=1 is set). Works on Colab CUDA.

Usage:
  HF_HUB_OFFLINE=1 uv run python -m tools.train_detector --train data/labeling/coco_train.json \
      --val data/labeling/coco_val.json --epochs 40 --batch 4
  uv run python -m tools.train_detector --smoke --train data/labeling/coco_prelabels_train.json \
      --val data/labeling/coco_prelabels_val.json      # 2 iterations on 4 images
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import tv_tensors  # noqa: E402
from torchvision.ops import box_iou  # noqa: E402
from torchvision.transforms import v2  # noqa: E402
from torchvision.transforms.v2 import functional as TF  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = "PekingU/rtdetr_v2_r18vd"

try:
    from tools.det_classes import CLASSES, COCO_CLASSES, NEW_CLASSES, STOCK_NAME
except ImportError:  # running the file directly (e.g. copied to Colab next to det_classes.py)
    from det_classes import CLASSES, COCO_CLASSES, NEW_CLASSES, STOCK_NAME  # type: ignore


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_transforms() -> v2.Compose:
    return v2.Compose([
        v2.RandomApply([v2.ColorJitter(0.3, 0.3, 0.3, 0.03)], p=0.8),
        v2.RandomPerspective(distortion_scale=0.15, p=0.3),
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.7, 1.3)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ClampBoundingBoxes(),
        v2.SanitizeBoundingBoxes(min_size=2),
    ])


class CocoDet(Dataset):
    def __init__(self, ann_file: Path, image_root: Path, processor, train: bool, limit: int = 0):
        coco = json.load(open(ann_file))
        names = [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]
        if names != CLASSES:
            raise ValueError(f"{ann_file}: categories {names} != {CLASSES}")
        self.images = coco["images"][:limit] if limit else coco["images"]
        self.anns: dict[int, list] = {im["id"]: [] for im in self.images}
        for a in coco["annotations"]:
            if a["image_id"] in self.anns:
                self.anns[a["image_id"]].append(a)
        self.root, self.proc, self.train = image_root, processor, train
        self.tf = train_transforms() if train else None

    def __len__(self) -> int:
        return len(self.images)

    def gt(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Ground truth in original pixels, xyxy, 0-based labels (for eval)."""
        anns = self.anns[self.images[i]["id"]]
        b = torch.tensor([a["bbox"] for a in anns], dtype=torch.float32).reshape(-1, 4)
        b[:, 2:] += b[:, :2]
        return b, torch.tensor([a["category_id"] - 1 for a in anns], dtype=torch.int64)

    def __getitem__(self, i: int):
        info = self.images[i]
        img = Image.open(self.root / info["file_name"]).convert("RGB")
        W, H = img.size
        boxes, labels = self.gt(i)
        if self.train:
            out = self.tf({"image": tv_tensors.Image(TF.pil_to_tensor(img)),
                           "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(H, W)),
                           "labels": labels})
            arr = out["image"].permute(1, 2, 0).numpy()
            boxes, labels = out["boxes"].as_subclass(torch.Tensor), out["labels"]
        else:
            arr = np.asarray(img)
        xywh = boxes.clone()
        xywh[:, 2:] -= xywh[:, :2]
        target = {"image_id": info["id"], "annotations": [
            {"bbox": b.tolist(), "category_id": int(l), "area": float(b[2] * b[3]), "iscrowd": 0}
            for b, l in zip(xywh, labels)]}
        enc = self.proc(images=arr, annotations=target, return_tensors="pt")
        return enc["pixel_values"][0], enc["labels"][0], i


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch], [b[2] for b in batch]


def build_model(device: torch.device):
    from transformers import RTDetrV2ForObjectDetection
    id2label = dict(enumerate(CLASSES))
    stock = RTDetrV2ForObjectDetection.from_pretrained(BASE)
    stock_sd, stock_l2i = stock.state_dict(), {v: k for k, v in stock.config.id2label.items()}
    model = RTDetrV2ForObjectDetection.from_pretrained(
        BASE, id2label=id2label, label2id={v: k for k, v in id2label.items()}, num_labels=len(CLASSES),
        ignore_mismatched_sizes=True)
    # Copy the classifier rows of the kept COCO classes; new classes get the stock mean prior bias.
    n_old, K = stock.config.num_labels, len(CLASSES)
    src = {CLASSES.index(c): stock_l2i[STOCK_NAME[c]] for c in COCO_CLASSES}
    sd = model.state_dict()
    with torch.no_grad():
        for k, v in sd.items():
            old = stock_sd.get(k)
            if old is None or old.shape == v.shape or old.shape[1:] != v.shape[1:]:
                continue
            if old.shape[0] in (n_old, n_old + 1) and v.shape[0] in (K, K + 1):
                if old.dim() == 1:  # bias
                    v[:K] = old[:n_old].mean()
                for new_i, old_i in src.items():
                    v[new_i] = old[old_i]
                if old.shape[0] == n_old + 1:  # denoising embedding: last row = padding class
                    v[K] = old[n_old]
    model.load_state_dict(sd)
    del stock
    return model.to(device)


def average_precision(preds, gts, n_cls: int, iou_thrs=np.arange(0.5, 0.96, 0.05)) -> dict:
    """COCO-style 101-point AP per class. preds/gts: per image (boxes xyxy, scores?, labels)."""
    ap = np.full((n_cls, len(iou_thrs)), np.nan)
    for c in range(n_cls):
        n_gt = sum(int((g[1] == c).sum()) for g in gts)
        if n_gt == 0:
            continue
        recs = []  # (score, image index, box)
        for i, (pb, ps, pl) in enumerate(preds):
            m = pl == c
            recs += [(float(s), i, b) for s, b in zip(ps[m], pb[m])]
        recs.sort(key=lambda r: -r[0])
        for ti, thr in enumerate(iou_thrs):
            used = [torch.zeros(int((g[1] == c).sum()), dtype=torch.bool) for g in gts]
            tp = np.zeros(len(recs))
            for j, (_, i, b) in enumerate(recs):
                gb = gts[i][0][gts[i][1] == c]
                if len(gb) == 0:
                    continue
                iou = box_iou(b[None], gb)[0]
                iou[used[i]] = -1
                k = int(iou.argmax())
                if iou[k] >= thr:
                    used[i][k] = True
                    tp[j] = 1
            ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
            rec = ctp / n_gt
            prec = ctp / np.maximum(ctp + cfp, 1e-9)
            prec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
            q = [prec[np.searchsorted(rec, r, side="left")] if np.searchsorted(rec, r, side="left") < len(rec)
                 else 0.0 for r in np.linspace(0, 1, 101)]
            ap[c, ti] = float(np.mean(q))
    with np.errstate(all="ignore"):
        per_cls = {CLASSES[c]: {"AP50": float(ap[c, 0]), "AP": float(np.nanmean(ap[c]))
                                if not np.isnan(ap[c]).all() else float("nan")} for c in range(n_cls)}
        new = [CLASSES.index(c) for c in NEW_CLASSES]
        return {"mAP50": float(np.nanmean(ap[:, 0])), "mAP": float(np.nanmean(ap)),
                "mAP50_new": float(np.nanmean(ap[new, 0])), "per_class": per_cls}


@torch.no_grad()
def evaluate(model, loader, ds: CocoDet, proc, device) -> dict:
    model.eval()
    preds, gts = [], []
    for px, _, idx in loader:
        out = model(pixel_values=px.to(device))
        sizes = torch.tensor([[ds.images[i]["height"], ds.images[i]["width"]] for i in idx])
        res = proc.post_process_object_detection(out, threshold=0.01, target_sizes=sizes)
        for r, i in zip(res, idx):
            preds.append((r["boxes"].float().cpu(), r["scores"].float().cpu(), r["labels"].cpu()))
            gts.append(ds.gt(i))
    model.train()
    return average_precision(preds, gts, len(CLASSES))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, default=ROOT / "data/labeling/coco_train.json")
    ap.add_argument("--val", type=Path, default=ROOT / "data/labeling/coco_val.json")
    ap.add_argument("--image-root", type=Path, default=ROOT, help="COCO file_name is relative to this")
    ap.add_argument("--out", type=Path, default=ROOT / "weights/detector_ft")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100, help="warm-up iterations")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="2 iterations on 4 images, eval on 4, save to *_smoke")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    from transformers import RTDetrImageProcessor
    device = pick_device()
    proc = RTDetrImageProcessor.from_pretrained(BASE)  # 640x640, rescale only (matches the stock model)
    limit = 4 if args.smoke else 0
    if args.smoke:
        args.epochs, args.batch, args.workers, args.warmup = 1, 2, 0, 1
        args.out = args.out.with_name(args.out.name + "_smoke")
    ds_tr = CocoDet(args.train, args.image_root, proc, train=True, limit=limit)
    ds_va = CocoDet(args.val, args.image_root, proc, train=False, limit=limit)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, collate_fn=collate,
                       num_workers=args.workers, drop_last=len(ds_tr) > args.batch,
                       persistent_workers=args.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, collate_fn=collate,
                       num_workers=args.workers)
    print(f"device={device} train={len(ds_tr)} val={len(ds_va)} images, classes={CLASSES}")

    model = build_model(device)
    model.train()
    bb = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr_backbone}, {"params": rest, "lr": args.lr}],
                            weight_decay=args.weight_decay)
    total = 2 if args.smoke else args.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (
        1 + math.cos(math.pi * min(s, total) / total)))

    best, step, history = -1.0, 0, []
    args.out.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        t0, losses = time.time(), []
        for px, labels, _ in dl_tr:
            labels = [{k: v.to(device) for k, v in l.items()} for l in labels]
            out = model(pixel_values=px.to(device), labels=labels)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            sched.step()
            losses.append(out.loss.item())
            step += 1
            if args.smoke and step >= 2:
                break
        msg = f"epoch {ep + 1}/{args.epochs} loss={np.mean(losses):.4f} ({time.time() - t0:.0f}s)"
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            m = evaluate(model, dl_va, ds_va, proc, device)
            history.append({"epoch": ep + 1, "loss": float(np.mean(losses)), **m})
            msg += f" val mAP50={m['mAP50']:.3f} mAP={m['mAP']:.3f} mAP50_new={m['mAP50_new']:.3f} " + " ".join(
                f"{c}={v['AP50']:.2f}" for c, v in m["per_class"].items() if not math.isnan(v["AP50"]))
            score = m["mAP50_new"] if not math.isnan(m["mAP50_new"]) else -0.5
            if score > best:
                best = score
                model.save_pretrained(args.out)
                proc.save_pretrained(args.out)
                json.dump({"epoch": ep + 1, **m, "classes": CLASSES, "base": BASE, "args": {
                    k: str(v) for k, v in vars(args).items()}}, open(args.out / "metrics.json", "w"), indent=1)
                msg += "  [saved]"
        print(msg, flush=True)
    json.dump(history, open(args.out / "history.json", "w"), indent=1)
    print(f"best mAP50 (new classes)={best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
