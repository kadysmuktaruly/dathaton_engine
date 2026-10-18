"""Sample frames for detector labeling.

Per clip: N frames spread evenly over the clip (6-10, depending on its length) plus frames
around every labeled violation timecode (t-3 .. t+3 s). A frame is dropped when it is a
near-duplicate of a frame already kept from the same clip (mean abs grayscale diff on a
downscaled copy below --dup-thresh). Timecode frames are considered first, so they survive.

Split is by clip (never by frame), ~80/20 train/val, seeded. Val clips are drawn separately from
each of the 3 furniture layouts (see LAYOUTS), alternating violation / clean clips, so val covers
every layout.

Usage:
  uv run python -m tools.extract_frames            # -> data/labeling/images/*.jpg, frames.csv
  uv run python -m tools.extract_frames --val-frac 0.2 --seed 0 --dup-thresh 2.0
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

from engine.config import ROOT
from engine.video import iter_frames_at, probe
from tools.labels import load_clips

OUT_DIR = ROOT / "data" / "labeling"
TC_OFFSETS = (-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0)


def thumb(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (116, 212), interpolation=cv2.INTER_AREA).astype(np.float32)


def plan_times(duration: float, timecodes: list[float]) -> list[tuple[float, str]]:
    """Return (t, source) candidates: timecode neighbourhoods first, then uniform samples."""
    n = int(min(10, max(6, round(duration / 4))))
    end = max(duration - 0.3, 0.1)
    out: list[tuple[float, str]] = []
    for tc in timecodes:
        for off in TC_OFFSETS:
            t = tc + off
            if 0 <= t < end:
                out.append((round(t, 2), "tc"))
    # uniform: centres of n equal bins, so we never take the very first/last frame
    for k in range(n):
        out.append((round((k + 0.5) * end / n, 2), "uniform"))
    return out


# Furniture layout per clip, from a visual review of the first frame of every clip
# (unsupervised clustering of median frames grouped clips by recording session/lighting instead
# of furniture, so the table is hand-made). a = desk across the room, cashier faces away, safe on a
# white stand on the right; b = cashier faces the camera, counting machine on the left, safe behind
# the desk; c = desk rotated, counting machine on the right (white/blue chair).
LAYOUTS = {
    "a": [1, 2, 3, 4, 5, 16, 17, 18, 19, 34, 35, 37, 38, 39],
    "b": [8, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    "c": [30, 31, 32, 33, 40, 43, 44, 45, 46],
}
LAYOUT_OF = {f"X-{n}": k for k, ns in LAYOUTS.items() for n in ns}


def layout_of(clip_id: str) -> str:
    return LAYOUT_OF.get(clip_id, "?")  # unknown clips (e.g. a new test set) form their own stratum


# Clips that are the only/main example of a rare object state stay in train:
# X-17 = collection bag taken without an ID check (D7), X-20 = safe door left open (B5).
TRAIN_ONLY = {"X-17", "X-20"}


def split_clips(clips, layouts: dict[str, str], val_frac: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    split = {c.clip_id: "train" for c in clips}
    for lay in sorted(set(layouts.values())):
        group = [c for c in clips if layouts[c.clip_id] == lay]
        n_val = max(1, round(val_frac * len(group)))
        group = [c for c in group if c.clip_id not in TRAIN_ONLY]
        viol = [c for c in group if c.violation]
        clean = [c for c in group if not c.violation]
        rng.shuffle(viol)
        rng.shuffle(clean)
        picked = []
        while len(picked) < n_val and (viol or clean):  # alternate violation / clean
            src = viol if (len(picked) % 2 == 0 and viol) or not clean else clean
            picked.append(src.pop())
        for c in picked:
            split[c.clip_id] = "val"
    return split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dup-thresh", type=float, default=2.0,
                    help="min mean abs gray diff (0-255) to an already-kept frame of the same clip")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    clips = load_clips()
    layouts = {c.clip_id: layout_of(c.clip_id) for c in clips}
    split = split_clips(clips, layouts, args.val_frac, args.seed)

    rows = []
    for c in clips:
        info = probe(c.path)
        cand = plan_times(info.duration, c.timecodes)
        src_of = {}
        for t, s in cand:
            src_of.setdefault(t, s)
        frames = {t: (ft, f) for t, ft, f in iter_frames_at(c.path, list(src_of), info.crop)}
        kept: list[np.ndarray] = []
        kept_actual: set[float] = set()
        for t, s in cand:  # priority order: tc first
            if t not in frames or src_of[t] != s:
                continue
            ft, f = frames[t]
            if round(ft, 3) in kept_actual:
                continue
            th = thumb(f)
            if any(np.abs(th - k).mean() < args.dup_thresh for k in kept):
                continue
            kept.append(th)
            kept_actual.add(round(ft, 3))
            name = f"{c.clip_id}_{ft:07.2f}.jpg"
            path = img_dir / name
            cv2.imwrite(str(path), f, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            rows.append({"clip_id": c.clip_id, "t": round(ft, 3), "path": str(path.relative_to(ROOT)),
                         "split": split[c.clip_id], "layout": layouts[c.clip_id], "source": s,
                         "width": f.shape[1], "height": f.shape[0]})
        print(f"{c.clip_id:6s} layout={layouts[c.clip_id]} {split[c.clip_id]:5s} "
              f"dur={info.duration:5.1f}s kept={len(kept)}/{len(cand)}")

    rows.sort(key=lambda r: (int(r["clip_id"].split("-")[1]), r["t"]))
    with open(args.out / "frames.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for sp in ("train", "val"):
        cl = sorted({r["clip_id"] for r in rows if r["split"] == sp}, key=lambda s: int(s[2:]))
        n = sum(r["split"] == sp for r in rows)
        print(f"{sp}: {n} frames from {len(cl)} clips: {' '.join(cl)}")
    for lay in sorted(set(layouts.values())):
        cl = [c.clip_id for c in clips if layouts[c.clip_id] == lay]
        print(f"layout {lay}: {' '.join(cl)}  (val: {' '.join(x for x in cl if split[x] == 'val')})")
    print(f"total {len(rows)} frames -> {args.out / 'frames.csv'}")


if __name__ == "__main__":
    main()
