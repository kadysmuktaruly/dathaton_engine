"""Make 'new camera angle' copies of a clip to test robustness.

Variants: perspective warp (camera moved sideways/up), rotation, crop+zoom, mirror (layout
flipped), plus black side bars (letterboxing like the brief describes). Written as mp4 into
cache/augment/<variant>/; constant frame rate (timestamps are preserved within ~1 frame).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from engine import config

VARIANTS = ["persp", "rot10", "crop80", "mirror", "bars"]


def _transform(variant: str, w: int, h: int):
    if variant == "persp":
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[0.08 * w, 0.04 * h], [0.97 * w, 0], [w, h], [0.02 * w, 0.93 * h]])
        M = cv2.getPerspectiveTransform(src, dst)
        return lambda f: cv2.warpPerspective(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE), (w, h)
    if variant == "rot10":
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1.12)
        return lambda f: cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE), (w, h)
    if variant == "crop80":
        x0, y0 = int(0.1 * w), int(0.05 * h)
        return lambda f: cv2.resize(f[y0:y0 + int(0.8 * h), x0:x0 + int(0.8 * w)], (w, h)), (w, h)
    if variant == "mirror":
        return lambda f: cv2.flip(f, 1), (w, h)
    if variant == "bars":
        bw = int(0.18 * w)
        W = w + 2 * bw
        def bars(f):
            out = np.zeros((h, W, 3), np.uint8)
            out[:, bw:bw + w] = f
            return out
        return bars, (W, h)
    raise ValueError(variant)


def make_variant(path: str | Path, variant: str) -> str:
    path = Path(path)
    out_dir = config.CACHE_DIR / "augment" / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (path.stem + ".mp4")
    if out.exists():
        return str(out)
    cap = cv2.VideoCapture(str(path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fn, size = _transform(variant, w, h)
    tmp = out.with_suffix(".tmp.mp4")
    wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        wr.write(fn(f))
    wr.release()
    cap.release()
    tmp.rename(out)
    return str(out)


if __name__ == "__main__":
    import sys
    from tools.labels import video_index
    v = video_index()
    for cid in sys.argv[1:] or ["X-1"]:
        for var in VARIANTS:
            print(make_variant(v[cid], var))
