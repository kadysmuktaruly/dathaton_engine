"""Ask the VLM one question on frames from several clips (experiments).

Usage: uv run python -m tools.vlm_probe "<question>" X-31@10 X-32@10 ... [--crop y0,y1] [--scale 2]
"""
import argparse
import time

import cv2

from engine import vlm
from engine.video import iter_frames_at
from tools.labels import video_index

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("items", nargs="+")
    ap.add_argument("--crop", default=None, help="y0,y1,x0,x1 as fractions")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--max-side", type=int, default=768)
    a = ap.parse_args()
    vids = video_index()
    for it in a.items:
        cid, ts = it.split("@")
        times = [float(x) for x in ts.split("+")]
        imgs = []
        for _, t, f in iter_frames_at(vids[cid], times):
            if a.crop:
                y0, y1, x0, x1 = map(float, a.crop.split(","))
                h, w = f.shape[:2]
                f = f[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
            if a.scale != 1.0:
                f = cv2.resize(f, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_CUBIC)
            imgs.append(vlm.to_pil(f, a.max_side))
        t0 = time.time()
        ans = vlm.ask(imgs, a.question, max_tokens=120)
        print(f"{it:14s} {time.time()-t0:4.1f}s  {ans.strip()[:300]!r}", flush=True)
