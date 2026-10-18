"""Ask one question on frames, to check prompts against frames verified by eye.

  python -m pure_vlm.probe door X-32@8 X-31@14
  python -m pure_vlm.probe actions X-3@11 X-2@9          (a window of 4 frames starting at t)
  python -m pure_vlm.probe layout X-3@0
  --save DIR writes the crops that were sent, for a visual check (local only).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from pure_vlm import config, observe, prompts, vlm
from tools.labels import video_index

_clips: dict = {}


def clip(cid: str) -> observe.Clip:
    if cid not in _clips:
        _clips[cid] = observe.load_clip(str(video_index()[cid]))
    return _clips[cid]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", help="layout | actions | " + " | ".join(observe.STATE_QUESTIONS))
    ap.add_argument("specs", nargs="+", help="CLIP@t")
    ap.add_argument("--save")
    args = ap.parse_args()
    zcache: dict = {}
    for spec in args.specs:
        cid, t = spec.split("@")
        t = float(t)
        c = clip(cid)
        t0 = time.time()
        if args.question == "layout":
            ans = observe._parse_layout(vlm.ask_json([vlm.pil(c.frame(t), 848)], prompts.LAYOUT, 260)[0], c.w, c.h)
            print(spec, f"{time.time() - t0:.1f}s", json.dumps(ans))
            continue
        if cid not in zcache:
            cp = observe.cache_path(c.md5)
            cached = json.loads(cp.read_text()).get("zones") if cp.exists() else None
            zcache[cid] = cached or observe.locate(c)
        boxes = observe.crops(zcache[cid], c.w, c.h)
        if args.question == "actions":
            q = observe.Question("work", prompts.ACTIONS.format(n=config.WINDOW_FRAMES, dt=config.GRID_S), 448, 110,
                                 "json")
            imgs = [vlm.crop(c.frame(t + k * config.GRID_S), boxes["work"]) for k in range(config.WINDOW_FRAMES)]
        else:
            q = observe.STATE_QUESTIONS[args.question]
            imgs = [vlm.crop(c.frame(t), boxes[q.zone])]
        t0 = time.time()
        ans = observe._ask(q, imgs)
        print(spec, f"{time.time() - t0:.1f}s", json.dumps(ans, ensure_ascii=False))
        if args.save:
            d = Path(args.save)
            d.mkdir(parents=True, exist_ok=True)
            for k, im in enumerate(imgs):
                cv2.imwrite(str(d / f"{cid}_{t:g}_{args.question}_{k}.jpg"), im)


if __name__ == "__main__":
    main()
