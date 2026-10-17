"""VLM desk scan: one yes/no question set per sampled frame, on a zoomed crop of the desk.

Output: a table t -> flags (banknotes_on_desk, counting_by_hand, handbag_or_coat, green_bag,
drink_cup, safe_open). Rules combine these flags with perception events; timecodes still come
from frame-level perception or from the scan frame times, never from VLM free text.
Results are cached per (video md5, prompt version, step).
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd

from engine import config, vlm
from engine.video import iter_frames_at

PROMPT_VERSION = "v2"
FLAGS = ["banknotes_on_desk", "bag_or_coat_on_desk", "green_bag_on_desk", "drink_cup"]
PROMPT = (
    "Zoomed view of a cash desk from a ceiling camera. Look only at objects LYING ON the desk "
    "surface, not things a person is carrying. Answer JSON only: "
    '{"banknotes_on_desk": true/false, "bag_or_coat_on_desk": true/false, '
    '"green_bag_on_desk": true/false, "drink_cup": true/false}. '
    "banknotes_on_desk = paper money lying on the desk. bag_or_coat_on_desk = a personal handbag, "
    "backpack or coat lying on the desk. green_bag_on_desk = a green cash-collection bag lying on "
    "the desk. drink_cup = a cup, mug or bottle on the desk or in a hand."
)
STEP_S = 2.0          # TUNE: scan step
MAX_FRAMES = 18       # TUNE: cap per video (budget)
MAX_SIDE = 640


def desk_crop_box(scene, w: int, h: int) -> tuple[int, int, int, int]:
    """Desk region with room for hands, items and a nearby safe; fallback: middle band."""
    d = scene.desk
    if d is None or d.w < 0.25 * w:
        return 0, int(0.25 * h), w, int(0.72 * h)
    x0 = max(0, int(d.x0 - 0.35 * d.w)); x1 = min(w, int(d.x1 + 0.35 * d.w))
    y0 = max(0, int(d.y0 - 0.9 * d.h)); y1 = min(h, int(d.y1 + 0.25 * d.h))
    if scene.safe is not None:
        s = scene.safe
        x0, y0 = min(x0, int(s.x0)), min(y0, int(s.y0))
        x1, y1 = max(x1, int(s.x1)), max(y1, int(s.y1))
    return x0, y0, x1, y1


def scan_times(duration: float) -> list[float]:
    n = int(duration // STEP_S) + 1
    if n > MAX_FRAMES:
        return [round(float(x), 2) for x in np.linspace(0.5, duration - 0.5, MAX_FRAMES)]
    return [round(min(k * STEP_S + 0.5, duration - 0.1), 2) for k in range(n)]


def _video_md5(video: str) -> str:
    h = hashlib.md5()
    with open(video, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_key(video_md5: str, box, crop) -> str:
    """The answers depend on the pixels asked about: the desk crop box and the black-bar crop
    are part of the key, so a changed desk or crop never reuses old answers."""
    b = "-".join(str(int(v)) for v in box)
    c = "-".join(str(int(v)) for v in crop)
    return f"{video_md5[:16]}_{PROMPT_VERSION}_s{STEP_S:g}_m{MAX_FRAMES}_b{b}_c{c}"


def ask_frames(video: str, times: list[float], box, crop, budget: vlm.Budget | None = None) -> list[dict]:
    import cv2
    rows = []
    x0, y0, x1, y1 = box
    for want, t, f in iter_frames_at(video, times, crop):
        if budget is not None and not budget.ok(1.5):
            break
        c = f[y0:y1, x0:x1]
        c = cv2.resize(c, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
        ans = vlm.parse_json(vlm.ask([vlm.to_pil(c, MAX_SIDE)], PROMPT, max_tokens=90)) or {}
        rows.append({"t": float(t), **{k: bool(ans.get(k, False)) for k in FLAGS}})
    return rows


def scan(video: str, perc, scene, budget: vlm.Budget | None = None, use_cache: bool | None = None) -> pd.DataFrame:
    use_cache = config.USE_CACHE if use_cache is None else use_cache
    crop = perc.info.crop
    w, h = crop[2] - crop[0], crop[3] - crop[1]
    box = desk_crop_box(scene, w, h)
    cache = config.CACHE_DIR / "scan" / f"{scan_key(_video_md5(video), box, crop)}.json"
    if use_cache and cache.exists():
        df = pd.DataFrame(json.loads(cache.read_text()), columns=["t"] + FLAGS)
        df.attrs["box"] = box
        return df
    t0 = time.time()
    times = scan_times(perc.info.duration)
    # coarse-to-fine: every other frame first, so a tight budget still covers the whole clip
    rows = ask_frames(video, times[::2], box, crop, budget)
    rows += ask_frames(video, times[1::2], box, crop, budget)
    rows.sort(key=lambda r: r["t"])
    df = pd.DataFrame(rows, columns=["t"] + FLAGS)
    complete = budget is None or len(rows) == len(scan_times(perc.info.duration))
    if complete:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows))
    df.attrs["scan_s"] = round(time.time() - t0, 2)
    df.attrs["box"] = box
    return df


def refine(video: str, perc, box, flag: str, t_neg: float | None, t_pos: float,
           budget: vlm.Budget | None = None, iters: int = 2) -> float:
    """Bisect between the last negative scan time and the first positive one."""
    if t_neg is None:
        return t_pos
    lo, hi = t_neg, t_pos
    for _ in range(iters):
        if budget is not None and not budget.ok(1.5):
            break
        mid = round((lo + hi) / 2, 2)
        r = ask_frames(video, [mid], box, perc.info.crop, budget)
        if r and r[0][flag]:
            hi = mid
        else:
            lo = mid
    return hi
