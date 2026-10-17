"""Idea 1: the engine's desk scan, asked by Qwen3.5-4B instead of Qwen3-VL-4B.

Same output as engine.vlm_scan.scan (t + engine.vlm_scan.FLAGS per sampled frame, attrs['box']),
so engine.rules.scan_rules (B1 / B3 / D1 / X0) run unchanged on it. Same frame times and desk
crop as the engine (vlm_scan.scan_times, vlm_scan.desk_crop_box); one combined JSON question
per frame. In the textflow benchmark Qwen3.5-4B got cash/bag on desk crops right where the
engine's scan missed (X-4, X-38 cash), at ~1.3 s per call.
"""
from __future__ import annotations

import json
import time

import cv2
import pandas as pd

from engine import config as engine_config
from engine import vlm_scan
from engine.video import iter_frames_at
from orig_and_vlm import config, eyes

PROMPT_VERSION = "ov1"
KEYS = {"cash": "banknotes_on_desk", "bag": "bag_or_coat_on_desk", "green_bag": "green_bag_on_desk",
        "drink": "drink_cup"}
assert set(KEYS.values()) == set(vlm_scan.FLAGS)
PROMPT = ("Zoomed ceiling CCTV view of a bank cash desk. Look only at objects LYING ON the desk surface, not "
          "things a person is carrying, except for the drink. For each key answer true or false. JSON only: "
          '{"cash": <paper money, a banknote or a bundle, lying on the desk>, '
          '"bag": <a personal handbag, backpack or coat lying on the desk>, '
          '"green_bag": <a green cash-collection bag lying on the desk>, '
          '"drink": <a cup, mug or bottle on the desk or in a hand>}')
SINGLE = {  # one flag at a time, for timecode bisection
    "banknotes_on_desk": "Zoomed ceiling CCTV view of a bank cash desk. Is there paper money (a banknote or a "
                         "bundle of banknotes) lying on the desk surface?",
    "bag_or_coat_on_desk": "Zoomed ceiling CCTV view of a bank cash desk. Is a handbag, backpack or coat lying on "
                           "the desk?",
    "green_bag_on_desk": "Zoomed ceiling CCTV view of a bank cash desk. Is a green cash-collection bag lying on "
                         "the desk?",
    "drink_cup": "Zoomed ceiling CCTV view of a bank cash desk. Is there a cup, mug or bottle on the desk or in a "
                 "hand?",
}
YESNO = " Answer 'yes' or 'no' first, then one short reason."


def _zoom(frame, box):
    c = eyes.crop(frame, box)
    return cv2.resize(c, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)  # as the engine's scan


def ask_frames(video: str, times: list[float], box, crop, budget=None) -> list[dict]:
    from engine.vlm import parse_json
    rows = []
    for _, t, f in iter_frames_at(video, times, crop):
        if budget is not None and not budget.ok(1.5):
            break
        text, _ = eyes.generate([eyes.pil(_zoom(f, box))], PROMPT, max_tokens=60)
        ans = parse_json(text) or {}
        rows.append({"t": float(t), **{flag: ans.get(k) is True for k, flag in KEYS.items()}})
    return rows


def key_moments(fa) -> list[float]:
    """Times the scan rules look at: just before each new client comes in (B1 'cash open when
    the next client arrives' reads the scan in [t-4, t+1]) and around each departure (B1 'cash
    left when the cashier walks away'). X-4: the uniform scan had no frame near the entry at
    55.3 s although it saw cash at 51.1 and 59.5 s."""
    from engine.rules.leave import departures
    from engine.rules.scan_rules import _entries
    ts = [t - 0.5 for t in _entries(fa, min_prev=1)]
    for d in departures(fa):
        ts += [d["t_depart"], d["t_exit"]]
    return sorted({round(max(t, 0.0), 2) for t in ts})


def scan(video: str, perc, scene, budget=None, use_cache: bool | None = None,
         extra_times: list[float] = ()) -> pd.DataFrame:
    """extra_times are asked first (key_moments), then the engine's grid coarse-to-fine."""
    use_cache = engine_config.USE_CACHE if use_cache is None else use_cache
    crop = perc.info.crop
    w, h = crop[2] - crop[0], crop[3] - crop[1]
    box = vlm_scan.desk_crop_box(scene, w, h)
    b = "-".join(str(int(v)) for v in box)
    c = "-".join(str(int(v)) for v in crop)
    x = "-".join(f"{t:g}" for t in extra_times)
    key = (f"{vlm_scan._video_md5(video)[:16]}_{PROMPT_VERSION}_s{vlm_scan.STEP_S:g}_m{vlm_scan.MAX_FRAMES}"
           f"_b{b}_c{c}_x{x}_{config.EYES_REPO.split('/')[-1]}")
    cache = engine_config.CACHE_DIR / "scan_ov" / f"{key}.json"
    if use_cache and cache.exists():
        df = pd.DataFrame(json.loads(cache.read_text()), columns=["t"] + vlm_scan.FLAGS)
        df.attrs["box"] = box
        return df
    t0 = time.time()
    grid = vlm_scan.scan_times(perc.info.duration)
    extra = [t for t in extra_times if all(abs(t - g) > 0.6 for g in grid)]
    rows = ask_frames(video, extra, box, crop, budget)
    rows += ask_frames(video, grid[::2], box, crop, budget)    # coarse: the whole clip is covered
    rows += ask_frames(video, grid[1::2], box, crop, budget)
    rows.sort(key=lambda r: r["t"])
    df = pd.DataFrame(rows, columns=["t"] + vlm_scan.FLAGS)
    if len(rows) == len(grid) + len(extra):
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows))
    df.attrs["scan_s"] = round(time.time() - t0, 2)
    df.attrs["box"] = box
    return df


def refine(video: str, perc, box, flag: str, t_neg: float | None, t_pos: float, budget=None, iters: int = 2) -> float:
    """Bisect between the last negative scan time and the first positive one (as vlm_scan.refine)."""
    if t_neg is None or flag not in SINGLE:
        return t_pos
    lo, hi = t_neg, t_pos
    for _ in range(iters):
        if budget is not None and not budget.ok(1.5):
            break
        mid = round((lo + hi) / 2, 2)
        _, actual, f = next(iter_frames_at(video, [mid], perc.info.crop))
        ans, _, _ = eyes.ask(_zoom(f, box), SINGLE[flag] + YESNO)
        if ans is True:
            hi = round(float(actual), 2)
        elif ans is False:
            lo = mid
        else:
            break
    return hi
