"""Targeted VLM questions at key moments (docs/master-document.md).

- Departures (phase 6): two frames just after the cashier leaves, cropped to the desk: stamps,
  cash or the collection bag left there; if the laptop screen is lit, is it the lock screen.
- UV detector (A2): is one on the desk, and are the notes ever brought to it while the cashier
  handles cash.

Rules (engine/rules/depart.py, engine/rules/client.py) read the answers; timecodes still come
from perception. Answers are cached like the desk scan (video md5 + prompt version + times +
crop box) and every call is budgeted.
"""
from __future__ import annotations

import json
import re

from engine import config, vlm
from engine.video import iter_frames_at

PROMPT_VERSION = "c2"
DEPART_PROMPT = (
    "Zoomed view of an office cash desk seen from a ceiling camera. List what lies on the desk surface. "
    'Answer JSON only: {"stamps": true/false, "cash": true/false, "collection_bag": true/false, '
    '"items": ["..."]}. stamps = small office rubber stamps / seal stamps (short cylinders with a handle). '
    "cash = paper money. collection_bag = a sealed cash-collection bag."
)
DEPART_FLAGS = ["stamps", "cash", "collection_bag"]
# In this dataset the Windows lock screen is a landscape photo and the open session shows the blue
# default wallpaper (X-23 locked vs X-22 / X-28 active); asked only when the screen is lit.
LOCK_PROMPT = (
    "Close-up of a laptop screen. Does the screen show a landscape photo (mountains, lake, nature) or a "
    'plain blue Windows wallpaper? Answer JSON only: {"landscape_photo": true/false, "blue_wallpaper": true/false}'
)
LAPTOP_PAD = 0.15
LAPTOP_SCALE = 5.0
AFTER_EXIT_S = (1.0, 2.0)   # TUNE: frames this long after the last frame with the cashier in the room
MAX_DEPARTURES = 3          # budget guard
SCALE = 2.0
MAX_SIDE = 768


TIGHT_PAD = 0.06
# The 4B VLM often lists stamps as 'two small cylindrical objects' but answers stamps=false (X-24);
# no clean clip lists such an item, so a listed stamp/cylinder counts as stamps too.
STAMP_ITEM = re.compile(r"stamp|cylind|seal|печат|штамп", re.I)


def tight_desk_box(scene, w: int, h: int) -> tuple[int, int, int, int]:
    """Desk surface + counting machine + laptop, lightly padded. Targeted questions need the small
    items large in the image (the wide scan crop also holds the safe and floor and gets shrunk:
    there the VLM listed X-24's stamps as 'cylindrical objects' but answered stamps=false)."""
    from engine.vlm_scan import desk_crop_box
    boxes = [b for b in (scene.desk, scene.machine, scene.laptop) if b is not None]
    if scene.desk is None or not boxes:
        return desk_crop_box(scene, w, h)
    x0, y0 = min(b.x0 for b in boxes), min(b.y0 for b in boxes)
    x1, y1 = max(b.x1 for b in boxes), max(b.y1 for b in boxes)
    px, py = TIGHT_PAD * (x1 - x0), TIGHT_PAD * (y1 - y0)
    return max(0, int(x0 - px)), max(0, int(y0 - py)), min(w, int(x1 + px)), min(h, int(y1 + py))


def _ask_desk(video: str, times: list[float], box, crop, budget) -> list[dict]:
    import cv2
    rows = []
    x0, y0, x1, y1 = box
    for _, t, f in iter_frames_at(video, times, crop):
        if budget is not None and not budget.ok(1.5):
            break
        c = cv2.resize(f[y0:y1, x0:x1], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
        ans = vlm.parse_json(vlm.ask([vlm.to_pil(c, MAX_SIDE)], DEPART_PROMPT, max_tokens=90)) or {}
        rows.append({"t": float(t), **{k: ans.get(k) is True for k in DEPART_FLAGS},
                     "items": [str(i) for i in ans.get("items", [])][:8] if isinstance(ans.get("items"), list) else []})
    return rows


def _ask_lock(video: str, times: list[float], laptop, crop, budget) -> list[bool]:
    import cv2
    b = laptop.expand(LAPTOP_PAD)
    w, h = crop[2] - crop[0], crop[3] - crop[1]
    x0, y0, x1, y1 = max(0, int(b.x0)), max(0, int(b.y0)), min(w, int(b.x1)), min(h, int(b.y1))
    out = []
    for _, _, f in iter_frames_at(video, times, crop):
        if budget is not None and not budget.ok(1.5):
            break
        c = cv2.resize(f[y0:y1, x0:x1], None, fx=LAPTOP_SCALE, fy=LAPTOP_SCALE, interpolation=cv2.INTER_CUBIC)
        ans = vlm.parse_json(vlm.ask([vlm.to_pil(c, MAX_SIDE)], LOCK_PROMPT, max_tokens=40)) or {}
        out.append(ans.get("landscape_photo") is True and ans.get("blue_wallpaper") is not True)
    return out


def departure_times(deps: list[dict], duration: float) -> list[list[float]]:
    return [[round(min(d["t_exit"] + k, duration - 0.1), 2) for k in AFTER_EXIT_S] for d in deps[:MAX_DEPARTURES]]


def run(video: str, perc, fa, budget: vlm.Budget | None = None, use_cache: bool | None = None) -> dict:
    """{'departures': [{'t_exit', 't_depart', 'frames', 'stamps', 'cash', 'collection_bag', 'lock_screen'}]}"""
    from engine.rules.leave import departures
    from engine.vlm_scan import _video_md5
    use_cache = config.USE_CACHE if use_cache is None else use_cache
    crop = perc.info.crop
    box = tight_desk_box(fa.scene, crop[2] - crop[0], crop[3] - crop[1])
    deps = departures(fa)[:MAX_DEPARTURES]
    times = departure_times(deps, perc.info.duration)
    key = "_".join([_video_md5(video)[:16], PROMPT_VERSION, "-".join(str(int(v)) for v in box),
                    "-".join(str(int(v)) for v in crop), "+".join(f"{t:g}" for ts in times for t in ts)])
    cache = config.CACHE_DIR / "checks" / f"{key}.json"
    if use_cache and cache.exists():
        rows = json.loads(cache.read_text())
    else:
        from engine.rules.depart import SCREEN_LIT_MAX_DARK, screen_dark_share
        rows = []
        for ts in times:
            r = {"desk": _ask_desk(video, ts, box, crop, budget), "lock": None}
            lap = fa.scene.laptop
            if lap is not None and min(screen_dark_share(video, perc, lap, ts)) <= SCREEN_LIT_MAX_DARK:
                r["lock"] = _ask_lock(video, ts, lap, crop, budget)
            rows.append(r)
        if all(len(r["desk"]) == len(ts) for r, ts in zip(rows, times)):
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(rows, ensure_ascii=False))
    out = []
    for d, r in zip(deps, rows):
        out.append({"t_exit": d["t_exit"], "t_depart": d["t_depart"], "frames": r["desk"],
                    **{k: any(x[k] for x in r["desk"]) for k in DEPART_FLAGS},
                    "stamps_listed": any(STAMP_ITEM.search(i) for x in r["desk"] for i in x.get("items", [])),
                    "lock_screen": bool(r["lock"]) and all(r["lock"])})
    return {"departures": out, "box": list(box)}


# --- A2: UV detector present but never used ---------------------------------------------------
UV_VERSION = "u4"
UV_PRESENT_PROMPT = (
    "Ceiling camera view of a cash desk. Is there a small UV banknote detector on the desk: a small black "
    "device with a glowing violet or teal light, separate from the larger counting machine? "
    'Answer JSON only: {"uv_detector_present": true/false}'
)
UV_USE_PROMPT = (
    "Ceiling camera view of a cash desk. On the desk there is a small black UV banknote detector with a "
    "glowing light, in front of a larger counting machine. Is the cashier holding banknotes at or under the "
    "UV detector right now (checking them on it), or counting/holding them away from it? "
    'Answer JSON only: {"uv_detector_present": true/false, "notes_at_uv_detector": true/false}'
)
UV_STEP_S = 1.5        # TUNE
UV_MAX_FRAMES = 10     # budget guard
UV_SCALE = 2.0
UV_PRESENCE_SCALE = 1.5


def presence_band(scene, w: int, h: int):
    """Full-width band around the desk for 'is there a UV detector at all': on the tight crop the
    VLM sees one in layouts that have none (clean X-8); here it must say yes on every frame."""
    d = scene.desk
    if d is None:
        return None
    return 0, max(0, int(d.y0 - 0.6 * d.h)), w, min(h, int(d.y1 + 0.3 * d.h))


def _ask_boxes(video, times, box, crop, prompt, budget, scale, max_tokens=40) -> list[dict]:
    import cv2
    x0, y0, x1, y1 = box
    out = []
    for _, t, f in iter_frames_at(video, times, crop):
        if budget is not None and not budget.ok(1.5):
            break
        c = cv2.resize(f[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        out.append({"t": float(t), **(vlm.parse_json(vlm.ask([vlm.to_pil(c, MAX_SIDE)], prompt, max_tokens=max_tokens)) or {})})
    return out


def run_uv(video: str, perc, fa, budget: vlm.Budget | None = None, use_cache: bool | None = None) -> dict:
    """{'present': bool|None, 'used': bool, 'cash_run': (t0, t1), 'frames': [...]}. Asked only when
    the cashier handles cash for a while (engine.rules.client.cash_runs)."""
    import numpy as np
    from engine.rules.client import cash_runs
    from engine.vlm_scan import _video_md5
    use_cache = config.USE_CACHE if use_cache is None else use_cache
    runs = cash_runs(fa)
    if not runs:
        return {"present": None}
    crop = perc.info.crop
    box = tight_desk_box(fa.scene, crop[2] - crop[0], crop[3] - crop[1])
    dur = perc.info.duration
    # ask about every stretch of cash handling (the check may happen outside the long one: X-43)
    span = cash_runs(fa, min_len=2) or runs
    t0, t1 = float(span[0][0]), float(min(max(span[-1][1], runs[-1][1]) + 1, dur - 0.1))
    pres_t = sorted({round(t0 + 0.5, 2), round(min((t0 + t1) / 2, dur - 0.1), 2), round(min(t1, dur - 0.1), 2)})
    pbox = presence_band(fa.scene, crop[2] - crop[0], crop[3] - crop[1]) or box
    use_t = [round(float(t), 2) for t in np.arange(t0, t1 + 1e-6, UV_STEP_S)]
    if len(use_t) > UV_MAX_FRAMES:
        use_t = [round(float(t), 2) for t in np.linspace(t0, t1, UV_MAX_FRAMES)]
    key = "_".join([_video_md5(video)[:16], UV_VERSION, "-".join(str(int(v)) for v in box),
                    "-".join(str(int(v)) for v in pbox), "+".join(f"{t:g}" for t in pres_t + use_t)])
    cache = config.CACHE_DIR / "checks" / f"uv_{key}.json"
    if use_cache and cache.exists():
        r = json.loads(cache.read_text())
    else:
        pres = _ask_boxes(video, pres_t, pbox, crop, UV_PRESENT_PROMPT, budget, UV_PRESENCE_SCALE)
        r = {"presence": pres, "use": []}
        if pres and all(p.get("uv_detector_present") is True for p in pres):
            r["use"] = _ask_boxes(video, use_t, box, crop, UV_USE_PROMPT, budget, UV_SCALE)
        complete = len(pres) == len(pres_t) and (not r["use"] or len(r["use"]) == len(use_t))
        if complete:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(r, ensure_ascii=False))
    pres = r["presence"]
    present = bool(pres) and all(p.get("uv_detector_present") is True for p in pres)
    used = any(u.get("notes_at_uv_detector") is True for u in r["use"])
    first = max(runs, key=lambda x: x[1] - x[0])  # the main handling stretch
    return {"present": present, "used": used, "cash_run": [float(first[0]), float(first[1])],
            "frames": len(r["use"]), "asked": bool(r["use"])}
