"""Video -> timestamped VLM answers. The VLM is the only perception model.

1. layout: the VLM locates the door, desk, safe, laptop, counting machine (and a UV detector) once per clip,
   so the crops follow the camera angle;
2. state pass: every `step` seconds, one short question per zone (people, door, safe, laptop, desk items).
   A zone whose crop did not change since it was last asked reuses that answer;
3. action pass: while the cashier is in the room, a window of 4 frames every 2 s is asked what happens
   (card / cash / bag handed over, counting machine, hand counting, UV check, typing).

Answers are cached per clip and question (key: video md5 + prompt version), so the timeline rules
(timeline.py) can be re-run and tuned without the VLM.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from engine.video import iter_frames_at, probe
from pure_vlm import config, prompts, vlm


@dataclass
class Clip:
    path: str
    duration: float
    times: np.ndarray            # frame grid (GRID_S apart)
    frames: list[np.ndarray]     # BGR frames on the grid
    md5: str

    @property
    def h(self) -> int:
        return self.frames[0].shape[0]

    @property
    def w(self) -> int:
        return self.frames[0].shape[1]

    def idx(self, t: float) -> int:
        return int(np.argmin(np.abs(self.times - t)))

    def frame(self, t: float) -> np.ndarray:
        return self.frames[self.idx(t)]


def video_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_clip(path: str) -> Clip:
    info = probe(path)
    times = np.round(np.arange(0.0, max(info.duration - 0.05, 0.1), config.GRID_S), 3)
    got = list(iter_frames_at(path, list(times)))
    return Clip(str(path), info.duration, np.array([g[0] for g in got]), [g[2] for g in got], video_md5(path))


# ---- layout ----------------------------------------------------------------------------------------------
ZONE_KEYS = {"door": "door", "desk": "desk", "safe": "safe", "laptop": "laptop", "counting": "machine",
             "machine": "machine", "uv": "uv"}


def _parse_layout(ans, w: int, h: int) -> dict:
    out = {}
    for item in ans if isinstance(ans, list) else []:
        if not isinstance(item, dict) or "bbox_2d" not in item:
            continue
        label = str(item.get("label", "")).lower()
        key = next((v for k, v in ZONE_KEYS.items() if k in label), None)
        b = item["bbox_2d"]
        if key is None or key in out or len(b) != 4:
            continue
        x0, y0, x1, y1 = (float(v) for v in b)  # Qwen grounding: 0..1000 relative coordinates
        box = [x0 / 1000 * w, y0 / 1000 * h, x1 / 1000 * w, y1 / 1000 * h]
        if box[2] - box[0] > 4 and box[3] - box[1] > 4:
            out[key] = [round(v, 1) for v in box]
    return out


def locate(clip: Clip) -> dict:
    """Zone boxes in pixels. Asked on two frames (people may hide objects in one); the first answer wins."""
    zones: dict = {}
    raw = []
    for t in (0.2, clip.duration * 0.5):
        ans, text = vlm.ask_json([vlm.pil(clip.frame(t), 848)], prompts.LAYOUT, max_tokens=260)
        raw.append(text)
        for k, v in _parse_layout(ans, clip.w, clip.h).items():
            zones.setdefault(k, v)
        if all(k in zones for k in ("door", "desk", "safe", "laptop")):
            break
    zones["_raw"] = raw
    return zones


def _union(*boxes):
    bs = [b for b in boxes if b]
    if not bs:
        return None
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _pad(box, w, h, left, top, right, bottom):
    """Grow a box by fractions of its own size; clamp to the frame."""
    if box is None:
        return None
    bw, bh = box[2] - box[0], box[3] - box[1]
    return [max(0, box[0] - left * bw), max(0, box[1] - top * bh), min(w, box[2] + right * bw),
            min(h, box[3] + bottom * bh)]


def crops(zones: dict, w: int, h: int) -> dict:
    """The crop each question looks at (pixels). Missing zones fall back to the full frame or the desk."""
    devices = _union(zones.get("laptop"), zones.get("machine"), zones.get("uv"))
    desk = zones.get("desk")
    if desk and devices and not _overlaps(desk, devices):
        desk = None  # X-4: the 9B grounded the white table in the foreground; the devices stand on the real desk
    desk_all = _union(desk, devices) if desk else _pad(devices, w, h, 0.4, 0.3, 0.4, 0.6)
    out = {
        "people": [0, 0, w, h],
        "door": _pad(zones.get("door"), w, h, 0.1, 0.05, 0.1, 0.05) or [0, 0, w, h * 0.6],
        "safe": _pad(zones.get("safe"), w, h, 0.3, 0.2, 0.3, 0.2),
        "laptop": _pad(zones.get("laptop"), w, h, 0.6, 0.6, 0.6, 0.4),
        "desk": _pad(desk_all, w, h, 0.08, 0.15, 0.08, 0.1) or [0, 0, w, h],
        # people working at the desk: the desk plus room above it for the visitor and the cashier
        "work": _pad(desk_all, w, h, 0.3, 1.0, 0.3, 0.4) or [0, 0, w, h],
    }
    if out["laptop"] is None:
        out["laptop"] = out["desk"]
    return out


# ---- questions ---------------------------------------------------------------------------------------------
@dataclass
class Question:
    zone: str
    prompt: str
    side: int          # longer side of the image sent to the VLM (crops are upscaled)
    max_tokens: int
    kind: str          # 'json', 'word' or 'names' (comma-separated names from `names`)
    names: list = field(default_factory=list)


def _people_key(a):
    a = a if isinstance(a, dict) else {}
    return (a.get("cashier"), _int(a.get("visitors")))


def _safe_key(a):
    a = a if isinstance(a, dict) else {}
    return (a.get("open") is True, a.get("key") is True)


def _list_key(a):
    return tuple(sorted(str(x) for x in a)) if isinstance(a, list) else ()


def parse_names(text: str, allowed: list[str]) -> list[str]:
    """'cash, drink' -> ['cash', 'drink']; anything not in `allowed` is dropped."""
    words = re.findall(r"[a-z_]+", text.lower())
    return sorted({w for w in words if w in allowed})


STATE_QUESTIONS = {
    "people": Question("people", prompts.PEOPLE, 576, 30, "json"),
    "door": Question("door", prompts.DOOR, 640, 4, "word"),
    "safe": Question("safe", prompts.SAFE, 512, 24, "json"),
    "laptop": Question("laptop", prompts.LAPTOP, 448, 4, "word"),
    "desk": Question("desk", prompts.DESK, 576, 30, "names", prompts.DESK_KEYS),
}
STATE_KEY = {"people": _people_key, "door": lambda a: a, "safe": _safe_key, "laptop": lambda a: a,
             "desk": _list_key}


def _grey(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)


def _ask(q: Question, images: list[np.ndarray]):
    pils = [vlm.pil(im, q.side) for im in images]
    if q.kind == "json":
        ans, text = vlm.ask_json(pils, q.prompt, q.max_tokens)
        return ans if ans is not None else {"_text": text}
    text = vlm.generate(pils, q.prompt, q.max_tokens)
    if q.kind == "names":
        return parse_names(text, q.names)
    return text.strip().strip(".").lower().split()[0] if text.strip() else ""


class Budget:
    def __init__(self, seconds: float):
        self.end = time.time() + seconds

    def left(self) -> float:
        return self.end - time.time()

    def ok(self, need: float = 2.0) -> bool:
        return self.left() > need


def coarse_times(clip: Clip) -> list[float]:
    step = max(config.STATE_STEP_S, clip.duration / config.MAX_STATE_FRAMES)
    step = max(config.GRID_S, round(step / config.GRID_S) * config.GRID_S)
    ts = [float(clip.times[clip.idx(t)]) for t in np.arange(0.0, clip.duration - 0.05, step)]
    if ts[-1] < clip.times[-1]:
        ts.append(float(clip.times[-1]))
    return sorted(set(ts))


class Series:
    """Answers of one state question over the clip. A crop that barely differs from an asked neighbour
    reuses that neighbour's answer instead of asking again."""

    def __init__(self, clip: Clip, name: str, box, budget: Budget, qcache: dict):
        self.clip, self.name, self.box, self.budget, self.qcache = clip, name, box, budget, qcache
        self.q = STATE_QUESTIONS[name]
        self.key_fn = STATE_KEY[name]
        self.answers: dict[float, object] = {}
        self.greys: dict[float, np.ndarray] = {}

    def get(self, t: float, neighbours=()) -> object:
        img = vlm.crop(self.clip.frame(t), self.box)
        g = _grey(img)
        self.greys[t] = g
        for n in neighbours:
            if n in self.greys and np.abs(g - self.greys[n]).mean() < config.REUSE_DIFF[self.name]:
                return self.answers[n]
        k = f"{self.name}@{t:.2f}"
        if k not in self.qcache:
            if not self.budget.ok():
                return self.answers[neighbours[0]] if neighbours else None
            self.qcache[k] = _ask(self.q, [img])
        return self.qcache[k]

    def coarse(self, t: float, prev: float | None) -> None:
        self.answers[t] = self.get(t, (prev,) if prev is not None else ())

    def refine_once(self) -> bool:
        """Ask the midpoint of every neighbouring pair whose answers differ; True if anything was added."""
        added = False
        ts = sorted(self.answers)
        for a, b in zip(ts, ts[1:]):
            if b - a > config.GRID_S + 1e-6 and self.key_fn(self.answers[a]) != self.key_fn(self.answers[b]):
                m = float(self.clip.times[self.clip.idx((a + b) / 2)])
                if m not in self.answers and a < m < b and self.budget.ok():
                    self.answers[m] = self.get(m, (a, b))
                    added = True
        return added


def key_moments(people: dict[float, object], duration: float) -> dict[str, list[float]]:
    """Times the departure / arrival rules read the desk, safe and laptop at: just after the cashier leaves
    the room (the view is free then) and just before a visitor comes in."""
    ts = sorted(people)
    dep, arr = [], []
    for a, b in zip(ts, ts[1:]):
        pa, pb = _people_key(people[a]), _people_key(people[b])
        if pa[0] not in (None, "absent") and pb[0] == "absent":
            dep += [b, min(b + 1.0, duration - 0.1)]
        if pa[1] == 0 and pb[1] > 0:
            arr += [max(a - 0.5, 0.0), a]
    return {"departure": dep, "arrival": arr}


def state_pass(clip: Clip, boxes: dict, budget: Budget, cache: dict) -> dict[str, dict[float, object]]:
    """{question: {t: answer}}.
    1. people and door on the coarse grid, then bisection wherever two neighbouring answers differ;
    2. safe and desk on a sparser grid (COARSE_EVERY), plus the key moments, then bisection;
    3. laptop only at departures (it matters only when the cashier leaves: B9).
    A tight budget still covers the whole clip because every question gets its coarse pass first."""
    qcache = cache.setdefault("state", {})
    series = {name: Series(clip, name, boxes[q.zone], budget, qcache)
              for name, q in STATE_QUESTIONS.items() if boxes.get(q.zone) is not None}
    grid = coarse_times(clip)
    dense = [s for n, s in series.items() if n in ("people", "door")]
    prev = None
    for t in grid:
        for s in dense:
            s.coarse(t, prev)
        prev = t
    while any([s.refine_once() for s in dense]):
        pass
    moments = key_moments(series["people"].answers, clip.duration) if "people" in series else {}
    for name, every in config.COARSE_EVERY.items():
        s = series.get(name)
        if s is None:
            continue
        ts = grid[::every] if every else []
        for m in config.KEY_MOMENTS.get(name, []):
            ts += [float(clip.times[clip.idx(t)]) for t in moments.get(m, [])]
        prev = None
        for t in sorted(set(ts)):
            s.coarse(t, prev)
            prev = t
    sparse = [s for n, s in series.items() if n in ("safe", "desk")]
    while any([s.refine_once() for s in sparse]):
        pass
    out = {name: series[name].answers if name in series else {} for name in STATE_QUESTIONS}
    out["_moments"] = moments
    return out


def at(series: dict[float, object], t: float):
    """Step function: the answer at the latest asked time <= t (or the first one)."""
    if not series:
        return None
    ts = sorted(series)
    i = int(np.searchsorted(ts, t + 1e-6)) - 1
    return series[ts[max(i, 0)]]


def busy(states: dict, t: float) -> bool:
    """Worth an action window: a visitor is in the room, or cash / a collection bag lies on the desk
    while the cashier is there (counting alone, opening a bag)."""
    p = at(states.get("people", {}), t)
    p = p if isinstance(p, dict) else {}
    if _int(p.get("visitors")) > 0:
        return True
    desk = at(states.get("desk", {}), t)
    items = set(_list_key(desk))
    return p.get("cashier") in ("desk", "safe") and bool(items & {"cash", "collection_bag"})


def action_windows(clip: Clip, states: dict) -> list[float]:
    """Window start times, every ACTION_STEP_S where the scene is busy at the window's start, middle or end."""
    span = (config.WINDOW_FRAMES - 1) * config.GRID_S
    out = []
    t = 0.0
    while t + span <= clip.times[-1] + 1e-6:
        if any(busy(states, x) for x in (t, t + span / 2, t + span)):
            out.append(round(t, 3))
        t += config.ACTION_STEP_S
    return out


def action_question() -> Question:
    return Question("work", prompts.ACTIONS.format(n=config.WINDOW_FRAMES, dt=config.GRID_S), config.ACTION_SIDE, 40,
                    "names", prompts.ACTION_KEYS)


def action_pass(clip: Clip, boxes: dict, starts: list[float], budget: Budget, cache: dict) -> list[dict]:
    q = action_question()
    acache = cache.setdefault("actions", {})
    rows = []
    for t in starts:
        key = f"{t:.2f}"
        times = [t + k * config.GRID_S for k in range(config.WINDOW_FRAMES)]
        if key in acache:
            ans = acache[key]
        elif budget.ok(4.0):
            ans = acache[key] = _ask(q, [vlm.crop(clip.frame(x), boxes["work"]) for x in times])
        else:
            continue
        got = set(_list_key(ans))
        rows.append({"t0": times[0], "t1": times[-1], **{k: k in got for k in prompts.ACTION_KEYS}})
    return rows


def _int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


# ---- entry point ---------------------------------------------------------------------------------------------
def cache_path(md5: str):
    return config.CACHE_DIR / f"{md5[:16]}_{prompts.VERSION}_{config.VLM_REPO.split('/')[-1]}.json"


def observe(path: str, budget_s: float | None = None) -> dict:
    """All VLM answers for a clip: {'duration', 'zones', 'boxes', 'states', 'actions', 'timing'}."""
    t_start = time.time()
    clip = load_clip(path)
    t_decode = time.time() - t_start
    budget = Budget(budget_s if budget_s is not None else config.BUDGET_S)
    cp = cache_path(clip.md5)
    cache = json.loads(cp.read_text()) if config.USE_CACHE and cp.exists() else {}
    c0, s0 = vlm.STATS["calls"], vlm.STATS["seconds"]

    if "zones" not in cache:
        cache["zones"] = locate(clip)
    zones = cache["zones"]
    boxes = crops(zones, clip.w, clip.h)
    states = state_pass(clip, boxes, budget, cache)
    moments = states.pop("_moments")
    t2 = time.time()
    actions = action_pass(clip, boxes, action_windows(clip, states), budget, cache)
    t3 = time.time()

    if config.USE_CACHE:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(cache, ensure_ascii=False))
    return {"path": str(path), "md5": clip.md5, "duration": clip.duration, "w": clip.w, "h": clip.h,
            "zones": {k: v for k, v in zones.items() if not k.startswith("_")}, "boxes": boxes,
            "states": {k: {f"{t:.2f}": a for t, a in sorted(v.items())} for k, v in states.items()},
            "actions": actions, "moments": moments,
            "timing": {"decode_s": round(t_decode, 2), "layout_states_s": round(t2 - t_start - t_decode, 2),
                       "actions_s": round(t3 - t2, 2), "total_s": round(time.time() - t_start, 2),
                       "vlm_calls": vlm.STATS["calls"] - c0, "vlm_s": round(vlm.STATS["seconds"] - s0, 2),
                       "budget_left_s": round(budget.left(), 1)}}
