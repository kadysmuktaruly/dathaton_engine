"""Turn per-frame boolean signals into events with hysteresis, min duration and merging."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Event:
    code: str
    start: float           # timecode candidate (sec)
    end: float
    score: float           # 0..1, rule confidence before VLM
    evidence_t: list[float] = field(default_factory=list)  # timestamps for evidence frames
    boxes: dict = field(default_factory=dict)  # t -> list of (label, [x0,y0,x1,y1])
    info: dict = field(default_factory=dict)   # debug details


def segments(times: np.ndarray, on: np.ndarray, min_on: float, max_gap: float) -> list[tuple[float, float]]:
    """Intervals where `on` holds; gaps shorter than max_gap are bridged; intervals shorter
    than min_on are dropped. Times must be sorted."""
    segs = []
    start = None
    last_on = None
    for t, v in zip(times, on):
        if v:
            if start is None:
                start = t
            elif last_on is not None and t - last_on > max_gap:
                segs.append((start, last_on))
                start = t
            last_on = t
    if start is not None:
        segs.append((start, last_on))
    return [(a, b) for a, b in segs if b - a >= min_on - 1e-6]


def merge(events: list[Event], gap: float = 2.0) -> list[Event]:
    """Merge same-code events that overlap or are closer than `gap` seconds."""
    out: list[Event] = []
    for e in sorted(events, key=lambda e: (e.code, e.start)):
        if out and out[-1].code == e.code and e.start <= out[-1].end + gap:
            p = out[-1]
            p.end = max(p.end, e.end)
            p.score = max(p.score, e.score)
            p.boxes.update(e.boxes)
            p.info.setdefault("merged", []).append(e.start)
        else:
            out.append(e)
    return sorted(out, key=lambda e: e.start)
