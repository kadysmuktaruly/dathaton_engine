"""Persistent background change between two time windows, with people masked out.

Used to see what was left changed after an event: a door left open, a safe door left open,
an object left on the desk. Cheap (grayscale, downscaled) and independent of object classes.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from engine.video import iter_frames_at

SCALE = 0.5          # work at half resolution
DIFF_THRESH = 28     # TUNE: grey-level difference counted as change
PERSON_PAD = 0.15    # TUNE: dilate person boxes by this share of their size


def _frames(video: str, times: list[float], crop) -> list[tuple[float, np.ndarray]]:
    out = []
    for _, t, f in iter_frames_at(video, times, crop):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        out.append((t, g.astype(np.float32)))
    return out


def _person_mask(shape, boxes: pd.DataFrame) -> np.ndarray:
    m = np.zeros(shape, bool)
    for r in boxes.itertuples():
        w, h = r.x1 - r.x0, r.y1 - r.y0
        x0 = int(max(0, (r.x0 - PERSON_PAD * w) * SCALE)); x1 = int(min(shape[1], (r.x1 + PERSON_PAD * w) * SCALE))
        y0 = int(max(0, (r.y0 - PERSON_PAD * h) * SCALE)); y1 = int(min(shape[0], (r.y1 + PERSON_PAD * h) * SCALE))
        m[y0:y1, x0:x1] = True
    return m


def background(video: str, times: list[float], persons: pd.DataFrame, crop) -> tuple[np.ndarray, np.ndarray]:
    """Median of frames at `times` ignoring pixels covered by people. Returns (bg, valid)."""
    fr = _frames(video, times, crop)
    if not fr:
        return None, None
    stack = []
    for t, g in fr:
        near = persons[(persons.t - t).abs() <= 0.2]
        g = g.copy()
        g -= np.median(g)  # remove global exposure shifts
        g[_person_mask(g.shape, near)] = np.nan
        stack.append(g)
    st = np.stack(stack)
    with np.errstate(all="ignore"):
        bg = np.nanmedian(st, axis=0)
    return bg, ~np.isnan(bg)


def change_map(video: str, before: list[float], after: list[float], persons: pd.DataFrame, crop
               ) -> np.ndarray | None:
    """Boolean map (at SCALE) of pixels that changed persistently between the windows."""
    b0, v0 = background(video, before, persons, crop)
    b1, v1 = background(video, after, persons, crop)
    if b0 is None or b1 is None:
        return None
    d = np.abs(b1 - b0)
    ch = (d > DIFF_THRESH) & v0 & v1
    ch = cv2.morphologyEx(ch.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return ch.astype(bool)


def blobs(ch: np.ndarray, min_area: int = 30) -> list[dict]:
    """Connected changed regions in full-frame (cropped) pixel coordinates."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ch.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        out.append({"box": [x / SCALE, y / SCALE, (x + w) / SCALE, (y + h) / SCALE], "area": a / SCALE ** 2})
    return sorted(out, key=lambda b: -b["area"])
