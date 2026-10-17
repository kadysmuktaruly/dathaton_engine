"""Video decoding with exact per-frame timestamps (clips are variable frame rate).

We decode sequentially with OpenCV, read each frame's own timestamp, and keep the frame
closest to each requested time. Black side/top bars are detected once per video and cropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float
    n_frames: int
    crop: tuple[int, int, int, int]  # x0, y0, x1, y1 of the non-black content


def detect_bars(frames: list[np.ndarray], thresh: int = 16) -> tuple[int, int, int, int]:
    """Find the content box by trimming near-black rows/cols that are black in every sample."""
    h, w = frames[0].shape[:2]
    mx = np.max(np.stack([f.max(axis=2) for f in frames]), axis=0)  # brightest value over samples
    cols = np.where((mx > thresh).mean(axis=0) > 0.02)[0]
    rows = np.where((mx > thresh).mean(axis=1) > 0.02)[0]
    if len(cols) == 0 or len(rows) == 0:
        return 0, 0, w, h
    x0, x1, y0, y1 = int(cols[0]), int(cols[-1]) + 1, int(rows[0]), int(rows[-1]) + 1
    # Only crop when the bar is substantial (>2% of the size); avoids nibbling dark edges.
    x0 = x0 if x0 > 0.02 * w else 0
    x1 = x1 if (w - x1) > 0.02 * w else w
    y0 = y0 if y0 > 0.02 * h else 0
    y1 = y1 if (h - y1) > 0.02 * h else h
    return x0, y0, x1, y1


def probe(path: str, n_samples: int = 8) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    samples = []
    for i in np.linspace(0, max(n - 2, 0), n_samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            samples.append(f)
    cap.release()
    crop = detect_bars(samples) if samples else (0, 0, w, h)
    return VideoInfo(w, h, n / fps, n, crop)


def iter_frames_at(path: str, times: Iterable[float], crop: tuple[int, int, int, int] | None = None
                   ) -> Iterator[tuple[float, float, np.ndarray]]:
    """Yield (requested_t, actual_frame_t, BGR frame) for each requested time, in order.

    The returned frame is the decoded frame whose own timestamp is closest to the request.
    """
    want = sorted(times)
    if not want:
        return
    cap = cv2.VideoCapture(str(path))
    i = 0
    prev_t, prev_f = None, None
    while i < len(want):
        ok, f = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        while i < len(want) and t >= want[i]:
            if prev_f is not None and abs(prev_t - want[i]) < abs(t - want[i]):
                yield want[i], prev_t, _crop(prev_f, crop)
            else:
                yield want[i], t, _crop(f, crop)
            i += 1
        prev_t, prev_f = t, f
    while i < len(want) and prev_f is not None:  # requests past the end get the last frame
        yield want[i], prev_t, _crop(prev_f, crop)
        i += 1
    cap.release()


def _crop(f: np.ndarray, crop):
    if crop is None:
        return f
    x0, y0, x1, y1 = crop
    return f[y0:y1, x0:x1]


def sample_times(duration: float, fps: float) -> list[float]:
    step = 1.0 / fps
    return [round(k * step, 4) for k in range(int(duration * fps) + 1) if k * step < duration]
