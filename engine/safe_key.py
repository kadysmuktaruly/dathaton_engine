"""Is the key left in the safe lock? (docs/master-document.md: B5, "close the safe and take the key")

The key is a few pixels at 464x848, too small for the detector. So we look at the lock directly:

1. Lock: the safe box (fine-tuned detector) gives a search area on the door's left part. The lock is
   the strongest small dark dot there: darker than a ring of pixels around it in every direction.
   A ring test, not a plain dark-pixel test, so the dark vertical door edge is not taken for it.
2. Key: on the side of the lock where the key head sticks out (left of it in these camera views),
   each lock row is compared with the rows above, the rows below and the line between them.
   Without a key the door edge, the gap and the background are lines (vertical, slanted or a
   horizontal edge), so a lock row matches at least one of the three. A key matches none.
   Tuned on the empty room after each departure: key left X-11, X-22 -> 0.085-0.095;
   no key X-12, X-13, X-15, X-23..X-29 -> 0.007-0.020.

Only used on frames where the room is empty (after the cashier leaves): averaging them gives a
clean picture of the lock, and the key may stay in the lock while the cashier is in the room.
All sizes scale with the safe box width (KEY_REF_W = the safe width this was tuned on).
"""
from __future__ import annotations

import numpy as np

KEY_REF_W = 73.0                 # safe box width (px) the sizes below are given for
SEARCH_X = (0.0, 0.25)           # lock search area, share of the safe box (x from the left edge)
SEARCH_Y = (0.40, 0.68)          # (y from the top)
RING_R = 3                       # px at KEY_REF_W: ring radius around a lock candidate
MIN_DOT = 0.12                   # TUNE: dot contrast (share of the door brightness) to trust the lock
KEY_COLS = (2, 6)                # px at KEY_REF_W: columns left of the lock centre checked for the key
KEY_ROWS = 1                     # px: lock rows = centre +- this
REF_ROWS = (3, 6)                # px: reference rows this far above and below the lock centre
KEY_MIN = 0.045                  # TUNE: key score above this = key in the lock


def _ring_offsets(r: int) -> list[tuple[int, int]]:
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    return sorted({(int(round(r * np.cos(a))), int(round(r * np.sin(a)))) for a in angles})


def find_lock(gray: np.ndarray, safe) -> tuple[int, int, float] | None:
    """(x, y, dot contrast) of the lock in a grey frame, or None. `safe` is a scene Box."""
    k = max(safe.w / KEY_REF_W, 0.5)
    r = max(2, int(round(RING_R * k)))
    H, W = gray.shape
    x0 = max(int(safe.x0 + SEARCH_X[0] * safe.w), r)
    x1 = min(int(safe.x0 + SEARCH_X[1] * safe.w), W - r - 1)
    y0 = max(int(safe.y0 + SEARCH_Y[0] * safe.h), r)
    y1 = min(int(safe.y0 + SEARCH_Y[1] * safe.h), H - r - 1)
    if x1 <= x0 or y1 <= y0:
        return None
    g = gray.astype(np.float32)
    c = g[y0:y1, x0:x1]
    ring = np.stack([g[y0 + dy:y1 + dy, x0 + dx:x1 + dx] for dx, dy in _ring_offsets(r)])
    # 2nd-darkest ring sample: one dark neighbour (a key, a shadow) is allowed, a line is not
    score = (np.sort(ring, axis=0)[1] - c) / max(float(np.median(c)), 1.0)
    iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
    return x0 + int(ix), y0 + int(iy), float(score[iy, ix])


def key_score(gray: np.ndarray, lock: tuple[int, int, float], safe) -> float:
    """Largest mean difference of a lock row from its best reference (rows above, rows below or
    the line between them), left of the lock, as a share of the door brightness."""
    k = max(safe.w / KEY_REF_W, 0.5)
    lx, ly, _ = lock
    g = gray.astype(np.float32)
    ca, cb = lx - int(round(KEY_COLS[1] * k)), lx - int(round(KEY_COLS[0] * k)) + 1
    a, c = int(round(REF_ROWS[0] * k)), int(round(REF_ROWS[1] * k))
    kr = max(1, int(round(KEY_ROWS * k)))
    if ca < 0 or ly - c < 0 or ly + c + 1 > g.shape[0] or lx + 8 * k > g.shape[1]:
        return 0.0
    up = g[ly - c:ly - a + 1, ca:cb].mean(axis=0)
    dn = g[ly + a:ly + c + 1, ca:cb].mean(axis=0)
    mid = (a + c) / 2
    worst = 0.0
    for dy in range(-kr, kr + 1):
        row = g[ly + dy, ca:cb]
        w = (dy + mid) / (2 * mid)
        line = (1 - w) * up + w * dn
        d = np.minimum.reduce([np.abs(row - up), np.abs(row - dn), np.abs(row - line)])
        worst = max(worst, float(d.mean()))
    door = float(np.median(g[ly - kr:ly + kr + 1, lx + 2:lx + 2 + int(round(6 * k))]))
    return worst / max(door, 1.0)


def check(video: str, perc, safe, times: list[float]) -> dict | None:
    """Key-in-lock check on the median of the given (empty-room) frames.
    Returns {'key': bool, 'score', 'dot', 'lock': [x, y], 'n_frames'} or None (no lock found)."""
    import cv2
    from engine.video import iter_frames_at
    frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for _, _, f in iter_frames_at(video, times, perc.info.crop)]
    if not frames or safe is None:
        return None
    med = np.median(np.stack(frames), axis=0).astype(np.float32)
    lock = find_lock(med, safe)
    if lock is None or lock[2] < MIN_DOT:
        return None
    s = key_score(med, lock, safe)
    return {"key": s > KEY_MIN, "score": round(s, 4), "dot": round(lock[2], 3),
            "lock": [lock[0], lock[1]], "n_frames": len(frames)}
