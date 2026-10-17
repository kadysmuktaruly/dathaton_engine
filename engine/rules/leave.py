"""Helpers for 'the cashier leaves the room' moments (used by B5 and others)."""
from __future__ import annotations

import numpy as np

from engine.facts import Facts

STILL_SPEED = 0.25  # TUNE: desk-widths per second; slower = standing/working
MIN_ABSENT_S = 1.5  # TUNE: room must stay empty this long (or until the clip ends)


def departures(f: Facts) -> list[dict]:
    """Moments when the last person leaves the room.

    Returns dicts: t_exit (last frame in the room), t_depart (last still moment before the
    walk out), absent_s, and the leaving person's last box.
    """
    fr = f.frames
    t = fr.t.to_numpy()
    zero = fr.n_smooth.to_numpy() == 0
    out = []
    for i in range(1, len(t)):
        if not (zero[i] and not zero[i - 1]):
            continue
        j = i
        while j + 1 < len(t) and zero[j + 1]:
            j += 1
        absent = t[j] - t[i] + (t[1] - t[0] if len(t) > 1 else 0)
        if absent < MIN_ABSENT_S and j != len(t) - 1:
            continue
        t_exit = float(t[i - 1])
        p = f.persons[(f.persons.t <= t_exit) & (f.persons.t >= t_exit - 12)]
        if p.empty:
            continue
        last = p[p.t == p.t.max()].iloc[0]
        tr = p[p.track_id == last.track_id].sort_values("t")
        t_dep = _depart_by_depth(p, t_exit, f.scene.person_h_ref)
        out.append({"t_exit": t_exit, "t_depart": t_dep, "absent_s": float(absent),
                    "box": [float(last.x0), float(last.y0), float(last.x1), float(last.y1)],
                    "track": int(last.track_id), "to_end": j == len(t) - 1})
    return out


def _extend_track(p, tr, unit):
    """Follow ID switches backwards: prepend the track that ended right before and nearby."""
    for _ in range(5):
        t0 = tr.t.min()
        r0 = tr.iloc[0]
        prev = p[(p.t < t0) & (p.t >= t0 - 1.0) & (~p.track_id.isin(tr.track_id.unique()))]
        if prev.empty:
            break
        d = np.hypot(prev.fx - r0.fx, prev.fy - r0.fy) / unit
        cand = prev[d <= 0.4]
        if cand.empty:
            break
        tid = cand.sort_values("t").iloc[-1].track_id
        tr = p[p.track_id == tid].sort_values("t")._append(tr) if hasattr(tr, "_append") else \
            __import__("pandas").concat([p[p.track_id == tid].sort_values("t"), tr])
    return tr.sort_values("t")


def _last_still(tr, unit, t_exit) -> float:
    ts = tr.t.to_numpy()
    xs, ys = tr.fx.to_numpy(), tr.fy.to_numpy()
    if len(ts) < 3:
        return float(ts[0])
    speed = np.zeros(len(ts))
    for k in range(len(ts)):
        m = np.abs(ts - ts[k]) <= 0.5
        idx = np.where(m)[0]
        a, b = idx[0], idx[-1]
        dt = max(ts[b] - ts[a], 1e-3)
        speed[k] = np.hypot(xs[b] - xs[a], ys[b] - ys[a]) / unit / dt
    still = np.where(speed <= STILL_SPEED)[0]
    if len(still) == 0:
        return float(ts[0])
    return float(ts[still[-1]])


def _depart_by_depth(p, t_exit: float, h_ref: float) -> float:
    """Start of the final walk to the door. With a high camera the door is at the back, so
    walking out moves the foot point up the image. Departure = last time the foot was at
    its usual working depth (median over the earlier part of the window)."""
    win = p[(p.t >= t_exit - 10) & (p.t <= t_exit)]
    if win.empty:
        return t_exit
    # one person in the room: take the lowest foot per frame (closest to camera)
    fy = win.groupby("t").fy.max().sort_index()
    early = fy[fy.index <= t_exit - 3]
    if early.empty:
        return float(fy.index[0])
    work_y = float(early.median())
    ok = fy[fy >= work_y - 0.15 * h_ref]  # TUNE
    return float(ok.index.max()) if len(ok) else float(fy.index[0])
