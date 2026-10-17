"""C1: an outsider enters the cash area.

Patterns seen in training:
  (b) a third person arrives while the cashier and a client are already present (X-18);
  (c) a newcomer comes to stand beside the cashier with no desk between them (X-30).
Normal clients stay across the desk: the line between their feet and the cashier's runs
through the desk (checked on all clean clips: 0-2% of frames 'beside' vs 100% for X-30).
Entering an empty room and leaving the door open is C2.

Timecode = the moment the newcomer reaches the cashier (labels mark this, e.g. X-18 at 20 s
when the visitor hands over money; the catalog says «момент входа ... в зону кассы»).
"""
from __future__ import annotations

import numpy as np

from engine.events import Event
from engine.facts import Facts

BESIDE_SEG_IN_DESK = 0.3   # TUNE: share of the feet-to-feet line inside the desk box
BESIDE_DIST_H = 1.0        # TUNE: feet distance in person heights
BESIDE_MIN_S = 2.0         # TUNE
BESIDE_DEPTH_H = 0.4       # TUNE: |foot y difference| in person heights (side by side, not behind)
THIRD_HOLD_S = 3.0         # TUNE: 3+ people must hold this long
REACH_DIST_H = 1.3         # TUNE: 'reached the cashier' distance for the timecode


def _seg_in_box(a, b, box, n=20) -> float:
    return float(np.mean([box.contains(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
                          for k in range(n + 1)]))


def _overlap(a, b) -> float:
    """Intersection over the smaller box (a split box of the same person overlaps a lot)."""
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    small = min((a.x1 - a.x0) * (a.y1 - a.y0), (b.x1 - b.x0) * (b.y1 - b.y0))
    return ix * iy / max(small, 1e-6)


def _cashier_track(f: Facts):
    p = f.persons
    return None if p.empty else p.groupby("track_id").t.nunique().idxmax()


def detect(f: Facts) -> list[Event]:
    p = f.persons
    if p.empty or f.scene.desk is None:
        return []
    h = max(f.scene.person_h_ref, 1.0)
    desk = f.scene.desk
    cashier = _cashier_track(f)
    events = []

    # (c) beside the cashier
    for tr, rows in p.groupby("track_id"):
        if tr == cashier:
            continue
        beside_t = []
        for r in rows.itertuples():
            others = p[(p.t == r.t) & (p.track_id != tr)]
            if others.empty:
                continue
            o = others.iloc[int(np.argmin(np.hypot(others.fx - r.fx, others.fy - r.fy)))]
            dist = np.hypot(r.fx - o.fx, r.fy - o.fy) / h
            ov = _overlap(r, o)
            same_depth = abs(r.fy - o.fy) / h < BESIDE_DEPTH_H
            if ov < 0.3 and same_depth and dist < BESIDE_DIST_H and _seg_in_box((r.fx, r.fy), (o.fx, o.fy), desk) < BESIDE_SEG_IN_DESK:
                beside_t.append(r.t)
        if len(beside_t) >= max(2, int(BESIDE_MIN_S * 3)):
            t0 = float(min(beside_t))
            r0 = rows.iloc[int(np.argmin(np.abs(rows.t - t0)))]
            events.append(Event("C1", t0, float(max(beside_t)), 0.8, evidence_t=[t0],
                                boxes={t0: [("посторонний", [r0.x0, r0.y0, r0.x1, r0.y1])]},
                                info={"reason": "to_staff_side", "track": int(tr)}))

    # (b) a third person
    fr = f.frames
    t = fr.t.to_numpy()
    n = fr.n_smooth.to_numpy()
    i = 0
    while i < len(t):
        if n[i] >= 3:
            j = i
            while j + 1 < len(t) and n[j + 1] >= 3:
                j += 1
            if t[j] - t[i] >= THIRD_HOLD_S:
                ev = _third(f, t[i], t[j], cashier, h)
                if ev:
                    events.append(ev)
            i = j + 1
        else:
            i += 1
    return events


def _third(f: Facts, t_up: float, t_end: float, cashier, h: float) -> Event | None:
    p = f.persons
    first = p.groupby("track_id").t.min()
    cand = first[(first >= t_up - 3.0) & (first <= t_up + 0.5)]
    cand = cand[cand.index != cashier]
    if cand.empty:
        return Event("C1", float(t_up), float(t_end), 0.6, evidence_t=[float(t_up)],
                     info={"reason": "third_person"})
    tr = cand.sub(t_up).abs().idxmin()
    rows = p[p.track_id == tr].sort_values("t")
    cash = p[p.track_id == cashier]
    t_reach = None
    for r in rows.itertuples():
        c = cash[(cash.t - r.t).abs() <= 0.2]
        if len(c) and np.hypot(r.fx - c.fx.iloc[0], r.fy - c.fy.iloc[0]) / h < REACH_DIST_H:
            t_reach = float(r.t)
            break
    t0 = t_reach if t_reach is not None else float(rows.t.iloc[0])
    r0 = rows.iloc[int(np.argmin(np.abs(rows.t - t0)))]
    return Event("C1", t0, float(t_end), 0.75, evidence_t=[t0],
                 boxes={t0: [("посторонний", [r0.x0, r0.y0, r0.x1, r0.y1])]},
                 info={"reason": "third_person", "track": int(tr), "t_enter": float(rows.t.iloc[0])})
