"""C1: an outsider enters the cash room.

Three patterns, all seen in training:
  (a) someone enters while the room was empty for a while (X-32, X-33);
  (b) a third person arrives while the cashier and a client are already present (X-18);
  (c) a newcomer goes to the cashier's side of the desk next to the cashier (X-30).
A normal client entering while only the cashier is present is NOT flagged (X-3, X-35).

Timecode = first frame the newcomer is inside the room (catalog: «момент входа постороннего»).
"""
from __future__ import annotations

import numpy as np

from engine.events import Event
from engine.facts import Facts

EMPTY_BEFORE_S = 2.0   # TUNE: room must be empty this long before an entry counts as (a)
START_GRACE_S = 1.0    # TUNE: people already there at the start are not entries
MIN_STAY_S = 1.0       # TUNE: newcomer must stay this long (filters flicker)
STAFF_SIDE_D = 0.6     # TUNE: within this many desk-widths of the cashier seat = staff side


def detect(f: Facts) -> list[Event]:
    fr = f.frames
    if fr.empty:
        return []
    t = fr.t.to_numpy()
    n = fr.n_smooth.to_numpy()
    events: list[Event] = []

    # Entry moments: smoothed head count goes up.
    for i in range(1, len(t)):
        if n[i] <= n[i - 1] or t[i] < START_GRACE_S:
            continue
        # the count must hold for MIN_STAY_S
        j = i
        while j + 1 < len(t) and n[j + 1] >= n[i] and t[j + 1] - t[i] < MIN_STAY_S:
            j += 1
        if t[j] - t[i] < MIN_STAY_S - 0.4 and j + 1 < len(t):
            continue
        before = n[(t >= t[i] - EMPTY_BEFORE_S) & (t < t[i])]
        prev = int(np.max(before)) if len(before) else 0
        newcomer = _newcomer(f, t[i])
        t_in = newcomer["t_first"] if newcomer else t[i]
        reason = None
        score = 0.0
        if prev == 0 and t[i] >= EMPTY_BEFORE_S:
            reason, score = "entered_empty_room", 0.7
        elif n[i] >= 3:
            reason, score = "third_person", 0.75
        elif newcomer and newcomer["min_d_seat"] <= STAFF_SIDE_D and newcomer["cashier_present"]:
            reason, score = "to_staff_side", 0.8
        if reason:
            boxes = {}
            if newcomer:
                boxes[t_in] = [("посторонний", newcomer["box"])]
            events.append(Event("C1", float(t_in), float(t_in + MIN_STAY_S), score,
                                evidence_t=[float(t_in)], boxes=boxes,
                                info={"reason": reason, "count": int(n[i]), "prev": prev,
                                      "track": newcomer["track"] if newcomer else None}))
    return events


def _newcomer(f: Facts, t_up: float) -> dict | None:
    """The in-room track that first appears closest to the moment the count went up."""
    p = f.persons
    if p.empty:
        return None
    first = p.groupby("track_id").t.min()
    cand = first[(first >= t_up - 1.5) & (first <= t_up + 0.5)]
    if cand.empty:
        return None
    tr = cand.sub(t_up).abs().idxmin()
    rows = p[(p.track_id == tr)]
    later = rows[rows.t <= rows.t.min() + 6.0]
    r0 = rows.sort_values("t").iloc[0]
    others = p[(p.t >= later.t.min()) & (p.t <= later.t.max()) & (p.track_id != tr)]
    return {"track": int(tr), "t_first": float(r0.t),
            "box": [float(r0.x0), float(r0.y0), float(r0.x1), float(r0.y1)],
            "min_d_seat": float(later.d_seat.min()),
            "cashier_present": bool((others.d_seat <= STAFF_SIDE_D).any())}
