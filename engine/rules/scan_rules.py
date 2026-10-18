"""B3 from the VLM desk scan (object flags per sampled frame): a non-allowed item on the desk.

  - a personal bag or coat lying on the desk (X-40);
  - a drink brought to the desk mid-clip while cash lies there (X-46: coffee next to cash), from
    the scan or from the fine-tuned detector's cup boxes (it sees the cup at 11 s, the scan only
    at 14.5 s and misses the cash). A cup present from the start, or with no cash around, is not
    flagged (clean X-25).

Allowed items (counting machine, UV detector, laptop, wired phone, client's ID, cash, sealed
collection bag, stamps, journal) are not flags of the scan. B1 checks live in rules/depart.py
and rules/client.py; this module only keeps the shared run helper for them.

Timecode: the first positive scan frame, refined by bisection with extra VLM calls
(engine.vlm_scan.refine).
"""
from __future__ import annotations

import pandas as pd

from engine.events import Event
from engine.facts import Facts

MIN_HITS = 2               # TUNE: consecutive positive scans for persistent states
BAG_SCORE, DRINK_SCORE = 0.6, 0.45
CUP_MIN_S = 3              # TUNE: detector: cup seen in >= this many seconds
CUP_WITH_CASH_S = 2        # TUNE: ... and cash seen in >= this many of those seconds
CUP_LATE_S = 2.0           # the cup must appear after the clip start


def _runs(scan: pd.DataFrame, flag, min_hits: int) -> list[tuple[float, float, float | None]]:
    """(t_first, t_last, t_prev_negative) for runs of >= min_hits consecutive positives.
    `flag` may be a column name or a boolean Series."""
    v = (scan[flag] if isinstance(flag, str) else flag).to_numpy()
    ts = scan.t.to_numpy()
    out, i = [], 0
    while i < len(ts):
        if v[i]:
            j = i
            while j + 1 < len(ts) and v[j + 1]:
                j += 1
            if j - i + 1 >= min_hits:
                out.append((float(ts[i]), float(ts[j]), float(ts[i - 1]) if i > 0 else None))
            i = j + 1
        else:
            i += 1
    return out


def drink_from_detector(f: Facts) -> list[Event]:
    from engine.rules.client import moving, seconds
    cup = moving(f, "cup")
    if cup.empty:
        return []
    cs = seconds(cup, 0.0, 1e9)
    if len(cs) < CUP_MIN_S or min(cs) < CUP_LATE_S:
        return []
    with_cash = cs & seconds(moving(f, "cash"), 0.0, 1e9)
    if len(with_cash) < CUP_WITH_CASH_S:
        return []
    t = float(cup.t.min())
    r = cup.sort_values("t").iloc[0]
    return [Event("B3", t, float(cup.t.max()), DRINK_SCORE, evidence_t=[t],
                  boxes={t: [("напиток", [r.x0, r.y0, r.x1, r.y1])]},
                  info={"reason": "drink_next_to_cash", "cup_s": len(cs), "with_cash_s": len(with_cash)})]


def detect(f: Facts) -> list[Event]:
    ev: list[Event] = drink_from_detector(f)
    scan = getattr(f, "scan", None)
    if scan is None or len(scan) == 0 or "bag_or_coat_on_desk" not in scan:
        return ev
    for t0, t1, tprev in _runs(scan, "bag_or_coat_on_desk", MIN_HITS):
        seg = scan[(scan.t >= t0) & (scan.t <= t1)]
        if seg.green_bag_on_desk.mean() > 0.5:
            continue  # the collection bag is allowed
        ev.append(Event("B3", t0, t1, BAG_SCORE, evidence_t=[t0],
                        info={"reason": "personal_items", "refine": ("bag_or_coat_on_desk", tprev)}))
    for t0, t1, tprev in _runs(scan, "drink_cup", MIN_HITS):
        if tprev is None:
            continue  # present from the start: part of the setting, not an action
        seg = scan[(scan.t >= t0) & (scan.t <= t1)]
        if not seg.banknotes_on_desk.any() or any(e.info.get("reason") == "drink_next_to_cash" for e in ev):
            continue
        ev.append(Event("B3", t0, t1, DRINK_SCORE, evidence_t=[t0],
                        info={"reason": "drink_next_to_cash", "refine": ("drink_cup", tprev)}))
        break
    return ev
