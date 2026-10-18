"""Collection-bag rules (docs/master-document.md, phase 5).

D7  the sealed bag is taken without checking the collector's ID / power of attorney: during the
    collector's visit no ID card is seen moving (clean X-16: the collector shows an ID for ~6 s;
    X-17: nothing). Timecode = the moment the cashier pulls the bag in (the bag box moves
    clearly toward the cashier's seat).
D6  the bag's cash goes into the safe with no count: the last lasting 'safe open' stretch after
    the bag was seen comes with almost no time at the counting machine since the bag appeared
    (X-13: 0 s; clean X-14: ~15 s at the machine first). Timecode = start of that stretch.
"""
from __future__ import annotations

import numpy as np

from engine.events import Event
from engine.facts import Facts
from engine.roles import dwell
from engine.rules.client import moving, seconds

D7_MAX_ID_S = 1           # TUNE: <= this many seconds with a moving id_card = no ID check seen
BAG_PULL_DROP = 0.25      # TUNE: bag-to-seat distance drop (desk widths) = cashier pulled it in
D6_MIN_BAG_S = 3          # TUNE: seconds with the bag before the safe stretch
D6_MAX_MACHINE_S = 3.0    # TUNE: machine time since the bag appeared below this = not counted
SAFE_RUN_MIN_S = 3        # TUNE: consecutive seconds with >= 2 safe_open boxes
SAFE_RUN_PER_S = 2

D7_SCORE, D6_SCORE = 0.6, 0.55


def safe_open_runs(f: Facts, min_len: int = SAFE_RUN_MIN_S) -> list[tuple[int, int]]:
    """(first, last) second of stretches where the safe is seen open (>= 2 boxes per second)."""
    o = f.objects
    if o.empty or "cls" not in o:
        return []
    so = o[o.cls == "safe_open"]
    per = so.groupby(np.floor(so.t)).size()
    secs = sorted(int(k) for k, v in per.items() if v >= SAFE_RUN_PER_S)
    out, i = [], 0
    while i < len(secs):
        j = i
        while j + 1 < len(secs) and secs[j + 1] - secs[j] <= 2:  # one weak second is bridged
            j += 1
        if secs[j] - secs[i] + 1 >= min_len:
            out.append((secs[i], secs[j]))
        i = j + 1
    return out


def _bag_taken(f: Facts, t0: float, t1: float) -> float | None:
    o = f.objects
    if o.empty or f.seat is None or f.scene.desk is None:
        return None
    b = o[(o.cls == "collection_bag") & (o.t >= t0) & (o.t <= t1)].sort_values("t")
    if b.empty:
        return None
    sx, sy = f.seat
    d = np.hypot((b.x0 + b.x1) / 2 - sx, (b.y0 + b.y1) / 2 - sy) / max(f.scene.desk.w, 1.0)
    far = float(d.max())
    near = b[d.to_numpy() <= far - BAG_PULL_DROP]
    return float(near.t.iloc[0]) if len(near) else None


def d7(f: Facts) -> list[Event]:
    ids = moving(f, "id_card")
    out = []
    for s in f.sessions or []:
        if s.role != "collector":
            continue
        n_id = len(seconds(ids, s.enter - 1.0, s.leave + 1.0))
        if n_id > D7_MAX_ID_S:
            continue
        t = _bag_taken(f, s.enter - 2.0, s.leave + 2.0)
        t = t if t is not None else s.enter
        out.append(Event("D7", t, s.leave, D7_SCORE, evidence_t=[t],
                         boxes={t: [("инкассатор", s.first_box)]} if s.first_box else {},
                         info={"reason": "bag_taken_no_id_check", "collector": s.idx, "id_card_s": n_id}))
    return out


def d6(f: Facts) -> list[Event]:
    o = f.objects
    if o.empty or "cls" not in o:
        return []
    bag = o[o.cls == "collection_bag"]
    if bag.empty:
        return []
    runs = safe_open_runs(f)
    if not runs:
        return []
    r0, r1 = runs[-1]  # the last stretch: cash goes in for storage
    bag_s = seconds(bag, 0.0, float(r0))
    if len(bag_s) < D6_MIN_BAG_S:
        return []
    t_bag = float(min(bag_s))
    m = dwell(f.hands, "machine", t_bag, float(r0)) if f.hands is not None else 0.0
    if f.scene.machine is None or m >= D6_MAX_MACHINE_S:
        return []
    t = float(r0)
    return [Event("D6", t, float(r1), D6_SCORE, evidence_t=[t],
                  info={"reason": "bag_cash_to_safe_no_count", "t_bag": t_bag, "machine_dwell_s": round(m, 2)})]


def detect(f: Facts) -> list[Event]:
    return d7(f) + d6(f)
