"""Client-service rules (docs/master-document.md, phases 3-4 and process-wide checks).

Built on visitor sessions (engine.roles): one session per client or collector.

A7  a second client is served while the first is still at the desk (two client sessions overlap
    and the newcomer reaches the cashier) - X-18
B1  cash already lying on the desk when a new client arrives: it is not the current client's - X-4, X-38
A1  the client leaves while the cashier is still at the counting machine - X-34
A6  a client operation with cash but the client's ID card is never seen moving - X-2.
    Kept below the verdict threshold: the detector misses most ID cards (val AP 0.29), clean
    services (X-19, X-34, X-35, X-39) look the same, and the 4B VLM cannot tell an ID card from
    a banknote at this resolution. It stays in the debug output as a candidate.
A2  a UV detector stands on the desk, the cashier handles cash, and the notes never go to the
    detector (VLM) - X-44 vs X-43
A4  (optional) the cashier leaves the room while a client is still there with cash on the desk

Detector classes are noisy in places (a shelf item seen as 'id_card' or 'handbag' in every
frame), so object evidence ignores static boxes: a box seen at the same place in most frames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.events import Event
from engine.facts import Facts
from engine.roles import Session, dwell, session_rows

STATIC_FRAC = 0.5          # TUNE: a box at the same place in >= this share of frames is furniture
STATIC_IOU = 0.5
A7_OVERLAP_S = 2.0         # TUNE: both clients in the room at least this long
REACH_DIST_H = 1.3         # TUNE: newcomer's feet within this many person heights of the cashier
B1_BEFORE_S, B1_AFTER_S = 4.0, 1.0
B1_LEFTOVER_AFTER_S = 3.0  # TUNE
B1_SAME_IOU = 0.3          # TUNE: same bundle, same place
A1_PAD_BEFORE, A1_PAD_AFTER = 1.0, 2.5
A1_MIN_AFTER_S = 1.5       # TUNE: counting goes on this long after the client left (X-34 2.3 s; clean X-39 0.7 s)
A6_MIN_CASH_S = 3          # TUNE: distinct seconds with cash during the visit
A6_MAX_ID_S = 2            # TUNE: <= this many distinct seconds with a moving id_card = no ID shown
A6_MIN_VISIT_S = 5.0
A2_MIN_RUN_S = 5          # TUNE: consecutive seconds with cash = a counting stretch (X-44: 6 s; clean X-8: 4 s, a phone at face level)
A4_SEAT_EMPTY_S = 3.0

A7_SCORE, B1_SCORE, B1_FIRST_SCORE, A1_SCORE, A2_SCORE, A4_SCORE = 0.75, 0.6, 0.45, 0.5, 0.5, 0.5
A6_SCORE = 0.1  # candidate only: stays below VERDICT_THRESHOLD even after a VLM 'yes' (+0.15 max)


# --- object evidence ---------------------------------------------------------------------------

def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def moving(f: Facts, cls: str) -> pd.DataFrame:
    """Confident detections of `cls` minus static ones (same place in most frames)."""
    o = f.objects
    if o.empty or "cls" not in o:
        return o.iloc[0:0]
    d = o[o.cls == cls]
    if d.empty:
        return d
    n_frames = max(len(f.frames), 1)
    boxes = d[["x0", "y0", "x1", "y1"]].to_numpy()
    ts = d.t.to_numpy()
    static = np.zeros(len(d), bool)
    for i in range(len(d)):
        same = [j for j in range(len(d)) if _iou(boxes[i], boxes[j]) >= STATIC_IOU]
        if len(set(ts[same])) >= STATIC_FRAC * n_frames:
            static[i] = True
    return d[~static]


def seconds(df: pd.DataFrame, t0: float, t1: float) -> set[int]:
    if df.empty:
        return set()
    return set(np.floor(df[(df.t >= t0) & (df.t <= t1)].t.to_numpy()).astype(int).tolist())


def _runs(secs: set[int], min_len: int) -> list[tuple[int, int]]:
    out, s = [], sorted(secs)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[j] + 1:
            j += 1
        if s[j] - s[i] + 1 >= min_len:
            out.append((s[i], s[j]))
        i = j + 1
    return out


def clients(f: Facts) -> list[Session]:
    return [s for s in (f.sessions or []) if s.role == "client"]


# --- rules ---------------------------------------------------------------------------------------

def a7(f: Facts) -> list[Event]:
    cl = clients(f)
    p = f.persons
    h = max(f.scene.person_h_ref, 1.0)
    out = []
    for s1 in cl:
        for s2 in cl:
            if s2.enter <= s1.enter + 1.0:
                continue
            overlap = min(s1.leave, s2.leave) - s2.enter
            if overlap < A7_OVERLAP_S:
                continue
            rows = session_rows(p, s2).sort_values("t")
            cash = p[p.role == "cashier"]
            t_reach = None
            for r in rows.itertuples():
                c = cash[(cash.t - r.t).abs() <= 0.2]
                if len(c) and np.hypot(r.fx - c.fx.iloc[0], r.fy - c.fy.iloc[0]) / h < REACH_DIST_H:
                    t_reach = float(r.t)
                    break
            t = t_reach if t_reach is not None else s2.enter
            box = s2.first_box if t_reach is None else \
                rows.iloc[int(np.argmin(np.abs(rows.t.to_numpy() - t)))][["x0", "y0", "x1", "y1"]].tolist()
            out.append(Event("A7", t, min(s1.leave, s2.leave), A7_SCORE, evidence_t=[t],
                             boxes={t: [("второй клиент", box)]} if box else {},
                             info={"reason": "second_client_while_first_present", "client1": s1.idx,
                                   "client2": s2.idx, "t_enter2": s2.enter, "overlap_s": round(overlap, 1)}))
            return out  # one per clip is enough for the verdict; the first overlap is the moment
    return out


def _cash_on_desk(f: Facts, t0: float, t1: float) -> bool:
    scan = f.scan
    if scan is not None and len(scan) and "banknotes_on_desk" in scan:
        w = scan[(scan.t >= t0) & (scan.t <= t1)]
        if len(w) and w.banknotes_on_desk.any():
            return True
    cash = _not_in_machine(f, moving(f, "cash"))  # notes in the machine are being counted, not lying
    return len(seconds(cash, t0, t1)) >= 2


def _not_in_machine(f: Facts, cash: pd.DataFrame) -> pd.DataFrame:
    m = f.scene.machine
    if cash.empty or m is None:
        return cash
    mb = m.expand(0.1)
    return cash[[not mb.contains((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2) for r in cash.itertuples()]]


def _leftover_cash(f: Facts, t_from: float, t_enter: float) -> bool:
    """The same bundle lies on the desk before and after the new client arrives (box overlap):
    the previous client's cash was not put away (X-38: one bundle from 26.7 s to 34.7 s)."""
    cash = _not_in_machine(f, moving(f, "cash"))
    if cash.empty:
        return False
    before = cash[(cash.t >= t_from) & (cash.t <= t_enter)][["x0", "y0", "x1", "y1"]].to_numpy()
    after = cash[(cash.t > t_enter) & (cash.t <= t_enter + B1_LEFTOVER_AFTER_S)][["x0", "y0", "x1", "y1"]].to_numpy()
    return any(_iou(a, b) >= B1_SAME_IOU for a in before for b in after)


def b1_new_client(f: Facts) -> list[Event]:
    out = []
    cl = clients(f)
    for s in cl:
        if s.enter < 1.0:
            continue  # already there when the clip starts: arrival not seen
        prev = [q for q in cl if q.enter < s.enter]
        t_from = max(q.leave for q in prev) - 1.0 if prev else s.enter - B1_BEFORE_S
        if not (_cash_on_desk(f, s.enter - B1_BEFORE_S, s.enter + B1_AFTER_S) or _leftover_cash(f, t_from, s.enter)):
            continue
        score = B1_SCORE if prev else B1_FIRST_SCORE
        out.append(Event("B1", s.enter, s.enter + 2.0, score, evidence_t=[s.enter],
                         boxes={s.enter: [("клиент", s.first_box)]} if s.first_box else {},
                         info={"reason": "cash_open_new_client", "client": s.idx, "previous_clients": len(prev)}))
    return out


def a1(f: Facts) -> list[Event]:
    out = []
    if f.hands is None or f.scene.machine is None:
        return out
    for s in clients(f):
        if not s.ended or s.duration < A6_MIN_VISIT_S:
            continue
        before = dwell(f.hands, "machine", s.leave - A1_PAD_BEFORE, s.leave)
        after = dwell(f.hands, "machine", s.leave, s.leave + A1_PAD_AFTER)
        if before <= 0 or after < A1_MIN_AFTER_S:
            continue
        busy = before + after
        out.append(Event("A1", s.leave, s.leave + A1_PAD_AFTER, A1_SCORE, evidence_t=[s.leave],
                         info={"reason": "client_left_during_count", "client": s.idx,
                               "machine_dwell_s": round(busy, 2)}))
    return out


def a6(f: Facts) -> list[Event]:
    out = []
    cash, ids = moving(f, "cash"), moving(f, "id_card")
    for s in clients(f):
        if s.duration < A6_MIN_VISIT_S:
            continue
        cs = seconds(cash, s.enter, s.leave + 1.0)
        if len(cs) < A6_MIN_CASH_S:
            continue
        ns = seconds(ids, s.enter, s.leave + 1.0)
        if len(ns) > A6_MAX_ID_S:
            continue
        t = float(min(cs))
        c = cash[(cash.t >= t) & (cash.t < t + 1)].sort_values("conf", ascending=False)
        boxes = {t: [("наличные", c.iloc[0][["x0", "y0", "x1", "y1"]].tolist())]} if len(c) else {}
        out.append(Event("A6", t, s.leave, A6_SCORE, evidence_t=[t], boxes=boxes,
                         info={"reason": "no_client_id", "client": s.idx, "cash_s": len(cs), "id_card_s": len(ns)}))
    return out


def a2(f: Facts) -> list[Event]:
    """Needs the VLM check (UV detector present and notes never at it)."""
    uv = (f.checks or {}).get("uv")
    if not uv or not uv.get("present") or uv.get("used"):
        return []
    run = uv.get("cash_run")
    if not run:
        return []
    t = float(run[0])
    return [Event("A2", t, float(run[1]), A2_SCORE, evidence_t=[t],
                  info={"reason": "uv_detector_not_used", "uv_frames": uv.get("frames")})]


def cash_runs(f: Facts, min_len: int = A2_MIN_RUN_S) -> list[tuple[int, int]]:
    """Stretches of consecutive seconds with (moving) cash: candidate counting episodes."""
    cash = moving(f, "cash")
    return _runs(seconds(cash, 0.0, 1e9), min_len)


def a4(f: Facts) -> list[Event]:
    """Cashier leaves the room while a client stays, with cash on the desk."""
    fr = f.frames
    out = []
    for s in clients(f):
        w = fr[(fr.t >= s.enter) & (fr.t <= s.leave)]
        if w.empty:
            continue
        alone = (w.n_smooth <= 1) & ~w.cashier_at_seat
        t_arr, a = w.t.to_numpy(), alone.to_numpy()
        i = 0
        while i < len(t_arr):
            if a[i]:
                j = i
                while j + 1 < len(t_arr) and a[j + 1]:
                    j += 1
                if t_arr[j] - t_arr[i] >= A4_SEAT_EMPTY_S and _cash_on_desk(f, t_arr[i] - 2.0, t_arr[i] + 2.0):
                    out.append(Event("A4", float(t_arr[i]), float(t_arr[j]), A4_SCORE, evidence_t=[float(t_arr[i])],
                                     info={"reason": "cashier_left_client_present", "client": s.idx}))
                    break
                i = j + 1
            else:
                i += 1
    return out


def detect(f: Facts) -> list[Event]:
    return a7(f) + b1_new_client(f) + a1(f) + a6(f) + a2(f) + a4(f)
