"""Rules that combine the VLM desk scan (object flags per sampled frame) with perception events.

B1  cash (or a cash packet) lying on the desk while the cashier walks away, or lying open when
    the next client comes in
B3  personal bag or coat lying on the cash desk
D1  green collection bag handed over and the courier leaves at once (no inspection; clean X-16
    shows a ~10 s inspection while the courier waits, X-17 the courier leaves within ~3 s)
X0  a drink appears at the cash desk mid-clip (catch-all, low score)

Not done with the scan: B7 ('counting alone') and B5 'safe open' - the VLM flags for these were
true on most clean clips (scan v1), so they would only add false positives.

Timecodes: perception event times when tied to one; otherwise the first positive scan frame,
refined by bisection with extra VLM calls (engine.vlm_scan.refine).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.events import Event
from engine.facts import Facts
from engine.rules.leave import departures

MIN_HITS = 2               # TUNE: consecutive positive scans for persistent states
B1_RUN_BEFORE_DEPART = 8.0 # TUNE: cash run may start this long before the cashier leaves
COURIER_LEAVE_S = 7.0      # TUNE: courier gone within this time after the bag appears = no check


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


def _n_at(f: Facts, t: float) -> int:
    fr = f.frames
    return int(fr.n_smooth.iloc[int(np.argmin(np.abs(fr.t.to_numpy() - t)))])


def _entries(f: Facts, min_prev: int = 1) -> list[float]:
    fr = f.frames
    t, n = fr.t.to_numpy(), fr.n_smooth.to_numpy()
    return [float(t[i]) for i in range(1, len(t)) if n[i] > n[i - 1] and n[i - 1] >= min_prev and t[i] > 1.0]


def detect(f: Facts) -> list[Event]:
    scan = getattr(f, "scan", None)
    if scan is None or len(scan) == 0 or "bag_or_coat_on_desk" not in scan:
        return []
    ev: list[Event] = []
    deps = departures(f)
    cashlike = scan.banknotes_on_desk | scan.green_bag_on_desk

    # B1a: cash/packet on the desk and the cashier walks away
    for d in deps:
        for t0, t1, tprev in _runs(scan, cashlike, 1):
            if t0 <= d["t_exit"] + 1.0 and t1 >= d["t_depart"] - 2.0:
                start_in_window = d["t_depart"] - B1_RUN_BEFORE_DEPART <= t0 <= d["t_depart"]
                t = t0 if start_in_window else d["t_depart"]
                ev.append(Event("B1", t, d["t_exit"], 0.75, evidence_t=[t],
                                info={"reason": "cash_left_cashier_away", "t_depart": d["t_depart"],
                                      "refine": ("banknotes_on_desk", tprev) if start_in_window and scan.banknotes_on_desk.any() else None,
                                      "seen": "на столе кассира остались наличные деньги без присмотра, кассир отошёл от рабочего места"}))
                break
    # B1b: cash lying open when the next client comes in
    for t in _entries(f, min_prev=1):
        before = scan[(scan.t <= t + 1.0) & (scan.t >= t - 4.0)]
        if len(before) and before.banknotes_on_desk.any():
            ev.append(Event("B1", t, t + 2, 0.6, evidence_t=[t],
                            info={"reason": "cash_open_new_client",
                                  "seen": "на столе открыто лежат наличные деньги, когда к кассе подходит следующий клиент"}))

    # B3: personal bag or coat lying on the desk
    for t0, t1, tprev in _runs(scan, "bag_or_coat_on_desk", MIN_HITS):
        seg = scan[(scan.t >= t0) & (scan.t <= t1)]
        if seg.green_bag_on_desk.mean() > 0.5:
            continue
        ev.append(Event("B3", t0, t1, 0.6, evidence_t=[t0],
                        info={"reason": "personal_items", "refine": ("bag_or_coat_on_desk", tprev),
                              "seen": "на рабочем месте кассира лежат личные вещи (сумка или верхняя одежда)"}))

    # D1: collection bag handed over, courier leaves at once
    for t0, t1, tprev in _runs(scan, "green_bag_on_desk", 1):
        if _n_at(f, t0) < 2:
            continue
        fr = f.frames
        after = fr[(fr.t > t0) & (fr.t <= t0 + COURIER_LEAVE_S)]
        if len(after) and (after.n_smooth < _n_at(f, t0)).any():
            ev.append(Event("D1", t0, t1, 0.6, evidence_t=[t0],
                            info={"reason": "bag_taken_courier_left", "refine": ("green_bag_on_desk", tprev),
                                  "seen": "инкассатор передаёт сумку и сразу уходит; осмотр пломбы и целостности сумки кассиром в кадре не виден"}))
        break

    # X0: a drink appears at the desk mid-clip and stays
    for t0, t1, tprev in _runs(scan, "drink_cup", MIN_HITS):
        if tprev is None:
            continue  # present from the start: part of the setting, not an action
        ev.append(Event("X0", t0, t1, 0.4, evidence_t=[t0],
                        info={"reason": "drink_at_cash_desk", "refine": ("drink_cup", tprev),
                              "seen": "кассир берёт напиток на рабочем месте рядом с наличными деньгами"}))
        break
    return ev
