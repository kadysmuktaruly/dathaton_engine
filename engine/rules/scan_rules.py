"""Rules that combine the VLM desk scan (object/action flags per sampled frame) with
perception events (departures, entries, head counts).

B1  cash lying open on the desk when the cashier walks away or a new client comes in
B3  personal bag/coat at the cash desk
D1  green collection bag brought in by another person
B7  the cashier alone counts banknotes by hand (no client, no second employee)
X0  drink at the cash desk while banknotes are on it (catch-all, low score)
B5  safe door open (scan flag) while the cashier is away from it

Timecodes: perception event times (departure/entry) when the rule is tied to one; otherwise the
first positive scan frame, refined by bisection with extra VLM calls (engine.vlm_scan).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.events import Event
from engine.facts import Facts
from engine.rules.leave import departures

NEAR_S = 2.5          # TUNE: scan frame counts as 'at' an event if within this many seconds
MIN_HITS = 2          # TUNE: consecutive positive scans for persistent states


def _runs(scan: pd.DataFrame, flag: str, min_hits: int) -> list[tuple[float, float, float | None]]:
    """(t_first, t_last, t_prev_negative) for runs of >= min_hits consecutive positives."""
    out = []
    ts, v = scan.t.to_numpy(), scan[flag].to_numpy()
    i = 0
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


def _flag_near(scan, flag, t, near=NEAR_S) -> bool:
    m = (scan.t - t).abs() <= near
    return bool(scan.loc[m, flag].any())


def _n_at(f: Facts, t: float) -> int:
    fr = f.frames
    i = int(np.argmin(np.abs(fr.t.to_numpy() - t)))
    return int(fr.n_smooth.iloc[i])


def _entries(f: Facts, min_prev: int = 1) -> list[float]:
    """Times the head count goes up from >= min_prev (someone joins people already there)."""
    fr = f.frames
    t, n = fr.t.to_numpy(), fr.n_smooth.to_numpy()
    return [float(t[i]) for i in range(1, len(t)) if n[i] > n[i - 1] and n[i - 1] >= min_prev and t[i] > 1.0]


def detect(f: Facts) -> list[Event]:
    scan = getattr(f, "scan", None)
    if scan is None or len(scan) == 0:
        return []
    ev: list[Event] = []
    deps = departures(f)

    # B1: cash on the desk when the cashier leaves, or when a new client comes in
    for d in deps:
        if _flag_near(scan, "banknotes_on_desk", d["t_depart"]) or _flag_near(scan, "banknotes_on_desk", d["t_exit"]):
            t = d["t_depart"]
            ev.append(Event("B1", t, d["t_exit"], 0.75, evidence_t=[t], boxes={t: [("кассир", d["box"])]},
                            info={"reason": "cash_left_cashier_away",
                                  "seen": "на столе кассира остались наличные деньги, кассир отошёл от рабочего места"}))
    for t in _entries(f, min_prev=1):
        before = scan[(scan.t <= t) & (scan.t >= t - 4.0)]
        if len(before) and before.banknotes_on_desk.any():
            ev.append(Event("B1", t, t + 2, 0.6, evidence_t=[t],
                            info={"reason": "cash_open_new_client",
                                  "seen": "на столе открыто лежат наличные деньги, когда к кассе подходит следующий клиент"}))

    # B3: personal bag or coat at the desk (not the green collection bag)
    for t0, t1, tprev in _runs(scan, "handbag_or_coat", MIN_HITS):
        seg = scan[(scan.t >= t0) & (scan.t <= t1)]
        if seg.green_bag.mean() > 0.5:
            continue
        ev.append(Event("B3", t0, t1, 0.6, evidence_t=[t0],
                        info={"reason": "personal_items", "refine": ("handbag_or_coat", tprev),
                              "seen": "на рабочем месте кассира лежат личные вещи (сумка или верхняя одежда)"}))

    # D1: green collection bag appears while another person is in the room
    for t0, t1, tprev in _runs(scan, "green_bag", 1):
        if _n_at(f, t0) >= 2:
            ev.append(Event("D1", t0, t1, 0.6, evidence_t=[t0],
                            info={"reason": "bag_taken", "refine": ("green_bag", tprev),
                                  "seen": "кассир принимает инкассаторскую сумку и сразу убирает её, осмотр пломбы и целостности сумки в кадре не виден"}))
            break

    # B7: alone and counting banknotes by hand
    for t0, t1, tprev in _runs(scan, "counting_by_hand", MIN_HITS):
        if _n_at(f, t0) == 1 and _n_at(f, t1) == 1:
            ev.append(Event("B7", t0, t1, 0.5, evidence_t=[t0],
                            info={"reason": "counting_alone", "refine": ("counting_by_hand", tprev),
                                  "seen": "кассир один пересчитывает наличные деньги вручную, контролирующего работника в кадре нет"}))

    # X0: drink at the desk next to banknotes
    both = scan.drink_cup & scan.banknotes_on_desk
    if both.any():
        t0 = float(scan.t[both].iloc[0])
        ev.append(Event("X0", t0, t0 + 2, 0.35, evidence_t=[t0],
                        info={"reason": "drink_near_cash",
                              "seen": "кассир пьёт или ставит напиток на стол рядом с наличными деньгами"}))

    # B5 (open variant): safe door open in persistent scans
    for t0, t1, tprev in _runs(scan, "safe_open", MIN_HITS):
        ev.append(Event("B5", t0, t1, 0.55, evidence_t=[t0],
                        info={"reason": "safe_open", "refine": ("safe_open", tprev),
                              "seen": "дверца сейфа остаётся открытой без присмотра"}))
    return ev
