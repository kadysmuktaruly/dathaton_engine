"""Rules checked when the cashier leaves the room (docs/master-document.md, phase 6).

Before leaving, the cashier puts all cash and the collection bag into the safe (B1), puts the
stamps away (B8), closes the safe and takes the key (B5) and closes or locks the laptop (B9).
Each departure is checked separately; one departure may yield several violations.

Signals:
  B1  cash / collection bag on the desk: VLM desk scan run that spans the departure, or the
      departure check (engine.vlm_checks) seeing cash or the bag on the desk after the exit
  B5  confident safe_open detections while the room is empty
  B8  departure check: stamps left on the desk (answered, or listed among the desk items)
  B9  laptop screen lit after the exit (few dark pixels in the screen area); with the VLM, a
      lit screen showing the lock-screen photo is not flagged

Timecode = the moment the cashier finishes at the workplace: the later of the start of the
final walk out and the last moment their hands were at the safe / laptop / counting machine.
B1 keeps the start of the cash run when it began shortly before the departure.
"""
from __future__ import annotations

import numpy as np

from engine.events import Event
from engine.facts import Facts
from engine.rules.leave import departures

B1_RUN_BEFORE_DEPART = 8.0  # TUNE: a cash run may start this long before the cashier leaves
SAFE_OPEN_MIN_FRAMES = 2    # TUNE
ABSENCE_PAD_S = 0.5
SAME_DEPARTURE_S = 6.0      # TUNE: a 2nd 'departure' this soon after the 1st is the same exit (flicker)
TOUCH_WINDOW_S = 15.0
SCREEN_TOP = 0.6            # upper share of the laptop box = the screen
SCREEN_DARK_REL = 0.45      # pixel darker than this x frame mean = black screen
SCREEN_LIT_MAX_DARK = 0.18  # TUNE: lit screen <= this dark share (lit 0.04-0.12, black 0.24-0.36)
AFTER_EXIT_S = (1.0, 2.0)

B1_SCORE, B5_SCORE, B8_SCORE = 0.75, 0.6, 0.6
B9_SCORE, B9_LOCKED_SCORE = 0.6, 0.1


def unique_departures(f: Facts) -> list[dict]:
    out = []
    for d in departures(f):
        if out and d["t_exit"] - out[-1]["t_exit"] < SAME_DEPARTURE_S:
            continue
        out.append(d)
    return out


def leave_time(f: Facts, d: dict) -> float:
    """Later of the final-walk start and the last time the cashier's hands were at a device."""
    t = d["t_depart"]
    h = f.hands
    if h is not None and len(h):
        w = h[(h.t <= d["t_exit"]) & (h.t >= d["t_exit"] - TOUCH_WINDOW_S) & (h.machine | h.laptop | h.safe)]
        if len(w):
            t = max(t, float(w.t.max()))
    return float(t)


def _check_for(f: Facts, d: dict) -> dict | None:
    for c in (f.checks or {}).get("departures", []):
        if abs(c["t_exit"] - d["t_exit"]) < 0.05:
            return c
    return None


def _scan_runs(scan, flag) -> list[tuple[float, float, float | None]]:
    from engine.rules.scan_rules import _runs
    return _runs(scan, flag, 1)


def _b1(f: Facts, d: dict, t_leave: float, chk: dict | None) -> Event | None:
    scan = f.scan
    if scan is not None and len(scan) and "banknotes_on_desk" in scan:
        cashlike = scan.banknotes_on_desk | scan.green_bag_on_desk
        for t0, t1, tprev in _scan_runs(scan, cashlike):
            if t0 <= d["t_exit"] + 1.0 and t1 >= d["t_depart"] - 2.0:
                in_window = d["t_depart"] - B1_RUN_BEFORE_DEPART <= t0 <= d["t_depart"]
                t = t0 if in_window else d["t_depart"]
                refine = ("banknotes_on_desk", tprev) if in_window and scan.banknotes_on_desk.any() else None
                return Event("B1", t, d["t_exit"], B1_SCORE, evidence_t=[t],
                             info={"reason": "cash_left_cashier_away", "t_depart": d["t_depart"], "refine": refine})
    if chk and (chk.get("cash") or chk.get("collection_bag")):
        reason = "bag_left_cashier_away" if chk.get("collection_bag") and not chk.get("cash") else "cash_left_cashier_away"
        return Event("B1", d["t_depart"], d["t_exit"], B1_SCORE, evidence_t=[d["t_depart"]],
                     info={"reason": reason, "t_depart": d["t_depart"], "check": "departure"})
    return None


def _b5(f: Facts, d: dict, t_leave: float) -> Event | None:
    o = f.objects
    if o.empty or "cls" not in o:
        return None
    t1 = d["t_exit"] + d["absent_s"] + ABSENCE_PAD_S
    so = o[(o.cls == "safe_open") & (o.t > d["t_exit"]) & (o.t <= t1)]
    n = int(so.t.nunique()) if len(so) else 0
    if n < SAFE_OPEN_MIN_FRAMES:
        return None
    r = so.sort_values("conf", ascending=False).iloc[0]
    return Event("B5", t_leave, d["t_exit"], B5_SCORE, evidence_t=[t_leave],
                 boxes={float(r.t): [("сейф открыт", [r.x0, r.y0, r.x1, r.y1])]},
                 info={"reason": "safe_open_cashier_left", "safe_open_frames": n, "t_exit": d["t_exit"]})


def screen_dark_share(video: str, perc, laptop, times: list[float]) -> list[float]:
    """Share of dark pixels in the screen part of the laptop box, per frame."""
    import cv2
    from engine.video import iter_frames_at
    x0, y0, x1, y1 = [int(round(v)) for v in laptop.as_list()]
    ys = y0 + max(1, int(SCREEN_TOP * (y1 - y0)))
    out = []
    for _, _, fr in iter_frames_at(video, times, perc.info.crop):
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        s = g[y0:ys, x0:x1]
        out.append(float((s < SCREEN_DARK_REL * g.mean()).mean()) if s.size else 1.0)
    return out


def _b9(f: Facts, d: dict, t_leave: float, chk: dict | None) -> Event | None:
    lap = f.scene.laptop
    if lap is None or f.video is None or f.perc is None:
        return None
    dur = f.perc.info.duration
    times = [min(d["t_exit"] + k, dur - 0.1) for k in AFTER_EXIT_S]
    dark = screen_dark_share(f.video, f.perc, lap, times)
    if not dark or float(np.median(dark)) > SCREEN_LIT_MAX_DARK:
        return None
    locked = bool(chk and chk.get("lock_screen"))
    return Event("B9", t_leave, d["t_exit"], B9_LOCKED_SCORE if locked else B9_SCORE, evidence_t=[t_leave],
                 boxes={t_leave: [("ноутбук", lap.as_list())]},
                 info={"reason": "laptop_active_cashier_left", "screen_dark": [round(x, 3) for x in dark],
                       "lock_screen": locked, "t_exit": d["t_exit"]})


def detect(f: Facts) -> list[Event]:
    out: list[Event] = []
    for d in unique_departures(f):
        t_leave = leave_time(f, d)
        chk = _check_for(f, d)
        for ev in (_b1(f, d, t_leave, chk), _b5(f, d, t_leave), _b9(f, d, t_leave, chk)):
            if ev is not None:
                out.append(ev)
        if chk and (chk.get("stamps") or chk.get("stamps_listed")):
            out.append(Event("B8", t_leave, d["t_exit"], B8_SCORE, evidence_t=[t_leave],
                             info={"reason": "stamps_left_cashier_away", "t_exit": d["t_exit"]}))
    return out
