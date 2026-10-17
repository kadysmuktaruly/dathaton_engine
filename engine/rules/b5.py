"""B5: safe / keys left unattended.

Candidate moment: the last person (the cashier) leaves the cash room and it stays empty
(rules/leave.py). Clean clips also contain proper close-outs, so leaving alone does not
separate violations from normal work.

Evidence gate (only when the loaded detector has a `safe_open` class, i.e. a fine-tuned
checkpoint): the candidate is reported only if a confident open safe is detected in at least
SAFE_OPEN_MIN_FRAMES sampled frames while the room is empty. Without that evidence it is kept
below the verdict threshold (visible in debug output as dropped). A detection only raises the
candidate to the usual score: it is evidence, not proof, and the VLM verifier still applies.
With the stock COCO detector (no safe classes) the rule behaves as before.

Timecode = moment the cashier leaves the workplace (catalog: «сейф остался ... без людей»).
"""
from __future__ import annotations

from engine.events import Event
from engine.facts import Facts
from engine.rules.leave import departures

LEAVE_SCORE = 0.5          # TUNE: candidate only; recall first
NO_EVIDENCE_SCORE = 0.1    # stays below VERDICT_THRESHOLD (0.3) even after a VLM 'yes' (+0.15 max)
SAFE_OPEN_MIN_FRAMES = 2   # TUNE
ABSENCE_PAD_S = 0.5        # look this far past the end of the empty stretch


def evidence_gate_on(f: Facts) -> bool:
    labels = getattr(f.perc, "labels", None)
    return bool(labels) and "safe_open" in labels


def _open_safe_frames(f: Facts, t0: float, t1: float):
    """Confident safe_open boxes (Facts.objects holds only confident ones) in (t0, t1]."""
    o = f.objects
    if o.empty or "cls" not in o:
        return o.iloc[0:0]
    return o[(o.cls == "safe_open") & (o.t > t0) & (o.t <= t1)]


def detect(f: Facts) -> list[Event]:
    gate = evidence_gate_on(f)
    out = []
    for d in departures(f):
        t = d["t_depart"]
        boxes = {t: [("кассир", d["box"])]}
        info = {"reason": "cashier_left_room", "t_exit": d["t_exit"],
                "absent_s": d["absent_s"], "to_end": d["to_end"]}
        score = LEAVE_SCORE
        if gate:
            so = _open_safe_frames(f, d["t_exit"], d["t_exit"] + d["absent_s"] + ABSENCE_PAD_S)
            n = int(so.t.nunique()) if len(so) else 0
            info["safe_open_frames"] = n
            if n >= SAFE_OPEN_MIN_FRAMES:
                info["reason"] = "safe_open_unattended"
                r = so.sort_values("conf", ascending=False).iloc[0]
                boxes.setdefault(float(r.t), []).append(("сейф открыт", [r.x0, r.y0, r.x1, r.y1]))
            else:
                score = NO_EVIDENCE_SCORE
        out.append(Event("B5", t, d["t_exit"], score, evidence_t=[t], boxes=boxes, info=info))
    return out
