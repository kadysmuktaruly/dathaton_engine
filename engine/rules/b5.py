"""B5: safe / keys left unattended.

Variant implemented now ("leave"): the last person (the cashier) leaves the cash room and the
room stays empty. Clean clips also contain proper close-outs, so this is only a candidate;
the VLM verifier (Step 6) must confirm what was left (key in the lock, open safe).
Variant "safe open, nobody near" needs a safe/safe_open detector (Step 5).

Timecode = moment the cashier leaves the workplace (catalog: «сейф остался ... без людей»).
"""
from __future__ import annotations

from engine.events import Event
from engine.facts import Facts
from engine.rules.leave import departures

LEAVE_SCORE = 0.5  # TUNE: candidate only; recall first


def detect(f: Facts) -> list[Event]:
    out = []
    for d in departures(f):
        t = d["t_depart"]
        out.append(Event("B5", t, d["t_exit"], LEAVE_SCORE, evidence_t=[t],
                         boxes={t: [("кассир", d["box"])]},
                         info={"reason": "cashier_left_room", "t_exit": d["t_exit"],
                               "absent_s": d["absent_s"], "to_end": d["to_end"]}))
    return out
