"""B5 while working (docs/master-document.md, phase 2, step 3): the safe stays open while the
cashier works at the desk instead of being closed after taking the cash out.

Signal: a 'safe open' stretch (fine-tuned detector) that lasts much longer than taking cash out
or putting it in does (X-20: open for the whole 12 s clip; clean X-14 loads the safe in 6-7 s).
Hands are no help here: the safe stands right next to the seat, so wrists look 'at the safe'.
Timecode = a few seconds into the stretch (opening and taking out the cash is allowed).

B5 when the cashier leaves (safe open / key left) is in rules/depart.py.
"""
from __future__ import annotations

from engine.events import Event
from engine.facts import Facts
from engine.rules.collection import safe_open_runs

OPEN_MIN_S = 8          # TUNE: open at least this long
GRACE_S = 4.0           # TUNE: time allowed to take cash out / put it in
SCORE = 0.6


def detect(f: Facts) -> list[Event]:
    fr = f.frames
    out = []
    for r0, r1 in safe_open_runs(f):
        if r1 - r0 + 1 < OPEN_MIN_S:
            continue
        w = fr[(fr.t >= r0) & (fr.t <= r1 + 1)]
        if w.empty or w.n_smooth.max() < 1:
            continue  # nobody in the room: that is the departure rule's case
        t = float(min(r0 + GRACE_S, r1))
        out.append(Event("B5", t, float(r1 + 1), SCORE, evidence_t=[t],
                         info={"reason": "safe_open_while_working", "open_from": r0, "open_to": r1 + 1}))
    return out
