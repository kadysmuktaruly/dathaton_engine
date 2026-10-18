"""C2: the cash-room door is left open after someone passes (docs/master-document.md).

Signal: a large persistent change of the static background on the back-wall side of the room
(farther than the desk's back edge) between the start and the end of the clip, with people
masked out. A door that is opened and closed again leaves no change (clean X-31); a door left
open leaves a door-sized blob (X-30, X-32, X-33). It is flagged even if nobody else comes in.
The door must have been opened by someone entering an empty room (the cashier arriving): every
training case is that, and changes that start while the cashier is already inside come from
clients arriving on clean clips (X-5, X-18, X-39), not from a door left open.

Timecode: the first visitor who walks in through the open door and reaches the desk (X-30: the
cashier left the door open at 3 s, a man walks in at 13 s); otherwise the moment the door is
fully open (X-32, X-33: the cashier walks in and leaves it open).
"""
from __future__ import annotations

import numpy as np

from engine import bgchange
from engine.events import Event
from engine.facts import Facts
from engine.roles import session_rows
from engine.video import iter_frames_at

MIN_AREA_REL = 0.05   # TUNE: blob area / person_h_ref^2
ONSET_FRAC = 0.4      # TUNE: share of blob pixels changed to count the door as open
HOLD_S = 2.0          # TUNE: must stay open at least this long
LIGHTING_FRAC = 0.25  # TUNE: changed share of the whole frame above which we assume lighting

def detect(f: Facts) -> list[Event]:
    video, perc = f.video, f.perc
    if video is None or f.scene.back_line is None:
        return []
    d = perc.info.duration
    persons = perc.df[perc.df.kind == "person"]
    before = [0.2, 0.6, 1.0, 1.4]
    after = [max(0.0, d - k) for k in (2.0, 1.5, 1.0, 0.5)]
    if d < 6:
        return []
    ch = bgchange.change_map(video, before, after, persons, perc.info.crop)
    if ch is None:
        return []
    if ch.mean() > LIGHTING_FRAC:
        return []  # most of the frame changed: lights switched or camera moved, not a door
    h2 = max(f.scene.person_h_ref, 1.0) ** 2
    events = []
    for b in bgchange.blobs(ch):
        x0, y0, x1, y1 = b["box"]
        if (y0 + y1) / 2 >= f.scene.back_line:      # must be on the far (back-wall) side
            continue
        if b["area"] < MIN_AREA_REL * h2:
            continue
        t_on = _onset(video, perc, ch, b, before)
        if t_on is None:
            continue
        around = f.frames[(f.frames.t >= t_on - 3.0) & (f.frames.t <= t_on + 1.0)]
        if len(around) and around.n_smooth.min() > 0:
            continue  # the room was not empty: no one walked in and left the door open
        t, reason, boxes = t_on, "door_left_open", {t_on: [("дверь", b["box"])]}
        s = first_visitor_through(f, t_on)
        if s is not None:
            t, reason = s.enter, "visitor_through_open_door"
            boxes = {t: [("дверь", b["box"])] + ([("посетитель", s.first_box)] if s.first_box else [])}
        events.append(Event("C2", t, d, min(0.9, 0.5 + b["area"] / h2), evidence_t=[t], boxes=boxes,
                            info={"reason": reason, "t_door_open": t_on,
                                  "blob_area_rel": round(b["area"] / h2, 3),
                                  "blob": [round(v) for v in b["box"]]}))
        break  # one door
    return events


VISITOR_REACH_DESK = 0.5   # TUNE: d_desk (desk widths) the visitor must reach


def first_visitor_through(f: Facts, t_open: float):
    """First visitor session that enters after the door opened and walks up to the desk."""
    p = f.persons
    for s in f.sessions or []:
        if s.enter < t_open + 1.0:
            continue
        rows = session_rows(p, s)
        if len(rows) and float(rows.d_desk.min()) <= VISITOR_REACH_DESK:
            return s
    return None


def _onset(video, perc, ch, blob, before) -> float | None:
    import cv2
    s = bgchange.SCALE
    x0, y0, x1, y1 = [int(v * s) for v in blob["box"]]
    region = ch[y0:y1, x0:x1]
    persons = perc.df[perc.df.kind == "person"]
    bg, valid = bgchange.background(video, before, persons, perc.info.crop)
    ts, fr = [], []
    for _, t, frame in iter_frames_at(video, perc.times, perc.info.crop):
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s,
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        g -= np.median(g)
        pm = bgchange._person_mask(g.shape, persons[(persons.t - t).abs() <= 0.2])
        diff = np.abs(g - bg) > bgchange.DIFF_THRESH
        m = region & valid[y0:y1, x0:x1] & ~pm[y0:y1, x0:x1]
        if m.sum() < 0.2 * region.sum():
            ts.append(t); fr.append(np.nan)  # mostly hidden by a person
            continue
        ts.append(t); fr.append(float(diff[y0:y1, x0:x1][m].mean()))
    ts, fr = np.array(ts), np.array(fr)
    tail = fr[ts >= ts[-1] - 3.0]
    tail = tail[~np.isnan(tail)]
    full = 0.8 * float(np.median(tail)) if len(tail) else ONSET_FRAC
    for i in range(len(ts)):
        if np.isnan(fr[i]) or fr[i] < ONSET_FRAC:
            continue
        rest = fr[(ts >= ts[i]) & (ts <= ts[i] + HOLD_S)]
        rest = rest[~np.isnan(rest)]
        if len(rest) and np.median(rest) >= ONSET_FRAC:
            # door starts opening at ts[i]; report when it is fully open (person through)
            for k in range(i, len(ts)):
                if not np.isnan(fr[k]) and fr[k] >= max(full, ONSET_FRAC):
                    return float(ts[k])
            return float(ts[i])
    return None
