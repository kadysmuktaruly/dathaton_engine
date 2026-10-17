"""C2: the cash-room door is left open during work.

Signal: a large persistent change of the static background on the back-wall side of the room
(farther than the desk's back edge) between the start and the end of the clip, with people
masked out. A door that is opened and closed again leaves no change (clean X-31); a door left
open leaves a door-sized blob (X-32, X-33).

Timecode = onset of that change (catalog: «момент, когда дверь открыта»), found by comparing
each frame's blob region to the 'before' background.
"""
from __future__ import annotations

import numpy as np

from engine import bgchange
from engine.events import Event
from engine.facts import Facts
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
    fr = f.frames
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
        # the door was opened by someone entering an EMPTY room (the cashier arriving and
        # leaving it open); doors opened by visitors while the cashier is in are C1's job
        around = fr[(fr.t >= t_on - 3.0) & (fr.t <= t_on + 1.0)]
        if len(around) and around.n_smooth.min() > 0:
            continue
        events.append(Event("C2", t_on, d, min(0.9, 0.5 + b["area"] / h2), evidence_t=[t_on],
                            boxes={t_on: [("дверь", b["box"])]},
                            info={"reason": "door_left_open", "blob_area_rel": round(b["area"] / h2, 3),
                                  "blob": [round(v) for v in b["box"]]}))
        break  # one door
    return events


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
