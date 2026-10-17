"""Per-frame facts derived from perception + scene. Rules only read these tables."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.scene import Scene, build_scene, foot_point, in_room_mask

SMOOTH_S = 1.0  # TUNE: window for the smoothed head count


@dataclass
class Facts:
    scene: Scene
    persons: pd.DataFrame   # one row per in-room person per frame (+ foot, rel. coords, is_cashier)
    frames: pd.DataFrame    # one row per sampled frame: t, n_room, n_smooth, cashier_present, ...
    objects: pd.DataFrame   # confident object detections (>= OBJ_MIN), for object rules
    seat: tuple[float, float] | None  # cashier's usual foot position
    video: str | None = None           # source video (for rules that look at pixels)
    perc: object | None = None         # perception result (times, info, raw detections)
    scan: pd.DataFrame | None = None   # VLM desk-scan flags per sampled frame (if run)


def _rolling_median(x: np.ndarray, t: np.ndarray, win: float) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    for i, ti in enumerate(t):
        m = np.abs(t - ti) <= win / 2 + 1e-6
        out[i] = np.median(x[m])
    return out


STITCH_GAP_S = 1.5   # TUNE: max time between one track ending and the next starting
STITCH_DIST_H = 0.6  # TUNE: max foot distance, in person heights
MAX_PERSONS = 3      # TUNE: most people counted in the room per frame (keeps the most confident)


def stitch_tracks(room: pd.DataFrame, h_ref: float) -> pd.DataFrame:
    """Join in-room tracks that are one person with a new ID (tracker lost them for a moment).

    Track B is merged into track A when A ends, B starts within STITCH_GAP_S after that,
    B's first foot point is near A's last one, and A and B never appear in the same frame.
    """
    if room.empty or room.track_id.nunique() < 2:
        return room
    unit = max(h_ref, 1.0)
    room = room.copy()
    ids = room.track_id.to_numpy().copy()
    while True:
        df = room.assign(track_id=ids).sort_values("t")
        frames = df.groupby("track_id").t.agg(lambda s: set(s.round(4)))
        first = df.groupby("track_id").head(1).set_index("track_id")
        last = df.groupby("track_id").tail(1).set_index("track_id")
        best = None
        for a, ra in last.iterrows():
            for b, rb in first.iterrows():
                gap = rb.t - ra.t
                if a == b or not (0 < gap <= STITCH_GAP_S) or frames[a] & frames[b]:
                    continue
                d = np.hypot(rb.fx - ra.fx, rb.fy - ra.fy) / unit
                if d <= STITCH_DIST_H and (best is None or d < best[0]):
                    best = (d, a, b)
        if best is None:
            break
        ids[ids == best[2]] = best[1]
    room["track_id"] = ids
    return room


def cap_per_frame(room: pd.DataFrame, n: int = MAX_PERSONS) -> pd.DataFrame:
    """Keep at most n people per frame, the most confident ones."""
    if room.empty or n <= 0:
        return room
    return room.sort_values("conf", ascending=False).groupby("t", sort=False).head(n).sort_index()


def build(df: pd.DataFrame, times: list[float], frame_w: int, frame_h: int,
          video: str | None = None, perc=None) -> Facts:
    n_frames = len(np.unique(np.asarray(times, dtype=float))) if len(times) else None
    scene = build_scene(df, frame_w, frame_h, n_frames)
    persons = df[df.kind == "person"].copy()
    from engine.scene import clean_persons
    persons = clean_persons(persons, scene.person_h_ref).copy()
    if len(persons):
        persons["in_room"] = in_room_mask(persons, scene)
        feet = np.array([foot_point(r) for r in persons.itertuples()])
        persons["fx"], persons["fy"] = feet[:, 0], feet[:, 1]
    else:
        # typed empty columns: an untyped empty mask would be read as column labels
        persons = persons.assign(in_room=pd.Series(dtype=bool), fx=pd.Series(dtype=float),
                                 fy=pd.Series(dtype=float))
    room = persons[persons["in_room"].astype(bool)].copy()
    room = stitch_tracks(room, scene.person_h_ref)
    room = cap_per_frame(room)

    # Cashier seat: the in-room track present longest; its median foot point.
    seat = None
    if len(room):
        counts = room.groupby("track_id")["t"].nunique().sort_values(ascending=False)
        top = room[room.track_id == counts.index[0]]
        seat = (float(top.fx.median()), float(top.fy.median()))
    unit = scene.scale
    if seat is not None and len(room):
        room["d_seat"] = np.hypot(room.fx - seat[0], room.fy - seat[1]) / unit
    else:
        room["d_seat"] = np.inf
    if scene.desk is not None and len(room):
        d = scene.desk
        room["d_desk"] = np.hypot(room.fx - d.cx, room.fy - d.cy) / unit
    else:
        room["d_desk"] = np.inf

    t_arr = np.unique(np.array(times, dtype=float))
    n_room = room.groupby("t").size().reindex(t_arr, fill_value=0).to_numpy()
    frames = pd.DataFrame({"t": t_arr, "n_room": n_room})
    frames["n_smooth"] = _rolling_median(frames.n_room.to_numpy(), t_arr, SMOOTH_S) if len(t_arr) else []
    # who is at the seat (cashier position) in each frame
    at_seat = room[room.d_seat <= 0.6].groupby("t").size()  # TUNE
    frames["seat_occupied"] = frames.t.map(at_seat).fillna(0).to_numpy() > 0
    obj = df[df.kind == "obj"]
    if "confident" in obj:
        obj = obj[obj.confident]  # weaker boxes only feed scene anchors
    return Facts(scene, room, frames, obj.copy(), seat, video, perc)
