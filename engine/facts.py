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
    objects: pd.DataFrame   # raw object detections (for object rules)
    seat: tuple[float, float] | None  # cashier's usual foot position
    video: str | None = None           # source video (for rules that look at pixels)
    perc: object | None = None         # perception result (times, info, raw detections)


def _rolling_median(x: np.ndarray, t: np.ndarray, win: float) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    for i, ti in enumerate(t):
        m = np.abs(t - ti) <= win / 2 + 1e-6
        out[i] = np.median(x[m])
    return out


def build(df: pd.DataFrame, times: list[float], frame_w: int, frame_h: int,
          video: str | None = None, perc=None) -> Facts:
    scene = build_scene(df, frame_w, frame_h)
    persons = df[df.kind == "person"].copy()
    from engine.scene import clean_persons
    persons = clean_persons(persons, scene.person_h_ref).copy()
    if len(persons):
        persons["in_room"] = in_room_mask(persons, scene)
        feet = np.array([foot_point(r) for r in persons.itertuples()])
        persons["fx"], persons["fy"] = feet[:, 0], feet[:, 1]
    else:
        persons = persons.assign(in_room=[], fx=[], fy=[])
    room = persons[persons.in_room].copy()

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
    return Facts(scene, room, frames, df[df.kind == "obj"].copy(), seat, video, perc)
