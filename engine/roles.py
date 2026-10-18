"""People roles and hand signals (docs/master-document.md, section 3).

One cashier works alone; everyone else in the room is a client (numbered by arrival) or the
collector. Roles are assigned per frame, not per track id (ID switches happen):
  - the cashier is the in-room person closest to the cashier's seat;
  - every other in-room person is a visitor.
Visitor rows are grouped into sessions (one per visitor, ID switches stitched). A session that
overlaps collection-bag evidence is the collector's.

`hands` says, per frame, whether the cashier's wrists are at the counting machine, the laptop or
the safe. Rules read only these tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.scene import L_WRIST, R_WRIST, Box

CASHIER_SEAT_MAX = 0.35   # TUNE: d_seat (desk widths) for 'cashier at the seat'; clients sit at 0.5-0.7
SESSION_MIN_S = 2.0       # TUNE: shorter visitor tracks are noise
SESSION_GAP_S = 2.0       # TUNE: a visitor hidden this long is still the same visit
SESSION_JOIN_H = 0.8      # TUNE: foot distance (person heights) to join two tracks into one visit
WRIST_CONF = 0.4
HAND_PAD = 0.15           # TUNE: device boxes grown by this share for the wrist test
BAG_NEAR_S = 2.0          # collection-bag evidence this close to a session tags it as the collector's


@dataclass
class Session:
    idx: int                   # 1-based arrival order among visitors
    enter: float               # first frame in the room
    leave: float               # last frame in the room
    tracks: list[int] = field(default_factory=list)
    role: str = "client"       # "client" | "collector"
    ended: bool = True         # left before the clip ended
    first_box: list[float] | None = None

    @property
    def duration(self) -> float:
        return self.leave - self.enter

    def covers(self, t: float, pad: float = 0.0) -> bool:
        return self.enter - pad <= t <= self.leave + pad


def split_people(persons: pd.DataFrame) -> pd.DataFrame:
    """Adds `role` ('cashier' / 'visitor') per in-room person row: per frame, the person nearest
    to the seat is the cashier, the others are visitors."""
    if persons.empty:
        return persons.assign(role=pd.Series(dtype=object))
    p = persons.copy()
    nearest = p.groupby("t").d_seat.transform("min")
    p["role"] = np.where(p.d_seat.to_numpy() <= nearest.to_numpy() + 1e-9, "cashier", "visitor")
    # ties (two rows at the same distance) - keep one cashier per frame
    dup = p[p.role == "cashier"].duplicated("t", keep="first")
    p.loc[dup[dup].index, "role"] = "visitor"
    return p


def cashier_at_seat(persons: pd.DataFrame, times) -> pd.Series:
    """Per sampled time: someone sits at the cashier's seat."""
    t_arr = np.unique(np.asarray(times, dtype=float))
    if persons.empty:
        return pd.Series(False, index=t_arr)
    at = persons[persons.d_seat <= CASHIER_SEAT_MAX].groupby("t").size()
    return pd.Series(t_arr, index=t_arr).map(at).fillna(0).gt(0)


def sessions(persons: pd.DataFrame, h_ref: float, duration: float) -> list[Session]:
    """Visitor sessions from visitor rows (split_people first)."""
    v = persons[persons.role == "visitor"] if "role" in persons else persons.iloc[0:0]
    if v.empty:
        return []
    unit = max(h_ref, 1.0)
    spans = []
    for tr, g in v.groupby("track_id"):
        g = g.sort_values("t")
        spans.append({"tracks": [int(tr)], "enter": float(g.t.iloc[0]), "leave": float(g.t.iloc[-1]),
                      "first": (float(g.fx.iloc[0]), float(g.fy.iloc[0])),
                      "last": (float(g.fx.iloc[-1]), float(g.fy.iloc[-1])),
                      "box": [float(g.x0.iloc[0]), float(g.y0.iloc[0]), float(g.x1.iloc[0]), float(g.y1.iloc[0])]})
    spans.sort(key=lambda s: s["enter"])
    # join a span to an earlier one that ended shortly before, nearby (same visitor, new track id)
    merged: list[dict] = []
    for s in spans:
        best = None
        for m in merged:
            gap = s["enter"] - m["leave"]
            d = np.hypot(s["first"][0] - m["last"][0], s["first"][1] - m["last"][1]) / unit
            if -0.01 <= gap <= SESSION_GAP_S and d <= SESSION_JOIN_H and (best is None or d < best[0]):
                best = (d, m)
        if best is None:
            merged.append(dict(s))
        else:
            m = best[1]
            m["tracks"] += s["tracks"]
            m["leave"] = max(m["leave"], s["leave"])
            m["last"] = s["last"]
    out = []
    for m in merged:
        if m["leave"] - m["enter"] < SESSION_MIN_S:
            continue
        out.append(Session(0, m["enter"], m["leave"], m["tracks"], ended=m["leave"] < duration - 1.0,
                           first_box=m["box"]))
    out.sort(key=lambda s: s.enter)
    for i, s in enumerate(out, 1):
        s.idx = i
    return out


def session_rows(persons: pd.DataFrame, s: Session) -> pd.DataFrame:
    """The visitor's own rows during the session."""
    if persons.empty:
        return persons
    m = persons.track_id.isin(s.tracks) & (persons.t >= s.enter) & (persons.t <= s.leave)
    if "role" in persons:
        m &= persons.role == "visitor"
    return persons[m]


def tag_collectors(sess: list[Session], objects: pd.DataFrame, scan: pd.DataFrame | None = None) -> None:
    """A session with collection-bag evidence (fine-tuned detector, or the VLM scan's green-bag
    flag) during the visit is the collector's."""
    bag_t = []
    if not objects.empty and "cls" in objects:
        bag_t += objects[objects.cls == "collection_bag"].t.round(2).tolist()
    if scan is not None and len(scan) and "green_bag_on_desk" in scan:
        bag_t += scan[scan.green_bag_on_desk].t.tolist()
    for s in sess:
        n = sum(s.covers(t, BAG_NEAR_S) for t in bag_t)
        if n >= 2:
            s.role = "collector"


def _wrists(kps) -> list[tuple[float, float]]:
    if kps is None:
        return []
    k = np.asarray(kps, dtype=np.float32).reshape(17, 3)
    return [(float(k[j, 0]), float(k[j, 1])) for j in (L_WRIST, R_WRIST) if k[j, 2] > WRIST_CONF]


def hands(persons: pd.DataFrame, scene, times) -> pd.DataFrame:
    """Per sampled time: the cashier's wrist is at the counting machine / laptop / safe."""
    t_arr = np.unique(np.asarray(times, dtype=float))
    out = pd.DataFrame({"t": t_arr, "machine": False, "laptop": False, "safe": False})
    if persons.empty or "kps" not in persons:
        return out
    boxes = {k: getattr(scene, k) for k in ("machine", "laptop", "safe")}
    boxes = {k: b.expand(HAND_PAD) for k, b in boxes.items() if isinstance(b, Box)}
    if not boxes:
        return out
    cash = persons[persons.role == "cashier"] if "role" in persons else persons
    idx = {t: i for i, t in enumerate(t_arr)}
    for r in cash.itertuples():
        i = idx.get(float(r.t))
        if i is None:
            continue
        for x, y in _wrists(r.kps):
            for k, b in boxes.items():
                if b.contains(x, y):
                    out.iat[i, out.columns.get_loc(k)] = True
    return out


def dwell(h: pd.DataFrame, col: str, t0: float, t1: float) -> float:
    """Seconds (sampled frames x step) with the cashier's wrist at `col` inside [t0, t1]."""
    if h.empty:
        return 0.0
    w = h[(h.t >= t0) & (h.t <= t1)]
    if len(w) < 2:
        return float(w[col].sum()) / 3.0
    step = float(np.median(np.diff(h.t.to_numpy()))) if len(h) > 1 else 1 / 3
    return float(w[col].sum()) * step
