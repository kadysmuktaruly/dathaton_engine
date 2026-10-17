"""Scene understanding that does not depend on fixed pixel zones.

Anchors (desk, laptop, counting machine, safe) are found per video from stable detections.
All distances are expressed relative to anchor sizes or the cashier's body height, so the
logic should survive a new camera angle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Object roles come from engine.classes.ROLES (detector class -> engine role).
STABLE_FRAC = 0.4  # TUNE: an anchor must be seen in >= this share of frames

# Keypoint indices (COCO-17)
L_WRIST, R_WRIST, L_ANKLE, R_ANKLE, L_HIP, R_HIP, L_SHO, R_SHO = 9, 10, 15, 16, 11, 12, 5, 6


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self): return self.x1 - self.x0
    @property
    def h(self): return self.y1 - self.y0
    @property
    def cx(self): return (self.x0 + self.x1) / 2
    @property
    def cy(self): return (self.y0 + self.y1) / 2

    def expand(self, k: float) -> "Box":
        dx, dy = self.w * k, self.h * k
        return Box(self.x0 - dx, self.y0 - dy, self.x1 + dx, self.y1 + dy)

    def contains(self, x, y) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def as_list(self): return [self.x0, self.y0, self.x1, self.y1]


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    u = a.w * a.h + b.w * b.h - inter
    return inter / u if u > 0 else 0.0


@dataclass
class Scene:
    frame_w: int
    frame_h: int
    desk: Box | None
    laptop: Box | None = None
    machine: Box | None = None
    safe: Box | None = None
    person_h_ref: float = 0.0          # typical in-room standing/seated person height
    notes: list[str] = field(default_factory=list)
    back_line: float | None = None     # image y of the desk's far edge (depth reference)
    # Where each anchor came from: 'explicit:<class>' (a fine-tuned class for that object) or
    # 'proxy:<how>' (geometry / stock look-alikes). A proxy is a location estimate only, never
    # proof of what the equipment is or whether a safe is open.
    provenance: dict = field(default_factory=dict)

    @property
    def scale(self) -> float:
        """Length unit for zones: desk width, else a person height."""
        if self.desk is not None:
            return max(self.desk.w, 1.0)
        return max(self.person_h_ref, 1.0)


def _with_roles(obj: pd.DataFrame) -> pd.DataFrame:
    if "role" in obj:
        return obj
    from engine.classes import role_map
    return obj.assign(role=obj.cls.map(role_map())).dropna(subset=["role"])


def _stable_clusters(obj: pd.DataFrame, n_frames: int) -> pd.DataFrame:
    """Group detections of the same role whose boxes overlap; return median box per group.

    A group whose top 10% of confidences stays below ANCHOR_MIN is dropped: one weak box
    seen in many frames is not trusted as an anchor (levels per role).
    """
    from engine.perception import ANCHOR_MIN
    out = []
    for role, g in _with_roles(obj).groupby("role"):
        boxes = g[["x0", "y0", "x1", "y1"]].to_numpy()
        ts = g["t"].to_numpy()
        conf = g["conf"].to_numpy()
        used = np.zeros(len(g), bool)
        order = np.argsort(-conf)
        for i in order:
            if used[i]:
                continue
            bi = Box(*boxes[i])
            members = [j for j in range(len(g)) if not used[j] and iou(bi, Box(*boxes[j])) > 0.4]
            for j in members:
                used[j] = True
            p90 = float(np.quantile(conf[members], 0.9))
            if p90 < ANCHOR_MIN.get(role, 0.5):
                continue
            frames = len(set(ts[members]))
            med = np.median(boxes[members], axis=0)
            out.append((role, frames / max(n_frames, 1), float(np.median(conf[members])), p90, *med))
    return pd.DataFrame(out, columns=["role", "frac", "conf", "conf_p90", "x0", "y0", "x1", "y1"])


def foot_point(row) -> tuple[float, float]:
    """Floor contact point: mean visible ankle, else the box bottom centre."""
    if row.kps is not None:
        k = np.asarray(row.kps, dtype=np.float32).reshape(17, 3)
        ank = k[[L_ANKLE, R_ANKLE]]
        ank = ank[ank[:, 2] > 0.3]
        if len(ank):
            return float(ank[:, 0].mean()), float(max(ank[:, 1].max(), row.y1 - 0.08 * (row.y1 - row.y0)))
    return float((row.x0 + row.x1) / 2), float(row.y1)


def _best(cl: pd.DataFrame, role: str) -> Box | None:
    """Most stable cluster of a role (ties: higher median confidence)."""
    c = cl[cl.role == role]
    if c.empty:
        return None
    r = c.sort_values(["frac", "conf"], ascending=False).iloc[0]
    return Box(r.x0, r.y0, r.x1, r.y1)


def build_scene(df: pd.DataFrame, frame_w: int, frame_h: int, n_frames: int | None = None) -> Scene:
    """n_frames: number of sampled frames, including frames with no detections (they count
    against an anchor's stability). Defaults to the frames present in df."""
    n = max(int(n_frames or 0), int(df["t"].nunique()))
    obj = df[df.kind == "obj"]
    cl = _stable_clusters(obj, n)
    cl = cl[cl.frac >= STABLE_FRAC]

    persons = df[df.kind == "person"]
    # Wrist cloud: where hands work most of the time -> the working desk.
    wrists = []
    for r in persons.itertuples():
        if r.kps is None:
            continue
        k = np.asarray(r.kps, dtype=np.float32).reshape(17, 3)
        for j in (L_WRIST, R_WRIST):
            if k[j, 2] > 0.4:
                wrists.append(k[j, :2])
    wrists = np.array(wrists) if wrists else np.zeros((0, 2))

    laptop = None
    lap = cl[cl.role == "laptop"].sort_values("frac", ascending=False)
    if len(lap):
        laptop = Box(*lap.iloc[0][["x0", "y0", "x1", "y1"]])

    # Work area = hull of stable devices (laptop + machine-like boxes) not touching the
    # bottom edge (foreground furniture) and not huge.
    devs, tall = [], []
    for r in cl[(cl.role.isin({"machine_like", "counting_machine", "laptop"})) & (cl.frac >= 0.5)].itertuples():
        b = Box(r.x0, r.y0, r.x1, r.y1)
        if b.y1 >= 0.95 * frame_h or b.w >= 0.4 * frame_w:
            continue  # foreground furniture
        if r.role == "machine_like" and b.h > 1.25 * b.w:  # TUNE: tall box = safe-like, not a desk device
            tall.append(b)
        else:
            devs.append(b)
    # keep devices that sit where hands work (wrist cloud), to drop furniture behind the glass
    if len(wrists) and devs:
        def act(b):
            return float(np.mean([b.expand(0.6).contains(x, y) for x, y in wrists]))
        devs = [b for b in devs if act(b) > 0.02] or devs
    hull = None
    if devs:
        hull = Box(min(b.x0 for b in devs), min(b.y0 for b in devs), max(b.x1 for b in devs), max(b.y1 for b in devs))
    # Desk = hull united with table detections that mostly overlap it.
    desk = hull
    prov = {"desk": "proxy:devices" if hull is not None else None, "machine": None, "safe": None}
    notes = []
    tables = [Box(r.x0, r.y0, r.x1, r.y1) for r in cl[cl.role == "table"].itertuples()]
    if hull is not None:
        for b in tables:
            ix = max(0.0, min(b.x1, hull.x1) - max(b.x0, hull.x0))
            iy = max(0.0, min(b.y1, hull.y1) - max(b.y0, hull.y0))
            # merge tables that mostly overlap the hull, or that cover most of a small hull
            if ix * iy >= 0.3 * b.w * b.h or (ix * iy >= 0.6 * hull.w * hull.h and b.y1 < 0.95 * frame_h):
                desk = Box(min(desk.x0, b.x0), min(desk.y0, b.y0), max(desk.x1, b.x1), max(desk.y1, b.y1))
                prov["desk"] = "proxy:devices+table"
    elif tables:
        def act_t(b):
            return float(np.mean([b.contains(x, y) for x, y in wrists])) if len(wrists) else 0.0
        desk = max([b for b in tables if b.y1 < 0.95 * frame_h] or tables, key=act_t)
        notes.append("no stable devices; desk from table detections")
        prov["desk"] = "proxy:table"
    elif _best(cl, "desk") is None:
        notes.append("desk not found")
    # Explicit (fine-tuned) desk / counting_machine / safe boxes win over the look-alike logic.
    explicit_desk = _best(cl, "desk")
    if explicit_desk is not None:
        desk = explicit_desk
        notes.append("desk from desk detections")
        prov["desk"] = "explicit:desk"

    machine = _best(cl, "counting_machine")
    if machine is not None:
        notes.append("machine from counting_machine detections")
        prov["machine"] = "explicit:counting_machine"
    elif desk is not None:
        cands = [b for b in devs if laptop is None or iou(b, laptop) < 0.3]
        if cands:
            machine = max(cands, key=lambda b: b.w * b.h)
            prov["machine"] = "proxy:tv/suitcase"
    safe = _best(cl, "safe")  # built from safe and safe_open boxes
    if safe is not None:
        notes.append("safe from safe detections")
        prov["safe"] = "explicit:safe"
    elif tall and desk is not None:
        near = [b for b in tall if np.hypot(b.cx - desk.cx, b.cy - desk.cy) < 1.2 * desk.w]
        if near:
            safe = max(near, key=lambda b: b.w * b.h)
            notes.append("safe guessed from a tall stable box")
            prov["safe"] = "proxy:tall box"
    # Back edge of the desk surface: devices stand on it, so it lies a bit above their bottoms.
    back_line = None
    if devs:
        back_line = min(b.y1 for b in devs) - 0.5 * float(np.median([b.h for b in devs]))  # TUNE
    elif desk is not None:
        back_line = desk.y0 + 0.1 * desk.h

    hs = (persons.y1 - persons.y0).to_numpy()
    person_h_ref = float(np.percentile(hs, 75)) if len(hs) else frame_h * 0.3
    return Scene(frame_w, frame_h, desk, laptop, machine, safe, person_h_ref, notes, back_line, prov)


def in_room_mask(persons: pd.DataFrame, scene: Scene) -> np.ndarray:
    """True for people physically in the room (not seen through the glass wall).

    With a high camera, depth grows upward in the image. People behind the glass wall (and
    their reflections) never reach the desk's back edge: their box bottom stays above it.
    The test is per frame, so a person walking toward the door behind the glass only counts
    once they are through the door.
    """
    if persons.empty:
        return np.zeros(0, bool)
    if scene.back_line is None:
        return np.ones(len(persons), bool)
    line = scene.back_line
    return persons.y1.to_numpy() >= line


def _contain(a, b) -> float:
    """Share of box b's area inside box a."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
    return ix * iy / area


def person_filter_reasons(persons: pd.DataFrame, h_ref: float) -> list[str | None]:
    """Why each person box is dropped before counting (None = kept): 'tiny box' (hands,
    fragments), 'group box' (one box spanning 2+ people), 'duplicate' (mostly inside a more
    confident box)."""
    n = len(persons)
    if n == 0:
        return []
    reason: list[str | None] = [None] * n
    h = (persons.y1 - persons.y0).to_numpy()
    for i in np.where(h < 0.3 * h_ref)[0]:  # TUNE
        reason[i] = "tiny box"
    boxes = persons[["x0", "y0", "x1", "y1"]].to_numpy()
    conf = persons.conf.to_numpy()
    for t, g in persons.groupby("t").indices.items():
        g = [i for i in g if reason[i] is None]
        for i in g:
            inside = [j for j in g if j != i and reason[j] is None and _contain(boxes[i], boxes[j]) > 0.8]
            if len(inside) >= 2:
                reason[i] = "group box"
        g = [i for i in g if reason[i] is None]
        for i in g:
            for j in g:
                if i != j and reason[i] is None and reason[j] is None and _contain(boxes[j], boxes[i]) > 0.75 \
                        and conf[i] <= conf[j]:
                    reason[i] = "duplicate"
    return reason


def clean_persons(persons: pd.DataFrame, h_ref: float) -> pd.DataFrame:
    """Drop tiny boxes (hands, fragments), group boxes spanning 2+ people, and near-duplicates."""
    if persons.empty:
        return persons
    keep = np.array([r is None for r in person_filter_reasons(persons, h_ref)], bool)
    return persons[keep]
