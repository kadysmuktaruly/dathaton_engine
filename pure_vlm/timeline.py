"""Timestamped VLM answers -> violations. Deterministic rules from the master document
(«Cashier routine, violations, and tracking list», 18.10.2026); no model is involved here.

Input: the dict from observe.observe(): per-question answer series {t: answer} and action windows.
Output: Event(code, t, reason, evidence_t) with t = the moment the violation becomes visible.

Rule map (master document section 2):
  departures (cashier leaves the room): B1 cash / sealed bag left on the desk, B8 stamps left, B5 safe open or
      key in the lock, B9 laptop left active, A4 a visitor is still in the room;
  door: C2 the door stays open;
  safe: B5 the safe stays open while the cashier works at the desk;
  desk: B3 personal items (bag, clothing, drink, phone with nobody else in the room);
  visitor sessions: B1 the previous client's cash is still on the desk when the next one comes in,
      A1 the client leaves while the counting machine is still running, A7 two visitors served at once,
      A2 cash taken without a machine count, A3 cash handed out without a recount, A6 cash taken without
      an ID card, D7 a collection bag taken without papers shown, D6 bag contents put into the safe without
      a count or a laptop entry.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---- thresholds -----------------------------------------------------------------------------------------
C2_MIN_S = 6.0          # TUNE: door open this long (X-4: 5.5 s while a client walks in past the door);
                        # a run reaching the clip end needs C2_MIN_END_S
C2_AFTER_PASS_S = 2.0   # the door should be shut this long after someone came in
C2_MIN_END_S = 5.0     # TUNE: X-23 (clean) reads 4.8 s of open door at the end after the cashier leaves
B5_DESK_S = 5.0         # TUNE: safe open while the cashier sits at the desk this long (X-5: 3 s at a
                        # safe beside the desk is normal use; X-20: 6.6 s)
ABSENT_MIN_S = 1.0      # a departure: the cashier is absent at least this long (or until the clip ends)
WALK_BACK_S = 4.0       # the departure time is the start of the walk out, at most this long before
DESK_BLIP_S = 1.0       # a 'desk' answer this short between walking answers is a misread
B3_MIN_SAMPLES = 2      # TUNE: a personal item seen on the desk at >= 2 asked times...
B3_MIN_SPAN_S = 1.0     # ... at least this far apart
TWO_VISITORS_S = 1.5    # TUNE: A7 needs two visitors for this long
MERGE_S = 3.0           # same code within this many seconds -> one event
DEPARTURE_MERGE_S = 10.0  # same departure code again within this long: the same walk-out seen twice
MAX_ITEMS = 5           # a desk answer naming this many items is noise
D7_AFTER_BAG_S = 4.0    # TUNE: the ID check may come right after the bag is put down (X-16)

PERSONAL = ("handbag", "clothing", "drink")


@dataclass
class Event:
    code: str
    t: float
    reason: str
    evidence_t: list[float] = field(default_factory=list)
    info: dict = field(default_factory=dict)


# ---- series helpers -----------------------------------------------------------------------------------------
def series(obs: dict, name: str) -> list[tuple[float, object]]:
    return sorted((float(t), a) for t, a in obs["states"].get(name, {}).items())


def at(s: list[tuple[float, object]], t: float):
    """Answer at the latest asked time <= t (the first answer before the first time)."""
    if not s:
        return None
    prev = s[0][1]
    for ts, a in s:
        if ts > t + 1e-6:
            break
        prev = a
    return prev


def runs(s: list[tuple[float, object]], pred, end: float) -> list[tuple[float, float]]:
    """Maximal [start, stop) intervals where pred(answer) holds; stop = the first asked time it fails."""
    out = []
    start = None
    for t, a in s:
        if pred(a) and start is None:
            start = t
        elif not pred(a) and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, end))
    return out


def cashier(a) -> str:
    c = str((a or {}).get("cashier", "")).lower() if isinstance(a, dict) else ""
    if c in ("absent", "none", "no", ""):
        return "absent" if c else "unknown"
    if c in ("desk", "safe"):
        return c
    return "walking"


def visitors(a) -> int:
    try:
        return int((a or {}).get("visitors", 0)) if isinstance(a, dict) else 0
    except (TypeError, ValueError):
        return 0


def items(a) -> set[str]:
    """Items on the desk. An answer naming MAX_ITEMS or more is a hallucination burst (X-13 lists all eight
    names twice) and counts as unknown (empty)."""
    s = set(a) if isinstance(a, list) else set()
    return set() if len(s) >= MAX_ITEMS else s


def safe_open(a) -> bool:
    return isinstance(a, dict) and a.get("open") is True


def safe_key(a) -> bool:
    return isinstance(a, dict) and a.get("key") is True


def windows(obs: dict, t0: float, t1: float, key: str) -> list[dict]:
    """Action windows overlapping [t0, t1] where `key` was seen."""
    return [w for w in obs["actions"] if w.get(key) and w["t1"] >= t0 and w["t0"] <= t1]


# ---- derived facts ------------------------------------------------------------------------------------------
def departures(obs: dict) -> list[dict]:
    """The cashier leaves the room: {t: start of the walk out, gone: first absent time, back: return time}."""
    ppl = series(obs, "people")
    end = obs["duration"]
    out = []
    for a, b in runs(ppl, lambda x: cashier(x) == "absent", end):
        if a <= ppl[0][0] + 1e-6:
            continue  # absent from the start: an arrival, not a departure
        if b - a < ABSENT_MIN_S and b < end:
            continue
        # walk back over the walk to the door, stepping over 'absent' blips (X-11 at 19.0 s) and 'desk' blips
        # of <= DESK_BLIP_S between walking samples (X-28 at 10-11 s)
        before = [(t, cashier(x)) for t, x in ppl if t < a]
        t_dep = before[-1][0]
        i = len(before) - 1
        while i >= 0 and a - before[i][0] <= WALK_BACK_S:
            t, c = before[i]
            if c == "walking":
                t_dep = t
                i -= 1
            elif c == "absent":
                i -= 1
            else:
                j = i
                while j >= 0 and before[j][1] == c:
                    j -= 1
                if j >= 0 and before[j][1] == "walking" and t - before[j + 1][0] <= DESK_BLIP_S:
                    i = j
                else:
                    break
        out.append({"t": t_dep, "gone": a, "back": b, "visitors": visitors(at(ppl, a))})
    return out


def sessions(obs: dict) -> list[dict]:
    """Visitor sessions: intervals with at least one visitor in the room."""
    ppl = series(obs, "people")
    end = obs["duration"]
    out = []
    for a, b in runs(ppl, lambda x: visitors(x) > 0, end):
        if b - a < 1.0 and b < end:
            continue  # a one-sample blip
        two = runs([(t, x) for t, x in ppl if a <= t < b], lambda x: visitors(x) >= 2, b)
        out.append({"start": a, "end": b, "two": [r for r in two if r[1] - r[0] >= TWO_VISITORS_S]})
    # merge sessions split by a short gap (a visitor hidden for a moment)
    merged = []
    for s in out:
        if merged and s["start"] - merged[-1]["end"] <= 1.5:
            merged[-1]["end"] = s["end"]
            merged[-1]["two"] += s["two"]
        else:
            merged.append(s)
    return merged


# ---- rules ----------------------------------------------------------------------------------------------------
def rule_departures(obs: dict) -> list[Event]:
    desk, safe, lap = series(obs, "desk"), series(obs, "safe"), series(obs, "laptop")
    ev = []
    for d in departures(obs):
        look = min(d["gone"] + 1.0, obs["duration"] - 0.1)  # the close-out frame: room empty, view free
        on_desk = items(at(desk, look)) | items(at(desk, d["gone"]))
        s = at(safe, look)
        lp = at(lap, look)
        ev_t = [d["t"], d["gone"], look]
        if "cash" in on_desk:
            ev.append(Event("B1", d["t"], "cash_left", ev_t))
        if "collection_bag" in on_desk:
            ev.append(Event("B1", d["t"], "bag_left", ev_t))
        if "stamps" in on_desk:
            ev.append(Event("B8", d["t"], "stamps_left", ev_t))
        if safe_open(s):
            ev.append(Event("B5", d["t"], "safe_open_left", ev_t))
        elif safe_key(s):
            ev.append(Event("B5", d["t"], "key_left", ev_t))
        if lp == "active":
            ev.append(Event("B9", d["t"], "laptop_active_left", ev_t))
        if d["visitors"] > 0:
            ev.append(Event("A4", d["t"], "left_with_visitor", ev_t))
    return ev


def entries(obs: dict) -> list[float]:
    """Times someone comes in: the visitor count goes up, or the cashier goes from absent to present."""
    ppl = series(obs, "people")
    out = []
    for (_, x), (t, y) in zip(ppl, ppl[1:]):
        if visitors(y) > visitors(x) or (cashier(x) == "absent" and cashier(y) not in ("absent", "unknown")):
            out.append(t)
    return out


def rule_door(obs: dict) -> list[Event]:
    """The door stays open after someone passes. Timecode: the first entry during the open run + C2_AFTER_PASS_S,
    when the door should have been shut again (X-32: entry 3.0 s, label 5; X-33: entry 4.5 s, label 6)."""
    door = series(obs, "door")
    end = obs["duration"]
    for a, b in runs(door, lambda x: x == "open", end):
        if b - a >= C2_MIN_S or (b >= end - 1e-6 and b - a >= C2_MIN_END_S):
            ins = [t for t in entries(obs) if a - 1.0 <= t < b]
            tc = min((ins[0] if ins else a) + C2_AFTER_PASS_S, b - 0.5, end - 0.1)
            return [Event("C2", tc, "door_left_open", [tc, min(tc + 2.0, end - 0.1), min(tc + 4.0, end - 0.1)],
                          {"open": [a, round(b, 2)]})]
    return []


def rule_safe_open(obs: dict) -> list[Event]:
    """Safe open while the cashier works at the desk, B5_DESK_S or longer (walking to and from the safe is
    part of using it)."""
    safe, ppl = series(obs, "safe"), series(obs, "people")
    end = obs["duration"]
    for a, b in runs(safe, safe_open, end):
        pts = [(t, cashier(x)) for t, x in ppl if a <= t < b]
        away = runs([(t, c) for t, c in pts], lambda c: c == "desk", b)
        for x, y in away:
            if y - x >= B5_DESK_S:
                return [Event("B5", x, "safe_open_at_work", [x, min(x + 1.5, end - 0.1)], {"open": [a, b]})]
    return []


def rule_personal(obs: dict) -> list[Event]:
    """A personal item put on the desk. Only items that appear during the clip count: an item named from the
    first frame on is usually a device misread (X-10: the second black device read as a handbag all clip).
    A phone is not used: the client's ID card and the desk telephone are read as 'phone' too (X-8)."""
    desk, ppl = series(obs, "desk"), series(obs, "people")
    first = items(desk[0][1]) if desk else set()
    ev = []
    for item in PERSONAL:
        if item in first:
            continue
        # with a visitor in the room the item may be the visitor's (X-5: a client's handbag)
        hits = [t for t, a in desk if item in items(a) and visitors(at(ppl, t)) == 0]
        if len(hits) >= B3_MIN_SAMPLES and hits[-1] - hits[0] >= B3_MIN_SPAN_S:
            ev.append(Event("B3", hits[0], f"personal_{item}", [hits[0], hits[1]], {"item": item}))
    return ev[:1] if ev else []


def rule_sessions(obs: dict) -> list[Event]:
    desk = series(obs, "desk")
    end = obs["duration"]
    ev = []
    sess = sessions(obs)
    for i, s in enumerate(sess):
        a, b = s["start"], s["end"]
        # B1: cash from before lies on the desk when the next client comes in
        # (+1.5 s: the bundle is often hidden by the person walking in; X-4 sees it 0.5 s after the arrival)
        before = [t for t, x in desk if a - 2.5 <= t <= a + 1.5]
        if i > 0 and any("cash" in items(at(desk, t)) for t in before):
            ev.append(Event("B1", a, "cash_on_arrival", [a, min(a + 1.5, end - 0.1)]))
        # A7: two visitors in the room and a handover while both are there
        for x, y in s["two"]:
            if windows(obs, x, y, "cash_in") or windows(obs, x, y, "card_in") or windows(obs, x, y, "bag_in"):
                ev.append(Event("A7", x, "two_visitors", [x, min(x + 1.5, end - 0.1)]))
                break
        # A1: the client leaves while the counting machine is still running
        # (counting both in the window that ends just before the exit and in the one around it: X-34 12-15.5 s;
        # X-35, clean, has one stray 'machine' window as the client leaves and nothing in the 6 s before)
        if b < end:
            late = [w for w in obs["actions"] if w.get("machine") and b - 1.0 <= w["t0"] <= b + 2.0]
            earlier = [w for w in obs["actions"] if w.get("machine") and b - 3.5 <= w["t0"] and w["t1"] <= b]
            if late and earlier:
                ev.append(Event("A1", max(a, b - 1.0), "left_during_count", [max(a, b - 1.0), b]))
        # D7: a collection bag taken with no ID card / papers shown. X-16 (clean): the collector puts the bag
        # down at 6.5 s and hands the cashier his ID card at 8-9 s, before she takes the bag; X-17: she pulls
        # the bag in at once. So: no card or papers from the arrival until D7_AFTER_BAG_S after the bag lands.
        bag_desk = [t for t, x in desk if a <= t <= b and "collection_bag" in items(x)]
        bag = windows(obs, a, b, "bag_in")
        if bag_desk or bag:
            tb = bag_desk[0] if bag_desk else bag[0]["t1"]
            shown = windows(obs, a, tb + D7_AFTER_BAG_S, "card_in") + windows(obs, a, tb + D7_AFTER_BAG_S, "papers")
            if not shown:
                ev.append(Event("D7", tb, "bag_without_papers", [tb, min(tb + 1.5, end - 0.1)]))
        # A6: the client's cash lies on the desk before any ID card did. The desk question reports the
        # client's ID card as a 'phone' (a small flat card); the action windows over-report card handovers,
        # so only the desk is used here.
        elif not before_cash(desk, a):
            tc = first_robust(desk, "cash", a, b)
            if tc is not None and not any("phone" in items(x) for t, x in desk if a - 0.5 <= t < tc):
                ev.append(Event("A6", tc, "cash_without_id", [tc, min(tc + 1.5, end - 0.1)]))
    return ev


def before_cash(desk, a: float) -> bool:
    """Cash already on the desk when the session starts (it cannot be told apart from the client's)."""
    return any("cash" in items(x) for t, x in desk if a - 2.5 <= t <= a)


def first_robust(desk, item: str, a: float, b: float) -> float | None:
    """First time `item` is on the desk in two neighbouring asked samples within [a, b]."""
    pts = [(t, item in items(x)) for t, x in desk if a <= t <= b]
    for (t1, h1), (_, h2) in zip(pts, pts[1:]):
        if h1 and h2:
            return t1
    return None


def rule_collection(obs: dict) -> list[Event]:
    """D6: contents of a collection bag go into the safe with no count and no laptop entry."""
    desk, safe = series(obs, "desk"), series(obs, "safe")
    end = obs["duration"]
    bag_t = [t for t, a in desk if "collection_bag" in items(a)]
    bag_t += [w["t0"] for w in obs["actions"] if w.get("bag_open")]
    if not bag_t:
        return []
    t_bag = min(bag_t)
    for a, b in runs(safe, safe_open, end):
        if a <= t_bag:
            continue
        counted = windows(obs, t_bag, a, "machine")
        typed = windows(obs, t_bag, min(b + 10.0, end), "typing")
        if not counted and not typed:
            return [Event("D6", a, "bag_to_safe_unregistered", [a, min(a + 1.5, end - 0.1)])]
        return []
    return []


RULES = [rule_departures, rule_door, rule_safe_open, rule_personal, rule_sessions, rule_collection]


def merge(events: list[Event]) -> list[Event]:
    out: list[Event] = []
    for e in sorted(events, key=lambda e: e.t):
        if any(o.code == e.code and abs(o.t - e.t) <= MERGE_S for o in out):
            continue
        if any(o.code == e.code and o.reason == e.reason and e.reason.endswith("_left")
               and e.t - o.t <= DEPARTURE_MERGE_S for o in out):
            continue
        out.append(e)
    return out


def detect(obs: dict, enabled: set[str] | None = None) -> list[Event]:
    ev = []
    for rule in RULES:
        ev += rule(obs)
    if enabled is not None:
        ev = [e for e in ev if e.code in enabled]
    for e in ev:
        e.t = round(float(e.t), 2)
        e.evidence_t = [round(float(x), 2) for x in (e.evidence_t or [e.t])][:3]
        e.evidence_t[0] = e.t
    return merge(ev)

