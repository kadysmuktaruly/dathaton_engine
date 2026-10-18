"""Second look, with narrow yes/no questions on high-resolution crops at an event's own frames.

1. checks: every candidate event from timeline.py gets the questions in CHECKS; it is dropped if they say no.
   The list-style state questions over-report (X-10: a device read as a handbag and clothing all clip); a direct
   question about one object is more reliable.
2. close-out probe: at each departure the empty desk is asked whether anything is left on it. The state questions
   miss X-24's teal stamps (listed as 'drink'/'phone'); this probe sees 'a tablet'. It can add events.

Answers are cached in the clip's answer cache under 'verify', so the eval can replay them without the model.
"""
from __future__ import annotations

import json

from pure_vlm import config, observe, prompts, timeline, vlm

YESNO = " Answer yes or no."
DESK = ("Zoomed ceiling CCTV view of a bank cashier's desk. Ignore the counting machine, the banknote detector, "
        "the laptop and the desk telephone. ")
SAFE_OPEN = ("safe", "Zoomed ceiling CCTV view of a grey metal safe. Is the safe door open?" + YESNO, 512, "all")
# X-20: at the moment she reaches the desk she is still sitting down ('no' at 5.0 s, 'yes' at 6.5 s): look later
SITTING = ("people", "Ceiling CCTV view of a bank cash room. Is the cashier sitting on a chair at the desk (not "
           "standing at the safe)?" + YESNO, 640, "any", (1.5, 3.0))
CASH = ("desk", DESK + "Is paper money (banknotes or a bundle of banknotes) lying on the desk?" + YESNO, 768, "any")
# (code, reason) -> questions; every question must pass. rule 'all': every frame says yes, 'any': one is enough.
CHECKS = {
    ("B3", "personal_handbag"): [("desk", DESK + "Is a personal handbag, purse or backpack lying on the desk?"
                                  + YESNO, 768, "any")],
    ("B3", "personal_clothing"): [("desk", DESK + "Is a jacket, coat or other piece of clothing lying on the desk?"
                                   + YESNO, 768, "any")],
    ("B3", "personal_drink"): [("desk", DESK + "Is there a cup, mug, glass or bottle on the desk or in the "
                                "cashier's hand?" + YESNO, 768, "any")],
    ("B1", "cash_left"): [CASH],
    ("B1", "cash_on_arrival"): [CASH],
    ("B1", "bag_left"): [("desk", DESK + "Is a sealed cash-collection bag or sack lying on the desk?" + YESNO, 768,
                          "any")],
    ("B8", "stamps_left"): [("desk", DESK + "Are office rubber stamps or seals lying on the desk?" + YESNO, 768,
                             "any")],
    # X-25: the cashier stands at a safe right beside the desk (answered 'desk'); X-20: she sits at the desk
    ("B5", "safe_open_at_work"): [SAFE_OPEN, SITTING],
    ("B5", "safe_open_left"): [SAFE_OPEN[:3] + ("any",)],
    ("C2", "door_left_open"): [("door", prompts.DOOR, 640, "all")],
}
# The empty desk after the cashier left. Checked on the departure frames: yes for X-24 ('a tablet': the stamps),
# X-15 (the bag), X-26 (cash); no for X-11, X-12, X-22, X-23, X-25, X-27, X-28, X-29. The stamp- and key-specific
# questions said no everywhere, X-24 and X-22 included.
PROBE_LEFT = ("desk", "Zoomed ceiling CCTV view of a bank cashier's desk after the cashier has left. Ignore the laptop, "
              "the desk telephone, the counting machine and the small banknote detector: they always stand there. Is "
              "anything else left lying on the desk, such as stamps, papers, a tablet, a cup or cash? Answer yes or no "
              "first, then name it.", 768)
CASH_WORDS = ("cash", "banknote", "money", "bill", "note")
BAG_WORDS = ("bag", "sack", "folder", "case", "pouch")


class Frames:
    """Full-resolution frames at the requested times, decoded on first use."""

    def __init__(self, path: str):
        self.path = path
        self._cache: dict[float, object] = {}

    def frame(self, t: float):
        t = round(float(t), 2)
        if t not in self._cache:
            from engine.video import iter_frames_at
            self._cache[t] = next(iter_frames_at(self.path, [t]))[2]
        return self._cache[t]


def _yes(ans) -> bool:
    return isinstance(ans, str) and ans.strip().lower().startswith(("yes", "open"))


def _boxes(obs: dict) -> dict:
    b = dict(obs["boxes"])
    s = obs["zones"].get("safe")
    if s:  # the lock is a few pixels: a tight crop of the safe, upscaled
        w, h = obs["w"], obs["h"]
        pw, ph = 0.08 * (s[2] - s[0]), 0.08 * (s[3] - s[1])
        b["safe_tight"] = [max(0, s[0] - pw), max(0, s[1] - ph), min(w, s[2] + pw), min(h, s[3] + ph)]
    return b


def _answer(frames, box, key: str, question: str, side: int, t: float, vcache: dict, ask: bool):
    k = f"{key}@{t:.2f}"
    if k not in vcache:
        if not ask:
            return None
        q = observe.Question("verify", question, side, 4, "word")
        vcache[k] = observe._ask(q, [vlm.crop(frames.frame(t), box)])
    return vcache[k]


def check(obs: dict, frames, e, vcache: dict, ask: bool = True) -> tuple[bool | None, list]:
    """(keep?, answers). None = not checked (no check for this event, or no model and nothing cached)."""
    specs = CHECKS.get((e.code, e.reason))
    if not specs:
        return None, []
    boxes = _boxes(obs)
    if e.code == "C2":  # well after the pass, when the door should have been shut again
        times = e.evidence_t[1:3] or e.evidence_t
    else:
        times = e.evidence_t[:2]
    answers, verdicts = [], []
    for i, spec in enumerate(specs):
        zone, question, side, rule = spec[:4]
        box = boxes.get(zone)
        if box is None:
            continue
        ts = times[-1:] if rule == "any" and e.code == "B1" else times
        if len(spec) > 4:  # offsets from the event time
            ts = [round(min(e.t + o, obs["duration"] - 0.1) * 2) / 2 for o in spec[4]]
        got = []
        for t in ts:
            a = _answer(frames, box, f"{e.code}:{e.reason}:{i}", question, side, t, vcache, ask)
            if a is None:
                return None, answers
            got.append(_yes(a))
            answers.append((zone, t, a))
        verdicts.append(all(got) if rule == "all" else any(got))
    return (all(verdicts) if verdicts else None), answers


def closeout(obs: dict, frames, vcache: dict, ask: bool = True) -> list:
    """Departure probe on the empty room 1 s after the cashier is gone: anything left on the desk. Cash -> B1,
    a bag -> B1, anything else -> B8 (the stamps, which the state questions call 'drink' or 'phone')."""
    box = obs["boxes"].get("desk")
    if box is None:
        return []
    out = []
    for d in timeline.departures(obs):
        look = round(min(d["gone"] + 1.0, obs["duration"] - 0.1) * 2) / 2
        a = _answer(frames, box, "probe:left", PROBE_LEFT[1], PROBE_LEFT[2], look, vcache, ask) or ""
        if not _yes(a):
            continue
        name = a.lower()
        ev_t = [d["t"], d["gone"], look]
        if any(w in name for w in CASH_WORDS):
            out.append(timeline.Event("B1", d["t"], "cash_left", ev_t, {"probe": a}))
        elif any(w in name for w in BAG_WORDS):
            out.append(timeline.Event("B1", d["t"], "bag_left", ev_t, {"probe": a}))
        else:
            out.append(timeline.Event("B8", d["t"], "items_left", ev_t, {"probe": a}))
    return out


def filter_events(obs: dict, events: list, ask: bool = True) -> tuple[list, list[dict]]:
    """Close-out probes, then the checks: drop events whose check says no. ask=False replays cached answers
    only (events without one are kept). The cache lives next to the clip's other answers."""
    frames = Frames(obs["path"])
    cp = observe.cache_path(obs["md5"])
    cache = json.loads(cp.read_text()) if cp.exists() else {}
    vcache = cache.setdefault("verify", {})
    events = timeline.merge(list(events) + closeout(obs, frames, vcache, ask))
    kept, log = [], []
    for e in events:
        ok, answers = check(obs, frames, e, vcache, ask)
        log.append({"code": e.code, "t": e.t, "reason": e.reason, "keep": ok is not False, "answers": answers})
        if ok is not False:
            kept.append(e)
    if ask and config.USE_CACHE:
        cp.write_text(json.dumps(cache, ensure_ascii=False))
    return kept, log
