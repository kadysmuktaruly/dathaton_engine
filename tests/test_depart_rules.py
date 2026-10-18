"""Departure rules (B1, B5, B8, B9) on synthetic facts."""
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.rules import depart

DEP = {"t_exit": 10.0, "t_depart": 8.0, "absent_s": 3.0, "to_end": True, "box": [1, 2, 3, 4], "track": 1}
COLS = ["t", "cls", "conf", "x0", "y0", "x1", "y1"]


def _facts(objects=(), checks=None, laptop=None):
    hands = pd.DataFrame({"t": [7.0, 9.0], "machine": [False, False], "laptop": [False, True], "safe": [False, False]})
    return SimpleNamespace(objects=pd.DataFrame(list(objects), columns=COLS), checks=checks, scan=None, hands=hands,
                           scene=SimpleNamespace(laptop=laptop, safe=None), video=None, perc=None)


@pytest.fixture(autouse=True)
def one_departure(monkeypatch):
    monkeypatch.setattr(depart, "departures", lambda f: [DEP])


def _codes(evs):
    return sorted(e.code for e in evs)


def test_plain_departure_is_not_a_violation():
    assert depart.detect(_facts()) == []


def test_open_safe_while_room_empty_is_b5():
    so = [(11.0, "safe_open", 0.7, 380, 200, 440, 330), (11.33, "safe_open", 0.8, 380, 200, 440, 330)]
    evs = depart.detect(_facts(so))
    assert _codes(evs) == ["B5"] and evs[0].info["reason"] == "safe_open_cashier_left"


def test_open_safe_before_leaving_does_not_count():
    so = [(5.0, "safe_open", 0.9, 380, 200, 440, 330), (6.0, "safe_open", 0.9, 380, 200, 440, 330)]
    assert depart.detect(_facts(so)) == []


def test_tv_lookalike_is_never_an_open_safe():
    assert depart.detect(_facts([(11.0, "tvmonitor", 0.9, 380, 200, 440, 330)] * 3)) == []


def test_checks_give_b8_and_b1():
    chk = {"departures": [{"t_exit": 10.0, "stamps": True, "cash": True, "collection_bag": False}]}
    assert _codes(depart.detect(_facts(checks=chk))) == ["B1", "B8"]


def test_leave_time_uses_last_device_touch():
    # walk starts at 8.0, but the hands were at the laptop until 9.0
    assert depart.leave_time(_facts(), DEP) == 9.0


def test_b9_lit_screen_locked_vs_active(monkeypatch):
    monkeypatch.setattr(depart, "screen_dark_share", lambda *a: [0.05, 0.06])
    f = _facts(laptop=SimpleNamespace(as_list=lambda: [0, 0, 10, 10]))
    f.video, f.perc = "v.mov", SimpleNamespace(info=SimpleNamespace(duration=20.0))
    ev = [e for e in depart.detect(f) if e.code == "B9"][0]
    assert ev.score == depart.B9_SCORE
    f.checks = {"departures": [{"t_exit": 10.0, "lock_screen": True}]}
    ev = [e for e in depart.detect(f) if e.code == "B9"][0]
    assert ev.score == depart.B9_LOCKED_SCORE  # a lock screen is not an open session


def test_dark_screen_is_not_b9(monkeypatch):
    monkeypatch.setattr(depart, "screen_dark_share", lambda *a: [0.36, 0.35])
    f = _facts(laptop=SimpleNamespace(as_list=lambda: [0, 0, 10, 10]))
    f.video, f.perc = "v.mov", SimpleNamespace(info=SimpleNamespace(duration=20.0))
    assert depart.detect(f) == []


def test_c2_timecode_prefers_visitor_through_open_door():
    from engine.roles import Session
    from engine.rules.c2 import first_visitor_through
    p = pd.DataFrame({"t": [13.0, 14.0], "track_id": [3, 3], "d_desk": [0.4, 0.3], "role": ["visitor"] * 2})
    f = SimpleNamespace(persons=p, sessions=[Session(1, 13.0, 18.0, [3])])
    assert first_visitor_through(f, 3.0).enter == 13.0
    assert first_visitor_through(f, 12.5) is None  # entered with the opening itself: that's the door time


def _key_facts(monkeypatch, key: bool, provenance="explicit:safe"):
    seen = {}

    def fake_check(video, perc, safe, times):
        seen["times"] = times
        return {"key": key, "score": 0.09 if key else 0.01, "dot": 0.4, "lock": [198, 302], "n_frames": len(times)}

    from engine import safe_key
    monkeypatch.setattr(safe_key, "check", fake_check)
    f = _facts()
    f.scene = SimpleNamespace(laptop=None, safe=SimpleNamespace(w=73.0), provenance={"safe": provenance})
    f.video, f.perc = "v.mov", SimpleNamespace(info=SimpleNamespace(duration=20.0))
    return f, seen


def test_key_left_in_safe_is_b5(monkeypatch):
    f, seen = _key_facts(monkeypatch, key=True)
    evs = depart.detect(f)
    assert _codes(evs) == ["B5"] and evs[0].info["reason"] == "key_left_cashier_left"
    assert evs[0].start == 9.0  # leave time, like the other departure rules
    assert min(seen["times"]) > DEP["t_exit"] and max(seen["times"]) <= DEP["t_exit"] + DEP["absent_s"]


def test_key_taken_is_not_b5(monkeypatch):
    f, _ = _key_facts(monkeypatch, key=False)
    assert depart.detect(f) == []


def test_key_check_needs_fine_tuned_safe(monkeypatch):
    f, seen = _key_facts(monkeypatch, key=True, provenance="proxy:tall box")
    assert depart.detect(f) == [] and not seen
