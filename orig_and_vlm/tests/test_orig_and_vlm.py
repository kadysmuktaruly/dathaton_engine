from types import SimpleNamespace

from engine import vlm_scan
from engine.events import Event
from engine.scene import Box
from orig_and_vlm import closeout, scan

PERC = SimpleNamespace(info=SimpleNamespace(crop=(0, 0, 464, 848), duration=20.0))
FA = SimpleNamespace(scene=SimpleNamespace(desk=None, safe=None, laptop=Box(300, 300, 360, 340)))


def _leave(t, t_exit):
    return Event("B5", t, t_exit, 0.5, info={"reason": "cashier_left_room", "t_exit": t_exit})


def test_closeout_drops_only_failed_leave_alarms(monkeypatch):
    answers = {6.0: {"keep": False, "why": "someone is still in the room"},   # X-23: seated cashier
               15.0: {"keep": True, "why": "left exposed: screen"}}           # X-28: lit screen
    seen = []

    def fake_check(video, perc, desk, lap, t_exit):
        seen.append((t_exit, lap))
        return dict(answers[t_exit])

    monkeypatch.setattr(closeout, "check", fake_check)
    other = Event("B1", 9.0, 9.0, 0.75, info={"reason": "cash_left_cashier_away"})
    kept, dbg = closeout.filter_events("v.mov", PERC, FA, [_leave(5.0, 6.0), other, _leave(13.0, 15.0)])
    assert [(e.code, e.start) for e in kept] == [("B1", 9.0), ("B5", 13.0)]
    assert [d["t"] for d in dbg] == [5.0, 13.0]
    assert seen[0][1] == (180, 220, 464, 364)  # laptop box with room around it, clipped to the frame


def test_closeout_lifts_a_gated_alarm_it_confirms(monkeypatch):
    from engine.rules import b5
    monkeypatch.setattr(closeout, "check", lambda *a: {"keep": True, "why": "left exposed: screen"})
    gated = _leave(5.0, 6.0)
    gated.score = b5.NO_EVIDENCE_SCORE  # fine-tuned detector saw no open safe
    kept, _ = closeout.filter_events("v.mov", PERC, FA, [gated])
    assert kept[0].score == b5.LEAVE_SCORE and kept[0].info["closeout_lifted"]


def test_closeout_keeps_alarm_without_budget(monkeypatch):
    monkeypatch.setattr(closeout, "check", lambda *a: (_ for _ in ()).throw(AssertionError("must not run")))
    budget = SimpleNamespace(ok=lambda need: False)
    kept, dbg = closeout.filter_events("v.mov", PERC, FA, [_leave(5.0, 6.0)], budget)
    assert len(kept) == 1 and kept[0].info["closeout"] == "skipped (budget)" and dbg == []


def test_laptop_box_none_without_laptop():
    assert closeout.laptop_box(SimpleNamespace(laptop=None), 464, 848) is None


def test_scan_keys_cover_the_engine_flags():
    assert sorted(scan.KEYS.values()) == sorted(vlm_scan.FLAGS)
    assert set(scan.SINGLE) == set(vlm_scan.FLAGS)
