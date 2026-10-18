import numpy as np

from engine.events import Event, merge, segments


def test_segments_hysteresis():
    t = np.arange(0, 10, 0.5)
    on = np.zeros_like(t, bool)
    on[(t >= 1) & (t <= 3)] = True
    on[t == 2.0] = False          # short dropout is bridged
    on[(t >= 7) & (t <= 7.5)] = True  # too short
    assert segments(t, on, min_on=1.0, max_gap=1.0) == [(1.0, 3.0)]


def test_merge_same_code_close_events():
    ev = merge([Event("B1", 5, 6, 0.4), Event("B1", 7, 8, 0.7), Event("A7", 7, 8, 0.5), Event("B1", 20, 21, 0.5)])
    assert [(e.code, e.start, e.end, e.score) for e in ev] == [("B1", 5, 8, 0.7), ("A7", 7, 8, 0.5), ("B1", 20, 21, 0.5)]


def test_departure_b5_replaces_overlapping_open_while_working():
    from engine import _resolve
    work = Event("B5", 10.0, 16.7, 0.6, info={"reason": "safe_open_while_working"})
    key = Event("B5", 13.3, 16.7, 0.6, info={"reason": "key_left_cashier_left"})
    b9 = Event("B9", 13.3, 16.7, 0.6, info={"reason": "laptop_active_cashier_left"})
    out = _resolve([work, key, b9])
    assert sorted((e.code, e.info["reason"]) for e in out) == [("B5", "key_left_cashier_left"),
                                                              ("B9", "laptop_active_cashier_left")]
    early = Event("B5", 2.0, 8.0, 0.6, info={"reason": "safe_open_while_working"})  # closed long before
    assert len(_resolve([early, key])) == 2
