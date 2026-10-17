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
    ev = merge([Event("B1", 5, 6, 0.4), Event("B1", 7, 8, 0.7), Event("C1", 7, 8, 0.5), Event("B1", 20, 21, 0.5)])
    assert [(e.code, e.start, e.end, e.score) for e in ev] == [("B1", 5, 8, 0.7), ("C1", 7, 8, 0.5), ("B1", 20, 21, 0.5)]
