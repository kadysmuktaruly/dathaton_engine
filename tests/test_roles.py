"""Roles: cashier vs visitors per frame, visitor sessions, collector tagging, hand signals."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from engine import roles
from engine.scene import Box, L_WRIST


def _row(t, track, fx, fy, d_seat, d_desk=0.3, kps=None):
    return {"t": round(float(t), 4), "track_id": track, "fx": fx, "fy": fy, "d_seat": d_seat, "d_desk": d_desk,
            "x0": fx - 20, "y0": fy - 150, "x1": fx + 20, "y1": fy, "kps": kps}


def _people(rows):
    return roles.split_people(pd.DataFrame(rows))


def test_nearest_to_seat_is_cashier_each_frame():
    p = _people([_row(1.0, 1, 200, 570, 0.0), _row(1.0, 2, 150, 420, 0.6), _row(2.0, 7, 200, 570, 0.05)])
    assert p.role.tolist() == ["cashier", "visitor", "cashier"]  # new track id, still the cashier


def test_session_survives_an_id_switch():
    rows = [_row(t, 1, 200, 570, 0.0) for t in np.arange(0, 10, 1 / 3)]
    rows += [_row(t, 2, 150, 420, 0.6) for t in np.arange(2, 5, 1 / 3)]
    rows += [_row(t, 9, 152, 421, 0.6) for t in np.arange(17, 27) / 3]  # same visitor, new id (from 5.67 s)
    s = roles.sessions(_people(rows), h_ref=190, duration=10.0)
    assert len(s) == 1 and s[0].tracks == [2, 9] and s[0].idx == 1
    assert abs(s[0].enter - 2.0) < 1e-6 and s[0].ended


def test_two_visitors_two_sessions_and_short_noise_dropped():
    rows = [_row(t, 1, 200, 570, 0.0) for t in np.arange(0, 20, 1 / 3)]
    rows += [_row(t, 2, 150, 420, 0.6) for t in np.arange(2, 15, 1 / 3)]
    rows += [_row(t, 3, 330, 430, 0.8) for t in np.arange(12, 20, 1 / 3)]
    rows += [_row(t, 4, 60, 300, 1.2) for t in np.arange(5, 6, 1 / 3)]   # 1 s blip
    s = roles.sessions(_people(rows), h_ref=190, duration=20.0)
    assert [x.tracks for x in s] == [[2], [3]] and not s[1].ended


def test_collector_tagged_by_bag_evidence():
    s = [roles.Session(1, 4.0, 11.0, [2]), roles.Session(2, 20.0, 30.0, [3])]
    obj = pd.DataFrame({"t": [5.0, 6.0, 7.0], "cls": ["collection_bag"] * 3})
    roles.tag_collectors(s, obj)
    assert [x.role for x in s] == ["collector", "client"]


def test_hands_at_devices():
    k = np.zeros((17, 3), np.float32)
    k[L_WRIST] = [105, 405, 0.9]
    p = _people([_row(1.0, 1, 200, 570, 0.0, kps=k.ravel().tolist())])
    scene = SimpleNamespace(machine=Box(80, 380, 130, 430), laptop=Box(300, 380, 350, 430), safe=None)
    h = roles.hands(p, scene, [1.0, 1.33])
    assert h.machine.tolist() == [True, False] and not h.laptop.any()
    assert roles.dwell(h, "machine", 0.0, 2.0) > 0
