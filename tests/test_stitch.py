import pandas as pd

from engine.facts import stitch_tracks


def _rows(tid, ts, fx, fy):
    return [{"t": t, "track_id": tid, "fx": fx, "fy": fy} for t in ts]


def test_new_id_in_same_spot_is_joined():
    room = pd.DataFrame(_rows(1, [0.0, 0.33, 0.67], 100, 400) + _rows(7, [1.67, 2.0], 110, 405))
    out = stitch_tracks(room, h_ref=200)
    assert out.track_id.nunique() == 1


def test_two_people_at_once_stay_apart():
    room = pd.DataFrame(_rows(1, [0.0, 0.33, 0.67], 100, 400) + _rows(2, [0.67, 1.0], 105, 400))
    assert stitch_tracks(room, h_ref=200).track_id.nunique() == 2


def test_far_away_or_late_start_stays_apart():
    far = pd.DataFrame(_rows(1, [0.0, 0.33], 100, 400) + _rows(2, [1.0, 1.33], 400, 400))
    late = pd.DataFrame(_rows(1, [0.0, 0.33], 100, 400) + _rows(2, [5.0, 5.33], 100, 400))
    assert stitch_tracks(far, h_ref=200).track_id.nunique() == 2
    assert stitch_tracks(late, h_ref=200).track_id.nunique() == 2
