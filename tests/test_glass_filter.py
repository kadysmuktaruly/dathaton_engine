"""People seen through the glass wall must not count as being in the room."""
import numpy as np
import pandas as pd

from engine.scene import Box, Scene, clean_persons, in_room_mask


def _persons(rows):
    return pd.DataFrame([{"t": t, "kind": "person", "cls": "person", "conf": c, "x0": x0, "y0": y0,
                          "x1": x1, "y1": y1, "track_id": tr, "kps": None}
                         for t, tr, c, (x0, y0, x1, y1) in rows])


def _scene():
    # desk across the middle of a 464x848 frame; its far edge is at y ~ 400
    return Scene(464, 848, desk=Box(40, 380, 320, 550), person_h_ref=220.0, back_line=400.0)


def test_person_behind_glass_is_excluded():
    p = _persons([
        (1.0, 1, 0.9, (180, 380, 290, 590)),   # cashier in front of the desk
        (1.0, 2, 0.9, (110, 300, 210, 415)),   # client seated behind the desk (box bottom at desk edge)
        (1.0, 3, 0.8, (0, 220, 45, 300)),      # person in the office outside, seen through the glass
        (1.0, 4, 0.6, (60, 230, 100, 368)),    # reflection / person walking behind the glass
    ])
    m = in_room_mask(p, _scene())
    assert m.tolist() == [True, True, False, False]


def test_person_coming_through_door_counts_once_inside():
    # walking toward the door behind the glass, then through it
    p = _persons([(0.0, 7, 0.9, (20, 200, 70, 360)), (1.0, 7, 0.9, (20, 220, 80, 410))])
    assert in_room_mask(p, _scene()).tolist() == [False, True]


def test_clean_persons_drops_fragments_and_group_boxes():
    p = _persons([
        (1.0, 1, 0.9, (180, 380, 290, 590)),
        (1.0, 2, 0.9, (110, 300, 210, 415)),
        (1.0, 3, 0.5, (100, 290, 300, 600)),   # group box spanning both
        (1.0, 4, 0.4, (220, 365, 244, 397)),   # tiny fragment (a hand)
    ])
    out = clean_persons(p, 220.0)
    assert sorted(out.track_id.tolist()) == [1, 2]


def test_mask_without_desk_keeps_everyone():
    p = _persons([(1.0, 1, 0.9, (0, 0, 10, 10))])
    s = Scene(464, 848, desk=None)
    assert in_room_mask(p, s).tolist() == [True]
    assert isinstance(in_room_mask(p.iloc[0:0], s), np.ndarray)
