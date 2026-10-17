"""Regression tests for the perception review: context classes, raw vs tracked people, the
per-frame cap, anchor provenance. Synthetic data only; nothing here measures the model."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from engine import facts as F
from engine.classes import CONTEXT_ROLES, runtime_keep
from engine.perception import filter_detections, match_raw_to_tracks
from engine.rules import b5
from engine.scene import build_scene

T = [0.0, 0.33, 0.67, 1.0]


def _row(kind, cls, conf, t, box, track=-1):
    return {"t": t, "frame_idx": 0, "kind": kind, "cls": cls, "conf": conf,
            "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3], "track_id": track, "kps": None}


def _person(t, x, track, conf=0.9, kind="person", w=60, h=200):
    return _row(kind, "person", conf, t, (x, 300, x + w, 300 + h), track)


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setattr(F, "COUNT_UNTRACKED", False)
    monkeypatch.setattr(F, "MAX_PERSONS", 3)
    return monkeypatch


# --- 1. context classes ---------------------------------------------------------------------

def test_context_classes_kept_but_never_anchors():
    keep = runtime_keep({0: "handbag", 1: "backpack", 2: "cell phone", 3: "cup", 4: "chair"})
    assert keep == {"handbag", "backpack", "cell phone", "cup"}  # stock name maps to id_card
    rows = [_row("obj", c, 0.9, float(i), (300, 300, 360, 340)) for i in range(10) for c in ("handbag", "cup")]
    df = filter_detections(pd.DataFrame(rows))
    assert set(df.role) <= CONTEXT_ROLES
    s = build_scene(df, 464, 848, 10)
    assert s.desk is None and s.machine is None and s.safe is None and s.laptop is None


# --- 2. raw vs tracked people ---------------------------------------------------------------

def test_raw_boxes_match_tracks_by_iou():
    tracked = np.array([[100, 300, 160, 500]], dtype=np.float32)
    raw = np.array([[102, 302, 161, 498], [300, 300, 360, 500]], dtype=np.float32)
    assert match_raw_to_tracks(raw, tracked, [7]) == [7, -1]
    assert match_raw_to_tracks(raw, np.zeros((0, 4)), []) == [-1, -1]


def _two_people_one_untracked():
    rows = []
    for t in T:
        rows += [_person(t, 100, 1), _person(t, 100, 1, kind="person_raw")]   # tracked + its raw box
        rows += [_person(t, 300, -1, kind="person_raw", conf=0.6)]            # seen, never tracked
    return filter_detections(pd.DataFrame(rows))


def test_untracked_person_counted_only_when_enabled(settings):
    df = _two_people_one_untracked()
    off = F.build(df, T, 464, 848)
    assert off.frames.n_room.tolist() == [1, 1, 1, 1]
    assert (off.person_log.reason == "untracked (not counted)").sum() == 4
    settings.setattr(F, "COUNT_UNTRACKED", True)
    on = F.build(df, T, 464, 848)
    assert on.frames.n_room.tolist() == [2, 2, 2, 2]


def test_raw_box_of_a_tracked_person_is_not_counted_twice(settings):
    settings.setattr(F, "COUNT_UNTRACKED", True)
    rows = [r for t in T for r in (_person(t, 100, 1), _person(t, 100, 1, kind="person_raw"))]
    fa = F.build(filter_detections(pd.DataFrame(rows)), T, 464, 848)
    assert fa.frames.n_room.tolist() == [1, 1, 1, 1]


def test_overlapping_untracked_boxes_count_once(settings):
    settings.setattr(F, "COUNT_UNTRACKED", True)
    rows = [r for t in T for r in (_person(t, 300, -1, kind="person_raw", conf=0.8),
                                   _person(t, 305, -1, kind="person_raw", conf=0.6, w=50, h=190))]
    fa = F.build(filter_detections(pd.DataFrame(rows)), T, 464, 848)
    assert fa.frames.n_room.tolist() == [1, 1, 1, 1]
    assert "duplicate" in set(fa.person_log.reason)


# --- 3. the per-frame cap -------------------------------------------------------------------

def _four_people():
    return filter_detections(pd.DataFrame([_person(t, 20 + 110 * k, k + 1, conf=0.95 - 0.1 * k)
                                           for t in T for k in range(4)]))


def test_cap_keeps_most_confident_and_logs_the_rest(settings):
    fa = F.build(_four_people(), T, 464, 848)
    assert fa.frames.n_room.tolist() == [3, 3, 3, 3]
    capped = fa.person_log[fa.person_log.reason == "over per-frame cap"]
    assert len(capped) == 4 and set(capped.track_id) == {4}           # least confident one


def test_no_cap_keeps_everyone_and_still_drops_duplicates(settings):
    settings.setattr(F, "MAX_PERSONS", None)
    rows = [_person(t, 20 + 110 * k, k + 1) for t in T for k in range(4)]
    rows += [_person(t, 22, 9, conf=0.5, w=55, h=190) for t in T]      # duplicate of person 1
    fa = F.build(filter_detections(pd.DataFrame(rows)), T, 464, 848)
    assert fa.frames.n_room.tolist() == [4, 4, 4, 4]
    assert set(fa.person_log[fa.person_log.track_id == 9].reason) == {"duplicate"}


# --- 4. anchor provenance -------------------------------------------------------------------

def _obj(cls, conf, box, n=10):
    return [_row("obj", cls, conf, float(i), box) for i in range(n)]


def test_provenance_explicit_vs_proxy():
    proxy = build_scene(filter_detections(pd.DataFrame(
        _obj("laptop", 0.9, (300, 300, 360, 340)) + _obj("tv", 0.6, (200, 300, 250, 340)) +
        _obj("tv", 0.6, (380, 200, 420, 330)))), 464, 848, 10)
    assert proxy.provenance["desk"].startswith("proxy:")
    assert proxy.provenance["machine"] == "proxy:tv/suitcase"
    assert proxy.provenance["safe"] in (None, "proxy:tall box")
    explicit = build_scene(filter_detections(pd.DataFrame(
        _obj("desk", 0.8, (100, 280, 420, 480)) + _obj("counting_machine", 0.8, (200, 300, 250, 340)) +
        _obj("safe", 0.8, (380, 200, 440, 330)))), 464, 848, 10)
    assert explicit.provenance == {"desk": "explicit:desk", "machine": "explicit:counting_machine",
                                   "safe": "explicit:safe"}


def test_proxy_safe_is_never_an_open_safe(monkeypatch):
    dep = {"t_exit": 10.0, "t_depart": 8.0, "absent_s": 3.0, "to_end": True, "box": [1, 2, 3, 4], "track": 1}
    monkeypatch.setattr(b5, "departures", lambda f: [dep])
    scene = SimpleNamespace(safe=object(), provenance={"safe": "proxy:tall box"})
    objects = pd.DataFrame([(11.0, "tvmonitor", 0.9, 380, 200, 440, 330)] * 3,
                           columns=["t", "cls", "conf", "x0", "y0", "x1", "y1"])
    # trained-detector vocabulary: a proxy safe plus tv boxes are no evidence of an open safe
    f = SimpleNamespace(perc=SimpleNamespace(labels=frozenset({"safe", "safe_open"})), objects=objects, scene=scene)
    ev = b5.detect(f)[0]
    assert ev.score == b5.NO_EVIDENCE_SCORE and ev.info["reason"] == "cashier_left_room"


# --- 7. inspector reasons -------------------------------------------------------------------

def test_inspector_explains_each_box():
    from tools.inspect_detections import explain_detections
    keep = runtime_keep({0: "person", 1: "laptop", 2: "cup", 3: "chair"})
    plog = pd.DataFrame([
        {"t": 1.0, "track_id": 4, "conf": 0.9, "x0": 100, "y0": 300, "x1": 160, "y1": 500, "source": "tracked", "reason": "counted"},
        {"t": 1.0, "track_id": -1, "conf": 0.6, "x0": 300, "y0": 300, "x1": 360, "y1": 500, "source": "untracked",
         "reason": "untracked (not counted)"}])
    dets = [("person", 0.9, (101, 301, 160, 499)), ("person", 0.6, (300, 300, 360, 500)), ("person", 0.3, (10, 10, 50, 90)),
            ("laptop", 0.8, (1, 1, 30, 30)), ("laptop", 0.4, (1, 1, 30, 30)), ("cup", 0.7, (5, 5, 15, 15)),
            ("chair", 0.9, (5, 5, 40, 40))]
    ex = explain_detections(dets, plog, keep)
    assert [e["reason"] for e in ex] == ["counted", "untracked (not counted)", "below person floor", "reported",
                                         "layout only (below reporting cutoff)", "reported (context only)", "class not used"]
    assert ex[0]["track"] == 4 and ex[1]["track"] is None
    assert [e["retained"] for e in ex] == [True, False, False, True, True, True, False]
