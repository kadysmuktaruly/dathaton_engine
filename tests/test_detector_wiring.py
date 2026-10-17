"""Wiring of the fine-tuned detector into the engine. Code tests only: no model is loaded and
nothing here measures detection accuracy."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from engine import config
from engine.classes import CLASSES, IGNORED, NEW_CLASSES, role_of, runtime_keep
from engine.perception import cache_key, clip_box, filter_detections  # noqa: F401
from engine.rules import b5
from engine.scene import build_scene
from tools.inspect_detections import frame_rows, to_original


def _det(cls, conf, t=0.0, box=(100, 100, 200, 160)):
    kind = "person" if cls == "person" else "obj"
    return {"t": t, "frame_idx": 0, "kind": kind, "cls": cls, "conf": conf,
            "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3], "track_id": -1, "kps": None}


def _ckpt(tmp_path, labels=CLASSES, weights=True):
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"id2label": {str(i): n for i, n in enumerate(labels)}}))
    if weights:
        (d / "model.safetensors").write_bytes(b"0" * 16)
    return d


# --- checkpoint selection -------------------------------------------------------------------

def test_default_is_stock(monkeypatch):
    monkeypatch.delenv("ENGINE_DETECTOR_PATH", raising=False)
    assert config.detector_source() == config.DETECTOR_REPO
    assert config.detector_fingerprint() == "stock"
    assert config.detector_labels() is None


def test_missing_checkpoint_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.setenv("ENGINE_DETECTOR_PATH", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="ENGINE_DETECTOR_PATH"):
        config.detector_source()


def test_checkpoint_without_weights_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("ENGINE_DETECTOR_PATH", str(_ckpt(tmp_path, weights=False)))
    with pytest.raises(FileNotFoundError, match="no weights"):
        config.detector_source()


def test_valid_checkpoint_selected_and_fingerprinted(monkeypatch, tmp_path):
    d = _ckpt(tmp_path)
    monkeypatch.setenv("ENGINE_DETECTOR_PATH", str(d))
    assert config.detector_source() == str(d.resolve())
    assert config.detector_labels() == frozenset(CLASSES)
    fp1 = config.detector_fingerprint()
    (d / "config.json").write_text(json.dumps({"id2label": {"0": "cash"}}))
    assert config.detector_fingerprint() != fp1


# --- class definitions ----------------------------------------------------------------------

def test_training_and_runtime_share_classes():
    from tools import det_classes
    assert det_classes.CLASSES is CLASSES
    for name in CLASSES:
        assert role_of(name) is not None or name in IGNORED, name


def test_custom_classes_survive_runtime_filter():
    keep = runtime_keep({i: n for i, n in enumerate(CLASSES)})
    assert set(NEW_CLASSES) <= keep
    assert not keep & IGNORED
    df = pd.DataFrame([_det(c, 0.8) for c in NEW_CLASSES])
    out = filter_detections(df).set_index("cls").role.to_dict()
    assert out == {"cash": "cash", "safe": "safe", "safe_open": "safe", "counting_machine": "counting_machine",
                   "collection_bag": "collection_bag", "desk": "desk"}


# --- boxes ----------------------------------------------------------------------------------

def test_clip_box():
    assert clip_box([-10, -5, 50, 40], 100, 80).tolist() == [0, 0, 50, 40]
    assert clip_box([90, 70, 130, 120], 100, 80).tolist() == [90, 70, 100, 80]
    assert clip_box([150, 10, 200, 40], 100, 80) is None        # fully outside
    assert clip_box([10, 10, 11, 40], 100, 80) is None          # thinner than 2 px


def test_crop_offsets_and_empty_frame_rows():
    crop = (12, 40, 476, 888)
    assert to_original([10, 20, 30, 40], crop) == [22, 60, 42, 80]
    rows = frame_rows("X-1", 3, 1.0, 1.02, [("safe_open", 0.7, np.array([10, 20, 30, 40.0]))], crop)
    r = rows[0]
    assert (r["cx0"], r["cy0"], r["ox0"], r["oy0"]) == (10, 20, 22, 60)
    assert r["cls"] == "safe_open" and r["role"] == "safe" and r["reported"] == 1
    assert r["actual_t"] == 1.02 and r["requested_t"] == 1.0
    empty = frame_rows("X-1", 4, 1.33, 1.35, [], crop)
    assert len(empty) == 1 and empty[0]["cls"] == "" and empty[0]["actual_t"] == 1.35


# --- scene anchors --------------------------------------------------------------------------

def test_explicit_desk_beats_lookalike_hull():
    rows = []
    for i in range(10):
        t = float(i)
        rows += [_det("laptop", 0.9, t, (300, 300, 360, 340)), _det("tv", 0.6, t, (200, 300, 250, 340)),
                 _det("desk", 0.8, t, (100, 280, 420, 480))]
    s = build_scene(filter_detections(pd.DataFrame(rows)), 464, 848, 10)
    assert [round(v) for v in s.desk.as_list()] == [100, 280, 420, 480]
    assert "desk from desk detections" in s.notes


def test_safe_open_sets_safe_anchor_and_keeps_class():
    rows = [_det("safe_open" if i % 2 else "safe", 0.8, float(i), (380, 200, 440, 330)) for i in range(10)]
    df = filter_detections(pd.DataFrame(rows))
    assert set(df.cls) == {"safe", "safe_open"}
    s = build_scene(df, 464, 848, 10)
    assert s.safe is not None and "safe from safe detections" in s.notes


def test_frames_without_detections_count_against_stability():
    rows = [_det("desk", 0.8, float(i), (100, 280, 420, 480)) for i in range(3)]
    df = filter_detections(pd.DataFrame(rows))
    assert build_scene(df, 464, 848).desk is not None             # 3 of the 3 frames in df
    assert build_scene(df, 464, 848, n_frames=10).desk is None    # 3 of 10 sampled frames


# --- cache keys -----------------------------------------------------------------------------

def test_perception_cache_key():
    from engine.perception import perception_settings
    md5 = "a" * 32
    base = perception_settings(3.0, (0, 0, 464, 848), fingerprint="stock", pose_sig="m:1")
    assert cache_key(md5, base) == cache_key(md5, perception_settings(3.0, (0, 0, 464, 848), "stock", "m:1"))
    assert cache_key(md5, base) != cache_key(md5, perception_settings(3.0, (0, 0, 464, 848), "f" * 40, "m:1"))
    assert cache_key(md5, base) != cache_key(md5, perception_settings(3.0, (12, 0, 476, 848), "stock", "m:1"))
    assert cache_key(md5, base) != cache_key(md5, perception_settings(3.0, (0, 0, 464, 848), "stock", "m:2"))


def test_scan_cache_key_changes_with_crop():
    from engine.vlm_scan import scan_key
    md5 = "b" * 32
    k = scan_key(md5, (10, 20, 300, 400), (0, 0, 464, 848))
    assert k != scan_key(md5, (10, 20, 300, 420), (0, 0, 464, 848))
    assert k != scan_key(md5, (10, 20, 300, 400), (12, 0, 476, 848))


# --- B5 evidence gate -----------------------------------------------------------------------

def _facts(labels, objects):
    cols = ["t", "cls", "conf", "x0", "y0", "x1", "y1"]
    return SimpleNamespace(perc=SimpleNamespace(labels=labels), objects=pd.DataFrame(objects, columns=cols))


@pytest.fixture
def one_departure(monkeypatch):
    dep = {"t_exit": 10.0, "t_depart": 8.0, "absent_s": 3.0, "to_end": True,
           "box": [1, 2, 3, 4], "track": 1}
    monkeypatch.setattr(b5, "departures", lambda f: [dep])


def test_b5_without_open_safe_stays_below_threshold(one_departure):
    from engine import VERDICT_THRESHOLD
    ev = b5.detect(_facts(frozenset(CLASSES), []))[0]
    assert ev.score + 0.15 < VERDICT_THRESHOLD     # even a VLM 'yes' cannot report it
    assert ev.info["safe_open_frames"] == 0


def test_b5_with_open_safe_while_empty(one_departure):
    so = [(11.0, "safe_open", 0.7, 380, 200, 440, 330), (11.33, "safe_open", 0.8, 380, 200, 440, 330)]
    ev = b5.detect(_facts(frozenset(CLASSES), so))[0]
    assert ev.score == b5.LEAVE_SCORE and ev.info["reason"] == "safe_open_unattended"


def test_b5_open_safe_before_leaving_does_not_count(one_departure):
    so = [(5.0, "safe_open", 0.9, 380, 200, 440, 330), (6.0, "safe_open", 0.9, 380, 200, 440, 330)]
    assert b5.detect(_facts(frozenset(CLASSES), so))[0].score == b5.NO_EVIDENCE_SCORE


def test_b5_unchanged_with_stock_detector(one_departure):
    ev = b5.detect(_facts(None, []))[0]
    assert ev.score == b5.LEAVE_SCORE and ev.info["reason"] == "cashier_left_room"


# --- objects but no person ------------------------------------------------------------------

def test_objects_without_people_run_through_all_rules():
    from engine import facts as F
    from engine.rules import RULES
    times = [0.0, 0.33, 0.67, 1.0]
    rows = [_det("laptop", 0.8, t, (100, 300, 200, 360)) for t in times]
    fa = F.build(filter_detections(pd.DataFrame(rows)), times, 464, 848)
    assert fa.frames.n_room.tolist() == [0, 0, 0, 0] and fa.seat is None
    for name, fn in RULES.items():
        assert fn(fa) == [], name


def test_cached_boxes_clipped_like_fresh_ones():
    from engine.perception import clip_df
    df = pd.DataFrame([_det("laptop", 0.8, 0.0, (-1.8, -0.5, 50, 40)), _det("laptop", 0.8, 0.0, (463.5, 10, 470, 40))])
    out = clip_df(df, 464, 848)
    assert len(out) == 1 and out.iloc[0][["x0", "y0"]].tolist() == [0, 0]


def test_loaded_detector_must_match_selection(monkeypatch, tmp_path):
    from engine import perception
    monkeypatch.setenv("ENGINE_DETECTOR_PATH", str(_ckpt(tmp_path)))
    monkeypatch.setattr(perception, "probe", lambda v: SimpleNamespace(duration=1.0, crop=(0, 0, 464, 848)))
    monkeypatch.setattr(perception, "_cache_path", lambda v, fps, crop: tmp_path / "x.parquet")
    with pytest.raises(ValueError, match="ENGINE_DETECTOR_PATH changed"):
        perception.run("video.mov", models=SimpleNamespace(source=config.DETECTOR_REPO))
