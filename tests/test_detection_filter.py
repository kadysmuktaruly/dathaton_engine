import pandas as pd

from engine.perception import filter_detections
from engine.scene import _stable_clusters, build_scene


def _det(cls, conf, t=0.0, box=(100, 100, 200, 160), kind=None):
    kind = kind or ("person" if cls == "person" else "obj")
    return {"t": t, "frame_idx": 0, "kind": kind, "cls": cls, "conf": conf,
            "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3], "track_id": -1, "kps": None}


def test_out_of_repertoire_classes_are_dropped():
    df = pd.DataFrame([_det("chair", 0.9), _det("cup", 0.9), _det("laptop", 0.9), _det("tv", 0.9)])
    out = filter_detections(df)
    assert list(out.cls) == ["laptop", "tv"]
    assert list(out.role) == ["laptop", "machine_like"]


def test_confidence_levels():
    df = pd.DataFrame([_det("laptop", 0.45), _det("laptop", 0.55), _det("laptop", 0.2),
                       _det("person", 0.35), _det("person", 0.6)])
    out = filter_detections(df)
    objs = out[out.kind == "obj"].set_index("conf").confident
    assert objs.to_dict() == {0.45: False, 0.55: True}   # 0.2 is below the candidate floor
    assert list(out[out.kind == "person"].conf) == [0.6]


def test_weak_cluster_is_not_an_anchor():
    weak = pd.DataFrame([_det("laptop", 0.3, t=float(i)) for i in range(10)])
    assert _stable_clusters(filter_detections(weak), 10).empty


def test_cluster_with_some_confident_frames_is_an_anchor():
    rows = [_det("laptop", 0.3, t=float(i)) for i in range(8)] + \
           [_det("laptop", 0.6, t=float(i)) for i in range(8, 10)]
    cl = _stable_clusters(filter_detections(pd.DataFrame(rows)), 10)
    assert list(cl.role) == ["laptop"] and cl.frac.iloc[0] == 1.0


def test_scene_with_only_foreground_tables_does_not_crash():
    # every table touches the bottom edge, no devices: used to raise ValueError
    rows = [_det("diningtable", 0.6, t=float(i), box=(0, 300, 200, 480)) for i in range(5)]
    scene = build_scene(filter_detections(pd.DataFrame(rows)), 640, 480)
    assert scene.desk is not None
