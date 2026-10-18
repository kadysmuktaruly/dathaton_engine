from types import SimpleNamespace

import pytest

from engine import catalog
from engine.backend import build_analysis
from engine.events import Event


def result(code="B5"):
    return {
        "violations": [
            {
                "type": catalog.violation(code)["type"],
                "timecode_sec": 2.0,
                "description": "test",
                "evidence_frames": [],
            }
        ]
    }


def test_boxes_map_crop_to_full_frame_and_preserve_indices():
    info = SimpleNamespace(
        width=1000, height=500, crop=(100, 50, 900, 450), duration=10
    )
    event = Event("B5", 2, 8, 0.9, boxes={2: [("safe", [100, 50, 300, 150])]})
    mapped = build_analysis(info, [event], result())
    assert mapped["duration_sec"] == 10
    assert mapped["violations"][0]["code"] == "B5"
    annotation = mapped["events"][0]
    assert annotation["violation_index"] == 0
    assert annotation["box"] == pytest.approx(
        {"x": 0.2, "y": 0.2, "width": 0.2, "height": 0.2}
    )
    assert annotation["start_sec"] == 1.8 and annotation["end_sec"] == 2.2
    assert annotation["rule"]["url"].startswith("https://adilet.zan.kz/")
    assert "code" not in result()["violations"][0]


def test_missing_and_invalid_boxes_are_not_fabricated():
    info = SimpleNamespace(width=1000, height=500, crop=(0, 0, 1000, 500), duration=10)
    event = Event(
        "B9", 2, 8, 0.9, boxes={20: [("x", [1, 1, 2, 2])], 2: [("x", [30, 30, 20, 20])]}
    )
    mapped = build_analysis(info, [event], result("B9"))
    assert not mapped["events"]
    event.boxes = {2: [("laptop", [-20, -10, 1100, 600])]}
    mapped = build_analysis(info, [event], result("B9"))
    assert mapped["events"][0]["box"] == {"x": 0, "y": 0, "width": 1, "height": 1}
    assert mapped["events"][0]["rule"] is None
    assert mapped["events"][0]["rule_note"]
