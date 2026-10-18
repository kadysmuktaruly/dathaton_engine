"""Strict output schema. Nothing outside these fields is allowed in the graded JSON."""
from __future__ import annotations

import base64

from engine import catalog

TOP_KEYS = {"job_id", "status", "verdict", "violations"}
V_KEYS = {"type", "timecode_sec", "description", "evidence_frames"}
E_KEYS = {"timestamp_sec", "image_base64"}


def validate(res: dict) -> None:
    assert set(res) == TOP_KEYS, f"top-level keys {set(res)}"
    assert res["status"] in {"done", "processing", "failed"}
    assert res["verdict"] in {"violation", "no_violation"}
    assert isinstance(res["violations"], list)
    assert (res["verdict"] == "violation") == (len(res["violations"]) > 0)
    types = catalog.active_types()  # inactive (reference-only) codes are never reported
    for v in res["violations"]:
        assert set(v) == V_KEYS, f"violation keys {set(v)}"
        assert v["type"] in types, v["type"]
        assert isinstance(v["timecode_sec"], (int, float))
        assert 1 <= len(v["evidence_frames"]) <= 3
        assert abs(v["evidence_frames"][0]["timestamp_sec"] - v["timecode_sec"]) < 1e-6
        assert "признаки возможного нарушения" in v["description"].lower()
        for e in v["evidence_frames"]:
            assert set(e) == E_KEYS
            raw = base64.b64decode(e["image_base64"], validate=True)
            assert raw[:2] == b"\xff\xd8", "not a JPEG"


def build(job_id: str, violations: list[dict]) -> dict:
    res = {"job_id": job_id, "status": "done",
           "verdict": "violation" if violations else "no_violation",
           "violations": violations}
    validate(res)
    return res
