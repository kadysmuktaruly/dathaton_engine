import base64

import cv2
import numpy as np
import pytest

from engine import catalog, output


def _jpeg_b64():
    ok, buf = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))
    return base64.b64encode(buf.tobytes()).decode()


def _violation(tc=44.2):
    return {"type": catalog.violation("A7")["type"], "timecode_sec": tc,
            "description": catalog.description("A7", "кассир принимает деньги второго клиента, пока первый ещё у кассы"),
            "evidence_frames": [{"timestamp_sec": tc, "image_base64": _jpeg_b64()}]}


def test_valid_result():
    res = output.build("job1", [_violation()])
    assert res["verdict"] == "violation" and set(res) == {"job_id", "status", "verdict", "violations"}


def test_no_violation():
    assert output.build("job2", [])["verdict"] == "no_violation"


def test_extra_field_rejected():
    v = _violation()
    v["code"] = "A7"
    with pytest.raises(AssertionError):
        output.build("job3", [v])


def test_first_evidence_timestamp_must_match():
    v = _violation()
    v["evidence_frames"][0]["timestamp_sec"] = 40.0
    with pytest.raises(AssertionError):
        output.build("job4", [v])


def test_description_wording_and_refs():
    d = catalog.description("B2", "купюра перемещена в личную сумку")
    assert d.startswith("Признаки возможного нарушения")
    assert "Правила № 231" in d and "хищ" not in d.lower()
    d = catalog.description("X0", "подозрительное действие с наличными")
    assert "хищ" not in d.lower()  # p. 14 quote is skipped for X0


def test_inactive_code_rejected():
    """Reference-only codes (C1, D1, B7, X0, ...) never reach the graded output."""
    v = _violation()
    v["type"] = catalog.violation("C1")["type"]
    v["description"] = catalog.description("C1", "в помещение кассы вошёл человек")
    with pytest.raises(AssertionError):
        output.build("job5", [v])


def test_internal_requirement_without_quote():
    d = catalog.description("B9", "кассир ушёл, ноутбук остался включённым")
    assert "Норма: внутренние документы банка." in d
    assert "о закрытии или блокировке ноутбука" in d and "«" not in d


def test_quote_and_internal_refs_together():
    d = catalog.description("B8", "кассир ушёл, печати остались на столе")
    assert "«порядок хранения, выдачи и передачи ценностей, печатей" in d
    assert "Правила № 231, п. 80; внутренние документы банка" in d


def test_only_if_refs_off_by_default():
    pts = [r["point"] for r in catalog.refs("C1")]
    assert "ст. 252-1 ч. 5" not in pts
