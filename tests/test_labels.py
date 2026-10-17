import pytest

from tools.labels import norm_id, parse_timecode


def test_norm_id_mixed_alphabets():
    assert norm_id("Х-44") == "X-44"  # Cyrillic
    assert norm_id("X-1.mov") == "X-1"  # Latin
    assert norm_id("Х-4.mov") == "X-4"


def test_parse_timecode():
    assert parse_timecode("22 секунда") == 22.0
    assert parse_timecode("1минута 8 секунда") == 68.0
    assert parse_timecode(None) is None
    with pytest.raises(ValueError):
        parse_timecode("скоро")
