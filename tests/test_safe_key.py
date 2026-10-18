"""Key-in-the-safe-lock check (engine.safe_key) on synthetic safe-door images."""
import numpy as np
import pytest

from engine import safe_key as K
from engine.scene import Box

SAFE = Box(190, 250, 263, 352)   # 73 px wide, like X-22
LOCK = (198, 302)                # x, y


def _door(edge_slope: float = 0.0, horizontal_edge: bool = False) -> np.ndarray:
    """Light door, dark gap left of it (a vertical or slanted line), the lock dot."""
    g = np.full((450, 300), 160.0, np.float32)
    for y in range(g.shape[0]):
        x = int(round(193 + edge_slope * (y - LOCK[1])))
        g[y, x - 8:x - 2] = 60.0     # background seen past the door
        g[y, x - 2:x] = 100.0        # the gap at the door edge
    if horizontal_edge:              # background edge at the lock height (layout of X-24..X-29)
        g[LOCK[1] - 1:, 180:189] = 50.0
    g[LOCK[1] - 1:LOCK[1] + 1, LOCK[0] - 1:LOCK[0] + 1] = 75.0  # lock dot
    return g


def _with_key(g: np.ndarray) -> np.ndarray:
    g = g.copy()
    g[LOCK[1]:LOCK[1] + 2, LOCK[0] - 6:LOCK[0] - 1] = 135.0  # key head sticking out to the left
    return g


@pytest.mark.parametrize("kw", [{}, {"edge_slope": 0.4}, {"horizontal_edge": True}])
def test_lock_found_and_no_key(kw):
    g = _door(**kw)
    lock = K.find_lock(g, SAFE)
    assert lock is not None and abs(lock[0] - LOCK[0]) <= 1 and abs(lock[1] - LOCK[1]) <= 1
    assert lock[2] >= K.MIN_DOT
    assert K.key_score(g, lock, SAFE) < K.KEY_MIN


@pytest.mark.parametrize("kw", [{}, {"edge_slope": 0.4}, {"horizontal_edge": True}])
def test_key_in_lock(kw):
    g = _with_key(_door(**kw))
    lock = K.find_lock(g, SAFE)
    assert K.key_score(g, lock, SAFE) > K.KEY_MIN


def test_dark_door_edge_is_not_the_lock():
    g = _door()
    g[LOCK[1] - 1:LOCK[1] + 1, LOCK[0] - 1:LOCK[0] + 1] = 160.0  # no lock dot at all
    lock = K.find_lock(g, SAFE)
    assert lock is None or lock[2] < K.MIN_DOT  # the vertical gap is a line, not a dot
