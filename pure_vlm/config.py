"""Settings for the pure-VLM pipeline. Lines marked `# TUNE` were checked on the 40 training clips."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
CACHE_DIR = ROOT.parent / "cache" / "pure_vlm"

VLM_REPO = os.environ.get("PV_VLM_REPO", "mlx-community/Qwen3.5-9B-MLX-4bit")
USE_CACHE = os.environ.get("ENGINE_NO_CACHE", "0") != "1"

# Frame grid. Every question reads frames from this grid (decoded once per clip).
GRID_S = 0.5
# State questions run every STATE_STEP_S; long clips get a coarser step so the clip fits the budget.
STATE_STEP_S = 2.0          # TUNE: coarse grid; changes are bisected down to GRID_S
MAX_STATE_FRAMES = 36       # TUNE: step = max(STATE_STEP_S, duration / MAX_STATE_FRAMES), rounded to GRID_S
# A crop is asked again only if it changed: mean absolute grey difference (0-255) to the crop that was
# last asked. Below this the previous answer is reused (door, safe and laptop barely change).
REUSE_DIFF = {"people": 3.0, "door": 4.0, "safe": 4.0, "laptop": 4.0, "desk": 3.0}  # TUNE
# Slow-changing zones: every Nth coarse time (0 = never) plus the key moments, then bisection on changes.
COARSE_EVERY = {"safe": 2, "desk": 2, "laptop": 0}      # TUNE
KEY_MOMENTS = {"safe": ["departure"], "desk": ["departure", "arrival"], "laptop": ["departure"]}
# Action windows: WINDOW_FRAMES frames GRID_S apart, every ACTION_STEP_S, while the cashier is in the room.
ACTION_STEP_S = 2.0         # TUNE
WINDOW_FRAMES = 4
ACTION_SIDE = 384      # TUNE: longer side of each window frame (cost grows with it)

BUDGET_S = float(os.environ.get("PV_BUDGET_S", "230"))  # per-video VLM budget (the API limit is 300 s)
VERIFY_RESERVE_S = 12.0  # part of the budget kept for the candidate checks (verify.py) and evidence frames
