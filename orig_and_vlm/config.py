"""Settings for the merged pipeline. Lines marked `# TUNE` are first guesses checked on the 40-clip eval."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"

# The eyes: a second, newer VLM next to the engine's Qwen3-VL-4B (which keeps doing the verifier).
EYES_REPO = os.environ.get("OV_EYES_REPO", "mlx-community/Qwen3.5-4B-MLX-4bit")

USE_SCAN = os.environ.get("OV_SCAN", "1") == "1"          # idea 1: Qwen3.5 desk scan feeds the rules
USE_CLOSEOUT = os.environ.get("OV_CLOSEOUT", "1") == "1"  # idea 2: close-out check on 'cashier left'

VLM_BUDGET_S = 65.0   # TUNE: per-video budget for scan + close-out + the engine's verifier
SCAN_BUDGET_S = 42.0  # TUNE: share of it for the desk scan (<= 18 grid frames + key moments, ~2.1 s each)
CLOSEOUT_NEED_S = 4.5 # a close-out check needs this much budget left, else the alarm is kept as is
CROP_SIDE = 640
