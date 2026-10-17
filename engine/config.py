"""Paths and tunable thresholds. Lines marked `# TUNE` were chosen to favour recall."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "videos"))
LABELS_XLSX = Path(os.environ.get("LABELS_XLSX", ROOT / "open_labels.xlsx"))
CATALOG_PATH = ROOT / "rules" / "rules_catalog.json"
CACHE_DIR = ROOT / "cache"
RUNS_DIR = ROOT / "runs"
WEIGHTS_DIR = ROOT / "weights"

DETECTOR_REPO = "PekingU/rtdetr_v2_r18vd"
VLM_REPO = "mlx-community/Qwen3-VL-4B-Instruct-4bit"

SAMPLE_FPS = 3.0  # TUNE
