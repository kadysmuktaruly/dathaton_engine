"""Paths and tunable thresholds. Lines marked `# TUNE` were chosen to favour recall."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "videos"))
LABELS_XLSX = Path(os.environ.get("LABELS_XLSX", ROOT / "open_labels.xlsx"))
CATALOG_PATH = ROOT / "rules" / "rules_catalog.json"
CACHE_DIR = Path(os.environ.get("ENGINE_CACHE_DIR", ROOT / "cache"))
RUNS_DIR = Path(os.environ.get("ENGINE_RUNS_DIR", ROOT / "runs"))
WEIGHTS_DIR = Path(os.environ.get("ENGINE_WEIGHTS_DIR", ROOT / "weights"))

DETECTOR_REPO = "PekingU/rtdetr_v2_r18vd"  # stock COCO detector (default)
VLM_BACKEND = os.environ.get("ENGINE_VLM_BACKEND", "mlx" if sys.platform == "darwin" else "transformers")
if VLM_BACKEND not in {"mlx", "transformers"}:
    raise ValueError("ENGINE_VLM_BACKEND must be mlx or transformers")
VLM_REPO = os.environ.get("ENGINE_VLM_REPO", "mlx-community/Qwen3-VL-4B-Instruct-4bit"
                         if VLM_BACKEND == "mlx" else "Qwen/Qwen3-VL-4B-Instruct")
USE_VLM = os.environ.get("ENGINE_USE_VLM", "1") == "1"
DEVICE = os.environ.get("ENGINE_DEVICE") or None
CPU_THREADS = int(os.environ.get("ENGINE_CPU_THREADS", "8"))
ONNX_THREADS = int(os.environ.get("ENGINE_ONNX_THREADS", "2"))

SAMPLE_FPS = 3.0  # TUNE
USE_CACHE = os.environ.get("ENGINE_NO_CACHE", "0") != "1"  # set ENGINE_NO_CACHE=1 for cold timings

# A local fine-tuned checkpoint (save_pretrained dir, e.g. weights/detector_ft) replaces the stock
# detector when ENGINE_DETECTOR_PATH is set. There is no silent fallback to stock.
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def detector_source() -> str:
    """Stock repo id, or the checked local checkpoint dir from ENGINE_DETECTOR_PATH."""
    path = os.environ.get("ENGINE_DETECTOR_PATH", "").strip()
    if not path:
        return DETECTOR_REPO
    p = Path(path).expanduser()
    if not p.is_absolute() and not p.exists():
        p = ROOT / p  # relative to the project root when not found from the current directory
    p = p.resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"ENGINE_DETECTOR_PATH={path}: directory not found ({p})")
    if not (p / "config.json").is_file():
        raise FileNotFoundError(f"ENGINE_DETECTOR_PATH={path}: config.json missing in {p}")
    if not any((p / w).is_file() for w in WEIGHT_FILES):
        raise FileNotFoundError(f"ENGINE_DETECTOR_PATH={path}: no weights ({' or '.join(WEIGHT_FILES)}) in {p}")
    return str(p)


def detector_fingerprint() -> str:
    """'stock', or a hash of the checkpoint's config + weight file size/mtime (no model load)."""
    import hashlib
    src = detector_source()
    if src == DETECTOR_REPO:
        return "stock"
    p = Path(src)
    h = hashlib.sha1((p / "config.json").read_bytes())
    for w in WEIGHT_FILES:
        f = p / w
        if f.is_file():
            st = f.stat()
            h.update(f"{w}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


def detector_labels() -> frozenset[str] | None:
    """Class names of the selected local checkpoint (read from its config.json, no model load);
    None for the stock COCO detector."""
    import json
    src = detector_source()
    if src == DETECTOR_REPO:
        return None
    cfg = json.loads((Path(src) / "config.json").read_text())
    return frozenset(str(v) for v in cfg.get("id2label", {}).values())
