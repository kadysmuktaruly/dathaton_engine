"""Download all model weights once. After this, everything runs with HF_HUB_OFFLINE=1.

Usage: uv run python -m tools.download_weights [--with-owlv2]
"""
import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

from engine import config

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = config.WEIGHTS_DIR

HF_MODELS = {
    "detector": ("PekingU/rtdetr_v2_r18vd", None),
    "vlm": (config.VLM_REPO, None),
}
OWLV2 = ("google/owlv2-base-patch16-ensemble", None)

# RTMPose-m body7, top-down (person boxes come from our detector). Apache-2.0 (OpenMMLab mmpose).
RTMPOSE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)


def fetch_rtmpose() -> Path:
    out = WEIGHTS / "rtmpose"
    existing = list(out.rglob("*.onnx"))
    if existing:
        return existing[0]
    out.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(RTMPOSE_URL, timeout=120).read()
    zipfile.ZipFile(io.BytesIO(data)).extractall(out)
    return next(out.rglob("*.onnx"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-owlv2", action="store_true", help="pre-labeling model (Step 5 only)")
    args = ap.parse_args()
    models = dict(HF_MODELS)
    if not config.USE_VLM:
        models.pop("vlm")
    if args.with_owlv2:
        models["owlv2"] = OWLV2
    for name, (repo, rev) in models.items():
        path = snapshot_download(repo, revision=rev)
        print(f"{name}: {repo} -> {path}")
    print("rtmpose:", fetch_rtmpose())


if __name__ == "__main__":
    main()
