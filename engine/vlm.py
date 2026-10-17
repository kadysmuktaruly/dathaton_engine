"""Local VLM (Qwen3-VL-4B, MLX 4-bit) for verification, timing refinement and descriptions.

The VLM never writes legal references and never sets timecodes on its own: it answers yes/no
questions about frames that perception picked, and writes a short 'what is seen' sentence.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from engine import config

_model = None
_processor = None


def load():
    global _model, _processor
    if _model is None:
        from mlx_vlm import load as mlx_load
        _model, _processor = mlx_load(config.VLM_REPO)
    return _model, _processor


def to_pil(bgr: np.ndarray, max_side: int = 768) -> Image.Image:
    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def ask(images: list[Image.Image], prompt: str, max_tokens: int = 160) -> str:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor = load()
    p = apply_chat_template(processor, model.config, prompt, num_images=len(images))
    out = generate(model, processor, p, image=images, max_tokens=max_tokens, temperature=0.0, verbose=False)
    return getattr(out, "text", out)


def parse_json(text: str):
    """First JSON object or array in the text, else None."""
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


@dataclass
class Budget:
    seconds: float
    start: float = 0.0

    def __post_init__(self):
        self.start = time.time()

    def left(self) -> float:
        return self.seconds - (time.time() - self.start)

    def ok(self, need: float = 2.0) -> bool:
        return self.left() >= need


def verify(video_path, perc, fa, events):
    """Placeholder until Step 6: keep events as they are."""
    return events
