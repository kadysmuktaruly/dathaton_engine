"""The only perception model: Qwen3.5 (MLX 4-bit), asked short structured questions on frames and crops."""
from __future__ import annotations

import json
import re
import time

import cv2
import numpy as np
from PIL import Image

from pure_vlm import config

_loaded: dict = {}
STATS = {"calls": 0, "seconds": 0.0}


def load(repo: str | None = None):
    repo = repo or config.VLM_REPO
    if repo not in _loaded:
        from mlx_vlm import load as mlx_load
        _loaded[repo] = mlx_load(repo)
    return _loaded[repo]


def generate(images: list[Image.Image], prompt: str, max_tokens: int = 60) -> str:
    """Greedy generation with thinking off."""
    from mlx_vlm import generate as mlx_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor = load()
    p = apply_chat_template(processor, model.config, prompt, num_images=len(images), enable_thinking=False)
    t0 = time.time()
    out = mlx_generate(model, processor, p, image=images or None, max_tokens=max_tokens, temperature=0.0,
                       verbose=False)
    STATS["calls"] += 1
    STATS["seconds"] += time.time() - t0
    return re.sub(r"<think>.*?(</think>|$)", "", getattr(out, "text", out), flags=re.S).strip()


def parse_json(text: str):
    """First JSON object or array in the text; None if there is none."""
    text = re.sub(r"```(json)?", "", text)
    for open_, close in (("{", "}"), ("[", "]")):
        i = text.find(open_)
        j = text.rfind(close)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None


def ask_json(images: list[Image.Image], prompt: str, max_tokens: int = 80):
    text = generate(images, prompt, max_tokens)
    return parse_json(text), text


def yes_no(text: str) -> bool | None:
    m = re.search(r"\b(yes|no)\b", text.lower())
    return None if m is None else m.group(1) == "yes"


def pil(bgr: np.ndarray, max_side: int = 640) -> Image.Image:
    """Resize so the longer side is max_side (up or down); BGR -> PIL RGB."""
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    bgr = cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))),
                     interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def crop(frame: np.ndarray, box, pad: float = 0.0) -> np.ndarray:
    """Crop a box (x0, y0, x1, y1 in pixels), optionally padded by a fraction of its size."""
    if box is None:
        return frame
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    x0, y0 = max(0, int(x0 - px)), max(0, int(y0 - py))
    x1, y1 = min(w, int(x1 + px)), min(h, int(y1 + py))
    return frame[y0:y1, x0:x1]
