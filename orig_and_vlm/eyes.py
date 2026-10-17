"""Qwen3.5-4B (MLX 4-bit) asked targeted yes/no or JSON questions on frames and crops.

Kept from the textflow experiment (see TEXTFLOW_REPORT.md): targeted questions on crops were the
part that worked (15/16 on zoomed crops in a verified benchmark), unlike free captions.
"""
from __future__ import annotations

import re
import time

import cv2
import numpy as np
from PIL import Image

from orig_and_vlm import config

_loaded: dict = {}


def load(repo: str | None = None):
    repo = repo or config.EYES_REPO
    if repo not in _loaded:
        from mlx_vlm import load as mlx_load
        _loaded[repo] = mlx_load(repo)
    return _loaded[repo]


def generate(images: list[Image.Image], prompt: str, max_tokens: int = 30) -> tuple[str, float]:
    """Greedy generation, thinking off; returns (text, seconds)."""
    from mlx_vlm import generate as mlx_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor = load()
    p = apply_chat_template(processor, model.config, prompt, num_images=len(images), enable_thinking=False)
    t0 = time.time()
    out = mlx_generate(model, processor, p, image=images or None, max_tokens=max_tokens, temperature=0.0,
                       verbose=False)
    text = re.sub(r"<think>.*?(</think>|$)", "", getattr(out, "text", out), flags=re.S).strip()
    return text, time.time() - t0


def yes_no(text: str) -> bool | None:
    m = re.search(r"\b(yes|no)\b", text.lower())
    return None if m is None else m.group(1) == "yes"


def pil(bgr: np.ndarray, max_side: int = config.CROP_SIDE) -> Image.Image:
    """Resize so the longer side is max_side (up or down); BGR -> PIL RGB."""
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    bgr = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))),
                     interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def crop(frame: np.ndarray, box) -> np.ndarray:
    if box is None:
        return frame
    x0, y0, x1, y1 = (int(v) for v in box)
    return frame[max(0, y0):y1, max(0, x0):x1]


def ask(image_bgr: np.ndarray, question: str) -> tuple[bool | None, str, float]:
    text, sec = generate([pil(image_bgr)], question)
    return yes_no(text), text, sec
