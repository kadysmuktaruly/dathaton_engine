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


APPEARANCE = re.compile(r"розов|бел[аоы]|ч[её]рн|сер[аоы]|волос|кофт|рубаш|пиджак|кардиган|девушк|женщин|мужчин|"
                        r"молод|парен|очк|блондин|брюнет|красив|лиц[оа]", re.I)
UNSEEN = re.compile(r"документ|подпис|провод|квитанц|сверк|чек[аи]?\b|зачисл", re.I)
YES_LIFT = 0.15  # most a VLM 'yes' can add to an event's score
NO_FLOOR = 0.3   # a 'no' never pushes an event below the verdict threshold: on the training
                 # clips the 4B VLM said 'not visible' on clear cases (cash on the desk in X-26),
                 # so it may lower confidence but cannot veto (recall first)
ECHO_MAX = 0.5   # reject a sentence when this share of its words comes from the cue text

VERIFY_PROMPT = (
    "Это кадры с камеры видеонаблюдения над кассой (кадр 1 — момент события, кадр 2 — через секунду). "
    "Проверь, видно ли на них следующее: {cues} "
    "Не считать нарушением, если: {not_flag} "
    "Людей называй только по роли: кассир, клиент, инкассатор, посторонний. Не описывай внешность и одежду. "
    'Ответь только JSON: {{"visible": true/false, "confidence": 0..1, '
    '"what_is_seen": "одно короткое предложение по-русски о том, что видно"}}'
)


def _words(t: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[а-яё]{4,}", t.lower())}


def clean_sentence(text: str | None, cues: str = "") -> str | None:
    """Accept a VLM sentence only if it is short Russian text with roles only and no claims
    about things that cannot be seen; otherwise return None (the rule template is used)."""
    if not text or not isinstance(text, str):
        return None
    t = " ".join(text.split()).strip().rstrip(".")
    if not (15 <= len(t) <= 220) or not re.search(r"[а-яА-Я]", t) or re.search(r"[A-Za-z]{3,}", t):
        return None
    from engine.catalog import BANNED
    if APPEARANCE.search(t) or UNSEEN.search(t) or BANNED.search(t):
        return None
    w = _words(t)
    if cues and w and len(w & _words(cues)) / len(w) > ECHO_MAX:
        return None  # the model repeated the question instead of describing the frame
    return t


def verify(video_path, perc, fa, events, budget=None):
    """Ask a yes/no question per event (2 frames) built from the catalog cues; adjust scores
    and take a sanitized 'what is seen' sentence. Never touches timecodes or legal refs."""
    from engine import catalog
    from engine.video import iter_frames_at
    crop = perc.info.crop
    for ev in sorted(events, key=lambda e: -e.score):
        if budget is not None and not budget.ok(2.5):
            ev.info["vlm"] = "skipped (budget)"
            continue
        v = catalog.violation(ev.code)
        times = [ev.start, min(ev.start + 1.0, perc.info.duration - 0.05)]
        imgs = [to_pil(f, 640) for _, _, f in iter_frames_at(video_path, times, crop)]
        prompt = VERIFY_PROMPT.format(cues=v["visual_cues"], not_flag=v["not_flag"])
        ans = parse_json(ask(imgs, prompt, max_tokens=120)) or {}
        visible = ans.get("visible")
        conf = float(ans.get("confidence", 0.5) or 0.5)
        ev.info["vlm"] = {"visible": visible, "confidence": conf, "raw_seen": ans.get("what_is_seen")}
        if visible is True:
            ev.score = min(1.0, ev.score + YES_LIFT * conf)
        elif visible is False:
            ev.score = max(min(ev.score, NO_FLOOR), ev.score - 0.25 * conf)
        s = clean_sentence(ans.get("what_is_seen"), v["visual_cues"] + " " + v["not_flag"])
        if s and visible is not False:
            ev.info["seen"] = s
    return events


CATCHALL_PROMPT = (
    "Кадр с камеры видеонаблюдения над кассой. Есть ли на нём ЯВНОЕ отступление от кассовых процедур "
    "из списка: {codes}. Если ничего такого явно не видно, ответь code \"none\". "
    "Людей называй только по роли. Ответь только JSON: "
    '{{"code": "<код или none>", "confidence": 0..1, "what_is_seen": "одно предложение по-русски"}}'
)
CATCHALL_STEP_S = 5.0      # TUNE
CATCHALL_MIN_CONF = 0.85   # TUNE
CATCHALL_ENABLED = False   # off by default: flagged nothing on 12 client clips (6 violation, 6 clean)


def catchall(video_path, perc, fa, budget=None, step: float = CATCHALL_STEP_S) -> list[dict]:
    """One frame every `step` s while a client is present (2+ people in the room). Returns raw
    answers [{t, code, confidence, seen}] - callers decide what (if anything) becomes an event."""
    from engine import catalog
    from engine.video import iter_frames_at
    codes = "; ".join(f"{v['code']} — {v['type']}" for v in catalog.load()["violations"]
                      if v.get("active"))
    fr = fa.frames
    busy = fr[fr.n_smooth >= 2].t.to_numpy()
    if len(busy) == 0:
        return []
    times, last = [], -1e9
    for t in busy:
        if t - last >= step:
            times.append(float(t)); last = t
    out = []
    for _, t, f in iter_frames_at(video_path, times, perc.info.crop):
        if budget is not None and not budget.ok(2.0):
            break
        ans = parse_json(ask([to_pil(f, 640)], CATCHALL_PROMPT.format(codes=codes), max_tokens=100)) or {}
        out.append({"t": float(t), "code": str(ans.get("code", "none")), "confidence": float(ans.get("confidence", 0) or 0),
                    "seen": ans.get("what_is_seen")})
    return out
