"""Evidence frames: re-decode at full resolution, draw thin role boxes, encode as bare JPEG base64."""
from __future__ import annotations

import base64

import cv2
import numpy as np

from engine.video import iter_frames_at

COLORS = {"кассир": (60, 180, 75), "клиент": (255, 160, 0), "посторонний": (0, 0, 255),
          "инкассатор": (200, 0, 200)}


def render(video: str, times: list[float], boxes: dict, crop=(0, 0, 0, 0), quality: int = 85
           ) -> list[dict]:
    """times: evidence timestamps (first = timecode). boxes: {t: [(label, [x0,y0,x1,y1]), ...]}
    in cropped-frame coordinates; they are shifted back to full-frame coordinates here."""
    ox, oy = crop[0], crop[1]
    out = []
    for want, actual, frame in iter_frames_at(video, times, None):
        img = frame.copy()
        # boxes drawn for the closest annotated time within 0.7 s
        near = [k for k in boxes if abs(k - want) <= 0.7]
        if near:
            k = min(near, key=lambda k: abs(k - want))
            for label, b in boxes[k]:
                x0, y0, x1, y1 = (int(b[0] + ox), int(b[1] + oy), int(b[2] + ox), int(b[3] + oy))
                c = COLORS.get(label, (0, 255, 255))
                cv2.rectangle(img, (x0, y0), (x1, y1), c, 1)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        out.append({"timestamp_sec": round(float(want), 1),
                    "image_base64": base64.b64encode(buf.tobytes()).decode("ascii")})
    return out
