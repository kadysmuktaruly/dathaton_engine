"""Render a debug overlay video (and a contact sheet) from perception results.

Usage: uv run python -m tools.overlay X-1 X-17 ...   -> runs/overlays/<clip>.mp4 and <clip>_sheet.jpg
"""
from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import cv2
import numpy as np

from engine import config, perception
from engine.video import iter_frames_at
from tools.contact_sheets import grid
from tools.labels import video_index

WRISTS = (9, 10)


def color_for(key) -> tuple[int, int, int]:
    h = (hash(str(key)) % 997) / 997
    r, g, b = colorsys.hsv_to_rgb(h, 0.8, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw(frame: np.ndarray, rows, extra_text: str = "") -> np.ndarray:
    img = frame.copy()
    for r in rows.itertuples():
        x0, y0, x1, y1 = map(int, (r.x0, r.y0, r.x1, r.y1))
        if r.kind == "person":
            c = color_for(r.track_id)
            cv2.rectangle(img, (x0, y0), (x1, y1), c, 1)
            cv2.putText(img, f"P{r.track_id} {r.conf:.2f}", (x0, max(10, y0 - 3)), 0, 0.4, c, 1, cv2.LINE_AA)
            if r.kps is not None:
                k = np.asarray(r.kps, dtype=np.float32).reshape(17, 3)
                for j in WRISTS:
                    if k[j, 2] > 0.3:
                        cv2.circle(img, (int(k[j, 0]), int(k[j, 1])), 4, (0, 0, 255), -1)
                for j in (15, 16):  # ankles
                    if k[j, 2] > 0.3:
                        cv2.circle(img, (int(k[j, 0]), int(k[j, 1])), 3, (255, 255, 0), -1)
        else:
            c = color_for(r.cls)
            cv2.rectangle(img, (x0, y0), (x1, y1), c, 1)
            cv2.putText(img, f"{r.cls} {r.conf:.2f}", (x0, max(10, y0 - 3)), 0, 0.35, c, 1, cv2.LINE_AA)
    if extra_text:
        cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
        cv2.putText(img, extra_text, (3, 13), 0, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return img


def render(video: Path, perc, out_dir: Path, name: str, sheet_every: float = 2.0) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = perc.df
    by_t = {t: g for t, g in df.groupby("t")}
    writer = None
    sheet_tiles = []
    next_sheet = 0.0
    for _, t, frame in iter_frames_at(str(video), perc.times, perc.info.crop):
        rows = by_t.get(t, df.iloc[0:0])
        img = draw(frame, rows, f"{name} t={t:.2f}")
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(out_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     config.SAMPLE_FPS, (w, h))
        writer.write(img)
        if t >= next_sheet:
            sheet_tiles.append(cv2.resize(img, None, fx=0.6, fy=0.6))
            next_sheet += sheet_every
    if writer:
        writer.release()
    if sheet_tiles:
        cv2.imwrite(str(out_dir / f"{name}_sheet.jpg"), grid(sheet_tiles, 8), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out_dir / f"{name}.mp4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--out", default=str(config.RUNS_DIR / "overlays"))
    args = ap.parse_args()
    vids = video_index()
    models = perception.Models()
    for cid in args.clips:
        perc = perception.run(vids[cid], models=models)
        print(cid, perc.timings, "rows", len(perc.df))
        print(render(vids[cid], perc, Path(args.out), cid))


if __name__ == "__main__":
    main()
