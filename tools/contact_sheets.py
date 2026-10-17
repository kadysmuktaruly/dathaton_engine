"""Contact sheets for the dataset review (Step 1).

For each labeled timecode t: frames at t-4 ... t+4 s every 1 s (the brief asks for t±2; the wider
window shows what leads up to the event). For every clip: an overview strip, one frame every
`--every` seconds. Frames are labeled with drawn text (ffmpeg here has no drawtext).

Usage: uv run python -m tools.contact_sheets [--out reports/sheets] [--every 2]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from engine.video import iter_frames_at, probe
from tools.labels import load_clips


def label(img: np.ndarray, text: str, color=(0, 255, 255)) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


def grid(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    out = np.zeros((rows * h, cols * w, 3), np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = cv2.resize(t, (w, h))
    return out


def sheet(path, times, scale, highlight=None, cols=9):
    tiles = []
    for want, actual, f in iter_frames_at(path, times):
        f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        is_hl = highlight is not None and abs(want - highlight) < 1e-6
        tag = f"t={actual:.1f}" + (" LABEL" if is_hl else "")
        tile = label(f, tag, (0, 0, 255) if is_hl else (0, 255, 255))
        if is_hl:
            cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (0, 0, 255), 4)
        tiles.append(tile)
    return grid(tiles, cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/sheets")
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--clips", nargs="*")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for c in load_clips():
        if args.clips and c.clip_id not in args.clips:
            continue
        info = probe(c.path)
        for tc in c.timecodes:
            times = [max(0.0, min(info.duration - 0.05, tc + d)) for d in range(-4, 5)]
            img = sheet(c.path, times, 0.5, highlight=times[4])
            cv2.imwrite(str(out / f"{c.clip_id}_tc{int(tc)}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        n = int(info.duration // args.every) + 1
        times = [min(k * args.every, info.duration - 0.05) for k in range(n)]
        img = sheet(c.path, times, 0.35, cols=12)
        cv2.imwrite(str(out / f"{c.clip_id}_overview.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(c.clip_id, "done")



def zoom_sheets(out: Path, half_window: float = 3.0, step: float = 0.5, y0f=0.2, y1f=0.66, cols=4) -> None:
    """Full-res crops of the room's upper part around each labeled timecode."""
    for c in load_clips():
        info = probe(c.path)
        for tc in c.timecodes:
            n = int(2 * half_window / step) + 1
            times = [max(0.0, min(info.duration - 0.05, tc - half_window + k * step)) for k in range(n)]
            tiles = []
            for want, actual, f in iter_frames_at(c.path, times):
                h = f.shape[0]
                f = f[int(y0f * h):int(y1f * h)]
                is_hl = abs(want - tc) < 1e-6
                tile = label(f, f"t={actual:.1f}" + (" LABEL" if is_hl else ""), (0, 0, 255) if is_hl else (0, 255, 255))
                if is_hl:
                    cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (0, 0, 255), 5)
                tiles.append(tile)
            cv2.imwrite(str(out / f"{c.clip_id}_tc{int(tc)}_zoom.jpg"), grid(tiles, cols), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
