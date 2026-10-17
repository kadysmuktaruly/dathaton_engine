"""Run only the object detector on clips and save annotated frames + box tables, for local review.

Usage:
  uv run python -m tools.inspect_detections X-30 X-17 [--fps 1] [--out runs/inspect] [--all-classes]
  ENGINE_DETECTOR_PATH=weights/detector_ft uv run python -m tools.inspect_detections X-30

Per clip, in <out>/<clip>/ (runs/ is gitignored; keep these files local):
  NNNN_t<actual>.jpg  every sampled frame, including frames with no detections. Boxes show class and
                      confidence: solid at >= OBJ_MIN (reported), dashed below (layout only).
  detections.csv      one row per box: requested and actual video time, class, role, confidence,
                      box in cropped-frame (c*) and original-frame (o*) pixels; frames with no
                      detections get one row with an empty class.
  summary.json        detector source/fingerprint, frame and box counts, timings.
  raw/NNNN.jpg        the same frames without drawings, used by the viewer.
<out>/index.html      interactive viewer: open it in a browser (local file, nothing is uploaded);
                      boxes are drawn live, with cutoff sliders and class toggles.
No tracker, no pose, no cache: this shows what the detector itself sees.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from engine import config
from engine.classes import COLORS, role_of
from engine.perception import OBJ_MIN, PERSON, PERSON_MIN
from engine.video import iter_frames_at, probe, sample_times

FIELDS = ["clip", "frame_idx", "requested_t", "actual_t", "cls", "role", "conf", "reported",
          "cx0", "cy0", "cx1", "cy1", "ox0", "oy0", "ox1", "oy1"]


def to_original(box, crop) -> list[float]:
    """Cropped-frame box -> original-frame box (add the black-bar crop offset)."""
    ox, oy = crop[0], crop[1]
    return [float(box[0]) + ox, float(box[1]) + oy, float(box[2]) + ox, float(box[3]) + oy]


def frame_rows(clip: str, frame_idx: int, requested_t: float, actual_t: float, dets, crop) -> list[dict]:
    """Table rows for one frame; a frame with no detections still gets one (empty) row."""
    base = {"clip": clip, "frame_idx": frame_idx, "requested_t": round(float(requested_t), 3),
            "actual_t": round(float(actual_t), 3)}
    if not dets:
        return [{**base, **{k: "" for k in FIELDS if k not in base}}]
    rows = []
    for name, conf, box in dets:
        floor = PERSON_MIN if name == PERSON else OBJ_MIN
        o = to_original(box, crop)
        rows.append({**base, "cls": name, "role": role_of(name) or "", "conf": round(float(conf), 3),
                     "reported": int(conf >= floor),
                     "cx0": round(float(box[0]), 1), "cy0": round(float(box[1]), 1),
                     "cx1": round(float(box[2]), 1), "cy1": round(float(box[3]), 1),
                     "ox0": round(o[0], 1), "oy0": round(o[1], 1), "ox1": round(o[2], 1), "oy1": round(o[3], 1)})
    return rows


def _bgr(name: str) -> tuple[int, int, int]:
    hx = COLORS.get(name, "#9e9e9e").lstrip("#")
    r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
    return b, g, r


def _dashed_rect(img, p0, p1, color, thick=1, dash=6):
    import cv2
    (x0, y0), (x1, y1) = p0, p1
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) // dash))
        for k in range(0, n, 2):
            s = (int(a[0] + (b[0] - a[0]) * k / n), int(a[1] + (b[1] - a[1]) * k / n))
            e = (int(a[0] + (b[0] - a[0]) * min(k + 1, n) / n), int(a[1] + (b[1] - a[1]) * min(k + 1, n) / n))
            cv2.line(img, s, e, color, thick)


def draw(frame, dets, header: str):
    import cv2
    img = frame.copy()
    for name, conf, box in sorted(dets, key=lambda d: d[1]):
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        col = _bgr(name)
        floor = PERSON_MIN if name == PERSON else OBJ_MIN
        if conf >= floor:
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
        else:
            _dashed_rect(img, (x0, y0), (x1, y1), col, 1)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ty = y0 - 3 if y0 > th + 6 else y0 + th + 3
        cv2.rectangle(img, (x0, ty - th - 2), (x0 + tw + 4, ty + 2), col, -1)
        cv2.putText(img, label, (x0 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    lines = [header] + ([] if dets else ["no detections"])
    for i, ln in enumerate(lines):
        y = 16 + i * 16
        cv2.putText(img, ln, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, ln, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def resolve_video(arg: str) -> tuple[str, Path]:
    p = Path(arg)
    if p.suffix and p.exists():
        return p.stem, p
    from tools.labels import norm_id, video_index
    cid = norm_id(arg)
    idx = video_index()
    if cid not in idx:
        raise FileNotFoundError(f"{arg}: no video for {cid} in {config.DATA_DIR}")
    return cid, idx[cid]


def inspect(models, clip: str, video: Path, out: Path, fps: float, batch: int = 8) -> dict:
    import cv2
    info = probe(str(video))
    crop = info.crop
    times = sample_times(info.duration, fps)
    d = out / clip
    (d / "raw").mkdir(parents=True, exist_ok=True)
    rows, counts, empty, frames = [], Counter(), 0, []
    t_dec = t_det = 0.0
    buf = []
    src = Path(models.source).name if models.source != config.DETECTOR_REPO else "stock"

    def flush():
        nonlocal t_det, empty
        if not buf:
            return
        t0 = time.time()
        dets = models.detect([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for *_, f in buf])
        t_det += time.time() - t0
        for (fi, want, actual, frame), ds in zip(buf, dets):
            empty += not ds
            counts.update(n for n, *_ in ds)
            rows.extend(frame_rows(clip, fi, want, actual, ds, crop))
            img = draw(frame, ds, f"{clip}  t={actual:.2f}s  #{fi}  [{src}]")
            cv2.imwrite(str(d / f"{fi:04d}_t{actual:07.2f}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            cv2.imwrite(str(d / "raw" / f"{fi:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            frames.append({"idx": fi, "requested_t": round(float(want), 3), "actual_t": round(float(actual), 3),
                           "w": int(frame.shape[1]), "h": int(frame.shape[0]), "img": f"{clip}/raw/{fi:04d}.jpg",
                           "dets": [{"cls": n, "role": role_of(n) or "", "conf": round(float(c), 3),
                                     "box": [round(float(v), 1) for v in b],
                                     "obox": [round(v, 1) for v in to_original(b, crop)]} for n, c, b in ds]})
        buf.clear()

    t0 = time.time()
    for fi, (want, actual, frame) in enumerate(iter_frames_at(str(video), times, crop)):
        buf.append((fi, want, actual, frame))
        if len(buf) >= batch:
            t_dec += time.time() - t0
            flush()
            t0 = time.time()
    t_dec += time.time() - t0
    flush()

    with open(d / "detections.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    n = len({r["frame_idx"] for r in rows})  # decoded frames (a short video yields fewer than requested)
    summary = {"clip": clip, "video": str(video), "crop": list(crop), "frames": n, "frames_requested": len(times),
               "frames_without_detections": empty, "boxes_by_class": dict(counts.most_common()),
               "reported_boxes": int(sum(int(r["reported"] or 0) for r in rows if r["cls"])),
               "decode_s": round(t_dec, 2), "detect_s": round(t_det, 2),
               "detect_ms_per_frame": round(1000 * t_det / max(n, 1), 1)}
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary, frames


def write_viewer(out: Path, run: dict, clips: list[dict]) -> Path:
    """index.html with the box data embedded; frames are loaded by relative path."""
    tpl = (Path(__file__).parent / "inspect_viewer.html").read_text()
    data = json.dumps({"run": run, "clips": clips, "colors": COLORS}).replace("</", "<\\/")
    page = out / "index.html"
    page.write_text(tpl.replace("__DATA__", data))
    return page


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", help="clip ids (X-30) or video paths")
    ap.add_argument("--fps", type=float, default=1.0, help="frames per second to sample (default 1)")
    ap.add_argument("--out", type=Path, default=config.RUNS_DIR / "inspect")
    ap.add_argument("--all-classes", action="store_true", help="also show classes the engine drops")
    args = ap.parse_args()

    from engine.perception import Models
    t0 = time.time()
    models = Models(pose=False)
    load_s = round(time.time() - t0, 2)
    if args.all_classes:
        models.keep = set(models.id2label.values())
    run = {"detector_source": models.source, "detector_fingerprint": config.detector_fingerprint(),
           "device": models.device, "model_load_s": load_s, "fps": args.fps,
           "obj_min": OBJ_MIN, "person_min": PERSON_MIN, "kept_classes": sorted(models.keep), "clips": [],
           "written": time.strftime("%Y-%m-%d %H:%M")}
    viewer_clips = []
    for c in args.clips:
        clip, video = resolve_video(c)
        s, frames = inspect(models, clip, video, args.out, args.fps)
        run["clips"].append(s)
        viewer_clips.append({"clip": clip, "summary": s, "frames": frames})
        print(f"{clip}: {s['frames']} frames, {s['frames_without_detections']} with no detections, "
              f"{sum(s['boxes_by_class'].values())} boxes ({s['reported_boxes']} reported), "
              f"{s['detect_ms_per_frame']} ms/frame detect")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run.json").write_text(json.dumps(run, indent=1))
    page = write_viewer(args.out, {k: v for k, v in run.items() if k != "clips"}, viewer_clips)
    print(f"wrote {args.out} (local only); open {page}")


if __name__ == "__main__":
    main()
