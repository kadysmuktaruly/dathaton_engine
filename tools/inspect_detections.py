"""Run only the object detector on clips and save annotated frames + box tables, for local review.

Usage:
  uv run python -m tools.inspect_detections X-30 X-17 [--fps 1] [--out runs/inspect] [--detector-only]
  uv run python -m tools.inspect_detections all          # every labelled clip
  ... --report reports/eval_reviewed_stock.md            # labels + engine predictions shown in the viewer
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
Every box the detector returns is shown (all classes), with the reason the engine keeps or drops
it. Unless --detector-only, the engine pipeline also runs locally (perception with tracking and
pose, then facts): people get their track id and counting reason from Facts.person_log, and the
scene anchors (desk, laptop, machine, safe) are shown with their provenance (explicit / proxy).
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from engine import config
from engine.classes import COLORS, CONTEXT_ROLES, role_of
from engine.perception import CANDIDATE_MIN, OBJ_MIN, PERSON, PERSON_MIN
from engine.video import iter_frames_at, probe, sample_times

FIELDS = ["clip", "frame_idx", "requested_t", "actual_t", "cls", "role", "conf", "reported",
          "reason", "track", "cx0", "cy0", "cx1", "cy1", "ox0", "oy0", "ox1", "oy1"]
MATCH_IOU = 0.5     # a displayed person box belongs to the engine's person row it overlaps this much
MATCH_DT = 0.2      # s: displayed frame <-> nearest perception frame


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def explain_detections(dets, person_rows, engine_keep) -> list[dict]:
    """Why the engine keeps or drops each raw detector box.

    person_rows: the engine's Facts.person_log rows for this frame (or None without the pipeline).
    Returns per box: reason, retained (kept by the engine at all), reported (>= the reporting
    cutoff), track (ByteTrack id or None)."""
    out = []
    for name, conf, box in dets:
        role = role_of(name)
        info = {"reason": "", "retained": False, "reported": False, "track": None}
        if name == PERSON:
            if conf < PERSON_MIN:
                info["reason"] = "below person floor"
            elif person_rows is None:
                info.update(reason="person (pipeline not run)", retained=True, reported=True)
            else:
                best, best_iou = None, MATCH_IOU
                for r in person_rows.itertuples():
                    iou = box_iou(box, (r.x0, r.y0, r.x1, r.y1))
                    if iou >= best_iou:
                        best, best_iou = r, iou
                if best is None:
                    info["reason"] = "no matching engine person row"
                else:
                    info["reason"] = best.reason
                    info["track"] = int(best.track_id) if best.source == "tracked" else None
                    info["retained"] = info["reported"] = best.reason == "counted"
        elif role is None or name not in engine_keep:
            info["reason"] = "class not used"
        elif conf < CANDIDATE_MIN:
            info["reason"] = "below candidate floor"
        elif conf < OBJ_MIN:
            info.update(reason="layout only (below reporting cutoff)", retained=True)
        else:
            info.update(reason="reported (context only)" if role in CONTEXT_ROLES else "reported",
                        retained=True, reported=True)
        out.append(info)
    return out


def to_original(box, crop) -> list[float]:
    """Cropped-frame box -> original-frame box (add the black-bar crop offset)."""
    ox, oy = crop[0], crop[1]
    return [float(box[0]) + ox, float(box[1]) + oy, float(box[2]) + ox, float(box[3]) + oy]


def frame_rows(clip: str, frame_idx: int, requested_t: float, actual_t: float, dets, crop,
               explained: list[dict] | None = None) -> list[dict]:
    """Table rows for one frame; a frame with no detections still gets one (empty) row."""
    base = {"clip": clip, "frame_idx": frame_idx, "requested_t": round(float(requested_t), 3),
            "actual_t": round(float(actual_t), 3)}
    if not dets:
        return [{**base, **{k: "" for k in FIELDS if k not in base}}]
    rows = []
    for k, (name, conf, box) in enumerate(dets):
        floor = PERSON_MIN if name == PERSON else OBJ_MIN
        ex = explained[k] if explained else {}
        o = to_original(box, crop)
        rows.append({**base, "cls": name, "role": role_of(name) or "", "conf": round(float(conf), 3),
                     "reported": int(ex["reported"]) if ex else int(conf >= floor),
                     "reason": ex.get("reason", ""), "track": "" if ex.get("track") is None else ex["track"],
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


def load_truth(report: Path) -> dict:
    """Per clip from an eval report's per-video table: labelled violation types and times, and
    what the engine predicted. Labels come from the labels file via tools/eval."""
    out = {}
    if not report or not Path(report).exists():
        return out
    text = Path(report).read_text().split("## Per video", 1)[-1]
    for line in text.splitlines():
        c = [x.strip() for x in line.split("|")[1:-1]]
        if len(c) < 6 or not c[0][:2] in ("X-", "Х-"):
            continue
        try:
            preds, label_t, wanted = (ast.literal_eval(c[3]), ast.literal_eval(c[4]), ast.literal_eval(c[5]))
        except (ValueError, SyntaxError):
            continue
        out[c[0].replace("Х", "X")] = {"labelled": c[1] == "V", "predicted_flag": c[2] == "V",
                                       "label_t": [float(t) for t in label_t], "wanted": list(wanted),
                                       "predicted": [[float(t), str(code)] for t, code in preds]}
    return out


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


def run_pipeline(models, video: Path):
    """Engine perception (tracking + pose, cached locally) and facts for one clip."""
    from engine import facts as F
    from engine import perception
    perc = perception.run(str(video), models=models)
    crop = perc.info.crop
    fa = F.build(perc.df, perc.times, crop[2] - crop[0], crop[3] - crop[1], video=str(video), perc=perc)
    return perc, fa


def inspect(models, clip: str, video: Path, out: Path, fps: float, batch: int = 8,
            engine_keep: set | None = None, pipeline: bool = True) -> dict:
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
    engine_keep = set(models.keep) if engine_keep is None else engine_keep
    fa = None
    if pipeline:
        _, fa = run_pipeline(models, video)
    plog = fa.person_log if fa is not None else None
    plog_t = np.unique(plog.t.to_numpy()) if plog is not None and len(plog) else np.zeros(0)
    reasons = Counter()

    def person_rows_at(t):
        if plog is None:
            return None
        if not len(plog_t):
            return plog.iloc[0:0]
        k = int(np.argmin(np.abs(plog_t - t)))
        if abs(plog_t[k] - t) > MATCH_DT:
            return plog.iloc[0:0]
        return plog[np.isclose(plog.t, plog_t[k])]

    def flush():
        nonlocal t_det, empty
        if not buf:
            return
        t0 = time.time()
        models.keep = set(models.id2label.values())  # raw view: every class the detector outputs
        try:
            dets = models.detect([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for *_, f in buf])
        finally:
            models.keep = engine_keep
        t_det += time.time() - t0
        for (fi, want, actual, frame), ds in zip(buf, dets):
            empty += not ds
            counts.update(n for n, *_ in ds)
            ex = explain_detections(ds, person_rows_at(actual), engine_keep)
            reasons.update(f"{n}: {e['reason']}" if n == PERSON else e["reason"] for (n, *_), e in zip(ds, ex))
            rows.extend(frame_rows(clip, fi, want, actual, ds, crop, ex))
            img = draw(frame, ds, f"{clip}  t={actual:.2f}s  #{fi}  [{src}]")
            cv2.imwrite(str(d / f"{fi:04d}_t{actual:07.2f}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            cv2.imwrite(str(d / "raw" / f"{fi:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            frames.append({"idx": fi, "requested_t": round(float(want), 3), "actual_t": round(float(actual), 3),
                           "w": int(frame.shape[1]), "h": int(frame.shape[0]), "img": f"{clip}/raw/{fi:04d}.jpg",
                           "dets": [{"cls": n, "role": role_of(n) or "", "conf": round(float(c), 3),
                                     "box": [round(float(v), 1) for v in b],
                                     "obox": [round(v, 1) for v in to_original(b, crop)], **e}
                                    for (n, c, b), e in zip(ds, ex)]})
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
               "detect_ms_per_frame": round(1000 * t_det / max(n, 1), 1),
               "reasons_on_displayed_frames": dict(reasons.most_common())}
    if fa is not None:
        sc = fa.scene
        summary["anchors"] = {k: (getattr(sc, k).as_list() if getattr(sc, k) is not None else None)
                              for k in ("desk", "laptop", "machine", "safe")}
        summary["provenance"] = dict(sc.provenance, laptop="explicit:laptop" if sc.laptop is not None else None)
        pl = fa.person_log
        summary["person_log_all_frames"] = dict(Counter(pl.reason).most_common()) if pl is not None else {}
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
    ap.add_argument("clips", nargs="+", help="clip ids (X-30), video paths, or 'all' for every labelled clip")
    ap.add_argument("--report", type=Path, default=config.ROOT / "reports" / "eval_reviewed_stock.md",
                    help="eval report whose labels and predictions are shown in the viewer")
    ap.add_argument("--fps", type=float, default=1.0, help="frames per second to sample (default 1)")
    ap.add_argument("--out", type=Path, default=config.RUNS_DIR / "inspect")
    ap.add_argument("--detector-only", action="store_true",
                    help="skip the engine pipeline (no track ids, counting reasons or anchors)")
    args = ap.parse_args()

    from engine.perception import Models
    t0 = time.time()
    models = Models(pose=not args.detector_only)
    load_s = round(time.time() - t0, 2)
    engine_keep = set(models.keep)
    run = {"detector_source": models.source, "detector_fingerprint": config.detector_fingerprint(),
           "device": models.device, "model_load_s": load_s, "fps": args.fps,
           "obj_min": OBJ_MIN, "person_min": PERSON_MIN, "kept_classes": sorted(engine_keep),
           "pipeline": not args.detector_only, "clips": [],
           "written": time.strftime("%Y-%m-%d %H:%M")}
    truth = load_truth(args.report)
    run["report"] = str(args.report) if truth else None
    clip_args = args.clips
    if clip_args == ["all"]:
        from tools.labels import load_clips
        clip_args = [c.clip_id for c in load_clips()]
    viewer_clips = []
    for c in clip_args:
        clip, video = resolve_video(c)
        s, frames = inspect(models, clip, video, args.out, args.fps, engine_keep=engine_keep,
                            pipeline=not args.detector_only)
        run["clips"].append(s)
        viewer_clips.append({"clip": clip, "summary": s, "frames": frames, "truth": truth.get(clip),
                             "anchors": s.get("anchors"), "provenance": s.get("provenance"),
                             "duration": probe(str(video)).duration})
        print(f"{clip}: {s['frames']} frames, {s['frames_without_detections']} with no detections, "
              f"{sum(s['boxes_by_class'].values())} boxes ({s['reported_boxes']} reported), "
              f"{s['detect_ms_per_frame']} ms/frame detect")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run.json").write_text(json.dumps(run, indent=1))
    page = write_viewer(args.out, {k: v for k, v in run.items() if k != "clips"}, viewer_clips)
    print(f"wrote {args.out} (local only); open {page}")


if __name__ == "__main__":
    main()
