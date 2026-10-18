"""Frame-level perception: RT-DETRv2 objects, ByteTrack person tracks, RTMPose keypoints.

Output is one table with a row per detection:
  t, frame_idx, kind, cls, conf, x0, y0, x1, y1, track_id, kps (17x3 flat, tracked persons only)
kind: 'obj' (objects), 'person' (ByteTrack output: identity over time, with pose), 'person_raw'
(every person box the detector returned; track_id = the tracked box it matches by IoU, else -1).
Raw rows keep people that tracking dropped or has not confirmed yet visible downstream.
Coordinates are in the cropped frame's pixels. Results are cached per (video md5, settings,
detector). The detector is the stock COCO model unless ENGINE_DETECTOR_PATH names a local
fine-tuned checkpoint (see config.detector_source).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from engine import config
from engine.classes import IGNORED, ROLES, role_map, runtime_keep  # noqa: F401  (ROLES re-exported)
from engine.video import VideoInfo, iter_frames_at, probe, sample_times

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PERSON = "person"
DET_THRESH = 0.25  # TUNE: low on purpose, small objects at 464 px

PERSON_MIN = 0.40     # TUNE: people below this are dropped
OBJ_MIN = 0.50        # TUNE: object boxes at or above this are 'confident' (reported)
# An anchor cluster's top 10% of frames must reach this. Stock look-alikes for the counting
# machine score low, so their level is lower.
ANCHOR_MIN = {"laptop": 0.50, "desk": 0.50, "table": 0.50, "counting_machine": 0.50,  # TUNE
              "machine_like": 0.40, "safe": 0.50}
CANDIDATE_MIN = 0.25  # TUNE: weaker object boxes still go into anchor clustering


def filter_detections(df: pd.DataFrame) -> pd.DataFrame:
    """Keep classes that map to a role; add `role` and `confident` columns.

    People below PERSON_MIN and objects below CANDIDATE_MIN are dropped. Objects between
    CANDIDATE_MIN and OBJ_MIN stay (scene anchors cluster them) but are not confident.
    """
    df = df.assign(role=df.cls.map(role_map()))  # `cls` stays as detected (e.g. safe_open vs safe)
    df = df[df.role.notna()]
    is_p = df.kind.isin(["person", "person_raw"])
    keep = (is_p & (df.conf >= PERSON_MIN)) | (~is_p & (df.conf >= CANDIDATE_MIN))
    df = df[keep]
    df = df.assign(confident=df.kind.isin(["person", "person_raw"]) | (df.conf >= OBJ_MIN))
    return df.reset_index(drop=True)


POSE_BATCH_MIN_H = 30  # px; skip pose for tiny boxes
POSE_INPUT = (192, 256)
# ByteTrack settings (part of the cache key). lost_track_buffer is in seconds here.
TRACKER = {"lost_track_buffer_s": 4.0, "track_activation_threshold": 0.3,  # TUNE
           "minimum_matching_threshold": 0.8}
RAW_MATCH_IOU = 0.5    # a raw person box belongs to the tracked box it overlaps this much
SCHEMA_VERSION = "v3"  # v3: person_raw rows, context classes kept
MIN_BOX_PX = 2.0       # boxes thinner than this after clipping are dropped


def clip_box(b, w: int, h: int) -> np.ndarray | None:
    """Clip an xyxy box to the frame; None if it is degenerate after clipping."""
    x0, y0, x1, y1 = (float(v) for v in b)
    x0, x1 = min(max(x0, 0.0), w), min(max(x1, 0.0), w)
    y0, y1 = min(max(y0, 0.0), h), min(max(y1, 0.0), h)
    if x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
        return None
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def clip_df(df: pd.DataFrame, w: int, h: int) -> pd.DataFrame:
    """clip_box for a detection table (cached results were written before clipping existed)."""
    if df.empty:
        return df
    df = df.assign(x0=df.x0.clip(0, w), x1=df.x1.clip(0, w), y0=df.y0.clip(0, h), y1=df.y1.clip(0, h))
    return df[((df.x1 - df.x0) >= MIN_BOX_PX) & ((df.y1 - df.y0) >= MIN_BOX_PX)].reset_index(drop=True)


@dataclass
class Perception:
    info: VideoInfo
    times: list[float]
    df: pd.DataFrame
    timings: dict = field(default_factory=dict)
    labels: frozenset | None = None  # detector class names; None = stock COCO detector


class Models:
    """Loads models once; reuse across videos (API worker keeps one instance)."""

    def __init__(self, device: str | None = None, pose: bool = True):
        import torch
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

        self.torch = torch
        self.device = device or config.DEVICE or ("mps" if torch.backends.mps.is_available() else
                                 "cuda" if torch.cuda.is_available() else "cpu")
        self.source = config.detector_source()  # raises if ENGINE_DETECTOR_PATH is bad
        self.proc = RTDetrImageProcessor.from_pretrained(self.source)
        self.det = RTDetrV2ForObjectDetection.from_pretrained(self.source).to(self.device).eval()
        self.id2label = {int(k): str(v) for k, v in self.det.config.id2label.items()}
        self.keep = runtime_keep(self.id2label)
        self.pose = None
        if pose:  # the detector-only inspection tool skips pose
            from rtmlib import RTMPose
            onnx = glob.glob(str(config.WEIGHTS_DIR / "rtmpose" / "**" / "end2end.onnx"), recursive=True)
            if not onnx:
                raise FileNotFoundError("RTMPose weights missing: run `python -m tools.download_weights`")
            self.pose = RTMPose(onnx[0], model_input_size=POSE_INPUT, backend="onnxruntime", device="cpu")
            # rtmlib does not accept SessionOptions. Replace its default session so pose
            # does not create a thread pool spanning all server cores.
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = config.ONNX_THREADS
            options.inter_op_num_threads = 1
            self.pose.session = ort.InferenceSession(
                onnx[0], sess_options=options, providers=["CPUExecutionProvider"],
            )

    def detect(self, frames_rgb: list[np.ndarray]) -> list[list[tuple[str, float, np.ndarray]]]:
        torch = self.torch
        with torch.no_grad():
            inp = self.proc(images=frames_rgb, return_tensors="pt").to(self.device)
            out = self.det(**inp)
            sizes = [f.shape[:2] for f in frames_rgb]
            res = self.proc.post_process_object_detection(out, threshold=DET_THRESH, target_sizes=sizes)
        batch = []
        for r, (h, w) in zip(res, sizes):
            dets = []
            for s, l, b in zip(r["scores"].cpu().numpy(), r["labels"].cpu().numpy(), r["boxes"].cpu().numpy()):
                name = self.id2label.get(int(l))
                if name not in self.keep:
                    continue
                b = clip_box(b, w, h)
                if b is not None:
                    dets.append((name, float(s), b))
            batch.append(dets)
        return batch


def pose_signature() -> str:
    onnx = sorted(glob.glob(str(config.WEIGHTS_DIR / "rtmpose" / "**" / "end2end.onnx"), recursive=True))
    if not onnx:
        return "none"
    return f"{Path(onnx[0]).parent.name}:{Path(onnx[0]).stat().st_size}"


def perception_settings(fps: float, crop, fingerprint: str | None = None, pose_sig: str | None = None) -> dict:
    """Everything that changes the cached table besides the video itself."""
    return {
        "schema": SCHEMA_VERSION, "fps": float(fps), "crop": [int(v) for v in crop],
        "detector": fingerprint if fingerprint is not None else config.detector_fingerprint(),
        "det_thresh": DET_THRESH, "classes": sorted(set().union(*ROLES.values())),
        "tracker": {**TRACKER, "frame_rate": int(round(fps))}, "raw_match_iou": RAW_MATCH_IOU,
        "pose": {"model": pose_sig if pose_sig is not None else pose_signature(),
                 "input": list(POSE_INPUT), "min_h": POSE_BATCH_MIN_H},
    }


def cache_key(video_md5: str, settings: dict) -> str:
    h = hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    return f"{video_md5[:16]}_{settings['schema']}_{h}"


def _cache_path(video: Path, fps: float, crop) -> Path:
    h = hashlib.md5()
    with open(video, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    key = cache_key(h.hexdigest(), perception_settings(fps, crop))
    return config.CACHE_DIR / "perception" / f"{key}.parquet"


def match_raw_to_tracks(raw_boxes: np.ndarray, tracked_boxes: np.ndarray, tracked_ids) -> list[int]:
    """Track id of the tracked box each raw person box overlaps by >= RAW_MATCH_IOU, else -1."""
    out = []
    for b in raw_boxes:
        best, best_iou = -1, RAW_MATCH_IOU
        for tb, tid in zip(tracked_boxes, tracked_ids):
            ix = max(0.0, min(b[2], tb[2]) - max(b[0], tb[0]))
            iy = max(0.0, min(b[3], tb[3]) - max(b[1], tb[1]))
            inter = ix * iy
            union = (b[2] - b[0]) * (b[3] - b[1]) + (tb[2] - tb[0]) * (tb[3] - tb[1]) - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= best_iou:
                best, best_iou = int(tid), iou
        out.append(best)
    return out


def run(video: str | Path, models: Models | None = None, fps: float = config.SAMPLE_FPS,
        use_cache: bool | None = None, batch: int = 8) -> Perception:
    import cv2
    import supervision as sv

    video = Path(video)
    info = probe(str(video))
    times = sample_times(info.duration, fps)
    cache = _cache_path(video, fps, info.crop)
    tcache = cache.with_suffix(".times.npy")
    use_cache = config.USE_CACHE if use_cache is None else use_cache
    fw, fh = info.crop[2] - info.crop[0], info.crop[3] - info.crop[1]
    if models is not None and models.source != config.detector_source():
        raise ValueError(f"loaded detector {models.source} != selected {config.detector_source()} "
                         "(ENGINE_DETECTOR_PATH changed after the models were loaded)")
    if use_cache and cache.exists() and tcache.exists():
        df = clip_df(pd.read_parquet(cache), fw, fh)
        return Perception(info, np.load(tcache).tolist(), filter_detections(df),
                          {"cached": True}, config.detector_labels())

    models = models or Models()
    tracker = sv.ByteTrack(frame_rate=int(round(fps)), lost_track_buffer=int(TRACKER["lost_track_buffer_s"] * fps),
                           track_activation_threshold=TRACKER["track_activation_threshold"],
                           minimum_matching_threshold=TRACKER["minimum_matching_threshold"])
    rows = []
    t_det = t_pose = t_dec = 0.0
    buf: list[tuple[int, float, np.ndarray]] = []

    def flush():
        nonlocal t_det, t_pose
        if not buf:
            return
        t0 = time.time()
        dets = models.detect([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for _, _, f in buf])
        t_det += time.time() - t0
        for (fi, t, frame), d in zip(buf, dets):
            persons = [(c, b) for n, c, b in d if n == PERSON]
            for n, c, b in d:
                if n != PERSON:
                    rows.append((t, fi, "obj", n, c, *map(float, b), -1, None))
            if persons:
                svd = sv.Detections(xyxy=np.array([b for _, b in persons], dtype=np.float32),
                                    confidence=np.array([c for c, _ in persons], dtype=np.float32),
                                    class_id=np.zeros(len(persons), dtype=int))
            else:
                svd = sv.Detections.empty()
            tracked = tracker.update_with_detections(svd)
            kps = [None] * len(tracked)
            boxes = [b for b in tracked.xyxy if (b[3] - b[1]) >= POSE_BATCH_MIN_H]
            if len(boxes):
                t1 = time.time()
                k, s = models.pose(frame, bboxes=[list(map(float, b)) for b in boxes])
                t_pose += time.time() - t1
                j = 0
                for i, b in enumerate(tracked.xyxy):
                    if (b[3] - b[1]) >= POSE_BATCH_MIN_H:
                        kps[i] = np.concatenate([k[j], s[j][:, None]], axis=1).astype(np.float32).ravel().tolist()
                        j += 1
            for i in range(len(tracked)):
                b = tracked.xyxy[i]
                rows.append((t, fi, "person", PERSON, float(tracked.confidence[i]), *map(float, b),
                             int(tracked.tracker_id[i]), kps[i]))
            if persons:
                raw = np.array([b for _, b in persons], dtype=np.float32)
                tids = match_raw_to_tracks(raw, tracked.xyxy, tracked.tracker_id if len(tracked) else [])
                for (c, b), tid in zip(persons, tids):
                    rows.append((t, fi, "person_raw", PERSON, float(c), *map(float, b), tid, None))
        buf.clear()

    actual: list[float] = []
    t0 = time.time()
    for fi, (_, t, frame) in enumerate(iter_frames_at(str(video), times, info.crop)):
        t = round(float(t), 4)
        actual.append(t)
        buf.append((fi, t, frame))
        if len(buf) >= batch:
            t_dec += time.time() - t0
            flush()
            t0 = time.time()
    t_dec += time.time() - t0
    flush()

    df = pd.DataFrame(rows, columns=["t", "frame_idx", "kind", "cls", "conf", "x0", "y0", "x1", "y1",
                                     "track_id", "kps"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    np.save(tcache, np.array(actual))
    return Perception(info, actual, filter_detections(df), {"decode_s": round(t_dec, 2), "detect_s": round(t_det, 2),
                                        "pose_s": round(t_pose, 2), "frames": len(times)},
                      None if models.source == config.DETECTOR_REPO else frozenset(models.id2label.values()))
