"""Frame-level perception: RT-DETRv2 objects, ByteTrack person tracks, RTMPose keypoints.

Output is one table with a row per detection:
  t, frame_idx, kind ('obj' | 'person'), cls, conf, x0, y0, x1, y1, track_id, kps (17x3 flat, persons only)
Coordinates are in the cropped frame's pixels. Results are cached per (video md5, settings).
"""
from __future__ import annotations

import glob
import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from engine import config
from engine.video import VideoInfo, iter_frames_at, probe, sample_times

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# COCO classes we keep (names as RT-DETR's id2label spells them)
KEEP = {"person", "laptop", "handbag", "backpack", "suitcase", "cup", "cell phone", "diningtable",
        "dining table", "tvmonitor", "tv", "chair", "book", "bottle", "remote", "keyboard", "mouse"}
PERSON = "person"
DET_THRESH = 0.25  # TUNE: low on purpose, small objects at 464 px
POSE_BATCH_MIN_H = 30  # px; skip pose for tiny boxes


@dataclass
class Perception:
    info: VideoInfo
    times: list[float]
    df: pd.DataFrame
    timings: dict = field(default_factory=dict)


class Models:
    """Loads models once; reuse across videos (API worker keeps one instance)."""

    def __init__(self, device: str | None = None):
        import torch
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
        from rtmlib import RTMPose

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else
                                 "cuda" if torch.cuda.is_available() else "cpu")
        self.proc = RTDetrImageProcessor.from_pretrained(config.DETECTOR_REPO)
        self.det = RTDetrV2ForObjectDetection.from_pretrained(config.DETECTOR_REPO).to(self.device).eval()
        self.id2label = self.det.config.id2label
        onnx = glob.glob(str(config.WEIGHTS_DIR / "rtmpose" / "**" / "end2end.onnx"), recursive=True)
        if not onnx:
            raise FileNotFoundError("RTMPose weights missing: run `python -m tools.download_weights`")
        self.pose = RTMPose(onnx[0], model_input_size=(192, 256), backend="onnxruntime", device="cpu")

    def detect(self, frames_rgb: list[np.ndarray]) -> list[list[tuple[str, float, np.ndarray]]]:
        torch = self.torch
        with torch.no_grad():
            inp = self.proc(images=frames_rgb, return_tensors="pt").to(self.device)
            out = self.det(**inp)
            sizes = [f.shape[:2] for f in frames_rgb]
            res = self.proc.post_process_object_detection(out, threshold=DET_THRESH, target_sizes=sizes)
        batch = []
        for r in res:
            dets = []
            for s, l, b in zip(r["scores"].cpu().numpy(), r["labels"].cpu().numpy(), r["boxes"].cpu().numpy()):
                name = self.id2label[int(l)]
                if name in KEEP:
                    dets.append((name, float(s), b))
            batch.append(dets)
        return batch


def _cache_path(video: Path, fps: float) -> Path:
    h = hashlib.md5()
    with open(video, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    key = f"{h.hexdigest()[:16]}_fps{fps:g}_v2"
    return config.CACHE_DIR / "perception" / f"{key}.parquet"


def run(video: str | Path, models: Models | None = None, fps: float = config.SAMPLE_FPS,
        use_cache: bool | None = None, batch: int = 8) -> Perception:
    import cv2
    import supervision as sv

    video = Path(video)
    info = probe(str(video))
    times = sample_times(info.duration, fps)
    cache = _cache_path(video, fps)
    tcache = cache.with_suffix(".times.npy")
    use_cache = config.USE_CACHE if use_cache is None else use_cache
    if use_cache and cache.exists() and tcache.exists():
        return Perception(info, np.load(tcache).tolist(), pd.read_parquet(cache), {"cached": True})

    models = models or Models()
    tracker = sv.ByteTrack(frame_rate=int(round(fps)), lost_track_buffer=int(4 * fps),  # TUNE
                           track_activation_threshold=0.3, minimum_matching_threshold=0.8)
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
    return Perception(info, actual, df, {"decode_s": round(t_dec, 2), "detect_s": round(t_det, 2),
                                        "pose_s": round(t_pose, 2), "frames": len(times)})
