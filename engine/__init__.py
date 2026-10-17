"""Cash-desk video analysis engine. Entry point: analyze(video_path) -> dict (graded schema)."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

SEEN_TEXT = {  # rule-based 'что видно' sentences (the VLM may replace them later)
    ("C1", "entered_empty_room"): "в помещение кассы, где не было кассира, вошёл человек и прошёл к рабочему месту кассира",
    ("C1", "third_person"): "во время обслуживания клиента в помещение кассы вошёл ещё один человек и подошёл к столу кассира",
    ("C1", "to_staff_side"): "в помещение кассы вошёл посторонний и прошёл на сторону рабочего места кассира",
    ("C2", "door_left_open"): "кассир вошёл в помещение кассы и оставил дверь открытой; дверь остаётся открытой во время работы",
    ("B5", "cashier_left_room"): "кассир покинул помещение кассы, рабочее место и сейф остались без присмотра",
}
DEFAULT_SEEN = "в кадре видны действия, похожие на отступление от кассовых процедур"
VERDICT_THRESHOLD = 0.3  # TUNE: low on purpose (recall first)

_models = None


def _get_models():
    global _models
    if _models is None:
        from engine import perception
        _models = perception.Models()
    return _models


def detect_events(perc, frame_w: int, frame_h: int, video: str | None = None):
    """Run facts + all rules. Returns (facts, merged events, per-rule timings)."""
    from engine import facts as F
    from engine.events import merge
    from engine.rules import RULES

    fa = F.build(perc.df, perc.times, frame_w, frame_h, video=video, perc=perc)
    events, timings = [], {}
    for name, fn in RULES.items():
        t0 = time.time()
        events += fn(fa)
        timings[name] = round(time.time() - t0, 3)
    return fa, merge(events), timings


def _add_cashier_boxes(fa, ev):
    if fa.seat is None:
        return
    p = fa.persons
    for t in list(ev.evidence_t):
        near = p[(p.t - t).abs() <= 0.4]
        near = near[near.d_seat <= 0.6]
        if len(near):
            r = near.sort_values("d_seat").iloc[0]
            ev.boxes.setdefault(t, []).append(("кассир", [r.x0, r.y0, r.x1, r.y1]))


def analyze(video_path: str | Path, job_id: str | None = None, use_vlm: bool = True,
            debug_dir: str | Path | None = None, models=None) -> dict:
    from engine import catalog, config, evidence, output, perception

    job_id = job_id or uuid.uuid4().hex[:12]
    video_path = str(video_path)
    T = {}
    t0 = time.time()
    perc = perception.run(video_path, models=models or _get_models())
    T["perception_s"] = round(time.time() - t0, 2)
    crop = perc.info.crop
    fw, fh = crop[2] - crop[0], crop[3] - crop[1]

    t0 = time.time()
    fa, events, rule_t = detect_events(perc, fw, fh, video_path)
    T["rules_s"] = round(time.time() - t0, 2)

    T["vlm_s"] = 0.0
    if use_vlm and events:
        from engine import vlm
        t0 = time.time()
        events = vlm.verify(video_path, perc, fa, events)
        T["vlm_s"] = round(time.time() - t0, 2)

    kept = [e for e in events if e.score >= VERDICT_THRESHOLD]
    violations, debug_events = [], []
    t0 = time.time()
    for ev in kept:
        tc = round(ev.start, 1)
        dur = perc.info.duration
        ev_times = [tc] + [round(min(tc + d, dur - 0.05), 1) for d in (1.0, 2.0) if tc + d < dur - 0.05]
        ev.evidence_t = ev_times
        _add_cashier_boxes(fa, ev)
        seen = ev.info.get("seen") or SEEN_TEXT.get((ev.code, ev.info.get("reason")), DEFAULT_SEEN)
        violations.append({
            "type": catalog.violation(ev.code)["type"],
            "timecode_sec": tc,
            "description": catalog.description(ev.code, seen),
            "evidence_frames": evidence.render(video_path, ev_times, ev.boxes, crop),
        })
        debug_events.append({"code": ev.code, "start": ev.start, "end": ev.end, "score": ev.score,
                             "refs": [f"{r['doc']} {r['point']}" for r in catalog.safe_refs(ev.code)],
                             "info": {k: v for k, v in ev.info.items() if k != "seen"}})
    T["evidence_s"] = round(time.time() - t0, 2)
    res = output.build(job_id, violations)

    dbg = {"job_id": job_id, "video": video_path, "timings": T, "rule_timings": rule_t,
           "scene": {"desk": fa.scene.desk and fa.scene.desk.as_list(),
                     "laptop": fa.scene.laptop and fa.scene.laptop.as_list(),
                     "machine": fa.scene.machine and fa.scene.machine.as_list(),
                     "safe": fa.scene.safe and fa.scene.safe.as_list(),
                     "seat": fa.seat, "notes": fa.scene.notes},
           "events": debug_events,
           "dropped": [{"code": e.code, "start": e.start, "score": e.score, "info": e.info}
                       for e in events if e not in kept]}
    out_dir = Path(debug_dir or config.RUNS_DIR / "jobs" / job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "debug.json").write_text(json.dumps(dbg, ensure_ascii=False, indent=1, default=str))
    return res
