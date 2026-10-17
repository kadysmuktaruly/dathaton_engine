"""orig_and_vlm: the rules engine (engine/, unchanged) + a newer VLM as better eyes.

Two changes to engine.analyze, both switchable (config.USE_SCAN, config.USE_CLOSEOUT):
  1. the desk scan the rules read (cash / bag / collection bag / drink) is asked by Qwen3.5-4B
     (orig_and_vlm.scan) instead of the engine's Qwen3-VL-4B scan;
  2. B5 'cashier_left_room' alarms go through a close-out check (orig_and_vlm.closeout).
Everything else (perception, rules, merge, the Qwen3-VL verifier, evidence, catalog wording,
output schema, partial result + deadline) is the engine's own code. Same signature and output
as engine.analyze, so tools.eval and api/main.py can run it (see eval.py, api.py).
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


def detect_events(perc, frame_w: int, frame_h: int, video: str | None = None, budget=None):
    """engine.detect_events with the Qwen3.5 desk scan and its bisection refine."""
    import engine
    from engine import facts as F
    from engine.events import merge
    from engine.rules import RULES
    from engine.vlm import Budget
    from orig_and_vlm import config, scan

    if budget is None or video is None or not config.USE_SCAN:
        return engine.detect_events(perc, frame_w, frame_h, video, budget)
    fa = F.build(perc.df, perc.times, frame_w, frame_h, video=video, perc=perc)
    timings = {}
    t0 = time.time()
    fa.scan = scan.scan(video, perc, fa.scene, Budget(min(config.SCAN_BUDGET_S, budget.left())),
                        extra_times=scan.key_moments(fa))
    timings["scan"] = round(time.time() - t0, 2)
    events = []
    for name, fn in RULES.items():
        t0 = time.time()
        events += fn(fa)
        timings[name] = round(time.time() - t0, 3)
    events = engine._resolve(events)
    box = fa.scan.attrs.get("box")
    for e in events:
        ref = e.info.get("refine")
        if ref and ref[1] is not None and box is not None and budget.ok(3.0):
            t_new = scan.refine(video, perc, box, ref[0], ref[1], e.start, budget)
            e.info["t_scan"] = e.start
            e.start = t_new
            e.evidence_t = [t_new]
    return fa, merge(events), timings


def analyze(video_path: str | Path, job_id: str | None = None, use_vlm: bool = True,
            debug_dir: str | Path | None = None, models=None, on_partial=None,
            deadline: float | None = None) -> dict:
    """engine.analyze with the two changes above (rules-only partial first, as the engine)."""
    import engine
    from engine import config as engine_config, perception
    from orig_and_vlm import closeout, config

    job_id = job_id or uuid.uuid4().hex[:12]
    video_path = str(video_path)
    T = {}
    t0 = time.time()
    perc = perception.run(video_path, models=models or engine._get_models())
    T["perception_s"] = round(time.time() - t0, 2)
    crop = perc.info.crop
    fw, fh = crop[2] - crop[0], crop[3] - crop[1]

    t0 = time.time()
    fa, events, rule_t = engine.detect_events(perc, fw, fh, video_path, None)
    T["rules_s"] = round(time.time() - t0, 2)
    if on_partial is not None:
        on_partial(engine._build(job_id, video_path, perc, fa, events)[0])

    vlm_budget = config.VLM_BUDGET_S
    if deadline is not None:
        vlm_budget = min(vlm_budget, deadline - time.time() - 15.0)
    T["vlm_s"] = 0.0
    closeout_dbg = []
    if use_vlm and vlm_budget > 5.0:
        from engine import vlm
        budget = vlm.Budget(vlm_budget)
        t0 = time.time()
        fa, events, rule_t = detect_events(perc, fw, fh, video_path, budget)
        T["scan_s"] = rule_t.get("scan")
        if config.USE_CLOSEOUT:
            t1 = time.time()
            n0 = len(events)
            events, closeout_dbg = closeout.filter_events(video_path, perc, fa, events, budget)
            T["closeout_s"] = round(time.time() - t1, 2)
            T["closeout_dropped"] = n0 - len(events)
        hopeful = [e for e in events if e.score + vlm.YES_LIFT >= engine.VERDICT_THRESHOLD]
        if hopeful and budget.ok(3.0):
            vlm.verify(video_path, perc, fa, hopeful, budget)  # the engine's Qwen3-VL verifier
        T["vlm_s"] = round(time.time() - t0, 2)
        T["vlm_budget_left_s"] = round(budget.left(), 1)
        try:
            import mlx.core as mx
            T["mlx_peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
        except Exception:
            pass
    elif use_vlm:
        T["vlm_skipped"] = "deadline"

    t0 = time.time()
    res, kept, debug_events = engine._build(job_id, video_path, perc, fa, events)
    T["evidence_s"] = round(time.time() - t0, 2)

    dbg = {"job_id": job_id, "video": video_path, "pipeline": "orig_and_vlm",
           "switches": {"scan": config.USE_SCAN, "closeout": config.USE_CLOSEOUT, "eyes": config.EYES_REPO},
           "timings": T, "rule_timings": rule_t,
           "scene": {"desk": fa.scene.desk and fa.scene.desk.as_list(),
                     "laptop": fa.scene.laptop and fa.scene.laptop.as_list(),
                     "safe": fa.scene.safe and fa.scene.safe.as_list(), "seat": fa.seat},
           "scan": None if fa.scan is None else fa.scan.to_dict("records"),
           "closeout": closeout_dbg,
           "events": debug_events,
           "dropped": [{"code": e.code, "start": e.start, "score": e.score, "info": e.info}
                       for e in events if e not in kept]}
    out_dir = Path(debug_dir or engine_config.RUNS_DIR / "jobs" / job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "debug.json").write_text(json.dumps(dbg, ensure_ascii=False, indent=1, default=str))
    return res
