"""Cash-desk video analysis engine. Entry point: analyze(video_path) -> dict (graded schema)."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

SEEN_TEXT = {  # rule-based 'что видно' sentences (the VLM may replace them); roles only, no unseen claims
    ("C2", "door_left_open"): "человек прошёл в помещение кассы, а дверь кассы осталась открытой",
    ("C2", "visitor_through_open_door"): "дверь кассы осталась открытой, и через неё в помещение кассы прошёл посетитель",
    ("B5", "safe_open_while_working"): "сейф остаётся открытым, пока кассир работает за столом",
    ("B5", "safe_open_cashier_left"): "кассир покинул помещение кассы, сейф остался открытым",
    ("B1", "cash_left_cashier_away"): "на столе кассира остались наличные деньги без присмотра, кассир отошёл от рабочего места",
    ("B1", "bag_left_cashier_away"): "на столе кассира осталась инкассаторская сумка, кассир покинул помещение кассы",
    ("B1", "cash_open_new_client"): "на столе открыто лежат наличные деньги, когда к кассе подходит следующий клиент",
    ("B8", "stamps_left_cashier_away"): "кассир покинул помещение кассы, печати остались на столе",
    ("B9", "laptop_active_cashier_left"): "кассир покинул помещение кассы, ноутбук остался открытым и включённым",
    ("A7", "second_client_while_first_present"): "кассир принимает второго клиента, пока первый клиент ещё находится у кассы",
    ("A1", "client_left_during_count"): "клиент уходит, пока его деньги ещё пересчитываются на счётной машине",
    ("A6", "no_client_id"): "кассир принимает деньги клиента; в кадре не видно, чтобы клиент передавал удостоверение личности",
    ("A2", "uv_detector_not_used"): "кассир пересчитывает наличные; в кадре не видно проверки купюр на УФ-детекторе, который стоит на столе",
    ("A4", "cashier_left_client_present"): "кассир покинул помещение кассы, пока клиент находится у кассы и на столе лежат деньги",
    ("D7", "bag_taken_no_id_check"): "кассир принимает инкассаторскую сумку; в кадре не видно, чтобы кассир проверял удостоверение или доверенность инкассатора",
    ("D6", "bag_cash_to_safe_no_count"): "кассир убирает деньги из инкассаторской сумки в сейф; в кадре не видно пересчёта на счётной машине",
    ("B3", "personal_items"): "на рабочем месте кассира лежат личные вещи (сумка или верхняя одежда)",
    ("B3", "drink_next_to_cash"): "кассир ставит напиток на рабочее место рядом с наличными деньгами",
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


SCAN_BUDGET_S = 18.0   # TUNE: VLM desk scan share of the per-video VLM budget
CHECKS_BUDGET_S = 22.0  # TUNE: targeted questions (departures, UV detector)
VLM_BUDGET_S = 60.0    # hard per-video VLM budget


def detect_events(perc, frame_w: int, frame_h: int, video: str | None = None, budget=None):
    """Facts (+ VLM desk scan when a budget is given) + all rules.
    Returns (facts, merged events, per-rule timings)."""
    from engine import facts as F
    from engine.events import merge
    from engine.rules import RULES

    fa = F.build(perc.df, perc.times, frame_w, frame_h, video=video, perc=perc)
    timings = {}
    if budget is not None and video is not None:
        from engine import roles, vlm, vlm_checks, vlm_scan
        t0 = time.time()
        fa.scan = vlm_scan.scan(video, perc, fa.scene, vlm.Budget(min(SCAN_BUDGET_S, budget.left())))
        roles.tag_collectors(fa.sessions, fa.objects, fa.scan)
        timings["scan"] = round(time.time() - t0, 2)
        t0 = time.time()
        cb = vlm.Budget(min(CHECKS_BUDGET_S, budget.left()))
        fa.checks = vlm_checks.run(video, perc, fa, cb)
        fa.checks["uv"] = vlm_checks.run_uv(video, perc, fa, cb)
        timings["checks"] = round(time.time() - t0, 2)
    events = []
    for name, fn in RULES.items():
        t0 = time.time()
        events += fn(fa)
        timings[name] = round(time.time() - t0, 3)
    events = _resolve(events)
    if budget is not None and fa.scan is not None and video is not None:
        from engine import vlm_scan
        crop = perc.info.crop
        box = vlm_scan.desk_crop_box(fa.scene, crop[2] - crop[0], crop[3] - crop[1])
        for e in events:
            ref = e.info.get("refine")
            if ref and ref[1] is not None and budget.ok(3.0):
                t_new = vlm_scan.refine(video, perc, box, ref[0], ref[1], e.start, budget)
                e.info["t_scan"] = e.start
                e.start = t_new
                e.evidence_t = [t_new]
    return fa, merge(events), timings


def _resolve(events):
    """One violation per (code, moment): the same departure can be found by two B1 signals."""
    out = []
    for e in sorted(events, key=lambda e: -e.score):
        if any(o.code == e.code and abs(o.start - e.start) <= 2.0 for o in out):
            continue
        out.append(e)
    return out


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


def _build(job_id, video_path, perc, fa, events):
    """Kept events -> graded JSON (+ debug event list)."""
    from engine import catalog, evidence, output
    crop = perc.info.crop
    kept = [e for e in events if e.score >= VERDICT_THRESHOLD]
    violations, debug_events = [], []
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
    return output.build(job_id, violations), kept, debug_events


def analyze(video_path: str | Path, job_id: str | None = None, use_vlm: bool = True,
            debug_dir: str | Path | None = None, models=None, on_partial=None,
            deadline: float | None = None, on_analysis=None) -> dict:
    """Analyze one video. on_partial(result) is called with a complete rules-only result before
    the VLM stage; `deadline` (epoch seconds) skips the VLM stage when time is short."""
    from engine import config, perception

    job_id = job_id or uuid.uuid4().hex[:12]
    video_path = str(video_path)
    T = {}
    t0 = time.time()
    perc = perception.run(video_path, models=models or _get_models())
    T["perception_s"] = round(time.time() - t0, 2)
    crop = perc.info.crop
    fw, fh = crop[2] - crop[0], crop[3] - crop[1]

    t0 = time.time()
    fa, events, rule_t = detect_events(perc, fw, fh, video_path, None)
    T["rules_s"] = round(time.time() - t0, 2)
    if on_partial is not None or on_analysis is not None:
        partial, partial_kept, _ = _build(job_id, video_path, perc, fa, events)
        if on_partial is not None:
            on_partial(partial)
        if on_analysis is not None:
            from engine.backend import build_analysis
            on_analysis(partial, build_analysis(perc.info, partial_kept, partial), True)

    vlm_budget = VLM_BUDGET_S
    if deadline is not None:
        vlm_budget = min(vlm_budget, deadline - time.time() - 15.0)
    T["vlm_s"] = 0.0
    if use_vlm and vlm_budget > 5.0:
        from engine import vlm
        budget = vlm.Budget(vlm_budget)
        t0 = time.time()
        fa, events, rule_t = detect_events(perc, fw, fh, video_path, budget)
        # events that even a VLM 'yes' cannot lift to the verdict threshold are not asked about
        hopeful = [e for e in events if e.score + vlm.YES_LIFT >= VERDICT_THRESHOLD]
        if hopeful and budget.ok(3.0):
            vlm.verify(video_path, perc, fa, hopeful, budget)  # updates scores in place
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
    res, kept, debug_events = _build(job_id, video_path, perc, fa, events)
    T["evidence_s"] = round(time.time() - t0, 2)

    dbg = {"job_id": job_id, "video": video_path, "timings": T, "rule_timings": rule_t,
           "scene": {"desk": fa.scene.desk and fa.scene.desk.as_list(),
                     "laptop": fa.scene.laptop and fa.scene.laptop.as_list(),
                     "machine": fa.scene.machine and fa.scene.machine.as_list(),
                     "safe": fa.scene.safe and fa.scene.safe.as_list(),
                     "seat": fa.seat, "notes": fa.scene.notes, "provenance": fa.scene.provenance},
           "sessions": [vars(x) for x in (fa.sessions or [])],
           "checks": fa.checks,
           "events": debug_events,
           "dropped": [{"code": e.code, "start": e.start, "score": e.score, "info": e.info}
                       for e in events if e not in kept]}
    out_dir = Path(debug_dir or config.RUNS_DIR / "jobs" / job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "debug.json").write_text(json.dumps(dbg, ensure_ascii=False, indent=1, default=str))
    if on_analysis is not None:
        from engine.backend import build_analysis
        on_analysis(res, build_analysis(perc.info, kept, res), False)
    return res
