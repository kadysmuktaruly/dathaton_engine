"""pure_vlm: a VLM (Qwen3.5-9B) is the only perception model; deterministic rules from the master document
turn its timestamped answers into violations.

  video -> observe.observe (layout, state questions, action windows; cached per clip)
        -> timeline.detect (departures, door, safe, desk items, visitor sessions, collection)
        -> graded JSON (same schema and wording rules as engine.analyze; legal refs only from the catalog)
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

# 'что видно' per (code, reason): facts only, roles only, no intent.
SEEN = {
    ("B1", "cash_left"): "кассир покинул помещение кассы, наличные остались лежать на столе",
    ("B1", "bag_left"): "кассир покинул помещение кассы, инкассаторская сумка осталась на столе",
    ("B1", "cash_on_arrival"): "в кассу вошёл следующий клиент, а на столе лежат наличные, оставшиеся от предыдущей операции",
    ("B8", "stamps_left"): "кассир покинул помещение кассы, печати или штампы остались на столе",
    ("B8", "items_left"): "кассир покинул помещение кассы, на столе остались предметы (похожие на печати или штампы)",
    ("B5", "safe_open_left"): "кассир покинул помещение кассы, сейф остался открытым",
    ("B5", "key_left"): "кассир покинул помещение кассы, ключ остался в замке сейфа",
    ("B5", "safe_open_at_work"): "кассир вернулся к столу и работает, а сейф остаётся открытым",
    ("B9", "laptop_active_left"): "кассир покинул помещение кассы, ноутбук остался включённым с открытой рабочей программой",
    ("A4", "left_with_visitor"): "кассир покинул помещение кассы, пока в нём оставался посетитель",
    ("C2", "door_left_open"): "дверь кассы открыта и остаётся открытой несколько секунд",
    ("B3", "personal_handbag"): "на рабочем столе кассира лежит сумка",
    ("B3", "personal_clothing"): "на рабочем столе кассира лежит одежда",
    ("B3", "personal_drink"): "на рабочем столе кассира стоит напиток",
    ("B3", "personal_phone"): "на рабочем столе кассира лежит телефон",
    ("A7", "two_visitors"): "в кассе находятся два посетителя, и кассир принимает деньги или документ второго до "
                            "завершения операции первого",
    ("A1", "left_during_count"): "клиент уходит, пока его деньги ещё пересчитываются на счётной машине",
    ("D7", "bag_without_papers"): "кассир принимает инкассаторскую сумку, служебное удостоверение или доверенность "
                                  "передающего в кадре не предъявлены",
    ("D6", "bag_to_safe_unregistered"): "содержимое инкассаторской сумки убирают в сейф без пересчёта на машине и "
                                        "без записи в ноутбуке",
}
DEFAULT_SEEN = "в кадре видны действия, похожие на отступление от кассовых процедур"
# the zone drawn on the evidence frames
EVIDENCE_ZONE = {"B1": "desk", "B8": "desk", "B3": "desk", "B5": "safe", "B9": "laptop", "C2": "door",
                 "D6": "safe", "D7": "desk", "A1": "desk", "A7": "desk", "A4": "desk"}
# Switched off after the training-clip eval: A4 is optional in the master document and fired only on a
# miscounted visitor (X-12).
DISABLED: set[str] = {"A4"}


def build(job_id: str, video_path: str, obs: dict, events) -> tuple[dict, list[dict]]:
    from engine import catalog, evidence, output
    violations, dbg = [], []
    dur = obs["duration"]
    for e in events:
        tc = round(e.t, 1)
        rest = sorted({round(min(max(t, 0.0), dur - 0.05), 1) for t in e.evidence_t} - {tc})
        times = [tc] + rest[:2]
        zone = obs["boxes"].get(EVIDENCE_ZONE.get(e.code, "desk"))
        boxes = {t: [("зона", zone)] for t in times} if zone else {}
        violations.append({
            "type": catalog.violation(e.code)["type"],
            "timecode_sec": tc,
            "description": catalog.description(e.code, SEEN.get((e.code, e.reason), DEFAULT_SEEN)),
            "evidence_frames": evidence.render(video_path, times, boxes),
        })
        dbg.append({"code": e.code, "t": e.t, "reason": e.reason, "evidence_t": times, "info": e.info,
                    "refs": [f"{r['doc']} {r['point']}" for r in catalog.safe_refs(e.code)]})
    return output.build(job_id, violations), dbg


def analyze(video_path: str | Path, job_id: str | None = None, use_vlm: bool = True,
            debug_dir: str | Path | None = None, models=None, on_partial=None,
            deadline: float | None = None) -> dict:
    """Same signature and graded output as engine.analyze. `deadline` (epoch seconds) caps the VLM budget;
    the budget ends the observation early, so the result is always complete (no partial result needed)."""
    from engine import config as engine_config
    from pure_vlm import config, observe, timeline, verify

    job_id = job_id or uuid.uuid4().hex[:12]
    video_path = str(video_path)
    t0 = time.time()
    budget = config.BUDGET_S
    if deadline is not None:
        budget = min(budget, deadline - time.time() - 20.0)
    obs = observe.observe(video_path, budget_s=max(budget - config.VERIFY_RESERVE_S, 5.0))
    events = [e for e in timeline.detect(obs) if e.code not in DISABLED]
    t1 = time.time()
    events, checks = verify.filter_events(obs, events, ask=True)
    res, dbg_events = build(job_id, video_path, obs, events)
    timings = {**obs["timing"], "verify_evidence_s": round(time.time() - t1, 2), "wall_s": round(time.time() - t0, 2)}
    try:
        import mlx.core as mx
        timings["mlx_peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    except Exception:
        pass
    dbg = {"job_id": job_id, "video": video_path, "pipeline": "pure_vlm", "vlm": config.VLM_REPO,
           "timings": timings, "zones": obs["zones"], "events": dbg_events, "checks": checks,
           "observations": {"states": obs["states"], "actions": obs["actions"], "moments": obs["moments"]}}
    out_dir = Path(debug_dir or engine_config.RUNS_DIR / "jobs" / job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "debug.json").write_text(json.dumps(dbg, ensure_ascii=False, indent=1, default=str))
    return res
