"""Evaluate the engine on all labeled clips.

Usage:
  uv run python -m tools.eval [--no-vlm] [--clips X-1 X-2] [--augment] [--tag name]
Writes reports/eval_<tag>.md and runs/eval/<tag>.json.

Timecode error: labels are whole seconds ("22 секунда"), so we report both the raw error
|pred - label| and the window error (distance to [label-1, label+1]).
"""
from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from pathlib import Path

from engine import analyze, catalog, config
from tools.labels import load_clips

# Expected type per labeled timecode (docs/master-document.md, section 2a; X-28 = B9 by the user;
# X-47 is a copy of X-4, so X-4@68 is X-47's B1).
CONFIRMED = {
    ("X-1", 22): "A3", ("X-2", 10): "A6", ("X-4", 57): "B1", ("X-4", 68): "B1", ("X-11", 18): "B9",
    ("X-13", 47): "D6", ("X-15", 15): "B1", ("X-17", 7): "D7", ("X-18", 20): "A7", ("X-20", 5): "B5",
    ("X-22", 14): "B5", ("X-24", 8): "B8", ("X-26", 9): "B1", ("X-28", 7): "B9", ("X-30", 13): "C2",
    ("X-32", 5): "C2", ("X-33", 6): "C2", ("X-34", 13): "A1", ("X-38", 33): "B1", ("X-40", 17): "B3",
    ("X-44", 16): "A2", ("X-46", 11): "B3",
}
TYPE2CODE = {v["type"]: v["code"] for v in catalog.load()["violations"]}


def window_err(pred: float, label: float) -> float:
    return max(0.0, abs(pred - label) - 1.0)


def run(clips, use_vlm: bool, video_override=None):
    rows = []
    for c in clips:
        path = video_override(c) if video_override else c.path
        t0 = time.time()
        res = analyze(path, job_id=f"eval_{c.clip_id}", use_vlm=use_vlm)
        wall = time.time() - t0
        dbg = json.loads((config.RUNS_DIR / "jobs" / f"eval_{c.clip_id}" / "debug.json").read_text())
        preds = [(v["timecode_sec"], TYPE2CODE[v["type"]]) for v in res["violations"]]
        matches = []
        for tc in c.timecodes:
            if preds:
                p = min(preds, key=lambda p: abs(p[0] - tc))
                matches.append({"label_t": tc, "pred_t": p[0], "pred_code": p[1],
                                "want_code": CONFIRMED.get((c.clip_id, int(tc))),
                                "err": abs(p[0] - tc), "win_err": window_err(p[0], tc)})
            else:
                matches.append({"label_t": tc, "pred_t": None, "pred_code": None,
                                "want_code": CONFIRMED.get((c.clip_id, int(tc))), "err": None, "win_err": None})
        rows.append({"clip": c.clip_id, "label": c.violation, "pred": res["verdict"] == "violation",
                     "preds": preds, "matches": matches, "wall_s": round(wall, 2),
                     "timings": dbg["timings"], "events": dbg["events"]})
        print(c.clip_id, "L" if c.violation else "-", "P" if rows[-1]["pred"] else "-", preds,
              [m["label_t"] for m in matches], f"{wall:.1f}s", flush=True)
    return rows


def summarize(rows) -> dict:
    tp = sum(r["label"] and r["pred"] for r in rows)
    fp = sum((not r["label"]) and r["pred"] for r in rows)
    fn = sum(r["label"] and not r["pred"] for r in rows)
    tn = sum((not r["label"]) and not r["pred"] for r in rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    ms = [m for r in rows for m in r["matches"] if m["err"] is not None]
    errs = [m["err"] for m in ms]
    werrs = [m["win_err"] for m in ms]
    typed = [m for m in ms if m["want_code"]]
    return {
        "videos": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
        "timecodes_total": sum(len(r["matches"]) for r in rows), "timecodes_matched": len(ms),
        "err_median": round(statistics.median(errs), 2) if errs else None,
        "err_max": round(max(errs), 2) if errs else None,
        "within_1s": round(sum(e <= 1 for e in errs) / len(errs), 3) if errs else None,
        "within_2s": round(sum(e <= 2 for e in errs) / len(errs), 3) if errs else None,
        "win_err_median": round(statistics.median(werrs), 2) if werrs else None,
        "type_acc": round(sum(m["pred_code"] == m["want_code"] for m in typed) / len(typed), 3) if typed else None,
        "time_mean_s": round(statistics.mean(r["wall_s"] for r in rows), 2),
        "time_max_s": round(max(r["wall_s"] for r in rows), 2),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
    }


def per_code(rows) -> dict:
    out = {}
    for r in rows:
        for m in r["matches"]:
            w = m["want_code"] or "?"
            d = out.setdefault(w, {"n": 0, "hit_any": 0, "hit_type": 0, "errs": []})
            d["n"] += 1
            if m["pred_t"] is not None:
                d["hit_any"] += 1
                d["errs"].append(round(m["err"], 1))
                d["hit_type"] += m["pred_code"] == w
    return out


def write_report(tag: str, rows, summ, extra: dict | None = None) -> Path:
    lines = [f"# Eval `{tag}`", "", "## Summary", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summ.items()]
    lines += ["", "## Per confirmed code", "", "| code | labels | video hit | right type | errors (s) |", "|---|---|---|---|---|"]
    for k, d in sorted(per_code(rows).items()):
        lines.append(f"| {k} | {d['n']} | {d['hit_any']} | {d['hit_type']} | {d['errs']} |")
    lines += ["", "## Misses (violation clips predicted clean)", ""]
    for r in rows:
        if r["label"] and not r["pred"]:
            lines.append(f"- {r['clip']} labels {[m['label_t'] for m in r['matches']]} "
                         f"want {[m['want_code'] for m in r['matches']]}; overlay: runs/overlays/{r['clip']}.mp4")
    lines += ["", "## False positives (clean clips predicted violation)", ""]
    for r in rows:
        if r["pred"] and not r["label"]:
            lines.append(f"- {r['clip']} preds {r['preds']} events {[(e['code'], e['info'].get('reason')) for e in r['events']]}; "
                         f"overlay: runs/overlays/{r['clip']}.mp4")
    lines += ["", "## Per video", "", "| clip | label | pred | preds (t, code) | label t | want | time s | perception | rules | vlm |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tm = r["timings"]
        lines.append(f"| {r['clip']} | {'V' if r['label'] else '-'} | {'V' if r['pred'] else '-'} | {r['preds']} | "
                     f"{[m['label_t'] for m in r['matches']]} | {[m['want_code'] for m in r['matches']]} | {r['wall_s']} | "
                     f"{tm.get('perception_s')} | {tm.get('rules_s')} | {tm.get('vlm_s')} |")
    if extra:
        lines += ["", "## Angle robustness", "", "| variant | recall | precision | f1 | err median |", "|---|---|---|---|---|"]
        for k, s in extra.items():
            lines.append(f"| {k} | {s['recall']} | {s['precision']} | {s['f1']} | {s['err_median']} |")
    p = Path("reports") / f"eval_{tag}.md"
    p.write_text("\n".join(lines) + "\n")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--augment", action="store_true", help="also run on warped/rotated/cropped copies")
    ap.add_argument("--tag", default="latest")
    args = ap.parse_args()
    clips = [c for c in load_clips() if not args.clips or c.clip_id in args.clips]
    rows = run(clips, use_vlm=not args.no_vlm)
    summ = summarize(rows)
    extra = None
    if args.augment:
        from tools.augment_angle import VARIANTS, make_variant
        extra = {}
        for v in VARIANTS:
            vr = run(clips, use_vlm=not args.no_vlm, video_override=lambda c, v=v: make_variant(c.path, v))
            extra[v] = summarize(vr)
    out = Path(config.RUNS_DIR / "eval")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps({"summary": summ, "rows": rows, "augment": extra},
                                                     ensure_ascii=False, indent=1, default=str))
    print(json.dumps(summ, indent=1))
    print("report:", write_report(args.tag, rows, summ, extra))


if __name__ == "__main__":
    main()
