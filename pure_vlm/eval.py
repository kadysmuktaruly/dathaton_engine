"""Metrics of the timeline rules on the cached VLM answers (cache/pure_vlm/obs_<clip>.json from observe_all).

  DATA_DIR=~/Downloads/open/videos python -m pure_vlm.eval [--tag NAME] [--codes B1 C2 ...]
Verdict precision / recall / F1 per video, type accuracy against the master document, timecode error.
Writes pure_vlm/reports/eval_<tag>.md. Runs in seconds: no model is loaded.
"""
from __future__ import annotations

import argparse
import json
import statistics

from pure_vlm import DISABLED, config, timeline, verify
from pure_vlm.labels import TYPES
from tools.labels import load_clips


def evaluate(clips, codes=None, verify_mode: str = "cache") -> tuple[list[dict], dict]:
    """verify_mode: 'off' (rules only), 'cache' (replay cached checks), 'ask' (load the VLM for missing checks)."""
    rows = []
    for c in clips:
        p = config.CACHE_DIR / f"obs_{c.clip_id}.json"
        if not p.exists():
            continue
        obs = json.loads(p.read_text())
        ev = [e for e in timeline.detect(obs, set(codes) if codes else None) if e.code not in DISABLED]
        dropped = []
        if verify_mode != "off":
            kept, log = verify.filter_events(obs, ev, ask=verify_mode == "ask")
            dropped = [(x["code"], x["t"], x["reason"]) for x in log if not x["keep"]]
            ev = kept
        matches = []
        for tc in c.timecodes:
            want = TYPES.get((c.clip_id, int(tc)))
            near = min(ev, key=lambda e: abs(e.t - tc)) if ev else None
            typed = [e for e in ev if e.code == want]
            near_typed = min(typed, key=lambda e: abs(e.t - tc)) if typed else None
            matches.append({"label_t": tc, "want": want, "pred_t": near and near.t, "pred": near and near.code,
                            "err": near and round(abs(near.t - tc), 2), "type_hit": near_typed is not None,
                            "type_err": near_typed and round(abs(near_typed.t - tc), 2)})
        rows.append({"clip": c.clip_id, "label": c.violation, "pred": bool(ev), "matches": matches,
                     "events": [(e.code, e.t, e.reason) for e in ev], "dropped": dropped,
                     "vlm_s": obs["timing"]["total_s"],
                     "duration": round(obs["duration"], 1)})
    tp = sum(r["label"] and r["pred"] for r in rows)
    fp = sum(not r["label"] and r["pred"] for r in rows)
    fn = sum(r["label"] and not r["pred"] for r in rows)
    tn = sum(not r["label"] and not r["pred"] for r in rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    ms = [m for r in rows for m in r["matches"]]
    errs = [m["err"] for m in ms if m["err"] is not None]
    terrs = [m["type_err"] for m in ms if m["type_err"] is not None]
    summ = {"videos": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(prec, 3),
            "recall": round(rec, 3), "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
            "labels": len(ms), "type_hit": sum(m["type_hit"] for m in ms),
            "type_hit_within_2s": sum(m["type_hit"] and m["type_err"] <= 2 for m in ms),
            "err_median": statistics.median(errs) if errs else None,
            "type_err_median": statistics.median(terrs) if terrs else None,
            "vlm_s_mean": round(statistics.mean(r["vlm_s"] for r in rows), 1) if rows else None,
            "vlm_s_max": max((r["vlm_s"] for r in rows), default=None)}
    return rows, summ


def report(tag: str, rows, summ) -> str:
    L = [f"# pure_vlm eval `{tag}`", "", "| metric | value |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in summ.items()]
    L += ["", "| clip | label | pred | want (t) | events (code, t, reason) | dropped by the check | type hit (err s) | "
          "VLM s / video s |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        want = ", ".join(f"{m['want']}@{m['label_t']:g}" for m in r["matches"])
        hit = ", ".join(("yes" if m["type_hit"] else "no") + (f" ({m['type_err']})" if m["type_hit"] else "")
                        for m in r["matches"])
        ev = "; ".join(f"{c}@{t:g} {why}" for c, t, why in r["events"])
        dr = "; ".join(f"{c}@{t:g} {why}" for c, t, why in r["dropped"])
        L.append(f"| {r['clip']} | {'V' if r['label'] else '-'} | {'V' if r['pred'] else '-'} | {want} | {ev} | {dr} | "
                 f"{hit} | {r['vlm_s']:.0f} / {r['duration']:.0f} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--codes", nargs="*")
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--verify", choices=["off", "cache", "ask"], default="cache")
    args = ap.parse_args()
    clips = [c for c in load_clips() if not args.clips or c.clip_id in args.clips]
    rows, summ = evaluate(clips, args.codes, args.verify)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    out = config.REPORTS_DIR / f"eval_{args.tag}.md"
    out.write_text(report(args.tag, rows, summ))
    for r in rows:
        flag = {(True, True): "TP", (False, True): "FP", (True, False): "FN", (False, False): "TN"}[(r["label"], r["pred"])]
        print(f"{r['clip']:5} {flag} want={[(m['want'], m['label_t']) for m in r['matches']]} ev={r['events']}"
              + (f" dropped={r['dropped']}" if r["dropped"] else ""))
    print(json.dumps(summ))
    print("report:", out)


if __name__ == "__main__":
    main()
