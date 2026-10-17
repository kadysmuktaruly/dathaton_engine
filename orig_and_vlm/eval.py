"""40-clip eval of orig_and_vlm with the engine's metrics (tools/eval.py); reports go to orig_and_vlm/reports/.

  DATA_DIR=~/Downloads/open/videos HF_HUB_OFFLINE=1 .venv/bin/python -m orig_and_vlm.eval --tag full
  OV_SCAN=0 ... --tag closeout_only        # idea 2 alone
  OV_CLOSEOUT=0 ... --tag scan_only        # idea 1 alone

Baseline (main, engine only): reports/eval_B0_uncached.md (13 TP / 6 FP / 8 FN, F1 0.65).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import orig_and_vlm
import tools.eval as E
from engine import config as engine_config
from orig_and_vlm import config
from tools.labels import load_clips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--augment", action="store_true", help="also run on warped/rotated/cropped copies")
    ap.add_argument("--tag", default="latest")
    args = ap.parse_args()
    E.analyze = orig_and_vlm.analyze  # tools.eval.run() calls this module-level name
    clips = [c for c in load_clips() if not args.clips or c.clip_id in args.clips]
    rows = E.run(clips, use_vlm=True)
    summ = E.summarize(rows)
    extra = None
    if args.augment:
        from tools.augment_angle import VARIANTS, make_variant
        extra = {v: E.summarize(E.run(clips, use_vlm=True, video_override=lambda c, v=v: make_variant(c.path, v)))
                 for v in VARIANTS}
    tag = f"ov_{args.tag}"
    out = engine_config.RUNS_DIR / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps({"summary": summ, "rows": rows, "augment": extra},
                                                ensure_ascii=False, indent=1, default=str))
    report = E.write_report(tag, rows, summ, extra)  # writes reports/eval_<tag>.md
    config.REPORTS_DIR.mkdir(exist_ok=True)
    dest = config.REPORTS_DIR / Path(report).name
    shutil.move(str(report), dest)
    print(json.dumps(summ, indent=1))
    print("report:", dest)


if __name__ == "__main__":
    main()
