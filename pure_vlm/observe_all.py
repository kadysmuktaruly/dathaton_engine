"""Run the VLM observation pass on every labeled clip (fills the answer cache used by timeline.py).

  DATA_DIR=~/Downloads/open/videos HF_HUB_OFFLINE=1 python -m pure_vlm.observe_all [--clips X-1 X-2]
Writes cache/pure_vlm/obs_<clip>.json (git-ignored: it holds no frames, only answers and timings).
"""
from __future__ import annotations

import argparse
import json
import time

from pure_vlm import config, observe
from tools.labels import load_clips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*")
    args = ap.parse_args()
    clips = [c for c in load_clips() if not args.clips or c.clip_id in args.clips]
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for c in clips:
        t0 = time.time()
        obs = observe.observe(str(c.path))
        (config.CACHE_DIR / f"obs_{c.clip_id}.json").write_text(json.dumps(obs, ensure_ascii=False))
        tm = obs["timing"]
        print(c.clip_id, f"{obs['duration']:.0f}s video", f"{time.time() - t0:.1f}s wall", f"calls={tm['vlm_calls']}",
              f"budget_left={tm['budget_left_s']}", flush=True)


if __name__ == "__main__":
    main()
