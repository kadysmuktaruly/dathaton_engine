"""Run the VLM desk scan on every clip and cache it. Usage: uv run python -m tools.scan_all"""
import time

from engine import facts, perception, vlm_scan
from tools.labels import load_clips

if __name__ == "__main__":
    for c in load_clips():
        p = perception.run(c.path)
        crop = p.info.crop
        f = facts.build(p.df, p.times, crop[2] - crop[0], crop[3] - crop[1])
        t0 = time.time()
        df = vlm_scan.scan(str(c.path), p, f.scene)
        on = {k: [round(t, 1) for t in df.t[df[k]]] for k in vlm_scan.FLAGS if df[k].any()}
        print(c.clip_id, "V" if c.violation else "-", c.timecodes, f"{time.time()-t0:.1f}s", on, flush=True)
