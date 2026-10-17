"""Run perception on every unique clip and cache it. Usage: uv run python -m tools.cache_all"""
import time

from engine import perception
from tools.labels import load_clips

if __name__ == "__main__":
    models = perception.Models()
    for c in load_clips():
        t0 = time.time()
        p = perception.run(c.path, models=models)
        print(c.clip_id, f"{p.info.duration:.0f}s video", f"{time.time()-t0:.1f}s", p.timings, flush=True)
