"""Print the smoothed in-room head count per second for cached clips (diagnostic)."""
import sys

import numpy as np

from engine import facts, perception
from tools.labels import load_clips

if __name__ == "__main__":
    want = set(sys.argv[1:])
    for c in load_clips():
        if want and c.clip_id not in want:
            continue
        cp = perception._cache_path(c.path, 3.0)
        if not cp.exists():
            continue
        p = perception.run(c.path)
        f = facts.build(p.df, p.times, p.info.crop[2] - p.info.crop[0], p.info.crop[3] - p.info.crop[1])
        fr = f.frames
        secs = np.arange(0, int(p.info.duration) + 1)
        n = [int(fr[(fr.t >= s) & (fr.t < s + 1)].n_smooth.max()) if len(fr[(fr.t >= s) & (fr.t < s + 1)]) else 0 for s in secs]
        seat = [int(fr[(fr.t >= s) & (fr.t < s + 1)].seat_occupied.any()) for s in secs]
        tc = ",".join(str(int(x)) for x in c.timecodes) or "-"
        print(f"{c.clip_id:5s} L={tc:6s} n:", "".join(map(str, n)))
        print(f"{'':5s} {'':8s} s:", "".join(map(str, seat)))
