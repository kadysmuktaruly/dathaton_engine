"""Print a clip's cached VLM answers as a compact timeline (for tuning the rules).

  python -m pure_vlm.show X-1 [X-2 ...]
"""
from __future__ import annotations

import json
import sys

from pure_vlm import config, timeline


def show(cid: str) -> None:
    obs = json.loads((config.CACHE_DIR / f"obs_{cid}.json").read_text())
    print(f"== {cid} dur={obs['duration']:.1f} timing={obs['timing']}")
    print("   zones:", {k: [round(v) for v in b] for k, b in obs["zones"].items()})
    names = ["people", "door", "safe", "laptop", "desk"]
    ser = {n: timeline.series(obs, n) for n in names}
    times = sorted({t for s in ser.values() for t, _ in s})
    for t in times:
        cells = []
        for n in names:
            hit = [a for x, a in ser[n] if abs(x - t) < 1e-6]
            if not hit:
                cells.append("")
                continue
            a = hit[0]
            if n == "people":
                a = f"{timeline.cashier(a)[:4]}+{timeline.visitors(a)}"
            elif n == "safe":
                a = ("OPEN" if timeline.safe_open(a) else "shut") + ("+key" if timeline.safe_key(a) else "")
            elif n == "desk":
                a = ",".join(a) if isinstance(a, list) else str(a)
            cells.append(str(a))
        print(f"  {t:6.2f} | " + " | ".join(f"{c:10}" for c in cells))
    for w in obs["actions"]:
        on = [k for k, v in w.items() if v is True]
        print(f"  act {w['t0']:5.1f}-{w['t1']:5.1f}: {', '.join(on)}")
    print("   moments:", obs.get("moments"))
    print("   events:", [(e.code, e.t, e.reason) for e in timeline.detect(obs)])


if __name__ == "__main__":
    for c in sys.argv[1:]:
        show(c)
