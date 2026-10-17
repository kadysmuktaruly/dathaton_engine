"""Submit several clips to a running API at once, like the jury might, and check each job.

  ENGINE_API_KEY=k .venv/bin/uvicorn orig_and_vlm.api:app --port 8000      # in another terminal
  .venv/bin/python -m orig_and_vlm.loadtest --key k X-26 X-20 X-34 X-12 X-30

Passes when every job is 'done' within --limit seconds of its upload and its JSON passes
engine.output.validate.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
import uuid

from engine import output
from tools.labels import video_index


def _post(url: str, key: str, path) -> dict:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: video/quicktime\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "X-API-Key": key, "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _get(url: str, key: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"X-API-Key": key}), timeout=30) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--key", required=True)
    ap.add_argument("--limit", type=float, default=300.0)
    args = ap.parse_args()
    idx = video_index()
    jobs = {}
    for c in args.clips:
        jobs[c] = {"id": _post(f"{args.base}/api/v1/analyze", args.key, idx[c])["job_id"], "t0": time.time()}
    pending = set(jobs)
    while pending:
        time.sleep(3)
        for c in list(pending):
            res = _get(f"{args.base}/api/v1/jobs/{jobs[c]['id']}", args.key)
            waited = time.time() - jobs[c]["t0"]
            if res["status"] in ("done", "failed") or waited > args.limit:
                jobs[c].update(res=res, waited=round(waited, 1))
                pending.discard(c)
    ok = True
    for c, j in jobs.items():
        res = j["res"]
        try:
            output.validate(res)
            valid = True
        except AssertionError as e:
            valid = f"invalid: {e}"
        good = res["status"] == "done" and valid is True and j["waited"] <= args.limit
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {c} {res['status']} {j['waited']}s verdict={res.get('verdict')} "
              f"types={[(v['type'][:30], v['timecode_sec']) for v in res.get('violations', [])]} schema={valid}")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
