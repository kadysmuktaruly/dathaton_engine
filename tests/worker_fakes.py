"""Lightweight spawn targets for testing actual process termination and recovery."""

import os
import time
from pathlib import Path


def worker(connection, settings):
    connection.send(("ready", None))
    while True:
        request = connection.recv()
        mode = Path(request["path"]).read_bytes()
        result = {
            "job_id": request["job_id"],
            "status": "done",
            "verdict": "no_violation",
            "violations": [],
        }
        bundle = {
            "result": result,
            "analysis": {"duration_sec": 12.5, "violations": [], "events": []},
        }
        if mode == b"crash":
            os._exit(3)
        if mode == b"partial":
            connection.send(("partial", bundle))
            time.sleep(60)
        if mode == b"hang":
            time.sleep(60)
        if mode == b"slow":
            time.sleep(0.5)
        if mode == b"fail":
            connection.send(
                ("failed", {"code": "ANALYSIS_FAILED", "message": "Bad video"})
            )
        else:
            connection.send(("done", bundle))


def startup_failure(connection, settings):
    connection.send(("startup_failed", None))


def startup_hang(connection, settings):
    time.sleep(60)
