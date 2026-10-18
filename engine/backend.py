"""Map engine events into datathon's Analysis schema without changing graded JSON."""

from __future__ import annotations

from engine import catalog


def build_analysis(info, kept, result: dict) -> dict:
    violations, annotations = [], []
    ox, oy = info.crop[:2]
    for index, (event, violation) in enumerate(
        zip(kept, result["violations"], strict=True)
    ):
        violations.append({**violation, "code": event.code})
        refs = catalog.safe_refs(event.code)
        docs = catalog.load()["documents"]
        external = next((r for r in refs if docs[r["doc"]].get("url")), None)
        rule = (
            None
            if external is None
            else {
                "title": docs[external["doc"]]["short"],
                "section": external.get("point", ""),
                "url": docs[external["doc"]]["url"],
            }
        )
        for timestamp, boxes in event.boxes.items():
            # Boxes are sparse observations, not continuous tracks. Show each only near
            # its observed timestamp; never stretch one box over the entire event.
            start = max(0.0, float(timestamp) - 0.2)
            end = min(info.duration, float(timestamp) + 0.2)
            if end <= start:
                continue
            for _label, (x0, y0, x1, y1) in boxes:
                left = min(1.0, max(0.0, (x0 + ox) / info.width))
                top = min(1.0, max(0.0, (y0 + oy) / info.height))
                right = min(1.0, max(0.0, (x1 + ox) / info.width))
                bottom = min(1.0, max(0.0, (y1 + oy) / info.height))
                if right <= left or bottom <= top:
                    continue
                annotations.append(
                    {
                        "violation_index": index,
                        "start_sec": start,
                        "end_sec": end,
                        "box": {
                            "x": left,
                            "y": top,
                            "width": right - left,
                            "height": bottom - top,
                        },
                        "rule": rule,
                        "rule_note": ""
                        if rule
                        else "Текст внутренних документов банка не предоставлен.",
                    }
                )
    return {
        "duration_sec": info.duration,
        "violations": violations,
        "events": annotations,
    }
