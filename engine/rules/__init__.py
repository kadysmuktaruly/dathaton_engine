"""Registry of rule functions: name -> fn(Facts) -> list[Event]. Only user-confirmed types."""
from engine.rules import b5, c1, c2, scan_rules

RULES = {
    "B5": b5.detect,
    "C1": c1.detect,
    "C2": c2.detect,
    "scan": scan_rules.detect,   # B1, B3, D1, B7, X0, B5-open (needs the VLM desk scan)
}
