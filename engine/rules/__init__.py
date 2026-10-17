"""Registry of rule functions: name -> fn(Facts) -> list[Event]. Only user-confirmed types."""
from engine.rules import b5, c1, c2

RULES = {
    "B5": b5.detect,
    "C1": c1.detect,
    "C2": c2.detect,
}
