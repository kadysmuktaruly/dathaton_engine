"""Registry of rule functions: name -> fn(Facts) -> list[Event].

Only codes marked active in rules/rules_catalog.json (docs/master-document.md) are produced.
"""
from engine.rules import b5, c2, client, collection, depart, scan_rules

RULES = {
    "C2": c2.detect,               # door left open after a pass
    "B5": b5.detect,               # safe left open while working
    "depart": depart.detect,       # cashier leaves: B1 cash/bag, B5 safe, B8 stamps, B9 laptop
    "client": client.detect,       # A7, B1 (new client), A1, A6, A2, A4
    "collection": collection.detect,  # D7, D6
    "objects": scan_rules.detect,  # B3 (VLM desk scan; drinks also from the detector)
}
