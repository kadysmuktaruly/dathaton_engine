"""Detector class names, shared by training (tools/det_classes.py re-exports this) and runtime.

The order of CLASSES defines the label ids (0..K-1) of the fine-tuned model and the COCO
category ids (1..K). NEW classes are pre-labeled with OWLv2; COCO classes are pre-labeled with the
stock RT-DETRv2 and their classifier rows are initialised from the stock COCO head.

At runtime the class ids always come from the loaded checkpoint's id2label; this module only
says which names the engine uses (ROLES) and which it knowingly ignores (IGNORED).
Pure constants: no imports, so it can be copied to Colab as det_classes.py.
"""
NEW_CLASSES = ["cash", "safe", "safe_open", "counting_machine", "collection_bag", "desk"]
COCO_CLASSES = ["person", "laptop", "handbag", "backpack", "id_card", "cup"]
CLASSES = NEW_CLASSES + COCO_CLASSES

# Name of each kept COCO class in the stock PekingU/rtdetr_v2_r18vd id2label.
# id_card (a client's ID document) starts from the stock "cell phone" row: both are small
# rectangles held in a hand. Phones are rare here and were labeled id_card too, so an id_card box
# may be a phone; docs/master-document.md counts only the ID card as a client ID (known limit).
STOCK_NAME = {c: c for c in COCO_CLASSES}
STOCK_NAME["id_card"] = "cell phone"

# Engine roles and the detector class names each one comes from. Explicit (fine-tuned) classes
# and stock look-alikes are separate roles, so scene anchors can prefer the explicit ones.
ROLES = {
    "person": {"person"},
    "laptop": {"laptop"},
    "desk": {"desk"},                                  # fine-tuned
    "table": {"diningtable", "dining table"},          # stock look-alike for the desk
    "counting_machine": {"counting_machine"},          # fine-tuned
    "machine_like": {"tvmonitor", "tv", "suitcase"},   # stock look-alikes for the machine
    "safe": {"safe", "safe_open"},                     # cls keeps open vs closed
    "cash": {"cash"},
    "collection_bag": {"collection_bag"},
    # Observation-only context: kept for inspection and downstream use, never scene anchors and
    # never a violation on their own.
    "bag": {"handbag", "backpack"},
    "id_card": {"id_card", "cell phone"},     # stock detector: its "cell phone" boxes
    "cup": {"cup"},
}
CONTEXT_ROLES = {"bag", "id_card", "cup"}
# Classes the detector may output that the engine knowingly drops after detection.
IGNORED: set[str] = set()

# Label Studio colours (only cosmetic).
COLORS = {
    "cash": "#00c853", "safe": "#607d8b", "safe_open": "#d50000", "counting_machine": "#ff6d00",
    "collection_bag": "#2e7d32", "desk": "#8d6e63", "person": "#2962ff", "laptop": "#aa00ff",
    "handbag": "#c51162", "backpack": "#6200ea", "id_card": "#00b8d4", "cup": "#ffd600",
}


def role_map() -> dict[str, str]:
    """Detector class name -> engine role (built on call, so a changed ROLES is respected)."""
    return {c: r for r, names in ROLES.items() for c in names}


def role_of(name: str) -> str | None:
    return role_map().get(name)


def runtime_keep(id2label: dict) -> set[str]:
    """Names from a checkpoint's id2label that the engine keeps after detection."""
    used = set().union(*ROLES.values())
    return {str(n) for n in id2label.values() if str(n) in used}
