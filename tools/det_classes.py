"""Class list for the fine-tuned detector (shared by the labeling and training tools).

The order defines the label ids (0..K-1) of the fine-tuned model and the COCO category ids (1..K).
NEW classes are pre-labeled with OWLv2; COCO classes are pre-labeled with the stock RT-DETRv2 and
their classifier rows are initialised from the stock COCO head in train_detector.py.
"""
NEW_CLASSES = ["cash", "safe", "safe_open", "counting_machine", "collection_bag", "desk"]
COCO_CLASSES = ["person", "laptop", "handbag", "backpack", "cell phone", "cup"]
CLASSES = NEW_CLASSES + COCO_CLASSES

# Name of each kept COCO class in the stock PekingU/rtdetr_v2_r18vd id2label.
STOCK_NAME = {c: c for c in COCO_CLASSES}

# Label Studio colours (only cosmetic).
COLORS = {
    "cash": "#00c853", "safe": "#607d8b", "safe_open": "#d50000", "counting_machine": "#ff6d00",
    "collection_bag": "#2e7d32", "desk": "#8d6e63", "person": "#2962ff", "laptop": "#aa00ff",
    "handbag": "#c51162", "backpack": "#6200ea", "cell phone": "#00b8d4", "cup": "#ffd600",
}
