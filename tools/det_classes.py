"""Class list for the fine-tuned detector (shared by the labeling and training tools).

The definitions live in engine/classes.py so training and runtime cannot drift apart.
On Colab, copy engine/classes.py next to train_detector.py as det_classes.py.
"""
from engine.classes import CLASSES, COCO_CLASSES, COLORS, NEW_CLASSES, STOCK_NAME  # noqa: F401
