"""Violation types per labeled timecode, from the master document (18.10.2026, clip notes X-1 ... X-48) and the
extended summary table (X-28 = B9). Verdicts and timecodes come from open_labels.xlsx (tools.labels)."""

TYPES = {
    ("X-1", 22): "A3", ("X-2", 10): "A6", ("X-4", 57): "B1", ("X-4", 68): "B1", ("X-11", 18): "B9",
    ("X-13", 47): "D6", ("X-15", 15): "B1", ("X-17", 7): "D7", ("X-18", 20): "A7", ("X-20", 5): "B5",
    ("X-22", 14): "B5", ("X-24", 8): "B8", ("X-26", 9): "B1", ("X-28", 7): "B9", ("X-30", 13): "C2",
    ("X-32", 5): "C2", ("X-33", 6): "C2", ("X-34", 13): "A1", ("X-38", 33): "B1", ("X-40", 17): "B3",
    ("X-44", 16): "A2", ("X-46", 11): "B3",
}
