# Cash-desk discipline engine (ForteBank Datathon 26)

Local, offline video analysis: `engine.analyze(video_path) -> dict`.

## Setup
```bash
uv sync                                   # Python 3.11 env
uv run python -m tools.download_weights   # one-time, needs network
export HF_HUB_OFFLINE=1                   # everything after this is offline
```
Videos are read from `videos/` (override with `DATA_DIR`), and labels from `open_labels.xlsx` (override with `LABELS_XLSX`).

## Tools
- `uv run python -m tools.labels` prints the parsed labels (duplicate files are merged).
- `uv run pytest` runs the tests.
