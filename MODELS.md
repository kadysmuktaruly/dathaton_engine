# Models and licenses

All models run locally. Weights are downloaded once (`uv run python -m tools.download_weights`), then the engine runs with `HF_HUB_OFFLINE=1`.
Licenses were checked on 2026-10-17 from the Hugging Face model card metadata (`card_data.license`) and the PyPI package metadata.

| Role | Model / package | Source | Revision | License | Verified from |
|---|---|---|---|---|---|
| Object detector | RT-DETRv2 R18vd (COCO) | `PekingU/rtdetr_v2_r18vd` | `56509617` | Apache-2.0 | HF card metadata |
| Pose (wrists) | RTMPose-m body7 256×192 ONNX | download.openmmlab.com (mmpose `rtmposev1/onnx_sdk`) | `e48f03d0_20230504` | Apache-2.0 (mmpose repo) | mmpose repo license; **the licenses of the training datasets were not checked** |
| VLM | Qwen3-VL-4B-Instruct, MLX 4-bit | `mlx-community/Qwen3-VL-4B-Instruct-4bit` (base `Qwen/Qwen3-VL-4B-Instruct`) | `2fd8dacb` (base `ebb281ec`) | Apache-2.0 (both) | HF card metadata |
| Pre-labeling only (not used at inference) | OWLv2 base patch16 ensemble | `google/owlv2-base-patch16-ensemble` | `cfd3195b` | Apache-2.0 | HF card metadata |

| Package | Version | License |
|---|---|---|
| torch | 2.14.1 | BSD-3-Clause family (see metadata) |
| transformers | 5.20.1 | Apache-2.0 |
| supervision (ByteTrack) | 0.31.0 | MIT |
| rtmlib | 0.0.16 | Apache-2.0 |
| onnxruntime | 1.31.0 | MIT |
| mlx / mlx-vlm | 0.32.4 / 0.7.8 | MIT |
| opencv-python-headless | 5.0.0 | Apache-2.0 |
| fastapi | 0.143.3 | MIT |

Not used, because of their licenses: Ultralytics YOLO (AGPL-3.0), Qwen2.5-VL-3B (non-commercial), RF-DETR A/F/P/XL/2XL (PML).
