from types import SimpleNamespace

import engine
from engine import config, evidence, perception, vlm, vlm_scan
from engine.events import Event


def test_analyze_reports_partial_and_final_backend_data_without_changing_result(
    monkeypatch, tmp_path
):
    info = SimpleNamespace(width=1000, height=500, duration=10, crop=(0, 0, 1000, 500))
    perc = SimpleNamespace(info=info)
    scene = SimpleNamespace(
        desk=None, laptop=None, machine=None, safe=None, notes=[], provenance={}
    )
    facts = SimpleNamespace(seat=None, scene=scene, sessions=[], checks={})
    events = [Event("B5", 2, 8, 0.9, boxes={2: [("safe", [100, 100, 200, 200])]})]
    monkeypatch.setattr(perception, "run", lambda *a, **kw: perc)
    monkeypatch.setattr(engine, "detect_events", lambda *a, **kw: (facts, events, {}))
    monkeypatch.setattr(engine, "_get_models", lambda: object())
    import base64

    import cv2
    import numpy as np

    _, jpeg = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    image = base64.b64encode(jpeg).decode()
    monkeypatch.setattr(
        evidence,
        "render",
        lambda _video, times, *a, **kw: [
            {"timestamp_sec": times[0], "image_base64": image},
        ],
    )
    reports, partials = [], []
    result = engine.analyze(
        "clip.mp4",
        use_vlm=False,
        debug_dir=tmp_path,
        on_partial=partials.append,
        on_analysis=lambda *args: reports.append(args),
    )
    assert [r[2] for r in reports] == [True, False]
    assert set(result) == {"job_id", "status", "verdict", "violations"}
    assert "code" not in result["violations"][0]
    assert reports[1][1]["violations"][0]["code"] == "B5"
    assert reports[1][1]["events"][0]["violation_index"] == 0
    assert partials[0] == reports[0][0]
    assert (tmp_path / "debug.json").is_file()


def test_transformers_generation_uses_images_and_decodes_only_new_tokens(monkeypatch):
    import torch

    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"][0]["image"] == "image-placeholder"
            assert messages[0]["content"][1]["text"] == "question"
            assert kwargs["tokenize"] and kwargs["return_dict"]
            return Inputs(
                input_ids=torch.tensor([[1, 2]]), token_type_ids=torch.tensor([[0, 0]])
            )

        def batch_decode(self, tokens, **kwargs):
            assert tokens.tolist() == [[3, 4]]
            return ['{"visible":true}']

    class Model:
        device = "cpu"

        def generate(self, **kwargs):
            assert not torch.is_grad_enabled()
            assert "token_type_ids" not in kwargs
            assert kwargs["max_new_tokens"] == 90 and kwargs["do_sample"] is False
            return torch.tensor([[1, 2, 3, 4]])

    monkeypatch.setattr(config, "VLM_BACKEND", "transformers")
    monkeypatch.setattr(vlm, "_model", Model())
    monkeypatch.setattr(vlm, "_processor", Processor())
    assert (
        vlm.ask(["image-placeholder"], "question", max_tokens=90) == '{"visible":true}'
    )


def test_vlm_cache_separates_runtime_and_checkpoint(monkeypatch):
    monkeypatch.setattr(config, "VLM_BACKEND", "mlx")
    first = vlm_scan.scan_key("abc", [0, 0, 100, 100], [0, 0, 200, 200])
    monkeypatch.setattr(config, "VLM_BACKEND", "transformers")
    second = vlm_scan.scan_key("abc", [0, 0, 100, 100], [0, 0, 200, 200])
    monkeypatch.setattr(config, "VLM_REPO", "another-checkpoint")
    third = vlm_scan.scan_key("abc", [0, 0, 100, 100], [0, 0, 200, 200])
    assert len({first, second, third}) == 3
