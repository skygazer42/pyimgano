from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from pyimgano.services.infer_context_service import (
    FromRunInferContextRequest,
    InferConfigContextRequest,
    prepare_from_run_context,
    prepare_infer_config_context,
)


def test_prepare_infer_config_context_returns_threshold_and_checkpoint(tmp_path) -> None:
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    ckpt_dir = run_dir / "checkpoints" / "custom"
    artifacts.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "model.pt").write_text("ckpt", encoding="utf-8")

    infer_cfg_path = artifacts / "infer_config.json"
    infer_cfg_path.write_text(
        json.dumps(
            {
                "from_run": str(run_dir),
                "category": "custom",
                "model": {
                    "name": "vision_ecod",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                    "preset": None,
                    "model_kwargs": {},
                    "checkpoint_path": None,
                },
                "adaptation": {"tiling": {}, "postprocess": None, "save_maps": False},
                "threshold": 0.7,
                "checkpoint": {"path": "checkpoints/custom/model.pt"},
            }
        ),
        encoding="utf-8",
    )

    context = prepare_infer_config_context(
        InferConfigContextRequest(config_path=str(infer_cfg_path))
    )

    assert context.model_name == "vision_ecod"
    assert context.threshold == pytest.approx(0.7)
    assert context.trained_checkpoint_path is not None
    assert context.trained_checkpoint_path.endswith("model.pt")


def test_prepare_from_run_context_applies_request_overrides_and_optional_payloads(
    tmp_path, monkeypatch
) -> None:
    import pyimgano.services.infer_context_service as infer_context_service

    @dataclass(frozen=True)
    class _DefectsConfig:
        enabled: bool
        min_area: int

    run_dir = tmp_path / "run"
    (run_dir / "artifacts").mkdir(parents=True)
    model_artifact = run_dir / "artifacts" / "backbone.onnx"
    model_artifact.write_text("onnx", encoding="utf-8")
    trained_checkpoint = run_dir / "checkpoints" / "custom" / "model.pt"
    trained_checkpoint.parent.mkdir(parents=True)
    trained_checkpoint.write_text("ckpt", encoding="utf-8")

    cfg = SimpleNamespace(
        model=SimpleNamespace(
            name="vision_onnx_ecod",
            preset="industrial-fast",
            device="cpu",
            contamination=0.1,
            pretrained=False,
            model_kwargs={"existing": 1},
            checkpoint_path="backbone.onnx",
        ),
        preprocessing=SimpleNamespace(
            illumination_contrast={"white_balance": "gray_world"},
        ),
        adaptation=SimpleNamespace(
            tiling=SimpleNamespace(
                tile_size=16,
                stride=8,
                score_reduce="topk_mean",
                score_topk=0.2,
                map_reduce="hann",
            )
        ),
        defects=_DefectsConfig(enabled=True, min_area=9),
    )

    monkeypatch.setattr(
        infer_context_service.workbench_run_service,
        "load_workbench_config_from_run",
        lambda _run_dir: cfg,
    )
    monkeypatch.setattr(
        infer_context_service.workbench_run_service,
        "load_report_from_run",
        lambda _run_dir: {"category": "custom"},
    )
    monkeypatch.setattr(
        infer_context_service.workbench_run_service,
        "select_category_report",
        lambda report, category=None: ("custom", {"report": report, "category": category}),
    )
    monkeypatch.setattr(
        infer_context_service.workbench_run_service,
        "extract_threshold",
        lambda _category_report: 0.7,
    )
    monkeypatch.setattr(
        infer_context_service.workbench_run_service,
        "resolve_checkpoint_path",
        lambda _run_dir, _category_report: trained_checkpoint,
    )

    context = prepare_from_run_context(
        FromRunInferContextRequest(
            run_dir=str(run_dir),
            preset="industrial-balanced",
            device="cuda",
            contamination=0.25,
            pretrained=True,
            model_kwargs={"new": 2},
        )
    )

    assert context.model_name == "vision_onnx_ecod"
    assert context.preset == "industrial-balanced"
    assert context.device == "cuda"
    assert context.contamination == pytest.approx(0.25)
    assert context.pretrained is True
    assert context.base_user_kwargs == {"existing": 1, "new": 2}
    assert context.checkpoint_path == str(model_artifact.resolve())
    assert context.trained_checkpoint_path == str(trained_checkpoint)
    assert context.threshold == pytest.approx(0.7)
    assert context.defects_payload == {"enabled": True, "min_area": 9}
    assert context.illumination_contrast_knobs == {"white_balance": "gray_world"}
    assert context.tiling_payload == {
        "tile_size": 16,
        "stride": 8,
        "score_reduce": "topk_mean",
        "score_topk": 0.2,
        "map_reduce": "hann",
    }
    assert context.warnings == ()


def test_prepare_from_run_context_prefers_explicit_request_checkpoint_path(tmp_path, monkeypatch) -> None:
    import pyimgano.services.infer_context_service as infer_context_service

    @dataclass(frozen=True)
    class _DefectsConfig:
        enabled: bool

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    explicit_checkpoint = run_dir / "manual.onnx"
    explicit_checkpoint.write_text("manual", encoding="utf-8")

    cfg = SimpleNamespace(
        model=SimpleNamespace(
            name="vision_onnx_ecod",
            preset=None,
            device="cpu",
            contamination=0.1,
            pretrained=False,
            model_kwargs={},
            checkpoint_path="ignored.onnx",
        ),
        preprocessing=SimpleNamespace(),
        adaptation=SimpleNamespace(),
        defects=_DefectsConfig(enabled=False),
    )

    monkeypatch.setattr(
        infer_context_service.workbench_run_service,
        "load_workbench_config_from_run",
        lambda _run_dir: cfg,
    )
    monkeypatch.setattr(
        infer_context_service.workbench_run_service,
        "load_report_from_run",
        lambda _run_dir: {"category": "custom"},
    )
    monkeypatch.setattr(
        infer_context_service.workbench_run_service,
        "select_category_report",
        lambda report, category=None: ("custom", {"report": report, "category": category}),
    )
    monkeypatch.setattr(
        infer_context_service.workbench_run_service,
        "extract_threshold",
        lambda _category_report: None,
    )
    monkeypatch.setattr(
        infer_context_service.workbench_run_service,
        "resolve_checkpoint_path",
        lambda _run_dir, _category_report: None,
    )

    context = prepare_from_run_context(
        FromRunInferContextRequest(
            run_dir=str(run_dir),
            checkpoint_path=str(explicit_checkpoint),
        )
    )

    assert context.checkpoint_path == str(explicit_checkpoint)
    assert context.warnings == ()


@pytest.mark.parametrize(
    ("payload_patch", "message"),
    [
        ({"defects": []}, "infer-config key 'defects' must be a JSON object/dict."),
        ({"preprocessing": []}, "infer-config key 'preprocessing' must be a JSON object/dict."),
        (
            {"preprocessing": {"illumination_contrast": []}},
            "infer-config key 'preprocessing.illumination_contrast' must be a JSON object/dict.",
        ),
    ],
)
def test_prepare_infer_config_context_rejects_invalid_optional_payload_shapes(
    tmp_path,
    payload_patch,
    message,
) -> None:
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    payload = {
        "from_run": str(run_dir),
        "category": "custom",
        "model": {
            "name": "vision_ecod",
            "device": "cpu",
            "pretrained": False,
            "contamination": 0.1,
            "preset": None,
            "model_kwargs": {},
            "checkpoint_path": None,
        },
        "adaptation": {"tiling": {}, "postprocess": None, "save_maps": False},
        "threshold": 0.7,
    }
    payload.update(payload_patch)

    infer_cfg_path = artifacts / "infer_config.json"
    infer_cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        prepare_infer_config_context(InferConfigContextRequest(config_path=str(infer_cfg_path)))


def test_prepare_infer_config_context_allows_missing_illumination_payload(tmp_path) -> None:
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    infer_cfg_path = artifacts / "infer_config.json"
    infer_cfg_path.write_text(
        json.dumps(
            {
                "from_run": str(run_dir),
                "category": "custom",
                "model": {
                    "name": "vision_ecod",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                    "preset": None,
                    "model_kwargs": {},
                    "checkpoint_path": None,
                },
                "preprocessing": {},
                "adaptation": {"tiling": {}, "postprocess": None, "save_maps": False},
                "threshold": 0.7,
            }
        ),
        encoding="utf-8",
    )

    context = prepare_infer_config_context(
        InferConfigContextRequest(config_path=str(infer_cfg_path))
    )

    assert context.illumination_contrast_knobs is None
