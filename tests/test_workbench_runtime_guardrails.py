from __future__ import annotations

import pytest

from pyimgano.models.registry import MODEL_REGISTRY
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.runtime_guardrails import validate_workbench_runtime_guardrails


def test_runtime_guardrails_reject_save_maps_without_save_run(tmp_path) -> None:
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {
                "name": "custom",
                "root": str(tmp_path / "dataset"),
                "category": "custom",
            },
            "model": {"name": "missing_detector"},
            "adaptation": {"save_maps": True},
            "output": {"save_run": False},
        }
    )

    with pytest.raises(ValueError, match="adaptation.save_maps requires output.save_run=true."):
        validate_workbench_runtime_guardrails(config=cfg)


def test_runtime_guardrails_reject_training_without_save_run(tmp_path) -> None:
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {
                "name": "custom",
                "root": str(tmp_path / "dataset"),
                "category": "custom",
            },
            "model": {"name": "missing_detector"},
            "training": {"enabled": True},
            "output": {"save_run": False},
        }
    )

    with pytest.raises(ValueError, match="training.enabled requires output.save_run=true."):
        validate_workbench_runtime_guardrails(config=cfg)


def test_runtime_guardrails_reject_pixel_map_features_on_non_pixel_map_model(tmp_path) -> None:
    class _DummyDetector:
        def __init__(self, **_kwargs):  # noqa: ANN003 - test stub
            pass

    MODEL_REGISTRY.register(
        "test_runtime_guardrails_non_pixel_map_detector",
        _DummyDetector,
        tags=("vision", "classical"),
        overwrite=True,
    )

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {
                "name": "custom",
                "root": str(tmp_path / "dataset"),
                "category": "custom",
            },
            "model": {"name": "test_runtime_guardrails_non_pixel_map_detector"},
            "adaptation": {
                "save_maps": True,
                "postprocess": {
                    "normalize": True,
                    "normalize_method": "minmax",
                },
            },
            "defects": {"enabled": True},
            "output": {"save_run": True},
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "These workbench options require a model that supports pixel maps: "
            "adaptation.save_maps, adaptation.postprocess, defects.enabled."
        ),
    ):
        validate_workbench_runtime_guardrails(config=cfg)


def test_runtime_guardrails_skip_registry_lookup_when_pixel_map_features_disabled(tmp_path) -> None:
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {
                "name": "custom",
                "root": str(tmp_path / "dataset"),
                "category": "custom",
            },
            "model": {"name": "missing_detector"},
            "output": {"save_run": True},
        }
    )

    validate_workbench_runtime_guardrails(config=cfg)
