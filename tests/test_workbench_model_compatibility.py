from __future__ import annotations

from pyimgano.models.registry import MODEL_REGISTRY
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.model_compatibility import (
    collect_workbench_pixel_map_requirements,
    load_workbench_model_capabilities,
)


def test_collect_workbench_pixel_map_requirements_returns_enabled_options(tmp_path) -> None:
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {
                "name": "custom",
                "root": str(tmp_path / "dataset"),
                "category": "custom",
            },
            "model": {"name": "unused_detector"},
            "adaptation": {
                "save_maps": True,
                "postprocess": {
                    "normalize": True,
                    "normalize_method": "minmax",
                },
            },
            "defects": {"enabled": True},
        }
    )

    requirements = collect_workbench_pixel_map_requirements(config=cfg)

    assert requirements == (
        "adaptation.save_maps",
        "adaptation.postprocess",
        "defects.enabled",
    )


def test_load_workbench_model_capabilities_reads_registry_summary() -> None:
    class _PixelMapDetector:
        def __init__(self, **_kwargs):  # noqa: ANN003 - test stub
            pass

    MODEL_REGISTRY.register(
        "test_workbench_model_compatibility_detector",
        _PixelMapDetector,
        tags=("vision", "classical", "pixel_map", "numpy"),
        overwrite=True,
    )

    caps = load_workbench_model_capabilities(
        model_name="test_workbench_model_compatibility_detector"
    )

    assert caps.model_name == "test_workbench_model_compatibility_detector"
    assert caps.input_modes == ("paths", "numpy")
    assert caps.supports_pixel_map is True
