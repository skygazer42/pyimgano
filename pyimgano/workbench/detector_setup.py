from __future__ import annotations

import pyimgano.services.workbench_service as workbench_service
from pyimgano.workbench.adaptation_runtime import apply_tiling
from pyimgano.workbench.config import WorkbenchConfig


def build_workbench_runtime_detector(*, config: WorkbenchConfig):
    detector = workbench_service.create_workbench_detector(config=config)
    detector = apply_tiling(detector, config.adaptation.tiling)

    illumination_contrast = getattr(
        getattr(config, "preprocessing", None),
        "illumination_contrast",
        None,
    )
    if illumination_contrast is not None:
        from pyimgano.inference.preprocessing import PreprocessingDetector

        detector = PreprocessingDetector(
            detector=detector,
            illumination_contrast=illumination_contrast,
        )

    return detector


__all__ = ["build_workbench_runtime_detector"]
