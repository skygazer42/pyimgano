from __future__ import annotations

"""Dynamic feature-extractor + core-detector pipeline model.

This registers :class:`pyimgano.pipelines.feature_pipeline.VisionFeaturePipeline` as a regular
model in the `pyimgano.models` registry so that it can be used from:

- `pyimgano.models.create_model(...)`
- Workbench JSON configs (`model.name = "vision_feature_pipeline"`)
"""

from pyimgano.models.registry import register_model
from pyimgano.pipelines.feature_pipeline import VisionFeaturePipeline


@register_model(
    "vision_feature_pipeline",
    tags=("vision", "classical", "pipeline"),
    metadata={
        "description": "Feature extractor + core detector pipeline (dynamic vision wrapper)."
    },
)
class VisionFeaturePipelineModel(VisionFeaturePipeline):
    pass

