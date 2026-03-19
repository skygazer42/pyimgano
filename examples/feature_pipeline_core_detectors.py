"""Example: VisionFeaturePipeline with core_* detectors.

This shows how to compose:
- a feature extractor (registered in `pyimgano.features`)
- a `core_*` detector (registered in `pyimgano.models`)

without writing a dedicated `vision_*` wrapper class.

Notes:
- For real image use, pass a path-capable feature extractor (e.g. HOG, torchvision_backbone).
- For this minimal example we use `feature_extractor="identity"` and feed feature vectors directly.
"""

from __future__ import annotations

import numpy as np

from pyimgano.models import create_model


def main() -> None:
    rng = np.random.default_rng(0)
    x = [rng.normal(size=(16,)).astype(np.float32) for _ in range(100)]

    pipe = create_model(
        "vision_feature_pipeline",
        contamination=0.1,
        feature_extractor="identity",
        core_detector="core_ecod",
        core_kwargs={},
    )
    pipe.fit(x)
    scores = pipe.decision_function(x[:5])
    print("scores:", scores)


if __name__ == "__main__":
    main()
