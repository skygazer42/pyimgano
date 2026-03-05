import numpy as np


def test_vision_feature_pipeline_glues_core_detector_and_extractor() -> None:
    import pyimgano.models  # noqa: F401 - ensure core detectors are registered
    from pyimgano.features.identity import IdentityExtractor
    from pyimgano.pipelines.feature_pipeline import VisionFeaturePipeline

    rng = np.random.RandomState(0)
    X = rng.normal(size=(80, 5))

    pipe = VisionFeaturePipeline(
        core_detector="core_ldof",
        core_kwargs={"n_neighbors": 10},
        feature_extractor=IdentityExtractor(),
        contamination=0.1,
    )
    pipe.fit(X)
    scores = pipe.decision_function(X[:10])
    preds = pipe.predict(X[:10])

    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(scores))
