import numpy as np


def test_model_kwargs_can_resolve_feature_extractor_from_json_spec() -> None:
    import pyimgano.models  # noqa: F401 - registry population
    from pyimgano.cli_common import build_model_kwargs
    from pyimgano.models import create_model

    user_kwargs = {
        "feature_extractor": {
            "name": "edge_stats",
            "kwargs": {},
        }
    }
    kwargs = build_model_kwargs(
        "vision_ecod",
        user_kwargs=user_kwargs,
        preset_kwargs=None,
        auto_kwargs={"contamination": 0.1},
    )

    det = create_model("vision_ecod", **kwargs)

    imgs = [
        np.zeros((32, 32, 3), dtype=np.uint8),
        np.ones((32, 32, 3), dtype=np.uint8) * 255,
    ]
    det.fit(imgs)
    scores = det.decision_function(imgs)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
