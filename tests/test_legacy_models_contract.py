import numpy as np
import pytest


@pytest.mark.parametrize(
    "model_name",
    [
        "lof_structure",
        "isolation_forest_struct",
        "kmeans_anomaly",
        "dbscan_anomaly",
        "ssim_template",
        "ssim_struct",
    ],
)
def test_legacy_models_follow_base_detector_contract(model_name: str) -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    if model_name in {"ssim_template", "ssim_struct"}:
        pytest.importorskip("skimage")

        base = np.zeros((96, 96, 3), dtype=np.uint8)
        base[20:70, 30:60, :] = 200
        train = [base for _ in range(6)]
        det = create_model(model_name, contamination=0.2, n_templates=1, resize_hw=(64, 64))
        det.fit(train)
        scores = det.decision_function([base, base])
        preds = det.predict([base, base])
    else:
        from pyimgano.features.identity import IdentityExtractor

        rng = np.random.RandomState(0)
        X = rng.normal(size=(80, 6))
        extra = {}
        if model_name == "kmeans_anomaly":
            extra = {"n_clusters": 5, "random_state": 0}
        elif model_name == "dbscan_anomaly":
            extra = {"eps": 1.5, "min_samples": 5}
        elif model_name == "isolation_forest_struct":
            extra = {"n_estimators": 25, "random_state": 0}
        elif model_name == "lof_structure":
            extra = {"n_neighbors": 10}

        det = create_model(
            model_name,
            feature_extractor=IdentityExtractor(),
            contamination=0.2,
            **extra,
        )
        det.fit(X)
        scores = det.decision_function(X[:9])
        preds = det.predict(X[:9])

    assert scores.shape[0] == preds.shape[0]
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})
    assert hasattr(det, "threshold_")
