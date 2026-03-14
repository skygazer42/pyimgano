from __future__ import annotations

import pytest

import pyimgano.models as models
from pyimgano.weights.manifest import validate_weights_manifest


def test_sonar_s7498_literal_kwargs_smoke_non_torch() -> None:
    # Cover the early-return path that uses an empty-tuple literal.
    report = validate_weights_manifest({}, check_files=False)
    assert report.ok is False
    assert report.entries == ()

    # Smoke: instantiate the wrappers we touched for S7498.
    # Using "identity" extractor avoids pulling heavy vision backends by default.
    cases: list[tuple[str, dict]] = [
        ("core_abod", {}),
        ("vision_abod", {"feature_extractor": "identity"}),
        ("core_cof", {}),
        ("vision_cof", {"feature_extractor": "identity"}),
        ("core_dbscan", {"eps": 0.6, "min_samples": 5}),
        ("vision_dbscan", {"feature_extractor": "identity", "eps": 0.6, "min_samples": 5}),
        ("core_hbos", {"n_bins": 8}),
        ("vision_hbos", {"feature_extractor": "identity", "n_bins": 8}),
        ("core_inne", {"n_estimators": 10, "random_state": 0}),
        ("vision_inne", {"feature_extractor": "identity", "n_estimators": 10, "random_state": 0}),
        ("core_lof", {"n_neighbors": 5}),
        ("vision_lof", {"feature_extractor": "identity", "n_neighbors": 5}),
        ("core_mcd", {"random_state": 0}),
        ("vision_mcd", {"feature_extractor": "identity", "random_state": 0}),
        ("core_ocsvm", {"kernel": "rbf"}),
        ("vision_ocsvm", {"feature_extractor": "identity", "kernel": "rbf"}),
        ("core_sampling", {"subset_size": 5}),
        ("vision_sampling", {"feature_extractor": "identity", "subset_size": 5}),
        ("core_feature_bagging", {"n_estimators": 2, "n_neighbors": 5}),
        (
            "vision_feature_bagging",
            {"feature_extractor": "identity", "n_estimators": 2, "n_neighbors": 5},
        ),
        ("core_feature_bagging_spec", {"n_estimators": 2}),
        (
            "vision_feature_bagging_spec",
            {"feature_extractor": "identity", "n_estimators": 2},
        ),
        ("core_oddoneout", {"n_neighbors": 3}),
        ("vision_oddoneout", {"feature_extractor": "identity", "n_neighbors": 3}),
        ("core_knn_cosine", {"n_neighbors": 3}),
        ("core_random_projection_knn", {"n_neighbors": 3}),
        ("vision_random_projection_knn", {"feature_extractor": "identity", "n_neighbors": 3}),
        ("vision_kpca", {"feature_extractor": "identity", "n_components": 2}),
    ]

    for name, kwargs in cases:
        det = models.create_model(name, **kwargs)
        assert det is not None


def test_sonar_s7498_literal_kwargs_smoke_torch_models() -> None:
    pytest.importorskip("torch")

    det = models.create_model("core_torch_autoencoder", epochs=1, batch_size=8)
    assert det is not None

    det = models.create_model(
        "vision_torch_autoencoder",
        feature_extractor="identity",
        epochs=1,
        batch_size=8,
    )
    assert det is not None

