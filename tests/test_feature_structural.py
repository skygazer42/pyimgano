from __future__ import annotations

import numpy as np


def test_structural_features_extractor_outputs_fixed_dim() -> None:
    from pyimgano.features.structural import StructuralFeaturesExtractor

    img0 = np.zeros((128, 128, 3), dtype=np.uint8)
    img1 = img0.copy()
    img1[32:96, 48:80, :] = 255  # simple high-contrast rectangle

    ext = StructuralFeaturesExtractor(max_size=256, error_mode="raise")
    feats = ext.extract([img0, img1])

    assert feats.shape == (2, 15)
    assert np.all(np.isfinite(feats))

    # The rectangle should increase edge density vs the blank image.
    assert float(feats[1, 0]) >= float(feats[0, 0])
    assert float(feats[1, 1]) >= float(feats[0, 1])


def test_structural_features_extractor_is_deterministic() -> None:
    from pyimgano.features.structural import StructuralFeaturesExtractor

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, 30:34, :] = 255
    ext = StructuralFeaturesExtractor(max_size=256, error_mode="raise")

    a = ext.extract([img, img])
    b = ext.extract([img, img])
    assert np.allclose(a, b)


def test_structural_features_extractor_error_mode_zeros() -> None:
    from pyimgano.features.structural import StructuralFeaturesExtractor

    ext = StructuralFeaturesExtractor(max_size=256, error_mode="zeros")
    feats = ext.extract(["/pyimgano__definitely_missing__nope.png"])
    assert feats.shape == (1, 15)
    assert np.allclose(feats, 0.0)

