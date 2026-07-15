from __future__ import annotations

import numpy as np

from pyimgano.models.spade import VisionSPADEDetector


def test_spade_localization_uses_only_retrieved_normal_images() -> None:
    detector = VisionSPADEDetector.__new__(VisionSPADEDetector)
    detector.k_neighbors = 1
    detector.feature_levels = ("layer1",)
    detector.align_features = False
    detector.image_size = 2
    detector.gaussian_sigma = 0.0
    detector.train_global_features_ = np.asarray([[0.0], [100.0]], dtype=np.float32)
    detector.train_feature_maps_ = {
        "layer1": np.asarray(
            [
                np.zeros((1, 2, 2), dtype=np.float32),
                np.full((1, 2, 2), 5.0, dtype=np.float32),
            ]
        )
    }
    detector._extract_feature_bundle = lambda _image: (  # type: ignore[method-assign]
        {"layer1": np.full((1, 2, 2), 5.0, dtype=np.float32)},
        np.asarray([0.0], dtype=np.float32),
    )

    anomaly_map = detector._compute_anomaly_map(np.zeros((2, 2, 3), dtype=np.float32))

    np.testing.assert_allclose(anomaly_map, 5.0)


def test_spade_concatenates_pyramid_levels_before_correspondence() -> None:
    detector = VisionSPADEDetector.__new__(VisionSPADEDetector)
    detector.k_neighbors = 1
    detector.feature_levels = ("layer1", "layer2")
    detector.align_features = False
    detector.image_size = 2
    detector.gaussian_sigma = 0.0
    detector.train_global_features_ = np.asarray([[0.0]], dtype=np.float32)
    detector.train_feature_maps_ = {
        "layer1": np.zeros((1, 1, 2, 2), dtype=np.float32),
        "layer2": np.zeros((1, 1, 1, 1), dtype=np.float32),
    }
    detector._extract_feature_bundle = lambda _image: (  # type: ignore[method-assign]
        {
            "layer1": np.zeros((1, 2, 2), dtype=np.float32),
            "layer2": np.full((1, 1, 1), 10.0, dtype=np.float32),
        },
        np.asarray([0.0], dtype=np.float32),
    )

    anomaly_map = detector._compute_anomaly_map(np.zeros((2, 2, 3), dtype=np.float32))

    np.testing.assert_allclose(anomaly_map, 10.0)


def test_spade_mvtec_preprocessing_resizes_then_center_crops() -> None:
    detector = VisionSPADEDetector.__new__(VisionSPADEDetector)
    detector.image_size = 256
    detector.crop_size = 224
    image = np.zeros((300, 500, 3), dtype=np.uint8)

    tensor = detector._preprocess(image)

    assert tuple(tensor.shape) == (3, 224, 224)
