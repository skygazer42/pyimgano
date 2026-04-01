from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np


def test_one_class_cnn_exposes_seeded_pca_random_state() -> None:
    from pyimgano.models.one_svm_cnn import ImageAnomalyDetector

    detector = ImageAnomalyDetector(feature_type="histogram", random_state=7)

    assert detector.pca.random_state == 7


def test_one_class_cnn_registry_contract_on_image_paths(tmp_path: Path) -> None:
    from PIL import Image

    from pyimgano.models import create_model

    train_paths: list[str] = []
    for idx, value in enumerate((60, 80, 100, 120), start=1):
        path = tmp_path / f"train_{idx}.png"
        img = np.ones((32, 32, 3), dtype=np.uint8) * int(value)
        Image.fromarray(img, mode="RGB").save(path)
        train_paths.append(str(path))

    test_paths: list[str] = []
    for idx, value in enumerate((70, 140), start=1):
        path = tmp_path / f"test_{idx}.png"
        img = np.ones((32, 32, 3), dtype=np.uint8) * int(value)
        Image.fromarray(img, mode="RGB").save(path)
        test_paths.append(str(path))

    detector = create_model("one_class_cnn", feature_type="histogram", random_state=0)
    detector.fit(train_paths)

    scores = np.asarray(detector.decision_function(test_paths), dtype=np.float64).reshape(-1)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))

    labels = np.asarray(detector.predict(test_paths), dtype=np.int64).reshape(-1)
    assert labels.shape == (2,)
    assert set(labels.tolist()).issubset({0, 1})


def test_one_class_cnn_cnn_feature_path_avoids_torchvision_pretrained_warning(
    tmp_path: Path,
) -> None:
    import pytest
    from PIL import Image

    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    from pyimgano.models.one_svm_cnn import ImageAnomalyDetector

    image_path = tmp_path / "cnn.png"
    Image.fromarray(np.ones((32, 32, 3), dtype=np.uint8) * 80, mode="RGB").save(image_path)

    detector = ImageAnomalyDetector(feature_type="cnn", cnn_pretrained=False, random_state=0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        features = detector.extract_cnn_features(str(image_path))

    assert features.ndim == 1
    messages = [str(item.message) for item in caught]
    assert not any(
        "parameter 'pretrained' is deprecated" in message.lower()
        or "arguments other than a weight enum or `none`" in message.lower()
        for message in messages
    ), messages
