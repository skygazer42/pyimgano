from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")


@pytest.mark.parametrize(
    ("model_name", "kwargs"),
    [
        (
            "vision_cutpaste",
            {
                "pretrained": False,
                "epochs": 1,
                "batch_size": 2,
                "device": "cpu",
            },
        ),
        (
            "vision_devnet",
            {
                "pretrained": False,
                "epochs": 1,
                "batch_size": 2,
                "device": "cpu",
            },
        ),
        (
            "vision_differnet",
            {
                "pretrained": False,
                "epochs": 1,
                "batch_size": 2,
                "device": "cpu",
                "random_state": 0,
            },
        ),
        (
            "vision_memseg",
            {
                "pretrained": False,
                "device": "cpu",
                "memory_size": 32,
                "k_neighbors": 1,
                "use_segmentation_head": False,
            },
        ),
        (
            "vision_patchcore",
            {
                "pretrained": False,
                "device": "cpu",
                "coreset_sampling_ratio": 1.0,
                "n_neighbors": 1,
            },
        ),
    ],
)
def test_selected_torchvision_detectors_do_not_use_deprecated_pretrained_api(
    model_name: str,
    kwargs: dict[str, object],
) -> None:
    from pyimgano.models import create_model

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        create_model(model_name, **kwargs)

    messages = [str(item.message) for item in caught]
    assert not any(
        "parameter 'pretrained' is deprecated" in message.lower()
        or "arguments other than a weight enum or `none`" in message.lower()
        for message in messages
    ), messages


def test_one_class_cnn_cnn_feature_extractor_does_not_use_deprecated_pretrained_api(
    tmp_path: Path,
) -> None:
    from PIL import Image

    from pyimgano.models.one_svm_cnn import ImageAnomalyDetector

    image_path = tmp_path / "sample.png"
    Image.fromarray(np.ones((32, 32, 3), dtype=np.uint8) * 64, mode="RGB").save(image_path)

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
