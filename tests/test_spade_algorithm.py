from __future__ import annotations

import cv2
import numpy as np
import pytest

from pyimgano.models.spade import VisionSPADEDetector, _build_resnet_backbone


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

    np.testing.assert_allclose(anomaly_map, 25.0)


def test_spade_concatenates_pyramid_levels_on_finest_selected_grid() -> None:
    detector = VisionSPADEDetector.__new__(VisionSPADEDetector)
    detector.k_neighbors = 1
    detector.feature_levels = ("layer2", "layer1")
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
            "layer1": np.asarray([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32),
            "layer2": np.full((1, 1, 1), 10.0, dtype=np.float32),
        },
        np.asarray([0.0], dtype=np.float32),
    )

    anomaly_map = detector._compute_anomaly_map(np.zeros((2, 2, 3), dtype=np.float32))

    np.testing.assert_allclose(
        anomaly_map,
        np.asarray([[100.0, 101.0], [104.0, 109.0]], dtype=np.float32),
    )


def test_spade_image_score_is_mean_squared_l2_distance() -> None:
    detector = VisionSPADEDetector.__new__(VisionSPADEDetector)
    detector.k_neighbors = 2
    detector.train_global_features_ = np.asarray([[0.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    detector.train_feature_maps_ = {}
    detector._iter_images = lambda _x: iter(  # type: ignore[method-assign]
        [np.zeros((2, 2, 3), dtype=np.float32)]
    )
    detector._extract_feature_bundle = lambda _image: (  # type: ignore[method-assign]
        {},
        np.asarray([1.0, 0.0], dtype=np.float32),
    )

    scores = detector.decision_function([np.zeros((2, 2, 3), dtype=np.float32)])

    np.testing.assert_allclose(scores, [5.0])


def test_spade_pins_paper_era_imagenet_v1_weights(monkeypatch) -> None:
    from pyimgano.utils import optional_deps

    selected = object()
    calls = {}

    class Weight:
        def transforms(self):
            return None

    class Weights:
        DEFAULT = Weight()
        IMAGENET1K_V1 = selected_weight = Weight()

    class Models:
        @staticmethod
        def get_model_weights(_name):
            return Weights

        @staticmethod
        def get_model(name, *, weights):
            calls.update(name=name, weights=weights)
            return selected

    monkeypatch.setattr(optional_deps, "require", lambda *_args, **_kwargs: Models)

    assert _build_resnet_backbone("wide_resnet50", pretrained=True) is selected
    assert calls == {"name": "wide_resnet50_2", "weights": Weights.selected_weight}


def test_spade_resizes_squared_distance_map_with_inter_area() -> None:
    detector = VisionSPADEDetector.__new__(VisionSPADEDetector)
    detector.k_neighbors = 1
    detector.feature_levels = ("layer1",)
    detector.align_features = False
    detector.image_size = 4
    detector.gaussian_sigma = 0.0
    detector.train_global_features_ = np.asarray([[0.0]], dtype=np.float32)
    detector.train_feature_maps_ = {"layer1": np.zeros((1, 1, 1, 1), dtype=np.float32)}
    detector._extract_feature_bundle = lambda _image: (  # type: ignore[method-assign]
        {"layer1": np.asarray([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32)},
        np.asarray([0.0], dtype=np.float32),
    )

    anomaly_map = detector._compute_anomaly_map(np.zeros((4, 4, 3), dtype=np.float32))

    expected = cv2.resize(
        np.asarray([[0.0, 1.0], [4.0, 9.0]], dtype=np.float32),
        (4, 4),
        interpolation=cv2.INTER_AREA,
    )
    np.testing.assert_allclose(anomaly_map, expected)


def test_spade_mvtec_preprocessing_resizes_then_center_crops() -> None:
    detector = VisionSPADEDetector.__new__(VisionSPADEDetector)
    detector.image_size = 256
    detector.crop_size = 224
    image = np.zeros((300, 500, 3), dtype=np.uint8)

    tensor = detector._preprocess(image)

    assert tuple(tensor.shape) == (3, 224, 224)


@pytest.mark.slow
def test_spade_paper_wrn50x2_feature_contract() -> None:
    torch = pytest.importorskip("torch")
    detector = VisionSPADEDetector(pretrained=False, device="cpu")

    assert detector.backbone_name == "wide_resnet50"
    assert detector.image_size == 256
    assert detector.crop_size == 224
    assert detector.k_neighbors == 50
    assert detector.feature_levels == ("layer1", "layer2", "layer3")
    assert detector.gaussian_sigma == 4.0
    assert all(not parameter.requires_grad for parameter in detector.feature_extractor.parameters())

    extractor = detector.feature_extractor
    assert [
        len(extractor.layer1[-1]),
        len(extractor.layer2[0]),
        len(extractor.layer3[0]),
        len(extractor.layer4[0]),
    ] == [3, 4, 6, 3]
    assert extractor.layer1[-1][0].conv2.out_channels == 128

    with torch.inference_mode():
        layer1, layer2, layer3, global_descriptor = extractor(torch.zeros(1, 3, 224, 224))

    assert tuple(layer1.shape) == (1, 256, 56, 56)
    assert tuple(layer2.shape) == (1, 512, 28, 28)
    assert tuple(layer3.shape) == (1, 1024, 14, 14)
    assert tuple(global_descriptor.shape) == (1, 2048)
