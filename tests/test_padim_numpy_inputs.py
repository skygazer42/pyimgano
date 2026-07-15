import numpy as np
import pytest

from pyimgano.models import create_model

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_padim_accepts_numpy_images_for_fit_scoring_and_maps():
    det = create_model(
        "vision_padim",
        pretrained=False,
        device="cpu",
        image_size=32,
        d_reduced=4,
        covariance_eps=0.1,
    )

    assert det.resize_size == 37
    assert [type(item).__name__ for item in det.transform.transforms] == [
        "ToPILImage",
        "Resize",
        "CenterCrop",
        "ToTensor",
        "Normalize",
    ]

    imgs = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
    det.fit(imgs)

    scores = det.decision_function(imgs)
    assert scores.shape == (2,)

    anomaly_map = det.get_anomaly_map(imgs[0])
    assert anomaly_map.shape == (32, 32)
    assert anomaly_map.dtype == np.float32
    assert np.isfinite(anomaly_map).all()


def test_padim_uses_fixed_random_channel_subset_not_projection(monkeypatch):
    det = create_model(
        "vision_padim",
        pretrained=False,
        device="cpu",
        image_size=32,
        d_reduced=2,
        covariance_eps=0.1,
        random_state=7,
    )
    features = np.arange(24, dtype=np.float32).reshape(6, 4)

    def fake_extract(_image):
        det.patch_shape = (2, 3)
        return features.copy()

    monkeypatch.setattr(det, "_extract_patch_features", fake_extract)

    images = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
    det.fit(images)

    expected_indices = np.sort(
        np.random.default_rng(7).choice(4, size=2, replace=False)
    )
    np.testing.assert_array_equal(det.feature_indices_, expected_indices)
    np.testing.assert_allclose(det.means, features[:, expected_indices])
