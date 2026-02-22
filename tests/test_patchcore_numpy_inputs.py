import numpy as np

from pyimgano.models import create_model


def test_patchcore_accepts_numpy_images(monkeypatch):
    det = create_model(
        "vision_patchcore",
        coreset_sampling_ratio=1.0,
        n_neighbors=1,
        pretrained=False,
        device="cpu",
    )

    def fake_extract(image):
        assert isinstance(image, np.ndarray)
        features = np.zeros((4, 2), dtype=np.float32)
        return features, (2, 2)

    monkeypatch.setattr(det, "_extract_patch_features", fake_extract)

    imgs = [np.zeros((10, 20, 3), dtype=np.uint8) for _ in range(2)]
    det.fit(imgs)

    scores = det.decision_function(imgs)
    assert scores.shape == (2,)

    monkeypatch.setattr(
        det._cv2,
        "imread",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("cv2.imread called")),
    )
    anomaly_map = det.get_anomaly_map(imgs[0])
    assert anomaly_map.shape == (10, 20)
    assert anomaly_map.dtype == np.float32


def test_patchcore_feature_projection_reduces_feature_dim(monkeypatch):
    det = create_model(
        "vision_patchcore",
        coreset_sampling_ratio=1.0,
        n_neighbors=1,
        pretrained=False,
        device="cpu",
        feature_projection_dim=1,
        projection_fit_samples=1,
        random_seed=0,
    )

    def fake_extract(_image):
        features = np.arange(8, dtype=np.float32).reshape(4, 2)
        return features, (2, 2)

    monkeypatch.setattr(det, "_extract_patch_features", fake_extract)

    imgs = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
    det.fit(imgs)

    assert det.memory_bank is not None
    assert det.memory_bank.ndim == 2
    assert det.memory_bank.shape[1] == 1


def test_patchcore_memory_bank_dtype_float16(monkeypatch):
    det = create_model(
        "vision_patchcore",
        coreset_sampling_ratio=1.0,
        n_neighbors=1,
        pretrained=False,
        device="cpu",
        memory_bank_dtype="float16",
    )

    def fake_extract(_image):
        features = np.random.randn(4, 2).astype(np.float32)
        return features, (2, 2)

    monkeypatch.setattr(det, "_extract_patch_features", fake_extract)

    imgs = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
    det.fit(imgs)

    assert det.memory_bank is not None
    assert det.memory_bank.dtype == np.float16
