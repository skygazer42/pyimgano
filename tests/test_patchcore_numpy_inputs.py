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
