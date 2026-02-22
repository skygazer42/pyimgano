import numpy as np

from pyimgano.models import create_model


def test_padim_accepts_numpy_images_for_fit_scoring_and_maps():
    det = create_model(
        "vision_padim",
        pretrained=False,
        device="cpu",
        image_size=32,
        d_reduced=4,
        projection_fit_samples=1,
        covariance_eps=0.1,
    )

    imgs = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
    det.fit(imgs)

    scores = det.decision_function(imgs)
    assert scores.shape == (2,)

    anomaly_map = det.get_anomaly_map(imgs[0])
    assert anomaly_map.shape == (32, 32)
    assert anomaly_map.dtype == np.float32
    assert np.isfinite(anomaly_map).all()
