import numpy as np

from pyimgano.models import create_model


def test_draem_accepts_numpy_images_for_scoring_and_maps():
    det = create_model(
        "vision_draem",
        image_size=32,
        epochs=1,
        batch_size=1,
        device="cpu",
    )

    # Avoid running full training in this unit test; enable inference path.
    det._is_fitted = True

    imgs = [np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(2)]
    scores = det.decision_function(imgs)
    assert scores.shape == (2,)

    anomaly_map = det.get_anomaly_map(imgs[0])
    assert anomaly_map.ndim == 2
    assert anomaly_map.shape == (20, 30)
    assert np.isfinite(anomaly_map).all()
