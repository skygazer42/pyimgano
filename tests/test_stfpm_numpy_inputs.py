import numpy as np

from pyimgano.models import create_model


def test_stfpm_accepts_numpy_images_for_scoring_and_maps():
    det = create_model(
        "vision_stfpm",
        pretrained_teacher=False,
        epochs=1,
        batch_size=1,
        device="cpu",
    )

    # Avoid running full training in this unit test; enable scoring path.
    det.mean_scores = 0.0
    det.std_scores = 1.0

    imgs = [np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(2)]
    scores = det.decision_function(imgs)
    assert scores.shape == (2,)

    anomaly_map = det.get_anomaly_map(imgs[0])
    assert anomaly_map.ndim == 2
    assert anomaly_map.shape == (32, 48)
    assert np.isfinite(anomaly_map).all()
