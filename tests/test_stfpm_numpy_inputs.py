import numpy as np
import pytest

from pyimgano.models import create_model

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_stfpm_accepts_numpy_images_for_scoring_and_maps():
    det = create_model(
        "vision_stfpm",
        pretrained_teacher=False,
        epochs=1,
        batch_size=1,
        device="cpu",
    )

    # Avoid running full training in this unit test; enable scoring path.
    det._is_fitted = True

    imgs = [np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(2)]
    scores = det.decision_function(imgs)
    assert scores.shape == (2,)

    anomaly_map = det.get_anomaly_map(imgs[0])
    assert anomaly_map.ndim == 2
    assert anomaly_map.shape == (32, 48)
    assert np.isfinite(anomaly_map).all()


def test_stfpm_uses_paper_train_validation_split() -> None:
    det = create_model(
        "vision_stfpm",
        pretrained_teacher=False,
        epochs=1,
        batch_size=2,
        validation_ratio=0.2,
        random_state=7,
        device="cpu",
    )
    inputs = [f"image-{index}.png" for index in range(10)]

    train, validation = det._split_training_inputs(inputs)

    assert len(train) == 8
    assert len(validation) == 2
    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == sorted(inputs)
