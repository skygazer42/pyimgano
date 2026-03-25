from __future__ import annotations

import numpy as np
import pytest


def test_spade_checkpoint_roundtrip_on_image_paths(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("cv2")

    import pyimgano.models  # noqa: F401
    from PIL import Image

    from pyimgano.models.registry import create_model
    from pyimgano.training.checkpointing import save_checkpoint
    from pyimgano.workbench.checkpoint_restore import load_checkpoint_into_detector

    train_paths = []
    for idx, value in enumerate((80, 82, 84, 86), start=1):
        path = tmp_path / f"train_{idx}.png"
        img = np.full((48, 48, 3), value, dtype=np.uint8)
        Image.fromarray(img, mode="RGB").save(path)
        train_paths.append(str(path))

    normal_path = tmp_path / "normal.png"
    anomaly_path = tmp_path / "anomaly.png"
    Image.fromarray(np.full((48, 48, 3), 83, dtype=np.uint8), mode="RGB").save(normal_path)
    anomaly = np.full((48, 48, 3), 83, dtype=np.uint8)
    anomaly[12:30, 12:30, :] = 240
    Image.fromarray(anomaly, mode="RGB").save(anomaly_path)

    def _make_detector():
        return create_model(
            "vision_spade",
            contamination=0.2,
            backbone="resnet18",
            pretrained=False,
            image_size=64,
            k_neighbors=1,
            feature_levels=("layer3",),
            align_features=False,
            gaussian_sigma=0.0,
            device="cpu",
        )

    detector = _make_detector()
    detector.fit(train_paths)
    expected_scores = np.asarray(
        detector.decision_function([str(normal_path), str(anomaly_path)]),
        dtype=np.float64,
    )
    expected_maps = np.asarray(
        detector.predict_anomaly_map([str(normal_path), str(anomaly_path)]),
        dtype=np.float32,
    )
    expected_threshold = float(detector.threshold_)

    ckpt_path = tmp_path / "spade.ckpt"
    save_checkpoint(detector, ckpt_path)

    restored = _make_detector()
    load_checkpoint_into_detector(restored, ckpt_path)

    restored_scores = np.asarray(
        restored.decision_function([str(normal_path), str(anomaly_path)]),
        dtype=np.float64,
    )
    restored_maps = np.asarray(
        restored.predict_anomaly_map([str(normal_path), str(anomaly_path)]),
        dtype=np.float32,
    )

    np.testing.assert_allclose(restored_scores, expected_scores, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(restored_maps, expected_maps, rtol=1e-5, atol=1e-5)
    assert float(restored.threshold_) == pytest.approx(expected_threshold)
