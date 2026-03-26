from __future__ import annotations

import numpy as np
import pytest


def test_student_teacher_lite_smoke_on_vectors() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    rng = np.random.default_rng(0)
    x = [rng.normal(size=(20,)).astype(np.float32) for _ in range(80)]

    det = create_model(
        "vision_student_teacher_lite",
        contamination=0.1,
        teacher_extractor="identity",
        student_extractor={
            "name": "pca_projector",
            "kwargs": {"base_extractor": "identity", "n_components": 0.8},
        },
        ridge=1e-6,
    )
    det.fit(x)
    scores = det.decision_function(x[:10])
    preds = det.predict(x[:10])

    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(preds)).issubset({0, 1})


def test_student_teacher_lite_checkpoint_roundtrip_on_image_paths(tmp_path) -> None:
    pytest.importorskip("torch")
    from PIL import Image

    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model
    from pyimgano.training.checkpointing import save_checkpoint
    from pyimgano.workbench.checkpoint_restore import load_checkpoint_into_detector

    train_paths = []
    for idx, value in enumerate((32, 48, 64, 80), start=1):
        path = tmp_path / f"train_{idx}.png"
        img = np.full((32, 32, 3), value, dtype=np.uint8)
        Image.fromarray(img, mode="RGB").save(path)
        train_paths.append(str(path))

    eval_paths = train_paths[:2]

    kwargs = {
        "contamination": 0.1,
        "teacher_extractor": "torchvision_multilayer",
        "teacher_kwargs": {
            "backbone": "resnet18",
            "pretrained": False,
            "device": "cpu",
            "batch_size": 2,
            "image_size": 32,
        },
        "student_extractor": "torchvision_backbone",
        "student_kwargs": {
            "backbone": "resnet18",
            "pretrained": False,
            "device": "cpu",
            "batch_size": 2,
            "image_size": 32,
        },
        "ridge": 1e-6,
    }

    detector = create_model("vision_student_teacher_lite", **kwargs)
    detector.fit(train_paths)
    expected_scores = detector.decision_function(eval_paths)
    expected_threshold = float(detector.threshold_)

    ckpt_path = tmp_path / "student_teacher.ckpt"
    save_checkpoint(detector, ckpt_path)

    restored = create_model("vision_student_teacher_lite", **kwargs)
    load_checkpoint_into_detector(restored, ckpt_path)

    restored_scores = restored.decision_function(eval_paths)
    np.testing.assert_allclose(restored_scores, expected_scores, rtol=1e-6, atol=1e-6)
    assert float(restored.threshold_) == pytest.approx(expected_threshold)
