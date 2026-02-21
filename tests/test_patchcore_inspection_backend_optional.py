import numpy as np
import pytest


def test_patchcore_inspection_checkpoint_wrapper_is_registered():
    from pyimgano.models import list_models

    assert "vision_patchcore_inspection_checkpoint" in list_models()


def test_patchcore_inspection_checkpoint_wrapper_requires_patchcore_if_no_inferencer():
    from pyimgano.models import create_model

    with pytest.raises(ImportError):
        create_model(
            "vision_patchcore_inspection_checkpoint",
            checkpoint_path="does-not-exist",
            device="cpu",
        )


class _FakePatchCoreInferencer:
    def __init__(self, *, scores: list[float], maps: list[np.ndarray]) -> None:
        self._scores = list(scores)
        self._maps = list(maps)

    def predict(self, images):
        # Ignore images; return fixed outputs in order for testing.
        return list(self._scores), list(self._maps)


def test_patchcore_inspection_wrapper_calibrates_threshold_and_maps(tmp_path):
    import cv2

    from pyimgano.models.patchcore_inspection_backend import VisionPatchCoreInspectionCheckpoint

    img = np.ones((32, 32, 3), dtype=np.uint8) * 128
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    cv2.imwrite(str(p1), img)
    cv2.imwrite(str(p2), img)

    inferencer = _FakePatchCoreInferencer(
        scores=[0.1, 0.9],
        maps=[
            np.zeros((224, 224), dtype=np.float32),
            np.ones((224, 224), dtype=np.float32),
        ],
    )

    model = VisionPatchCoreInspectionCheckpoint(
        checkpoint_path="ignored",
        inferencer=inferencer,
        contamination=0.5 - 1e-6,
        device="cpu",
        batch_size=8,
    )

    model.fit([str(p1), str(p2)])
    assert model.threshold_ is not None

    m = model.get_anomaly_map(str(p1))
    assert isinstance(m, np.ndarray)
    assert m.dtype == np.float32
    assert m.ndim == 2
