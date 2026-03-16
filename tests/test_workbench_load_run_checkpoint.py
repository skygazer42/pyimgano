from __future__ import annotations

import pytest


class _SerializableDetector:
    def __init__(self, marker: str) -> None:
        self.marker = marker
        self.threshold_ = None
        self.detector = {"fitted": False}

    def decision_function(self, X):  # noqa: ANN001 - test stub
        if not self.detector["fitted"]:
            raise RuntimeError("not fitted")
        return [0.5 for _ in X]


def test_load_checkpoint_into_detector_builds_model_when_missing(tmp_path):
    torch = pytest.importorskip("torch")

    from pyimgano.workbench.load_run import load_checkpoint_into_detector

    class _Detector:
        def __init__(self) -> None:
            self.model = None
            self.device = torch.device("cpu")

        def build_model(self):
            return torch.nn.Linear(4, 2)

    reference = _Detector()
    reference_model = reference.build_model()
    for param in reference_model.parameters():
        torch.nn.init.constant_(param, 0.5)

    ckpt = tmp_path / "model.pt"
    torch.save(reference_model.state_dict(), ckpt)

    det = _Detector()
    load_checkpoint_into_detector(det, ckpt)

    assert det.model is not None
    loaded_state = det.model.state_dict()
    for name, tensor in reference_model.state_dict().items():
        assert name in loaded_state
        assert torch.allclose(loaded_state[name], tensor)


def test_load_checkpoint_into_detector_restores_joblib_serialized_detector_state(tmp_path):
    from pyimgano.models.serialization import save_model
    from pyimgano.workbench.load_run import load_checkpoint_into_detector

    ckpt = tmp_path / "model.pt"
    fitted = _SerializableDetector(marker="trained")
    fitted.threshold_ = 0.42
    fitted.detector = {"fitted": True, "weights": [1, 2, 3]}
    save_model(fitted, ckpt)

    det = _SerializableDetector(marker="fresh")
    load_checkpoint_into_detector(det, ckpt)

    assert det.marker == "trained"
    assert det.threshold_ == pytest.approx(0.42)
    assert det.detector == {"fitted": True, "weights": [1, 2, 3]}
    assert det.decision_function([1, 2]) == [0.5, 0.5]


def test_load_checkpoint_into_detector_unwraps_runtime_tiling_wrapper_state(tmp_path):
    from pyimgano.inference.tiling import TiledDetector
    from pyimgano.models.serialization import save_model
    from pyimgano.workbench.load_run import load_checkpoint_into_detector

    ckpt = tmp_path / "wrapped.pt"
    fitted = _SerializableDetector(marker="trained")
    fitted.threshold_ = 0.24
    fitted.detector = {"fitted": True, "weights": [4, 5, 6]}
    save_model(TiledDetector(detector=fitted, tile_size=4, stride=4), ckpt)

    det = _SerializableDetector(marker="fresh")
    load_checkpoint_into_detector(det, ckpt)

    assert det.marker == "trained"
    assert det.threshold_ == pytest.approx(0.24)
    assert det.detector == {"fitted": True, "weights": [4, 5, 6]}
