from __future__ import annotations

from pathlib import Path

import pytest

from pyimgano.training.checkpointing import save_checkpoint


class _SerializableDetector:
    def __init__(self) -> None:
        self.name = "vision-ecod-like"
        self.state = {"threshold": 0.7, "scores": [0.1, 0.2, 0.3]}


def test_save_checkpoint_prefers_detector_method(tmp_path):
    class _Detector:
        def save_checkpoint(self, path):  # noqa: ANN001 - test stub
            Path(path).write_text("ok", encoding="utf-8")

    out = tmp_path / "ckpt.pt"
    saved = save_checkpoint(_Detector(), out)
    assert saved == out
    assert out.read_text(encoding="utf-8") == "ok"


def test_save_checkpoint_falls_back_to_torch_state_dict(tmp_path):
    torch = pytest.importorskip("torch")

    class _Detector:
        def __init__(self):
            self.model = torch.nn.Linear(4, 2)

    out = tmp_path / "model.pt"
    saved = save_checkpoint(_Detector(), out)
    assert saved == out
    assert out.exists()

    state = torch.load(out, map_location="cpu")
    assert "weight" in state
    assert "bias" in state


def test_save_checkpoint_falls_back_to_joblib_detector_serialization(tmp_path):
    out = tmp_path / "model.pt"
    saved = save_checkpoint(_SerializableDetector(), out)
    assert saved == out
    assert out.exists()


def test_save_checkpoint_unwraps_runtime_tiling_wrapper_before_joblib_fallback(tmp_path):
    from pyimgano.inference.tiling import TiledDetector
    from pyimgano.models.serialization import load_model

    wrapped = TiledDetector(detector=_SerializableDetector(), tile_size=4, stride=4)

    out = tmp_path / "wrapped.pt"
    saved = save_checkpoint(wrapped, out)
    assert saved == out

    loaded = load_model(out)
    assert isinstance(loaded, _SerializableDetector)


def test_save_checkpoint_raises_when_unsupported(tmp_path):
    class _Detector:
        def __getstate__(self):
            raise RuntimeError("no pickle")

    with pytest.raises(NotImplementedError) as exc:
        save_checkpoint(_Detector(), tmp_path / "nope.pt")

    msg = str(exc.value)
    assert "save_checkpoint" in msg or "state_dict" in msg or "joblib" in msg
