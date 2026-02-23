from __future__ import annotations

from pathlib import Path

import pytest

from pyimgano.training.checkpointing import save_checkpoint


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


def test_save_checkpoint_raises_when_unsupported(tmp_path):
    class _Detector:
        pass

    with pytest.raises(NotImplementedError) as exc:
        save_checkpoint(_Detector(), tmp_path / "nope.pt")

    msg = str(exc.value)
    assert "save_checkpoint" in msg or "state_dict" in msg

