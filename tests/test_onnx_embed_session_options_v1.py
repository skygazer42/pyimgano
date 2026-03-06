from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest


def test_onnx_embed_extractor_applies_session_options(monkeypatch, tmp_path: Path) -> None:
    """SessionOptions plumbing should be deterministic and not require real onnxruntime."""

    import pyimgano.features.onnx_embed as onnx_embed
    from pyimgano.features.onnx_embed import ONNXEmbedExtractor

    captured = {"sess_options": None}

    class _GraphOptimizationLevel:
        ORT_DISABLE_ALL = 0
        ORT_ENABLE_BASIC = 1
        ORT_ENABLE_EXTENDED = 2
        ORT_ENABLE_ALL = 99

    class _ExecutionMode:
        ORT_SEQUENTIAL = 0
        ORT_PARALLEL = 1

    class _SessionOptions:
        def __init__(self) -> None:
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0
            self.execution_mode = None
            self.graph_optimization_level = None
            self.enable_mem_pattern = True
            self.enable_cpu_mem_arena = True
            self._entries: dict[str, str] = {}

        def add_session_config_entry(self, key: str, value: str) -> None:
            self._entries[str(key)] = str(value)

    @dataclass(frozen=True)
    class _IO:
        name: str

    class _InferenceSession:
        def __init__(self, _path: str, *, providers, sess_options) -> None:  # noqa: ANN001
            _ = providers
            captured["sess_options"] = sess_options

        def get_inputs(self):
            return [_IO(name="input")]

        def get_outputs(self):
            return [_IO(name="output")]

    class _FakeORT:
        __version__ = "0.0"
        SessionOptions = _SessionOptions
        InferenceSession = _InferenceSession
        GraphOptimizationLevel = _GraphOptimizationLevel
        ExecutionMode = _ExecutionMode

        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(onnx_embed, "require", lambda *a, **k: _FakeORT)

    ckpt = tmp_path / "model.onnx"
    ckpt.write_text("fake", encoding="utf-8")

    extractor = ONNXEmbedExtractor(
        checkpoint_path=str(ckpt),
        device="cpu",
        batch_size=1,
        image_size=224,
        session_options={
            "intra_op_num_threads": 2,
            "inter_op_num_threads": 1,
            "execution_mode": "sequential",
            "graph_optimization_level": "all",
            "enable_mem_pattern": False,
            "enable_cpu_mem_arena": False,
            "session_config_entries": {"session.set_denormal_as_zero": "1"},
        },
    )

    extractor._ensure_ready()

    sess_options = captured["sess_options"]
    assert sess_options is not None
    assert sess_options.intra_op_num_threads == 2
    assert sess_options.inter_op_num_threads == 1
    assert sess_options.execution_mode == _ExecutionMode.ORT_SEQUENTIAL
    assert sess_options.graph_optimization_level == _GraphOptimizationLevel.ORT_ENABLE_ALL
    assert sess_options.enable_mem_pattern is False
    assert sess_options.enable_cpu_mem_arena is False
    assert sess_options._entries["session.set_denormal_as_zero"] == "1"


def test_onnx_embed_extractor_rejects_unknown_session_options_key(tmp_path: Path) -> None:
    from pyimgano.features.onnx_embed import ONNXEmbedExtractor

    ckpt = tmp_path / "model.onnx"
    ckpt.write_text("fake", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Unknown session_options key"):
        ONNXEmbedExtractor(
            checkpoint_path=str(ckpt),
            device="cpu",
            batch_size=1,
            image_size=224,
            session_options={"definitely_unknown": 1},
        )
