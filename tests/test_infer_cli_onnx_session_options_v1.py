from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import pyimgano.infer_cli as infer_cli


def _write_png(path: Path) -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


def test_infer_cli_onnx_session_options_shorthand_passes_to_model_kwargs(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    captured: dict[str, object] = {"kwargs": None}

    class _OK:
        def __init__(self) -> None:
            self.threshold_ = 0.5

        def decision_function(self, X):
            return np.asarray([0.1 for _ in list(X)], dtype=np.float32)

    def _create_model(name: str, **kwargs):  # noqa: ANN001, ANN201 - test stub
        _ = name
        captured["kwargs"] = dict(kwargs)
        return _OK()

    monkeypatch.setattr(infer_cli, "create_model", _create_model)

    rc = infer_cli.main(
        [
            "--model",
            "vision_onnx_ecod",
            "--checkpoint-path",
            str(tmp_path / "model.onnx"),
            "--onnx-session-options",
            json.dumps(
                {
                    "intra_op_num_threads": 2,
                    "graph_optimization_level": "all",
                    "enable_mem_pattern": False,
                }
            ),
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs.get("checkpoint_path") is not None
    assert kwargs.get("session_options") == {
        "intra_op_num_threads": 2,
        "graph_optimization_level": "all",
        "enable_mem_pattern": False,
    }


def test_infer_cli_onnx_session_options_shorthand_delegates_to_model_options(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    captured: dict[str, object] = {"kwargs": None}

    class _OK:
        def __init__(self) -> None:
            self.threshold_ = 0.5

        def decision_function(self, X):
            return np.asarray([0.1 for _ in list(X)], dtype=np.float32)

    monkeypatch.setattr(
        infer_cli,
        "create_model",
        lambda name, **kwargs: captured.update(kwargs=dict(kwargs)) or _OK(),
    )

    import pyimgano.services.model_options as model_options

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        model_options,
        "apply_onnx_session_options_shorthand",
        lambda *, model_name, user_kwargs, session_options: (
            calls.append(
                {
                    "model_name": model_name,
                    "user_kwargs": dict(user_kwargs),
                    "session_options": dict(session_options or {}),
                }
            )
            or {**dict(user_kwargs), "session_options": {"delegated": True}}
        ),
    )

    rc = infer_cli.main(
        [
            "--model",
            "vision_onnx_ecod",
            "--checkpoint-path",
            str(tmp_path / "model.onnx"),
            "--onnx-session-options",
            json.dumps({"intra_op_num_threads": 2}),
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert len(calls) == 1
    assert calls[0]["model_name"] == "vision_onnx_ecod"
    assert captured["kwargs"]["session_options"] == {"delegated": True}


def test_infer_cli_onnx_sweep_selects_best_session_options_and_applies_to_model(
    tmp_path: Path, monkeypatch
) -> None:
    """Sweep should run without real onnxruntime and apply selected session_options."""

    # Minimal checkpoint for the ONNX embed extractor (sweep uses ONNXEmbedExtractor directly).
    ckpt = tmp_path / "model.onnx"
    ckpt.write_text("fake", encoding="utf-8")

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")

    out_jsonl = tmp_path / "out.jsonl"

    import pyimgano.features.onnx_embed as onnx_embed

    @dataclass(frozen=True)
    class _IO:
        name: str

    class _GraphOptimizationLevel:
        ORT_DISABLE_ALL = 0
        ORT_ENABLE_BASIC = 1
        ORT_ENABLE_EXTENDED = 2
        ORT_ENABLE_ALL = 3

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

        def add_session_config_entry(self, key: str, value: str) -> None:  # noqa: ARG002
            return None

    class _InferenceSession:
        def __init__(self, _path: str, *, providers, sess_options) -> None:  # noqa: ANN001
            _ = _path
            _ = providers
            self._sess_options = sess_options

        def get_inputs(self):
            return [_IO(name="input")]

        def get_outputs(self):
            return [_IO(name="output")]

        def run(self, _output_names, feed):  # noqa: ANN001
            batch = int(np.asarray(next(iter(feed.values()))).shape[0])
            intra = int(getattr(self._sess_options, "intra_op_num_threads", 0))
            opt = getattr(self._sess_options, "graph_optimization_level", None)
            opt_penalty = 0 if opt == _GraphOptimizationLevel.ORT_ENABLE_ALL else 1
            work = (intra - 2) ** 2 + opt_penalty
            # Use wall-clock delay instead of CPU busy work so suite load does not
            # scramble the intended ordering of candidates.
            time.sleep(0.002 + (work * 0.02))
            return [np.zeros((batch, 8), dtype=np.float32)]

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

    captured: dict[str, object] = {"kwargs": None}

    class _OK:
        def __init__(self) -> None:
            self.threshold_ = 0.5

        def decision_function(self, X):
            return np.asarray([0.1 for _ in list(X)], dtype=np.float32)

    def _create_model(name: str, **kwargs):  # noqa: ANN001, ANN201 - test stub
        _ = name
        captured["kwargs"] = dict(kwargs)
        return _OK()

    monkeypatch.setattr(infer_cli, "create_model", _create_model)

    sweep_json = tmp_path / "sweep.json"
    rc = infer_cli.main(
        [
            "--model",
            "vision_onnx_ecod",
            "--checkpoint-path",
            str(ckpt),
            "--model-kwargs",
            json.dumps({"image_size": 32, "batch_size": 1}),
            "--onnx-sweep",
            "--onnx-sweep-intra",
            "1,2,4",
            "--onnx-sweep-opt-levels",
            "basic,all",
            "--onnx-sweep-repeats",
            "2",
            "--onnx-sweep-samples",
            "2",
            "--onnx-sweep-json",
            str(sweep_json),
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0

    # Best should be intra=2, opt=all (work=0) based on our FakeORT timing.
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    sess_opts = kwargs.get("session_options")
    assert isinstance(sess_opts, dict)
    assert int(sess_opts.get("intra_op_num_threads")) == 2
    assert str(sess_opts.get("graph_optimization_level")) == "all"

    assert sweep_json.exists()
