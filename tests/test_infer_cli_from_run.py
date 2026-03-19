from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import pyimgano.infer_cli as infer_cli


def _write_png(path: Path) -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


def test_infer_cli_supports_from_run(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints" / "custom").mkdir(parents=True)

    # Minimal workbench config artifact format: {"config": {...}}.
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "config": {
                    "recipe": "industrial-adapt",
                    "dataset": {
                        "name": "custom",
                        "root": "/tmp/data",
                        "category": "custom",
                        "resize": [16, 16],
                        "input_mode": "paths",
                    },
                    "model": {
                        "name": "vision_ecod",
                        "device": "cpu",
                        "pretrained": False,
                        "contamination": 0.1,
                    },
                    "output": {
                        "save_run": True,
                        "per_image_jsonl": False,
                        "output_dir": str(run_dir),
                    },
                }
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    ckpt_path = run_dir / "checkpoints" / "custom" / "model.pt"
    ckpt_path.write_text("ckpt", encoding="utf-8")

    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "dataset": "custom",
                "category": "custom",
                "model": "vision_ecod",
                "recipe": "industrial-adapt",
                "threshold": 0.7,
                "checkpoint": {"path": "checkpoints/custom/model.pt"},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")

    out_jsonl = tmp_path / "out.jsonl"

    class _DummyDetector:
        def __init__(self):
            self.threshold_ = None
            self.loaded = None

        def load_checkpoint(self, path):  # noqa: ANN001 - test stub
            self.loaded = str(path)

        def decision_function(self, x):  # noqa: ANN001
            return np.linspace(0.0, 1.0, num=len(list(x)), dtype=np.float32)

    det = _DummyDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--from-run",
            str(run_dir),
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert det.loaded == str(ckpt_path)
    assert np.isclose(det.threshold_, 0.7)

    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["label"] == 0
    assert second["label"] == 1


def test_infer_cli_from_run_resolves_model_checkpoint_path_relative_to_run_dir(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints" / "custom").mkdir(parents=True)
    (run_dir / "artifacts").mkdir(parents=True)

    model_artifact = run_dir / "artifacts" / "backbone.onnx"
    model_artifact.write_text("onnx", encoding="utf-8")

    # Minimal workbench config artifact format: {"config": {...}}.
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "config": {
                    "recipe": "industrial-adapt",
                    "dataset": {
                        "name": "custom",
                        "root": "/tmp/data",
                        "category": "custom",
                        "resize": [16, 16],
                        "input_mode": "paths",
                    },
                    "model": {
                        "name": "vision_onnx_ecod",
                        "device": "cpu",
                        "pretrained": False,
                        "contamination": 0.1,
                        "checkpoint_path": "artifacts/backbone.onnx",
                    },
                    "output": {
                        "save_run": True,
                        "per_image_jsonl": False,
                        "output_dir": str(run_dir),
                    },
                }
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "dataset": "custom",
                "category": "custom",
                "model": "vision_onnx_ecod",
                "recipe": "industrial-adapt",
                "threshold": 0.7,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")

    out_jsonl = tmp_path / "out.jsonl"

    seen: dict[str, object] = {}

    class _DummyDetector:
        def __init__(self):
            self.threshold_ = None

        def decision_function(self, x):  # noqa: ANN001
            return np.linspace(0.0, 1.0, num=len(list(x)), dtype=np.float32)

    def _create_model(name, **kwargs):  # noqa: ANN001 - test stub
        seen["name"] = str(name)
        seen["kwargs"] = dict(kwargs)
        return _DummyDetector()

    monkeypatch.setattr(infer_cli, "create_model", _create_model)

    rc = infer_cli.main(
        [
            "--from-run",
            str(run_dir),
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert seen["name"] == "vision_onnx_ecod"
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs.get("checkpoint_path") == str(model_artifact.resolve())


def test_infer_cli_from_run_applies_tiling_defaults_from_run_config(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints" / "custom").mkdir(parents=True)

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "config": {
                    "recipe": "industrial-adapt",
                    "dataset": {
                        "name": "custom",
                        "root": "/tmp/data",
                        "category": "custom",
                        "resize": [16, 16],
                        "input_mode": "paths",
                    },
                    "model": {
                        "name": "vision_ecod",
                        "device": "cpu",
                        "pretrained": False,
                        "contamination": 0.1,
                    },
                    "adaptation": {
                        "tiling": {
                            "tile_size": 4,
                            "stride": 3,
                            "score_reduce": "topk_mean",
                            "score_topk": 0.2,
                            "map_reduce": "hann",
                        }
                    },
                    "output": {
                        "save_run": True,
                        "per_image_jsonl": False,
                        "output_dir": str(run_dir),
                    },
                }
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    ckpt_path = run_dir / "checkpoints" / "custom" / "model.pt"
    ckpt_path.write_text("ckpt", encoding="utf-8")

    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "dataset": "custom",
                "category": "custom",
                "model": "vision_ecod",
                "recipe": "industrial-adapt",
                "threshold": 0.7,
                "checkpoint": {"path": "checkpoints/custom/model.pt"},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")

    out_jsonl = tmp_path / "out.jsonl"

    constructed: dict[str, object] = {}

    class _DummyDetector:
        def __init__(self) -> None:
            self.threshold_ = None
            self.loaded = None

        def load_checkpoint(self, path):  # noqa: ANN001 - test stub
            self.loaded = str(path)

        def decision_function(self, x):  # noqa: ANN001
            return np.linspace(0.0, 1.0, num=len(list(x)), dtype=np.float32)

    class _FakeTiledDetector:
        def __init__(
            self,
            *,
            detector,  # noqa: ANN001
            tile_size: int,
            stride: int | None,
            score_reduce: str,
            score_topk: float,
            map_reduce: str,
            u16_max: int | None,
        ) -> None:
            constructed.update(
                {
                    "detector": detector,
                    "tile_size": int(tile_size),
                    "stride": (None if stride is None else int(stride)),
                    "score_reduce": str(score_reduce),
                    "score_topk": float(score_topk),
                    "map_reduce": str(map_reduce),
                    "u16_max": (None if u16_max is None else int(u16_max)),
                }
            )
            self.detector = detector

        def __getattr__(self, name: str):  # pragma: no cover - test stub
            return getattr(self.detector, name)

        def decision_function(self, x):  # noqa: ANN001
            return self.detector.decision_function(x)

    det = _DummyDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    import pyimgano.inference.tiling as tiling

    monkeypatch.setattr(tiling, "TiledDetector", _FakeTiledDetector)

    rc = infer_cli.main(
        [
            "--from-run",
            str(run_dir),
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert det.loaded == str(ckpt_path)
    assert constructed == {
        "detector": det,
        "tile_size": 4,
        "stride": 3,
        "score_reduce": "topk_mean",
        "score_topk": 0.2,
        "map_reduce": "hann",
        "u16_max": None,
    }


def test_infer_cli_from_run_delegates_postprocess_summary_to_runtime_plan(
    tmp_path: Path, monkeypatch
) -> None:
    import pyimgano.services.infer_context_service as infer_context_service
    import pyimgano.services.infer_load_service as infer_load_service
    import pyimgano.services.infer_runtime_service as infer_runtime_service
    import pyimgano.services.infer_wrapper_service as infer_wrapper_service

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    class _Det:
        threshold_ = 0.5

        def decision_function(self, X):  # noqa: ANN001
            return np.asarray([0.1 for _ in X], dtype=np.float32)

    postprocess_summary = {
        "has_defects_payload": True,
        "defects_payload_source": "from_run",
        "pixel_threshold_in_payload": True,
        "pixel_threshold_strategy": "infer_config",
        "has_prediction_policy": False,
        "has_tiling": True,
        "has_map_postprocess": False,
        "maps_enabled_by_default": False,
    }

    monkeypatch.setattr(
        infer_context_service,
        "prepare_from_run_context",
        lambda _request: infer_context_service.ConfigBackedInferContext(
            model_name="vision_patchcore",
            preset=None,
            device="cpu",
            contamination=0.1,
            pretrained=False,
            base_user_kwargs={},
            checkpoint_path=None,
            trained_checkpoint_path=None,
            threshold=0.5,
            defects_payload={"pixel_threshold": 0.3},
            prediction_payload=None,
            defects_payload_source="from_run",
            illumination_contrast_knobs=None,
            tiling_payload={"tile_size": 4, "stride": 4, "score_reduce": "max", "score_topk": 0.1, "map_reduce": "max"},
            infer_config_postprocess=None,
            enable_maps_by_default=False,
            postprocess_summary=postprocess_summary,
            warnings=(),
        ),
    )
    monkeypatch.setattr(
        infer_load_service,
        "load_config_backed_infer_detector",
        lambda request, *, create_detector=None, load_checkpoint=None: infer_load_service.ConfigBackedInferLoadResult(
            model_name="vision_patchcore",
            detector=_Det(),
            model_kwargs={},
        ),
    )
    monkeypatch.setattr(
        infer_wrapper_service,
        "apply_infer_detector_wrappers",
        lambda request: infer_wrapper_service.InferDetectorWrapperResult(
            detector=request.detector
        ),
    )

    calls: list[object] = []
    monkeypatch.setattr(
        infer_runtime_service,
        "prepare_infer_runtime_plan",
        lambda request, *, run_inference_impl=None: calls.append(request)
        or infer_runtime_service.InferRuntimePlanResult(
            include_maps=False,
            postprocess=None,
            pixel_threshold_value=0.3,
            pixel_threshold_provenance={"source": "from_run"},
        ),
    )

    rc = infer_cli.main(
        [
            "--from-run",
            str(run_dir),
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert len(calls) == 1
    assert calls[0].postprocess_summary == postprocess_summary


def test_infer_cli_from_run_applies_prediction_defaults_from_run_config(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints" / "custom").mkdir(parents=True)

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "config": {
                    "recipe": "industrial-adapt",
                    "dataset": {
                        "name": "custom",
                        "root": "/tmp/data",
                        "category": "custom",
                        "resize": [16, 16],
                        "input_mode": "paths",
                    },
                    "model": {
                        "name": "vision_ecod",
                        "device": "cpu",
                        "pretrained": False,
                        "contamination": 0.1,
                    },
                    "prediction": {
                        "reject_confidence_below": 0.75,
                        "reject_label": -9,
                    },
                    "output": {
                        "save_run": True,
                        "per_image_jsonl": False,
                        "output_dir": str(run_dir),
                    },
                }
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    ckpt_path = run_dir / "checkpoints" / "custom" / "model.pt"
    ckpt_path.write_text("ckpt", encoding="utf-8")

    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "dataset": "custom",
                "category": "custom",
                "model": "vision_ecod",
                "recipe": "industrial-adapt",
                "threshold": 0.7,
                "checkpoint": {"path": "checkpoints/custom/model.pt"},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")

    out_jsonl = tmp_path / "out.jsonl"

    class _DummyDetector:
        def __init__(self):
            self.threshold_ = None
            self.loaded = None

        def load_checkpoint(self, path):  # noqa: ANN001 - test stub
            self.loaded = str(path)

        def decision_function(self, X):  # noqa: ANN001
            return np.asarray([0.8, 0.9], dtype=np.float32)

        def predict_confidence(self, X):  # noqa: ANN001
            return np.asarray([0.6, 0.95], dtype=np.float32)

    det = _DummyDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--from-run",
            str(run_dir),
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert det.loaded == str(ckpt_path)

    rows = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["label"] == -9
    assert rows[0]["rejected"] is True
    assert rows[0]["label_confidence"] == pytest.approx(0.6)
    assert rows[1]["label"] == 1
    assert rows[1]["rejected"] is False


def test_infer_cli_from_run_supports_preprocessing_for_base_vision_models(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints" / "custom").mkdir(parents=True)

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "config": {
                    "recipe": "industrial-adapt",
                    "dataset": {
                        "name": "custom",
                        "root": "/tmp/data",
                        "category": "custom",
                        "resize": [16, 16],
                        "input_mode": "paths",
                    },
                    "model": {
                        "name": "vision_ecod",
                        "device": "cpu",
                        "pretrained": False,
                        "contamination": 0.1,
                    },
                    "preprocessing": {
                        "illumination_contrast": {
                            "white_balance": "gray_world",
                        }
                    },
                    "output": {
                        "save_run": True,
                        "per_image_jsonl": False,
                        "output_dir": str(run_dir),
                    },
                }
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    ckpt_path = run_dir / "checkpoints" / "custom" / "model.pt"
    ckpt_path.write_text("ckpt", encoding="utf-8")

    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "dataset": "custom",
                "category": "custom",
                "model": "vision_ecod",
                "recipe": "industrial-adapt",
                "threshold": 0.7,
                "checkpoint": {"path": "checkpoints/custom/model.pt"},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")

    out_jsonl = tmp_path / "out.jsonl"

    class _NumpyOnlyDetector:
        def __init__(self) -> None:
            self.threshold_ = None
            self.loaded = None

        def load_checkpoint(self, path):  # noqa: ANN001 - test stub
            self.loaded = str(path)

        def decision_function(self, x):  # noqa: ANN001
            items = list(x)
            assert items
            assert all(isinstance(x, np.ndarray) for x in items)
            assert all(x.dtype == np.uint8 for x in items)
            return np.linspace(0.0, 1.0, num=len(items), dtype=np.float32)

    det = _NumpyOnlyDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--from-run",
            str(run_dir),
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert det.loaded == str(ckpt_path)

