from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import pyimgano.infer_cli as infer_cli


def _write_png(path: Path) -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


def test_infer_cli_supports_infer_config(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    ckpt_dir = run_dir / "checkpoints" / "custom"
    artifacts.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "model.pt"
    ckpt_path.write_text("ckpt", encoding="utf-8")

    infer_cfg_path = artifacts / "infer_config.json"
    infer_cfg_path.write_text(
        json.dumps(
            {
                "from_run": str(run_dir),
                "category": "custom",
                "model": {
                    "name": "vision_ecod",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                    "preset": None,
                    "model_kwargs": {},
                    "checkpoint_path": None,
                },
                "adaptation": {
                    "tiling": {
                        "tile_size": None,
                        "stride": None,
                        "score_reduce": "max",
                        "score_topk": 0.1,
                        "map_reduce": "max",
                    },
                    "postprocess": None,
                    "save_maps": False,
                },
                "threshold": 0.7,
                "checkpoint": {"path": "checkpoints/custom/model.pt"},
            }
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
            return np.linspace(0.0, 1.0, num=len(list(X)), dtype=np.float32)

    det = _DummyDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--infer-config",
            str(infer_cfg_path),
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0

    assert det.loaded == str(ckpt_path)
    assert det.threshold_ == 0.7

    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["label"] == 0
    assert second["label"] == 1


def test_infer_cli_supports_infer_config_preprocessing(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    ckpt_dir = run_dir / "checkpoints" / "custom"
    artifacts.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "model.pt"
    ckpt_path.write_text("ckpt", encoding="utf-8")

    infer_cfg_path = artifacts / "infer_config.json"
    infer_cfg_path.write_text(
        json.dumps(
            {
                "from_run": str(run_dir),
                "category": "custom",
                "model": {
                    "name": "vision_patchcore",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                    "preset": None,
                    "model_kwargs": {},
                    "checkpoint_path": None,
                },
                "preprocessing": {
                    "illumination_contrast": {
                        "white_balance": "gray_world",
                    }
                },
                "adaptation": {
                    "tiling": {
                        "tile_size": None,
                        "stride": None,
                        "score_reduce": "max",
                        "score_topk": 0.1,
                        "map_reduce": "max",
                    },
                    "postprocess": None,
                    "save_maps": False,
                },
                "threshold": 0.7,
                "checkpoint": {"path": "checkpoints/custom/model.pt"},
            }
        ),
        encoding="utf-8",
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")

    out_jsonl = tmp_path / "out.jsonl"

    class _NumpyOnlyDetector:
        def __init__(self):
            self.threshold_ = None
            self.loaded = None

        def load_checkpoint(self, path):  # noqa: ANN001 - test stub
            self.loaded = str(path)

        def decision_function(self, X):  # noqa: ANN001
            items = list(X)
            assert items
            assert all(isinstance(x, np.ndarray) for x in items)
            assert all(x.dtype == np.uint8 for x in items)
            assert all(x.ndim == 3 and x.shape[2] == 3 for x in items)
            return np.linspace(0.0, 1.0, num=len(items), dtype=np.float32)

    det = _NumpyOnlyDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--infer-config",
            str(infer_cfg_path),
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0

    assert det.loaded == str(ckpt_path)
    assert det.threshold_ == 0.7


def test_infer_cli_errors_when_preprocessing_enabled_on_non_numpy_model(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    ckpt_dir = run_dir / "checkpoints" / "custom"
    artifacts.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "model.pt"
    ckpt_path.write_text("ckpt", encoding="utf-8")

    infer_cfg_path = artifacts / "infer_config.json"
    infer_cfg_path.write_text(
        json.dumps(
            {
                "from_run": str(run_dir),
                "category": "custom",
                "model": {
                    "name": "vision_ecod",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                    "preset": None,
                    "model_kwargs": {},
                    "checkpoint_path": None,
                },
                "preprocessing": {
                    "illumination_contrast": {
                        "white_balance": "gray_world",
                    }
                },
                "adaptation": {
                    "tiling": {
                        "tile_size": None,
                        "stride": None,
                        "score_reduce": "max",
                        "score_topk": 0.1,
                        "map_reduce": "max",
                    },
                    "postprocess": None,
                    "save_maps": False,
                },
                "threshold": 0.7,
                "checkpoint": {"path": "checkpoints/custom/model.pt"},
            }
        ),
        encoding="utf-8",
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")

    out_jsonl = tmp_path / "out.jsonl"

    class _NumpyOnlyDetector:
        def __init__(self):
            self.threshold_ = None
            self.loaded = None

        def load_checkpoint(self, path):  # noqa: ANN001 - test stub
            self.loaded = str(path)

        def decision_function(self, X):  # noqa: ANN001
            items = list(X)
            assert items
            assert all(isinstance(x, np.ndarray) for x in items)
            assert all(x.dtype == np.uint8 for x in items)
            assert all(x.ndim == 3 and x.shape[2] == 3 for x in items)
            return np.linspace(0.0, 1.0, num=len(items), dtype=np.float32)

    det = _NumpyOnlyDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--infer-config",
            str(infer_cfg_path),
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "PREPROCESSING_REQUIRES_NUMPY_MODEL" in err


def test_infer_cli_supports_infer_config_defects(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    ckpt_dir = run_dir / "checkpoints" / "custom"
    artifacts.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "model.pt"
    ckpt_path.write_text("ckpt", encoding="utf-8")

    infer_cfg_path = artifacts / "infer_config.json"
    infer_cfg_path.write_text(
        json.dumps(
            {
                "from_run": str(run_dir),
                "category": "custom",
                "model": {
                    "name": "vision_ecod",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                    "preset": None,
                    "model_kwargs": {},
                    "checkpoint_path": None,
                },
                "adaptation": {"tiling": {}, "postprocess": None, "save_maps": False},
                "threshold": 0.7,
                "checkpoint": {"path": "checkpoints/custom/model.pt"},
                "defects": {"pixel_threshold": 0.5},
            }
        ),
        encoding="utf-8",
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")

    out_jsonl = tmp_path / "out.jsonl"
    masks_dir = tmp_path / "masks"

    class _DummyMapDetector:
        def __init__(self):
            self.threshold_ = None
            self.loaded = None

        def load_checkpoint(self, path):  # noqa: ANN001 - test stub
            self.loaded = str(path)

        def decision_function(self, X):  # noqa: ANN001
            return np.linspace(0.0, 1.0, num=len(list(X)), dtype=np.float32)

        def get_anomaly_map(self, item):  # noqa: ANN001 - test stub
            _ = item
            m = np.zeros((4, 4), dtype=np.float32)
            m[1:3, 1:3] = 1.0
            return m

    det = _DummyMapDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--infer-config",
            str(infer_cfg_path),
            "--input",
            str(input_dir),
            "--defects",
            "--save-masks",
            str(masks_dir),
            "--mask-format",
            "png",
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert det.loaded == str(ckpt_path)
    assert det.threshold_ == 0.7

    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["defects"]["pixel_threshold"] == 0.5
    assert first["defects"]["pixel_threshold_provenance"]["source"] == "infer_config"
    assert len(first["defects"]["regions"]) == 1

    saved_masks = sorted(masks_dir.glob("*.png"))
    assert len(saved_masks) == 2


def test_infer_cli_supports_infer_config_defects_border_ignore_px(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    ckpt_dir = run_dir / "checkpoints" / "custom"
    artifacts.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "model.pt"
    ckpt_path.write_text("ckpt", encoding="utf-8")

    infer_cfg_path = artifacts / "infer_config.json"
    infer_cfg_path.write_text(
        json.dumps(
            {
                "from_run": str(run_dir),
                "category": "custom",
                "model": {
                    "name": "vision_ecod",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                    "preset": None,
                    "model_kwargs": {},
                    "checkpoint_path": None,
                },
                "adaptation": {"tiling": {}, "postprocess": None, "save_maps": False},
                "threshold": 0.7,
                "checkpoint": {"path": "checkpoints/custom/model.pt"},
                "defects": {"pixel_threshold": 0.5, "border_ignore_px": 1},
            }
        ),
        encoding="utf-8",
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"
    masks_dir = tmp_path / "masks"

    class _DummyMapDetector:
        def __init__(self):
            self.threshold_ = None
            self.loaded = None

        def load_checkpoint(self, path):  # noqa: ANN001 - test stub
            self.loaded = str(path)

        def decision_function(self, X):  # noqa: ANN001
            return np.linspace(0.0, 1.0, num=len(list(X)), dtype=np.float32)

        def get_anomaly_map(self, item):  # noqa: ANN001 - test stub
            _ = item
            m = np.zeros((5, 5), dtype=np.float32)
            m[0, 0] = 1.0  # border FP candidate
            m[2:4, 2:4] = 1.0  # real defect (not 8-connected to (0,0))
            return m

    det = _DummyMapDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--infer-config",
            str(infer_cfg_path),
            "--input",
            str(input_dir),
            "--defects",
            "--save-masks",
            str(masks_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0

    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    first = json.loads(lines[0])
    assert len(first["defects"]["regions"]) == 1


def test_infer_cli_supports_infer_config_defects_map_smoothing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    ckpt_dir = run_dir / "checkpoints" / "custom"
    artifacts.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "model.pt"
    ckpt_path.write_text("ckpt", encoding="utf-8")

    infer_cfg_path = artifacts / "infer_config.json"
    infer_cfg_path.write_text(
        json.dumps(
            {
                "from_run": str(run_dir),
                "category": "custom",
                "model": {
                    "name": "vision_ecod",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                    "preset": None,
                    "model_kwargs": {},
                    "checkpoint_path": None,
                },
                "adaptation": {"tiling": {}, "postprocess": None, "save_maps": False},
                "threshold": 0.7,
                "checkpoint": {"path": "checkpoints/custom/model.pt"},
                "defects": {
                    "pixel_threshold": 0.5,
                    "map_smoothing": {"method": "median", "ksize": 3, "sigma": 0.0},
                },
            }
        ),
        encoding="utf-8",
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    class _DummyMapDetector:
        def __init__(self):
            self.threshold_ = None
            self.loaded = None

        def load_checkpoint(self, path):  # noqa: ANN001 - test stub
            self.loaded = str(path)

        def decision_function(self, X):  # noqa: ANN001
            return np.linspace(0.0, 1.0, num=len(list(X)), dtype=np.float32)

        def get_anomaly_map(self, item):  # noqa: ANN001 - test stub
            _ = item
            m = np.zeros((9, 9), dtype=np.float32)
            m[1, 1] = 1.0  # isolated noise pixel
            m[5:8, 5:8] = 1.0  # defect blob
            return m

    det = _DummyMapDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--infer-config",
            str(infer_cfg_path),
            "--input",
            str(input_dir),
            "--defects",
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0

    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    first = json.loads(lines[0])
    assert len(first["defects"]["regions"]) == 1


def test_infer_cli_supports_infer_config_defects_hysteresis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    ckpt_dir = run_dir / "checkpoints" / "custom"
    artifacts.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "model.pt"
    ckpt_path.write_text("ckpt", encoding="utf-8")

    infer_cfg_path = artifacts / "infer_config.json"
    infer_cfg_path.write_text(
        json.dumps(
            {
                "from_run": str(run_dir),
                "category": "custom",
                "model": {
                    "name": "vision_ecod",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                    "preset": None,
                    "model_kwargs": {},
                    "checkpoint_path": None,
                },
                "adaptation": {"tiling": {}, "postprocess": None, "save_maps": False},
                "threshold": 0.7,
                "checkpoint": {"path": "checkpoints/custom/model.pt"},
                "defects": {
                    "pixel_threshold": 0.9,
                    "hysteresis": {"enabled": True, "low": 0.5, "high": 0.9},
                },
            }
        ),
        encoding="utf-8",
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    class _DummyMapDetector:
        def __init__(self):
            self.threshold_ = None
            self.loaded = None

        def load_checkpoint(self, path):  # noqa: ANN001 - test stub
            self.loaded = str(path)

        def decision_function(self, X):  # noqa: ANN001
            return np.linspace(0.0, 1.0, num=len(list(X)), dtype=np.float32)

        def get_anomaly_map(self, item):  # noqa: ANN001 - test stub
            _ = item
            m = np.zeros((5, 5), dtype=np.float32)
            m[2, 2] = 1.0  # high seed
            m[2, 3] = 0.6  # low pixel connected to seed (should be kept)
            return m

    det = _DummyMapDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--infer-config",
            str(infer_cfg_path),
            "--input",
            str(input_dir),
            "--defects",
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0

    first = json.loads(out_jsonl.read_text(encoding="utf-8").strip().splitlines()[0])
    assert len(first["defects"]["regions"]) == 1
    assert first["defects"]["regions"][0]["bbox_xyxy"] == [2, 2, 3, 2]


def test_infer_cli_supports_infer_config_defects_shape_filters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    ckpt_dir = run_dir / "checkpoints" / "custom"
    artifacts.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "model.pt"
    ckpt_path.write_text("ckpt", encoding="utf-8")

    infer_cfg_path = artifacts / "infer_config.json"
    infer_cfg_path.write_text(
        json.dumps(
            {
                "from_run": str(run_dir),
                "category": "custom",
                "model": {
                    "name": "vision_ecod",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                    "preset": None,
                    "model_kwargs": {},
                    "checkpoint_path": None,
                },
                "adaptation": {"tiling": {}, "postprocess": None, "save_maps": False},
                "threshold": 0.7,
                "checkpoint": {"path": "checkpoints/custom/model.pt"},
                "defects": {
                    "pixel_threshold": 0.5,
                    "shape_filters": {"max_aspect_ratio": 3.0},
                },
            }
        ),
        encoding="utf-8",
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    class _DummyMapDetector:
        def __init__(self):
            self.threshold_ = None
            self.loaded = None

        def load_checkpoint(self, path):  # noqa: ANN001 - test stub
            self.loaded = str(path)

        def decision_function(self, X):  # noqa: ANN001
            return np.linspace(0.0, 1.0, num=len(list(X)), dtype=np.float32)

        def get_anomaly_map(self, item):  # noqa: ANN001 - test stub
            _ = item
            m = np.zeros((10, 10), dtype=np.float32)
            m[1, 1:7] = 1.0  # long thin line (filtered)
            m[5:8, 5:8] = 1.0  # square (kept)
            return m

    det = _DummyMapDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--infer-config",
            str(infer_cfg_path),
            "--input",
            str(input_dir),
            "--defects",
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0

    first = json.loads(out_jsonl.read_text(encoding="utf-8").strip().splitlines()[0])
    assert len(first["defects"]["regions"]) == 1
    assert first["defects"]["regions"][0]["bbox_xyxy"] == [5, 5, 7, 7]


def test_infer_cli_supports_infer_config_defects_merge_nearby(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    ckpt_dir = run_dir / "checkpoints" / "custom"
    artifacts.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "model.pt"
    ckpt_path.write_text("ckpt", encoding="utf-8")

    infer_cfg_path = artifacts / "infer_config.json"
    infer_cfg_path.write_text(
        json.dumps(
            {
                "from_run": str(run_dir),
                "category": "custom",
                "model": {
                    "name": "vision_ecod",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                    "preset": None,
                    "model_kwargs": {},
                    "checkpoint_path": None,
                },
                "adaptation": {"tiling": {}, "postprocess": None, "save_maps": False},
                "threshold": 0.7,
                "checkpoint": {"path": "checkpoints/custom/model.pt"},
                "defects": {
                    "pixel_threshold": 0.5,
                    "merge_nearby": {"enabled": True, "max_gap_px": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    class _DummyMapDetector:
        def __init__(self):
            self.threshold_ = None
            self.loaded = None

        def load_checkpoint(self, path):  # noqa: ANN001 - test stub
            self.loaded = str(path)

        def decision_function(self, X):  # noqa: ANN001
            return np.linspace(0.0, 1.0, num=len(list(X)), dtype=np.float32)

        def get_anomaly_map(self, item):  # noqa: ANN001 - test stub
            _ = item
            m = np.zeros((10, 10), dtype=np.float32)
            m[2:4, 2:4] = 1.0
            m[2:4, 5:7] = 1.0  # 1px gap between bboxes
            return m

    det = _DummyMapDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--infer-config",
            str(infer_cfg_path),
            "--input",
            str(input_dir),
            "--defects",
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0

    first = json.loads(out_jsonl.read_text(encoding="utf-8").strip().splitlines()[0])
    assert len(first["defects"]["regions"]) == 1
    assert first["defects"]["regions"][0]["bbox_xyxy"] == [2, 2, 6, 3]
    assert first["defects"]["regions"][0]["merged_from_ids"] == [1, 2]


def test_infer_cli_infer_config_recalibrates_pixel_threshold_when_train_dir_provided(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    ckpt_dir = run_dir / "checkpoints" / "custom"
    artifacts.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "model.pt"
    ckpt_path.write_text("ckpt", encoding="utf-8")

    infer_cfg_path = artifacts / "infer_config.json"
    infer_cfg_path.write_text(
        json.dumps(
            {
                "from_run": str(run_dir),
                "category": "custom",
                "model": {
                    "name": "vision_ecod",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                    "preset": None,
                    "model_kwargs": {},
                    "checkpoint_path": None,
                },
                "adaptation": {"tiling": {}, "postprocess": None, "save_maps": False},
                "threshold": 0.7,
                "checkpoint": {"path": "checkpoints/custom/model.pt"},
                "defects": {"pixel_threshold": 0.9},
            }
        ),
        encoding="utf-8",
    )

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    _write_png(train_dir / "train0.png")
    _write_png(train_dir / "train1.png")

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    class _DummyMapDetector:
        def __init__(self):
            self.threshold_ = None
            self.loaded = None

        def load_checkpoint(self, path):  # noqa: ANN001 - test stub
            self.loaded = str(path)

        def fit(self, X):  # noqa: ANN001
            _ = X
            return self

        def decision_function(self, X):  # noqa: ANN001
            return np.linspace(0.0, 1.0, num=len(list(X)), dtype=np.float32)

        def get_anomaly_map(self, item):  # noqa: ANN001 - test stub
            s = str(item)
            if "train0" in s:
                return np.zeros((4, 4), dtype=np.float32)
            if "train1" in s:
                return np.ones((4, 4), dtype=np.float32)
            return np.zeros((4, 4), dtype=np.float32)

    det = _DummyMapDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--infer-config",
            str(infer_cfg_path),
            "--train-dir",
            str(train_dir),
            "--input",
            str(input_dir),
            "--defects",
            "--pixel-normal-quantile",
            "0.5",
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0

    record = json.loads(out_jsonl.read_text(encoding="utf-8").strip().splitlines()[0])
    defects = record["defects"]
    assert defects["pixel_threshold"] == 0.5
    assert defects["pixel_threshold_provenance"]["method"] == "normal_pixel_quantile"
    assert defects["pixel_threshold_provenance"]["source"] == "train_dir"


def test_infer_cli_infer_config_applies_defects_defaults(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    ckpt_dir = run_dir / "checkpoints" / "custom"
    artifacts.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "model.pt"
    ckpt_path.write_text("ckpt", encoding="utf-8")

    infer_cfg_path = artifacts / "infer_config.json"
    infer_cfg_path.write_text(
        json.dumps(
            {
                "from_run": str(run_dir),
                "category": "custom",
                "model": {
                    "name": "vision_ecod",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                    "preset": None,
                    "model_kwargs": {},
                    "checkpoint_path": None,
                },
                "adaptation": {"tiling": {}, "postprocess": None, "save_maps": False},
                "threshold": 0.7,
                "checkpoint": {"path": "checkpoints/custom/model.pt"},
                "defects": {
                    "pixel_threshold": 0.5,
                    "roi_xyxy_norm": [0.25, 0.25, 0.75, 0.75],
                    "mask_format": "npy",
                },
            }
        ),
        encoding="utf-8",
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"
    masks_dir = tmp_path / "masks"

    class _DummyMapDetector:
        def __init__(self):
            self.threshold_ = None
            self.loaded = None

        def load_checkpoint(self, path):  # noqa: ANN001 - test stub
            self.loaded = str(path)

        def decision_function(self, X):  # noqa: ANN001
            return np.linspace(0.0, 1.0, num=len(list(X)), dtype=np.float32)

        def get_anomaly_map(self, item):  # noqa: ANN001 - test stub
            _ = item
            m = np.zeros((4, 4), dtype=np.float32)
            m[0, 3] = 1.0  # hotspot outside center ROI
            m[2, 2] = 1.0  # hotspot inside center ROI
            return m

    det = _DummyMapDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--infer-config",
            str(infer_cfg_path),
            "--input",
            str(input_dir),
            "--defects",
            "--save-masks",
            str(masks_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0

    record = json.loads(out_jsonl.read_text(encoding="utf-8").strip().splitlines()[0])
    defects = record["defects"]
    assert defects["pixel_threshold"] == 0.5
    assert defects["pixel_threshold_provenance"]["source"] == "infer_config"
    assert defects["mask"]["encoding"] == "npy"
    assert len(defects["regions"]) == 1

    saved_masks = sorted(masks_dir.glob("*.npy"))
    assert len(saved_masks) == 1


def test_infer_cli_applies_tiling_defaults_from_infer_config(tmp_path: Path, monkeypatch) -> None:
    import pyimgano.inference.tiling as tiling

    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    infer_cfg_path = artifacts / "infer_config.json"
    infer_cfg_path.write_text(
        json.dumps(
            {
                "category": "custom",
                "model": {
                    "name": "vision_ecod",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                    "preset": None,
                    "model_kwargs": {},
                    "checkpoint_path": None,
                },
                "adaptation": {
                    "tiling": {
                        "tile_size": 4,
                        "stride": 3,
                        "score_reduce": "topk_mean",
                        "score_topk": 0.2,
                        "map_reduce": "hann",
                    },
                    "postprocess": None,
                    "save_maps": False,
                },
                "threshold": 0.7,
            }
        ),
        encoding="utf-8",
    )

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    class _DummyDetector:
        def __init__(self):
            self.threshold_ = None

        def decision_function(self, X):  # noqa: ANN001 - test stub
            return np.linspace(0.0, 1.0, num=len(list(X)), dtype=np.float32)

    det = _DummyDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    constructed: dict[str, object] = {}

    class _FakeTiledDetector:
        def __init__(
            self,
            *,
            detector,
            tile_size: int,
            stride: int | None,
            score_reduce: str,
            score_topk: float,
            map_reduce: str,
        ) -> None:
            constructed.update(
                {
                    "tile_size": int(tile_size),
                    "stride": (int(stride) if stride is not None else None),
                    "score_reduce": str(score_reduce),
                    "score_topk": float(score_topk),
                    "map_reduce": str(map_reduce),
                }
            )
            self.detector = detector

        def decision_function(self, X):  # noqa: ANN001 - test stub
            return self.detector.decision_function(X)

        def __getattr__(self, name: str):  # noqa: D401 - test stub
            return getattr(self.detector, name)

    monkeypatch.setattr(tiling, "TiledDetector", _FakeTiledDetector)

    rc = infer_cli.main(
        [
            "--infer-config",
            str(infer_cfg_path),
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert constructed == {
        "tile_size": 4,
        "stride": 3,
        "score_reduce": "topk_mean",
        "score_topk": 0.2,
        "map_reduce": "hann",
    }


def test_infer_cli_infer_config_requires_category_when_ambiguous(tmp_path: Path, capsys) -> None:
    cfg_path = tmp_path / "infer_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "category": "all",
                "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
                "adaptation": {"tiling": {}, "postprocess": None, "save_maps": False},
                "per_category": {"a": {"threshold": 0.1}, "b": {"threshold": 0.2}},
            }
        ),
        encoding="utf-8",
    )

    rc = infer_cli.main(["--infer-config", str(cfg_path), "--input", str(tmp_path)])
    assert rc == 2
    err = capsys.readouterr().err.lower()
    assert "infer-category" in err
