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
                "defects": {"pixel_threshold": 0.5, "pixel_threshold_strategy": "fixed"},
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
                    "pixel_threshold_strategy": "fixed",
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
