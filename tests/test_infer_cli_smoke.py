import json
from pathlib import Path

import numpy as np
from PIL import Image

import pyimgano.infer_cli as infer_cli


class _DummyDetector:
    def __init__(self) -> None:
        self.threshold_ = 0.5
        self.fit_calls = 0

    def fit(self, X):
        _ = X
        self.fit_calls += 1
        return self

    def decision_function(self, X):
        return np.linspace(0.0, 1.0, num=len(X), dtype=np.float32)

    def get_anomaly_map(self, item):
        _ = item
        return np.zeros((4, 4), dtype=np.float32)


def _write_png(path: Path) -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


def test_infer_cli_smoke(tmp_path, monkeypatch):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    _write_png(train_dir / "train.png")

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")

    out_jsonl = tmp_path / "out.jsonl"
    maps_dir = tmp_path / "maps"

    det = _DummyDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--model",
            "vision_ecod",
            "--train-dir",
            str(train_dir),
            "--calibration-quantile",
            "0.95",
            "--input",
            str(input_dir),
            "--include-maps",
            "--save-maps",
            str(maps_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert det.fit_calls == 1

    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    record = json.loads(lines[0])
    assert record["index"] == 0
    assert "input" in record
    assert isinstance(record["score"], float)
    assert record["label"] in (0, 1)
    assert "anomaly_map" in record
    assert "path" in record["anomaly_map"]

    saved = sorted(maps_dir.glob("*.npy"))
    assert len(saved) == 2


def test_infer_cli_train_dir_auto_calibrates_when_threshold_missing(tmp_path, monkeypatch):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    _write_png(train_dir / "train.png")

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")

    out_jsonl = tmp_path / "out.jsonl"

    class _NoThresholdDetector:
        def __init__(self) -> None:
            self.fit_calls = 0

        def fit(self, X):
            _ = X
            self.fit_calls += 1
            return self

        def decision_function(self, X):
            return np.linspace(0.0, 1.0, num=len(X), dtype=np.float32)

    det = _NoThresholdDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--model",
            "vision_ecod",
            "--train-dir",
            str(train_dir),
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert det.fit_calls == 1

    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert "label" in record
