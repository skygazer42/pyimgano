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


def test_infer_cli_smoke_can_include_anomaly_map_values(tmp_path, monkeypatch):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    det = _DummyDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--model",
            "vision_ecod",
            "--input",
            str(input_dir),
            "--include-maps",
            "--include-anomaly-map-values",
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0

    record = json.loads(out_jsonl.read_text(encoding="utf-8").strip().splitlines()[0])
    assert "anomaly_map" in record
    assert "anomaly_map_values" in record
    assert isinstance(record["anomaly_map_values"], list)


def test_infer_cli_smoke_defects_export(tmp_path, monkeypatch):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"
    masks_dir = tmp_path / "masks"

    class _MapDetector:
        def __init__(self) -> None:
            self.threshold_ = 0.5

        def decision_function(self, X):
            return np.linspace(0.0, 1.0, num=len(X), dtype=np.float32)

        def get_anomaly_map(self, item):
            _ = item
            m = np.zeros((4, 4), dtype=np.float32)
            m[1:3, 1:3] = 1.0
            return m

    det = _MapDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--model",
            "vision_ecod",
            "--input",
            str(input_dir),
            "--defects",
            "--save-masks",
            str(masks_dir),
            "--mask-format",
            "png",
            "--pixel-threshold",
            "0.5",
            "--pixel-threshold-strategy",
            "fixed",
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0

    record = json.loads(out_jsonl.read_text(encoding="utf-8").strip().splitlines()[0])
    defects = record["defects"]
    assert defects["pixel_threshold"] == 0.5
    assert defects["pixel_threshold_provenance"]["source"] == "explicit"
    assert defects["mask"]["path"]
    assert len(defects["regions"]) == 1

    saved_masks = sorted(masks_dir.glob("*.png"))
    assert len(saved_masks) == 1


def test_infer_cli_smoke_defects_image_space_and_overlays(tmp_path, monkeypatch):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"
    masks_dir = tmp_path / "masks"
    overlays_dir = tmp_path / "overlays"

    class _MapDetector:
        def __init__(self) -> None:
            self.threshold_ = 0.5

        def decision_function(self, X):
            return np.linspace(0.0, 1.0, num=len(X), dtype=np.float32)

        def get_anomaly_map(self, item):
            _ = item
            m = np.zeros((4, 4), dtype=np.float32)
            m[1:3, 1:3] = 1.0
            return m

    det = _MapDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--model",
            "vision_ecod",
            "--input",
            str(input_dir),
            "--defects",
            "--defects-image-space",
            "--save-overlays",
            str(overlays_dir),
            "--save-masks",
            str(masks_dir),
            "--mask-format",
            "png",
            "--pixel-threshold",
            "0.5",
            "--pixel-threshold-strategy",
            "fixed",
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0

    record = json.loads(out_jsonl.read_text(encoding="utf-8").strip().splitlines()[0])
    regions = record["defects"]["regions"]
    assert len(regions) == 1
    assert regions[0]["bbox_xyxy"] == [1, 1, 2, 2]
    assert regions[0]["bbox_xyxy_image"] == [2, 2, 5, 5]

    saved_overlays = sorted(overlays_dir.glob("*.png"))
    assert len(saved_overlays) == 1
    with Image.open(saved_overlays[0]) as im:
        assert im.size == (8, 8)


def test_infer_cli_smoke_defects_roi_gates_defects_only(tmp_path, monkeypatch):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"
    masks_dir = tmp_path / "masks"

    class _ROIMapDetector:
        def __init__(self) -> None:
            self.threshold_ = 0.5

        def decision_function(self, X):
            _ = X
            return np.asarray([1.0], dtype=np.float32)

        def get_anomaly_map(self, item):
            _ = item
            m = np.zeros((4, 4), dtype=np.float32)
            m[0, 3] = 1.0  # hotspot outside ROI (right side)
            return m

    det = _ROIMapDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--model",
            "vision_ecod",
            "--input",
            str(input_dir),
            "--defects",
            "--save-masks",
            str(masks_dir),
            "--mask-format",
            "png",
            "--pixel-threshold",
            "0.5",
            "--pixel-threshold-strategy",
            "fixed",
            "--roi-xyxy-norm",
            "0.0",
            "0.0",
            "0.5",
            "1.0",
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0

    record = json.loads(out_jsonl.read_text(encoding="utf-8").strip().splitlines()[0])
    assert record["score"] == 1.0
    assert record["label"] == 1

    defects = record["defects"]
    assert defects["regions"] == []

    mask_path = Path(defects["mask"]["path"])
    loaded = np.asarray(Image.open(mask_path), dtype=np.uint8)
    assert int(loaded.max()) == 0


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
