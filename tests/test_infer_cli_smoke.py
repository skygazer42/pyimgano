import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import pyimgano.infer_cli as infer_cli
from pyimgano.inference.api import InferenceResult


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


def test_infer_cli_direct_mode_delegates_detector_setup_to_service(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    import pyimgano.services.infer_load_service as infer_load_service

    monkeypatch.setattr(
        infer_load_service,
        "load_direct_infer_detector",
        lambda request, *, create_detector=None: infer_load_service.DirectInferLoadResult(
            model_name="delegated-model",
            detector=_DummyDetector(),
            model_kwargs={},
        ),
    )

    rc = infer_cli.main(
        [
            "--model",
            "vision_ecod",
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0


def test_infer_cli_smoke_seed_calls_seed_everything(tmp_path, monkeypatch):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    det = _DummyDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    called = {"seed": None}
    import pyimgano.utils.seeding as seeding

    monkeypatch.setattr(seeding, "seed_everything", lambda s: called.update(seed=int(s)))

    rc = infer_cli.main(
        [
            "--model",
            "vision_ecod",
            "--seed",
            "123",
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert called["seed"] == 123


def test_infer_cli_smoke_delegates_to_inference_service(tmp_path, monkeypatch):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    det = _DummyDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    import pyimgano.services.inference_service as inference_service

    captured: dict[str, object] = {}

    def fake_iter_inference_records(**kwargs):
        captured.update(kwargs)
        yield InferenceResult(score=0.25, label=0, anomaly_map=None)

    monkeypatch.setattr(
        inference_service, "iter_inference_records", fake_iter_inference_records
    )

    rc = infer_cli.main(
        [
            "--model",
            "vision_ecod",
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert captured["detector"] is det
    assert captured["inputs"] == [str(input_dir / "a.png")]

    record = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
    assert record["score"] == pytest.approx(0.25)


def test_infer_cli_smoke_delegates_artifact_materialization_to_service(tmp_path, monkeypatch):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    det = _DummyDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    import pyimgano.services.infer_artifact_service as infer_artifact_service

    calls: list[object] = []
    monkeypatch.setattr(
        infer_artifact_service,
        "materialize_infer_result_artifacts",
        lambda request: calls.append(request)
        or infer_artifact_service.InferResultArtifactResult(
            record={
                "index": int(request.index),
                "input": str(request.input_path),
                "score": 0.1,
                "label": 0,
            },
            regions_payload=None,
        ),
    )

    rc = infer_cli.main(
        [
            "--model",
            "vision_ecod",
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert len(calls) == 1


def test_infer_cli_smoke_delegates_output_writing_to_service(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    det = _DummyDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    import pyimgano.services.infer_output_service as infer_output_service

    calls: list[object] = []

    monkeypatch.setattr(
        infer_output_service,
        "write_infer_output_payloads",
        lambda request, *, print_fn=print: calls.append(request)
        or infer_output_service.InferOutputWriteResult(
            output_written=int(request.output_written) + 1,
            regions_written=int(request.regions_written),
        ),
    )

    rc = infer_cli.main(
        [
            "--model",
            "vision_ecod",
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert len(calls) == 1


def test_infer_cli_smoke_delegates_defects_config_building_to_service(
    tmp_path, monkeypatch
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    det = _DummyDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    import pyimgano.services.infer_artifact_service as infer_artifact_service

    calls: list[object] = []
    monkeypatch.setattr(
        infer_artifact_service,
        "build_defects_artifact_config",
        lambda request: calls.append(request) or None,
    )

    rc = infer_cli.main(
        [
            "--model",
            "vision_ecod",
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
            "--defects",
            "--pixel-threshold",
            "0.5",
        ]
    )
    assert rc == 0
    assert len(calls) == 1


def test_infer_cli_smoke_delegates_artifact_request_building_to_service(
    tmp_path, monkeypatch
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    det = _DummyDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    import pyimgano.services.infer_artifact_service as infer_artifact_service

    calls: list[object] = []
    monkeypatch.setattr(
        infer_artifact_service,
        "build_infer_result_artifact_request",
        lambda request: calls.append(request)
        or infer_artifact_service.InferResultArtifactRequest(
            index=int(request.index),
            input_path=str(request.input_path),
            result=request.result,
            include_status=bool(request.include_status),
            include_anomaly_map_values=bool(request.include_anomaly_map_values),
            maps_dir=request.maps_dir,
            overlays_dir=request.overlays_dir,
            defects_config=None,
        ),
    )

    rc = infer_cli.main(
        [
            "--model",
            "vision_ecod",
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert len(calls) == 1


def test_infer_cli_smoke_delegates_final_artifact_request_building_to_service(
    tmp_path, monkeypatch
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"
    maps_dir = tmp_path / "maps"

    det = _DummyDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    import pyimgano.services.infer_artifact_service as infer_artifact_service

    calls: list[object] = []
    monkeypatch.setattr(
        infer_artifact_service,
        "build_infer_result_artifact_request_from_cli",
        lambda request: calls.append(request)
        or infer_artifact_service.InferResultArtifactRequest(
            index=int(request.index),
            input_path=str(request.input_path),
            result=request.result,
            include_status=bool(request.include_status),
            include_anomaly_map_values=False,
            maps_dir=str(maps_dir),
            overlays_dir=None,
            defects_config=None,
        ),
    )

    rc = infer_cli.main(
        [
            "--model",
            "vision_ecod",
            "--input",
            str(input_dir),
            "--include-maps",
            "--save-jsonl",
            str(out_jsonl),
            "--save-maps",
            str(maps_dir),
        ]
    )
    assert rc == 0
    assert len(calls) == 1


def test_infer_cli_defects_calibration_delegates_to_inference_service(tmp_path, monkeypatch):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    _write_png(train_dir / "train.png")

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    det = _DummyDetector()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    import pyimgano.services.inference_service as inference_service

    run_calls: list[list[str]] = []
    iter_calls: list[list[str]] = []

    def fake_run_inference(**kwargs):
        inputs = [str(item) for item in kwargs["inputs"]]
        run_calls.append(inputs)
        return inference_service.InferenceRunResult(
            records=[
                InferenceResult(
                    score=float(i),
                    label=0,
                    anomaly_map=np.ones((4, 4), dtype=np.float32),
                )
                for i in range(len(inputs))
            ],
            timing_seconds=0.0,
        )

    def fake_iter_inference_records(**kwargs):
        inputs = [str(item) for item in kwargs["inputs"]]
        iter_calls.append(inputs)
        yield InferenceResult(score=0.0, label=0, anomaly_map=np.ones((4, 4), dtype=np.float32))

    monkeypatch.setattr(inference_service, "run_inference", fake_run_inference)
    monkeypatch.setattr(
        inference_service, "iter_inference_records", fake_iter_inference_records
    )

    rc = infer_cli.main(
        [
            "--model",
            "vision_ecod",
            "--train-dir",
            str(train_dir),
            "--input",
            str(input_dir),
            "--defects",
            "--pixel-threshold-strategy",
            "normal_pixel_quantile",
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert [str(train_dir / "train.png")] in run_calls
    assert [str(input_dir / "a.png")] in iter_calls


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
    assert defects["pixel_threshold"] == pytest.approx(0.5)
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
    assert record["score"] == pytest.approx(1.0)
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
