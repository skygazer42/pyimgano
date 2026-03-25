import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pyimgano.models.registry import MODEL_REGISTRY


def test_train_cli_smoke(tmp_path, capsys):
    import cv2

    from pyimgano.train_cli import main

    class _DummyDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)

        def fit(self, x):  # noqa: ANN001
            self.fit_inputs = list(x)
            return self

        def decision_function(self, x):  # noqa: ANN001
            n = len(list(x))
            if n == 0:
                return np.asarray([], dtype=np.float32)
            return np.linspace(0.0, 1.0, num=n, dtype=np.float32)

    MODEL_REGISTRY.register(
        "test_train_cli_dummy_detector",
        _DummyDetector,
        tags=("classical",),
        overwrite=True,
    )

    root = tmp_path / "custom"
    for rel, value in [
        ("train/normal/train_0.png", 120),
        ("train/normal/train_1.png", 121),
        ("test/normal/good_0.png", 120),
        ("test/anomaly/bad_0.png", 240),
    ]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
        cv2.imwrite(str(p), img)

    out_dir = tmp_path / "run_out"
    cfg = {
        "recipe": "industrial-adapt",
        "seed": 123,
        "dataset": {
            "name": "custom",
            "root": str(root),
            "category": "all",
            "resize": [16, 16],
            "input_mode": "paths",
            "limit_train": 2,
            "limit_test": 2,
        },
        "model": {
            "name": "test_train_cli_dummy_detector",
            "device": "cpu",
            "pretrained": False,
            "contamination": 0.1,
        },
        "output": {
            "output_dir": str(out_dir),
            "save_run": True,
            "per_image_jsonl": True,
        },
    }
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    code = main(["--config", str(config_path)])
    assert code == 0

    out = capsys.readouterr().out
    assert "Train Run Summary" in out
    assert "[RUN]" in out
    assert "[CFG]" in out
    assert "[OUT]" in out
    assert "engine=train" in out
    assert "recipe=industrial-adapt" in out
    assert "data=custom/all" in out
    assert "model=test_train_cli_dummy_detector" in out
    assert "imgsz=16x16" in out
    assert "save_dir=" in out
    assert "[DATA custom]" in out
    assert "[CAL custom]" in out
    assert "[VAL custom]" in out
    assert "[DONE]" in out
    assert "[SAVE]" in out
    assert "[DONE] status=done" in out
    assert "[DONE] status=done save_dir=" in out
    assert "Results saved to " not in out
    assert "Artifacts" not in out
    assert "report=" in out
    assert "config=" in out
    assert "environment=" in out
    assert "per_image=" in out

    assert (out_dir / "report.json").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "environment.json").exists()
    assert (out_dir / "categories" / "custom" / "per_image.jsonl").exists()


def test_train_cli_training_logs_epoch_metrics_in_human_output(tmp_path, capsys):
    import cv2

    from pyimgano.train_cli import main

    class _TrainingDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)

        def fit(
            self,
            X,
            *,
            epochs=None,
            batch_size=None,
            num_workers=None,
            optimizer_name=None,
            criterion_name=None,
            scheduler_name=None,
        ):  # noqa: ANN001 - test stub
            self.fit_inputs = list(X)
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.optimizer_name = optimizer_name
            self.criterion_name = criterion_name
            self.scheduler_name = scheduler_name
            self.training_loss_history_ = [0.9, 0.6]
            self.training_lr_history_ = [0.01, 0.005]
            self.training_epochs_completed_ = int(epochs or 2)
            self.training_steps_completed_ = 4
            self.training_best_loss_ = 0.6
            self.training_last_lr_ = 0.005
            self.training_stop_reason_ = "completed"
            return self

        def decision_function(self, X):  # noqa: ANN001 - test stub
            n = len(list(X))
            if n == 0:
                return np.asarray([], dtype=np.float32)
            return np.linspace(0.0, 1.0, num=n, dtype=np.float32)

        def save_checkpoint(self, path):  # noqa: ANN001 - test stub
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("ok", encoding="utf-8")
            return target

    MODEL_REGISTRY.register(
        "test_train_cli_epoch_logger_detector",
        _TrainingDetector,
        tags=("classical",),
        overwrite=True,
    )

    root = tmp_path / "custom"
    for rel, value in [
        ("train/normal/train_0.png", 120),
        ("train/normal/train_1.png", 121),
        ("test/normal/good_0.png", 120),
        ("test/anomaly/bad_0.png", 240),
    ]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
        cv2.imwrite(str(p), img)

    cfg = {
        "recipe": "industrial-adapt",
        "seed": 123,
        "dataset": {
            "name": "custom",
            "root": str(root),
            "category": "custom",
            "resize": [16, 16],
            "input_mode": "paths",
            "limit_train": 2,
            "limit_test": 2,
        },
        "model": {
            "name": "test_train_cli_epoch_logger_detector",
            "device": "cpu",
            "pretrained": False,
            "contamination": 0.1,
        },
        "training": {
            "enabled": True,
            "epochs": 2,
            "batch_size": 8,
            "num_workers": 2,
            "optimizer_name": "adamw",
            "criterion_name": "mae",
            "scheduler_name": "cosine",
        },
        "output": {
            "output_dir": str(tmp_path / "run_train"),
            "save_run": True,
            "per_image_jsonl": False,
        },
    }
    config_path = tmp_path / "cfg_train.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    code = main(["--config", str(config_path)])

    assert code == 0
    out = capsys.readouterr().out
    assert "[TRAIN custom] status=enabled" in out
    assert "tracker=none" not in out
    assert "callbacks=none" not in out
    assert "Training\n" not in out
    assert "[RUN]" in out
    assert "[CFG]" in out
    assert "[OPT]" in out
    assert "[OUT]" in out
    assert "data=custom/custom" in out
    assert "imgsz=16x16" in out
    assert "Epoch" in out
    assert "Device" in out
    assert "loss" in out
    assert "lr" in out
    assert "Time" in out
    assert "ETA" in out
    assert "items/s" in out
    assert "batch=8" in out
    assert "workers=2" in out
    assert "optimizer=adamw" in out
    assert "scheduler=cosine" in out
    assert "criterion=mae" in out
    assert re.search(r"1/2\s+cpu\s+0\.9000\s+0\.0100\s+\d+\s+custom", out)
    assert re.search(r"2/2\s+cpu\s+0\.6000\s+0\.0050\s+\d+\s+custom", out)
    assert "[DONE custom]" in out
    assert "stage=training_complete" in out
    assert "epochs=2" in out
    assert "steps=4" in out
    assert "best_loss=0.6000" in out
    assert "last_lr=0.0050" in out
    assert "[DATA custom]" in out
    assert "[CAL custom]" in out
    assert "[VAL custom]" in out
    assert "[DONE]" in out
    assert "[DONE] status=done save_dir=" in out
    assert "[SAVE] checkpoint=" in out
    assert "Results saved to " not in out
    assert "Artifacts" not in out


def test_train_console_reporter_formats_epoch_timing_columns(capsys):
    from pyimgano.train_cli_presentation import TrainConsoleReporter

    reporter = TrainConsoleReporter()
    reporter.on_run_start(
        config=SimpleNamespace(
            recipe="industrial-adapt",
            dataset=SimpleNamespace(
                name="custom",
                category="widget",
                resize=[16, 16],
                input_mode="paths",
            ),
            model=SimpleNamespace(name="vision_dummy", device="cpu"),
            output=SimpleNamespace(save_run=True, per_image_jsonl=False),
            training=SimpleNamespace(
                epochs=2,
                batch_size=8,
                num_workers=2,
                optimizer_name="adamw",
                criterion_name="mae",
                scheduler_name="cosine",
            ),
        ),
        request=SimpleNamespace(
            config_path="cfg.json",
            export_infer_config=False,
            export_deploy_bundle=False,
        ),
    )
    reporter.on_run_context(run_dir="/tmp/run")
    reporter.on_category_start(category="widget", index=1, total=1)
    reporter.on_dataset_loaded(
        category="widget",
        train_count=12,
        calibration_count=4,
        test_count=4,
        anomaly_count=1,
        pixel_metrics_enabled=True,
        pixel_metrics_reason=None,
    )
    reporter.on_training_start(
        category="widget",
        enabled=True,
        fit_kwargs={
            "epochs": 2,
            "batch_size": 8,
            "num_workers": 2,
            "optimizer_name": "adamw",
            "criterion_name": "mae",
            "scheduler_name": "cosine",
        },
        tracker_backend=None,
        callback_names=[],
    )
    reporter.on_training_epoch(
        epoch=1,
        total_epochs=2,
        metrics={
            "loss": 0.9,
            "lr": 0.01,
            "train_items": 12,
            "epoch_s": 3.0,
            "eta_s": 3.0,
            "items_per_s": 4.0,
        },
        live=False,
    )
    reporter.on_training_end(
        category="widget",
        report={
            "timing": {"fit_s": 6.0},
            "detector_training_state": {
                "epochs_completed": 2,
                "steps_completed": 4,
                "best_loss": 0.6,
                "last_lr": 0.005,
                "stop_reason": "completed",
            },
        },
        checkpoint_meta={"path": "checkpoints/widget/model.pt"},
    )
    reporter.on_calibration_end(
        category="widget",
        threshold=0.9,
        quantile=0.9,
        source="contamination",
        score_summary={"mean": 0.5},
    )
    reporter.on_evaluation_end(
        category="widget",
        results={"auroc": 1.0, "average_precision": 1.0},
        dataset_summary=None,
    )
    reporter.on_artifact_written(kind="infer_config", path="artifacts/infer_config.json")
    reporter.on_run_end(report={"results": {"auroc": 1.0, "average_precision": 1.0}})

    out = capsys.readouterr().out
    assert "[RUN]" in out
    assert "[CFG]" in out
    assert "[OPT]" in out
    assert "[OUT]" in out
    assert "Time" in out
    assert "ETA" in out
    assert "items/s" in out
    assert "batch=8" in out
    assert "workers=2" in out
    assert "optimizer=adamw" in out
    assert "scheduler=cosine" in out
    assert "criterion=mae" in out
    assert re.search(r"1/2\s+cpu\s+0\.9000\s+0\.0100\s+12\s+widget\s+3\.0s\s+3\.0s\s+4\.0", out)
    assert "[TRAIN widget] status=enabled" in out
    assert "tracker=none" not in out
    assert "callbacks=none" not in out
    assert "Training\n" not in out
    assert "steps=4" in out
    assert "best_loss=0.6000" in out
    assert "last_lr=0.0050" in out
    assert "[DATA widget]" in out
    assert "Train" in out
    assert "Cal" in out
    assert "Test" in out
    assert "Anom" in out
    assert "Input" in out
    assert "Pixel" not in out
    assert re.search(r"12\s+4\s+4\s+1\s+16x16\s+paths", out)
    assert "[CAL widget]" in out
    assert "[VAL widget]" in out
    assert "Threshold" in out
    assert "Quantile" in out
    assert "ScoreMean" in out
    assert "Source" in out
    assert re.search(r"0\.900000\s+0\.900000\s+0\.500000\s+contamination", out)
    assert "AUROC" in out
    assert "AP" in out
    assert "pAUROC" in out
    assert re.search(r"1\.000000\s+1\.000000\s+-\s+-\s+-\s+-", out)
    assert "[DONE]" in out
    assert "[DONE widget]" in out
    assert "[DONE] status=done save_dir=/tmp/run" in out
    assert "[SAVE] checkpoint=" in out
    assert "[SAVE] infer_config=" in out


def test_train_console_reporter_formats_multi_category_transition_with_badge(capsys):
    from pyimgano.train_cli_presentation import TrainConsoleReporter

    reporter = TrainConsoleReporter()
    reporter.on_category_start(category="capsule", index=2, total=3)

    out = capsys.readouterr().out
    assert "[DATA capsule] category=2/3" in out
    assert "Category 2/3:" not in out


def test_train_console_reporter_keeps_nondefault_training_and_data_metadata(capsys):
    from pyimgano.train_cli_presentation import TrainConsoleReporter

    reporter = TrainConsoleReporter()
    reporter.on_run_start(
        config=SimpleNamespace(
            recipe="industrial-adapt",
            dataset=SimpleNamespace(
                name="custom",
                category="capsule",
                resize=[16, 16],
                input_mode="paths",
            ),
            model=SimpleNamespace(name="vision_dummy", device="cpu"),
            output=SimpleNamespace(save_run=True, per_image_jsonl=False),
            training=SimpleNamespace(
                epochs=3,
                batch_size=8,
                num_workers=2,
                optimizer_name="adamw",
                criterion_name="mae",
                scheduler_name="cosine",
            ),
        ),
        request=SimpleNamespace(
            config_path="cfg.json",
            export_infer_config=False,
            export_deploy_bundle=False,
        ),
    )
    reporter.on_category_start(category="capsule", index=1, total=1)
    reporter.on_dataset_loaded(
        category="capsule",
        train_count=8,
        calibration_count=2,
        test_count=4,
        anomaly_count=1,
        pixel_metrics_enabled=False,
        pixel_metrics_reason="missing masks for anomaly test samples",
    )
    reporter.on_training_start(
        category="capsule",
        enabled=True,
        fit_kwargs={"epochs": 3, "batch_size": 8},
        tracker_backend="mlflow",
        callback_names=["early_stop"],
    )

    out = capsys.readouterr().out
    assert "[TRAIN capsule] status=enabled epochs=3 tracker=mlflow callbacks=early_stop" in out
    assert "[DATA capsule]" in out
    assert "Pixel" in out
    assert re.search(r"8\s+2\s+4\s+1\s+16x16\s+paths\s+off", out)
    assert "note=missing masks for anomaly test samples" in out


def test_train_console_reporter_renders_live_progress_bar(capsys):
    from pyimgano.train_cli_presentation import TrainConsoleReporter

    reporter = TrainConsoleReporter()
    reporter.on_run_start(
        config=SimpleNamespace(
            recipe="industrial-adapt",
            dataset=SimpleNamespace(
                name="custom",
                category="widget",
                resize=[16, 16],
                input_mode="paths",
            ),
            model=SimpleNamespace(name="vision_dummy", device="cpu"),
            output=SimpleNamespace(save_run=True, per_image_jsonl=False),
            training=SimpleNamespace(
                epochs=2,
                batch_size=8,
                num_workers=2,
                optimizer_name="adamw",
                criterion_name="mae",
                scheduler_name="cosine",
            ),
        ),
        request=SimpleNamespace(
            config_path="cfg.json",
            export_infer_config=False,
            export_deploy_bundle=False,
        ),
    )
    reporter.on_run_context(run_dir="/tmp/run")
    reporter.on_category_start(category="widget", index=1, total=1)
    reporter.on_dataset_loaded(
        category="widget",
        train_count=12,
        calibration_count=4,
        test_count=4,
        anomaly_count=1,
        pixel_metrics_enabled=True,
        pixel_metrics_reason=None,
    )
    reporter.on_training_start(
        category="widget",
        enabled=True,
        fit_kwargs={"epochs": 2, "batch_size": 8},
        tracker_backend=None,
        callback_names=[],
    )
    reporter.on_training_epoch(
        epoch=1,
        total_epochs=2,
        metrics={
            "loss": 0.9,
            "lr": 0.01,
            "train_items": 12,
            "epoch_s": 3.0,
            "eta_s": 3.0,
            "items_per_s": 4.0,
        },
        live=True,
    )
    reporter.on_training_epoch(
        epoch=2,
        total_epochs=2,
        metrics={
            "loss": 0.6,
            "lr": 0.005,
            "train_items": 12,
            "epoch_s": 2.0,
            "eta_s": 0.0,
            "items_per_s": 6.0,
        },
        live=True,
    )
    reporter.on_training_end(
        category="widget",
        report={"timing": {"fit_s": 5.0}, "detector_training_state": {"epochs_completed": 2}},
        checkpoint_meta=None,
    )

    out = capsys.readouterr().out
    assert "[RUN]" in out
    assert "[CFG]" in out
    assert "[OPT]" in out
    assert "[OUT]" in out
    assert "\r[TRAIN widget]" in out
    assert "50%|" in out
    assert "100%|" in out
    assert "loss 0.9000" in out
    assert "loss 0.6000" in out
    assert "lr 0.0100" in out
    assert "lr 0.0050" in out
    assert "n  12" in out
    assert "ips  4.0" in out
    assert "ips  6.0" in out
    assert "items/s=" not in out
    assert "train=" not in out
    assert "[DONE widget]" in out
    assert "stage=training_complete" in out
