import json
from pathlib import Path

import numpy as np
from PIL import Image

from pyimgano.models.registry import MODEL_REGISTRY


def _write_png(path: Path) -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


def test_integration_train_then_infer_from_run(tmp_path, capsys):
    import cv2

    from pyimgano.infer_cli import main as infer_main
    from pyimgano.train_cli import main as train_main

    class _DummyDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)
            self.threshold_ = None

        def fit(self, X, *, epochs=None, lr=None):  # noqa: ANN001 - test stub
            _ = epochs, lr
            self.fit_inputs = list(X)
            return self

        def decision_function(self, X):  # noqa: ANN001
            return np.linspace(0.0, 1.0, num=len(list(X)), dtype=np.float32)

        def save_checkpoint(self, path):  # noqa: ANN001 - test stub
            Path(path).write_text("ckpt", encoding="utf-8")

        def load_checkpoint(self, path):  # noqa: ANN001 - test stub
            # Best-effort: just verify the file is readable.
            Path(path).read_text(encoding="utf-8")

    MODEL_REGISTRY.register(
        "test_integration_workbench_dummy_detector",
        _DummyDetector,
        tags=("vision",),
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

    run_dir = tmp_path / "run_out"
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
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
                    "name": "test_integration_workbench_dummy_detector",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                },
                "training": {"enabled": True, "epochs": 2, "lr": 0.001, "checkpoint_name": "model.pt"},
                "output": {"output_dir": str(run_dir), "save_run": True, "per_image_jsonl": False},
            }
        ),
        encoding="utf-8",
    )

    code = train_main(["--config", str(cfg_path)])
    assert code == 0
    capsys.readouterr()

    assert (run_dir / "report.json").exists()
    assert (run_dir / "checkpoints" / "custom" / "model.pt").exists()

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write_png(inputs / "a.png")
    _write_png(inputs / "b.png")

    out_jsonl = tmp_path / "out.jsonl"
    code = infer_main(["--from-run", str(run_dir), "--input", str(inputs), "--save-jsonl", str(out_jsonl)])
    assert code == 0

    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    rec1 = json.loads(lines[1])
    assert rec0["label"] == 0
    assert rec1["label"] == 1


def test_integration_train_then_infer_from_run_manifest(tmp_path, capsys):
    from pyimgano.infer_cli import main as infer_main
    from pyimgano.train_cli import main as train_main

    class _DummyDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)
            self.threshold_ = None

        def fit(self, X, *, epochs=None, lr=None):  # noqa: ANN001 - test stub
            _ = epochs, lr
            self.fit_inputs = list(X)
            return self

        def decision_function(self, X):  # noqa: ANN001
            return np.linspace(0.0, 1.0, num=len(list(X)), dtype=np.float32)

        def save_checkpoint(self, path):  # noqa: ANN001 - test stub
            Path(path).write_text("ckpt", encoding="utf-8")

        def load_checkpoint(self, path):  # noqa: ANN001 - test stub
            Path(path).read_text(encoding="utf-8")

    MODEL_REGISTRY.register(
        "test_integration_workbench_manifest_dummy_detector",
        _DummyDetector,
        tags=("vision",),
        overwrite=True,
    )

    mdir = tmp_path / "manifest"
    mdir.mkdir(parents=True, exist_ok=True)
    manifest = mdir / "manifest.jsonl"

    for name in ["train_0.png", "train_1.png", "good_0.png", "bad_0.png"]:
        (mdir / name).touch()

    manifest.write_text(
        "\n".join(
            [
                '{"image_path":"train_0.png","category":"bottle","split":"train"}',
                '{"image_path":"train_1.png","category":"bottle","split":"train"}',
                '{"image_path":"good_0.png","category":"bottle","split":"test","label":0}',
                '{"image_path":"bad_0.png","category":"bottle","split":"test","label":1}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "run_out"
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "industrial-adapt",
                "seed": 123,
                "dataset": {
                    "name": "manifest",
                    "root": str(tmp_path),
                    "manifest_path": str(manifest),
                    "category": "bottle",
                    "resize": [16, 16],
                    "input_mode": "paths",
                    "limit_train": 2,
                    "limit_test": 2,
                },
                "model": {
                    "name": "test_integration_workbench_manifest_dummy_detector",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                },
                "training": {"enabled": True, "epochs": 2, "lr": 0.001, "checkpoint_name": "model.pt"},
                "output": {"output_dir": str(run_dir), "save_run": True, "per_image_jsonl": False},
            }
        ),
        encoding="utf-8",
    )

    code = train_main(["--config", str(cfg_path)])
    assert code == 0
    capsys.readouterr()

    assert (run_dir / "report.json").exists()
    assert (run_dir / "checkpoints" / "bottle" / "model.pt").exists()

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write_png(inputs / "a.png")
    _write_png(inputs / "b.png")

    out_jsonl = tmp_path / "out.jsonl"
    code = infer_main(
        [
            "--from-run",
            str(run_dir),
            "--input",
            str(inputs),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert code == 0

    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    rec1 = json.loads(lines[1])
    assert rec0["label"] == 0
    assert rec1["label"] == 1
