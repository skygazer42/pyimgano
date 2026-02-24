from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pyimgano.cli import main as cli_main
from pyimgano.models.registry import MODEL_REGISTRY


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_cli_manifest_benchmark_supports_meta_in_per_image_jsonl(tmp_path: Path) -> None:
    class _DummyDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)

        def fit(self, X):  # noqa: ANN001
            self.fit_inputs = list(X)
            return self

        def decision_function(self, X):  # noqa: ANN001
            return np.linspace(0.0, 1.0, num=len(list(X)), dtype=np.float32)

    MODEL_REGISTRY.register(
        "test_cli_manifest_benchmark_dummy_detector",
        _DummyDetector,
        tags=("classical",),
        overwrite=True,
    )

    mdir = tmp_path / "m"
    mdir.mkdir(parents=True, exist_ok=True)
    manifest = mdir / "manifest.jsonl"
    (mdir / "train.png").touch()
    (mdir / "good.png").touch()
    (mdir / "bad.png").touch()

    _write_jsonl(
        manifest,
        [
            {"image_path": "train.png", "category": "bottle", "split": "train"},
            {
                "image_path": "good.png",
                "category": "bottle",
                "split": "test",
                "label": 0,
                "meta": {"camera": "c1"},
            },
            {"image_path": "bad.png", "category": "bottle", "split": "test", "label": 1},
        ],
    )

    out_dir = tmp_path / "run_out"
    code = cli_main(
        [
            "--dataset",
            "manifest",
            "--root",
            str(tmp_path),
            "--manifest-path",
            str(manifest),
            "--category",
            "bottle",
            "--model",
            "test_cli_manifest_benchmark_dummy_detector",
            "--device",
            "cpu",
            "--no-pretrained",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert code == 0

    per_image = out_dir / "categories" / "bottle" / "per_image.jsonl"
    assert per_image.exists()
    rows = [json.loads(line) for line in per_image.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(r.get("meta", None) == {"camera": "c1"} for r in rows)

    report_path = out_dir / "categories" / "bottle" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "dataset_summary" in report

