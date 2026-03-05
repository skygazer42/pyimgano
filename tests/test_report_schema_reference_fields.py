from __future__ import annotations

import json
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_benchmark_report_includes_reference_context_when_present(tmp_path: Path) -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.registry import MODEL_REGISTRY
    from pyimgano.pipelines.run_benchmark import RunConfig, run_benchmark_category

    class _DummyRefDetector:
        def __init__(
            self, *, reference_dir=None, match_mode="basename", contamination=0.1, **kwargs
        ):  # noqa: ANN001
            self.reference_dir = reference_dir
            self.match_mode = match_mode
            self.kwargs = dict(kwargs)

        def fit(self, X):  # noqa: ANN001
            return self

        def decision_function(self, X):  # noqa: ANN001
            return [0.0 for _ in list(X)]

    MODEL_REGISTRY.register(
        "test_reference_report_dummy",
        _DummyRefDetector,
        tags=("vision", "reference"),
        overwrite=True,
    )

    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.png").touch()
    (root / "good.png").touch()
    (root / "bad.png").touch()

    manifest = root / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {"image_path": "train.png", "category": "bottle", "split": "train"},
            {"image_path": "good.png", "category": "bottle", "split": "test", "label": 0},
            {"image_path": "bad.png", "category": "bottle", "split": "test", "label": 1},
        ],
    )

    ref_dir = tmp_path / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)

    payload = run_benchmark_category(
        config=RunConfig(
            dataset="manifest",
            root=str(root),
            manifest_path=str(manifest),
            category="bottle",
            model="test_reference_report_dummy",
            input_mode="paths",
            contamination=0.1,
            resize=(16, 16),
            model_kwargs={"reference_dir": str(ref_dir), "match_mode": "basename"},
        ),
        save_run=False,
        per_image_jsonl=False,
        write_top_level=False,
    )

    assert "reference" in payload
    ref = payload["reference"]
    assert ref["enabled"] is True
    assert ref["reference_dir"] == str(ref_dir)
    assert ref["match_mode"] == "basename"
    assert ref["reference_dir_exists"] is True
