from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _write_png(path: Path, *, value: int, size: int = 16) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((size, size, 3), dtype=np.uint8) * int(value)
    Image.fromarray(img, mode="RGB").save(path)


def test_infer_cli_reference_dir_flag_applies_to_detector(tmp_path: Path) -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.infer_cli import main as infer_main
    from pyimgano.models.registry import MODEL_REGISTRY

    class _DummyRefDetector:
        def __init__(self, contamination: float = 0.1, **kwargs):  # noqa: ANN003 - test stub
            self.contamination = float(contamination)
            self.kwargs = dict(kwargs)
            self.reference_dir = None
            self.match_mode = "basename"

        def set_reference_dir(self, reference_dir: str) -> None:
            self.reference_dir = str(reference_dir)

        def get_anomaly_map(self, item):  # noqa: ANN001
            from PIL import Image

            if self.reference_dir is None:
                raise RuntimeError("reference_dir not set")

            q = Path(str(item))
            r = Path(self.reference_dir) / q.name
            q_img = np.asarray(Image.open(str(q)).convert("L"), dtype=np.float32) / 255.0
            r_img = np.asarray(Image.open(str(r)).convert("L"), dtype=np.float32) / 255.0
            return np.abs(q_img - r_img).astype(np.float32)

        def decision_function(self, x):  # noqa: ANN001
            items = list(x)
            scores = []
            for it in items:
                m = np.asarray(self.get_anomaly_map(it), dtype=np.float32)
                scores.append(float(np.max(m)))
            return np.asarray(scores, dtype=np.float64)

    MODEL_REGISTRY.register(
        "test_infer_cli_reference_dir_dummy",
        _DummyRefDetector,
        tags=("vision", "reference", "pixel_map"),
        overwrite=True,
    )

    ref_dir = tmp_path / "ref"
    query_dir = tmp_path / "query"
    _write_png(ref_dir / "x.png", value=120, size=16)
    _write_png(query_dir / "x.png", value=240, size=16)

    out_jsonl = tmp_path / "out.jsonl"
    rc = infer_main(
        [
            "--model",
            "test_infer_cli_reference_dir_dummy",
            "--reference-dir",
            str(ref_dir),
            "--input",
            str(query_dir),
            "--include-maps",
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    rows = [
        json.loads(line)
        for line in out_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    assert rows[0]["anomaly_map"]["shape"] == [16, 16]


def test_infer_cli_reference_dir_rejects_non_reference_detectors(
    tmp_path: Path, capsys
) -> None:  # noqa: ANN001
    import pyimgano.models  # noqa: F401
    from pyimgano.infer_cli import main as infer_main
    from pyimgano.models.registry import MODEL_REGISTRY

    class _NoRefDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)

        def decision_function(self, x):  # noqa: ANN001
            return np.zeros(len(list(x)), dtype=np.float64)

    MODEL_REGISTRY.register(
        "test_infer_cli_no_reference_support",
        _NoRefDetector,
        tags=("vision",),
        overwrite=True,
    )

    img = tmp_path / "x.png"
    _write_png(img, value=120, size=16)

    rc = infer_main(
        [
            "--model",
            "test_infer_cli_no_reference_support",
            "--reference-dir",
            str(tmp_path),
            "--input",
            str(img),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err.lower()
    assert "reference-dir" in err

