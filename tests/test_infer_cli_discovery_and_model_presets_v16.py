from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


class _DummyDetector:
    def __init__(self) -> None:
        self.threshold_ = 0.5

    def decision_function(self, X):  # noqa: ANN001, ANN201 - test stub
        return np.linspace(0.0, 1.0, num=len(X), dtype=np.float32)


def test_infer_cli_can_list_models(capsys) -> None:
    from pyimgano.infer_cli import main as infer_main

    rc = infer_main(["--list-models"])
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert "vision_ecod" in out


def test_infer_cli_can_list_model_presets(capsys) -> None:
    from pyimgano.infer_cli import main as infer_main

    rc = infer_main(["--list-model-presets"])
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert "industrial-structural-ecod" in out


def test_infer_cli_accepts_model_preset_name_as_model(tmp_path: Path, monkeypatch) -> None:
    import pyimgano.infer_cli as infer_cli
    from pyimgano.infer_cli import main as infer_main

    input_dir = tmp_path / "inputs"
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"

    created: dict[str, object] = {}

    def _create_model(name: str, **kwargs):  # noqa: ANN001, ANN201 - test stub
        created["name"] = str(name)
        created["kwargs"] = dict(kwargs)
        return _DummyDetector()

    monkeypatch.setattr(infer_cli, "create_model", _create_model)

    rc = infer_main(
        [
            "--model",
            "industrial-structural-ecod",
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert created.get("name") == "vision_feature_pipeline"
