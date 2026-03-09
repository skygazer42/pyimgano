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


def test_infer_cli_can_list_models_by_year_and_type(capsys) -> None:
    from pyimgano.infer_cli import main as infer_main

    rc = infer_main(["--list-models", "--year", "2021", "--type", "deep-vision"])
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert "cutpaste" in out
    assert "core_qmcd" not in out


def test_infer_cli_can_list_models_by_specific_method_type(capsys) -> None:
    from pyimgano.infer_cli import main as infer_main

    rc = infer_main(["--list-models", "--type", "one-class-svm", "--year", "2001"])
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert "vision_ocsvm" in out
    assert "vision_lof" not in out


def test_infer_cli_can_list_models_by_density_estimation_type(capsys) -> None:
    from pyimgano.infer_cli import main as infer_main

    rc = infer_main(["--list-models", "--type", "density-estimation", "--year", "2019"])
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert "vision_rkde_anomalib" in out


def test_infer_cli_can_list_model_presets(capsys) -> None:
    from pyimgano.infer_cli import main as infer_main

    rc = infer_main(["--list-model-presets"])
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert "industrial-structural-ecod" in out


def test_infer_cli_can_list_model_presets_by_family(capsys) -> None:
    from pyimgano.infer_cli import main as infer_main

    rc = infer_main(["--list-model-presets", "--family", "graph"])
    assert rc == 0

    out = capsys.readouterr().out.strip().splitlines()
    assert "industrial-structural-rgraph" in out
    assert "industrial-structural-lof" not in out


def test_infer_cli_can_list_model_presets_by_family_as_json(capsys) -> None:
    import json

    from pyimgano.infer_cli import main as infer_main

    rc = infer_main(["--list-model-presets", "--family", "distillation", "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert any(
        item["name"] == "industrial-reverse-distillation" and "distillation" in item["tags"]
        for item in payload
    )


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
