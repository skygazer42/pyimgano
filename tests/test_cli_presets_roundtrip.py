from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def _write_rgb(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def test_cli_accepts_model_preset_name(tmp_path: Path) -> None:
    """CLI should accept a preset name as --model and resolve it internally."""

    from pyimgano.cli import main
    from pyimgano.cli_presets import resolve_model_preset

    preset = resolve_model_preset("industrial-structural-ecod")
    assert preset is not None
    assert preset.model
    assert isinstance(dict(preset.kwargs), dict)

    root = tmp_path / "custom"
    _write_rgb(root / "train" / "normal" / "train_0.png", color=(10, 10, 10))
    _write_rgb(root / "train" / "normal" / "train_1.png", color=(20, 20, 20))
    _write_rgb(root / "test" / "normal" / "good_0.png", color=(10, 10, 10))
    _write_rgb(root / "test" / "anomaly" / "bad_0.png", color=(250, 250, 250))

    out_dir = tmp_path / "run_out"
    code = main(
        [
            "--dataset",
            "custom",
            "--root",
            str(root),
            "--model",
            "industrial-structural-ecod",
            "--device",
            "cpu",
            "--no-pretrained",
            "--limit-train",
            "2",
            "--limit-test",
            "2",
            "--output-dir",
            str(out_dir),
            "--no-save-run",
            "--no-per-image-jsonl",
        ]
    )
    assert code == 0


def test_cli_preset_is_json_friendly() -> None:
    """Presets should be JSON-ready so they can be copied into config files."""

    from pyimgano.cli_presets import resolve_model_preset

    preset = resolve_model_preset("industrial-structural-ecod")
    assert preset is not None
    payload = {"name": preset.name, "model": preset.model, "kwargs": dict(preset.kwargs)}
    text = json.dumps(payload, sort_keys=True)
    assert "industrial-structural-ecod" in text
