from __future__ import annotations

from pathlib import Path

import numpy as np


def _write_png(path: Path, *, value: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
    Image.fromarray(img, mode="RGB").save(path)


def test_infer_cli_fails_fast_when_checkpoint_required(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    from pyimgano.infer_cli import main as infer_main

    img = tmp_path / "x.png"
    _write_png(img, value=120)

    # Pick a representative checkpoint-required model name; the test should fail
    # before optional backend imports matter.
    rc = infer_main(
        [
            "--model",
            "vision_patchcore_anomalib",
            "--device",
            "cpu",
            "--no-pretrained",
            "--input",
            str(img),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err.lower()
    assert "requires a checkpoint" in err

