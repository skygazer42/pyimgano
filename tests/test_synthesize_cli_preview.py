from __future__ import annotations

from pathlib import Path

from PIL import Image


def _write_rgb(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def test_synthesize_cli_preview_grid(tmp_path: Path) -> None:
    from pyimgano.synthesize_cli import main

    in_dir = tmp_path / "in"
    _write_rgb(in_dir / "a.png", color=(10, 10, 10))
    _write_rgb(in_dir / "b.png", color=(20, 20, 20))
    _write_rgb(in_dir / "c.png", color=(30, 30, 30))

    out_root = tmp_path / "out"
    preview_path = out_root / "preview.png"

    code = main(
        [
            "--in-dir",
            str(in_dir),
            "--out-root",
            str(out_root),
            "--preset",
            "scratch",
            "--preview",
            "--preview-out",
            str(preview_path),
            "--preview-n",
            "6",
            "--preview-cols",
            "3",
            "--seed",
            "0",
        ]
    )
    assert code == 0
    assert preview_path.exists()
    assert preview_path.stat().st_size > 0

