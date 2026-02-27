from __future__ import annotations

from pathlib import Path

from PIL import Image


def _write_rgb(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=color).save(path)


def test_synthesize_dataset_is_deterministic_with_preset_mix(tmp_path: Path) -> None:
    from pyimgano.synthesize_cli import synthesize_dataset

    in_dir = tmp_path / "normals"
    _write_rgb(in_dir / "n0.png", color=(10, 20, 30))
    _write_rgb(in_dir / "n1.png", color=(30, 40, 50))
    _write_rgb(in_dir / "n2.png", color=(60, 70, 80))

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    synthesize_dataset(
        in_dir=in_dir,
        out_root=out1,
        category="demo",
        preset="scratch",
        presets=["scratch", "stain", "tape"],
        seed=123,
        n_train=3,
        n_test_normal=2,
        n_test_anomaly=2,
    )
    synthesize_dataset(
        in_dir=in_dir,
        out_root=out2,
        category="demo",
        preset="scratch",
        presets=["scratch", "stain", "tape"],
        seed=123,
        n_train=3,
        n_test_normal=2,
        n_test_anomaly=2,
    )

    m1 = (out1 / "manifest.jsonl").read_text(encoding="utf-8")
    m2 = (out2 / "manifest.jsonl").read_text(encoding="utf-8")
    assert m1 == m2

