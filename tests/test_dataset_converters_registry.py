from __future__ import annotations

from pathlib import Path

from PIL import Image


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(path)


def test_dataset_converters_registry_has_expected_names(tmp_path: Path) -> None:
    from pyimgano.datasets.converters import convert_dataset_to_manifest, list_dataset_converters

    names = [c.name for c in list_dataset_converters()]
    assert "custom" in names
    assert "mvtec_ad2" in names
    assert "real_iad" in names
    assert "rad" in names

    # Smoke: run the custom converter via the registry.
    root = tmp_path / "custom"
    _write_rgb(root / "train" / "normal" / "n0.png")
    _write_rgb(root / "test" / "normal" / "x0.png")
    _write_rgb(root / "test" / "anomaly" / "a0.png")

    out = tmp_path / "out" / "manifest.jsonl"
    records = convert_dataset_to_manifest(
        dataset="custom",
        root=root,
        out_path=out,
        category="demo",
        absolute_paths=False,
        include_masks=False,
    )
    assert out.exists()
    assert records
