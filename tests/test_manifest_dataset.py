from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pyimgano.datasets.manifest import (
    ManifestSplitPolicy,
    iter_manifest_records,
    load_manifest_benchmark_split,
)


def _write_rgb(path: Path, *, size: tuple[int, int] = (8, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 20, 30)).save(path)


def _write_mask(path: Path, *, size: tuple[int, int] = (8, 6), value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=int(value)).save(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_iter_manifest_records_skips_blank_and_comments(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n# comment\n" + json.dumps({"image_path": "a.png", "category": "c"}) + "\n",
        encoding="utf-8",
    )
    recs = list(iter_manifest_records(manifest))
    assert len(recs) == 1
    assert recs[0].image_path == "a.png"
    assert recs[0].category == "c"


def test_iter_manifest_records_requires_required_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [{"category": "c"}])
    with pytest.raises(ValueError, match=r"missing required field 'image_path'"):
        list(iter_manifest_records(manifest))


def test_manifest_path_resolution_prefers_manifest_dir_then_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    mdir = tmp_path / "manifest_dir"
    manifest = mdir / "manifest.jsonl"

    # Same relative path exists in both places → manifest-dir wins.
    _write_rgb(root / "img.png")
    _write_rgb(mdir / "img.png")

    _write_jsonl(
        manifest,
        [
            {"image_path": "img.png", "category": "bottle"},
            {"image_path": "img.png", "category": "bottle", "label": 1},
        ],
    )

    split = load_manifest_benchmark_split(
        manifest_path=manifest,
        root_fallback=root,
        category="bottle",
        resize=(6, 8),
        load_masks=False,
        split_policy=ManifestSplitPolicy(seed=1, test_normal_fraction=0.5),
    )

    assert split.train_paths[0].startswith(str(mdir))
    assert split.test_paths[0].startswith(str(mdir))


def test_manifest_auto_split_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "root"
    mdir = tmp_path / "manifest_dir"
    manifest = mdir / "manifest.jsonl"

    rows = []
    # 8 normal records (no split/label).
    for i in range(8):
        name = f"n_{i}.png"
        _write_rgb(mdir / name)
        rows.append({"image_path": name, "category": "bottle"})
    # 2 anomaly records (label=1).
    for i in range(2):
        name = f"a_{i}.png"
        _write_rgb(mdir / name)
        rows.append({"image_path": name, "category": "bottle", "label": 1})

    _write_jsonl(manifest, rows)

    policy = ManifestSplitPolicy(seed=123, test_normal_fraction=0.25)
    s1 = load_manifest_benchmark_split(
        manifest_path=manifest,
        root_fallback=root,
        category="bottle",
        resize=(6, 8),
        load_masks=False,
        split_policy=policy,
    )
    s2 = load_manifest_benchmark_split(
        manifest_path=manifest,
        root_fallback=root,
        category="bottle",
        resize=(6, 8),
        load_masks=False,
        split_policy=policy,
    )
    assert s1.train_paths == s2.train_paths
    assert s1.test_paths == s2.test_paths
    assert s1.test_labels.tolist() == s2.test_labels.tolist()
    assert any(s1.test_labels == 1), "anomaly labels must be present in test"


def test_manifest_auto_split_changes_with_seed_when_choices_exist(tmp_path: Path) -> None:
    mdir = tmp_path / "m"
    manifest = mdir / "manifest.jsonl"
    mdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(30):
        name = f"n_{i}.png"
        (mdir / name).touch()
        rows.append({"image_path": name, "category": "bottle"})

    (mdir / "a_0.png").touch()
    rows.append({"image_path": "a_0.png", "category": "bottle", "label": 1})

    _write_jsonl(manifest, rows)

    s1 = load_manifest_benchmark_split(
        manifest_path=manifest,
        root_fallback=None,
        category="bottle",
        load_masks=False,
        split_policy=ManifestSplitPolicy(seed=1, test_normal_fraction=0.2),
    )
    s2 = load_manifest_benchmark_split(
        manifest_path=manifest,
        root_fallback=None,
        category="bottle",
        load_masks=False,
        split_policy=ManifestSplitPolicy(seed=2, test_normal_fraction=0.2),
    )

    normal_test_1 = {p for p, lab in zip(s1.test_paths, s1.test_labels) if int(lab) == 0}
    normal_test_2 = {p for p, lab in zip(s2.test_paths, s2.test_labels) if int(lab) == 0}
    assert normal_test_1, "expected some normal samples to be assigned to test"
    assert normal_test_2, "expected some normal samples to be assigned to test"
    assert normal_test_1 != normal_test_2, "different seeds should affect normal test sampling"


def test_group_aware_split_forces_entire_group_to_test(tmp_path: Path) -> None:
    mdir = tmp_path / "m"
    manifest = mdir / "manifest.jsonl"

    _write_rgb(mdir / "g1_a.png")
    _write_rgb(mdir / "g1_b.png")
    _write_rgb(mdir / "g2.png")

    _write_jsonl(
        manifest,
        [
            {"image_path": "g1_a.png", "category": "bottle", "group_id": "g1", "label": 1},
            {"image_path": "g1_b.png", "category": "bottle", "group_id": "g1"},
            {"image_path": "g2.png", "category": "bottle", "group_id": "g2"},
        ],
    )

    split = load_manifest_benchmark_split(
        manifest_path=manifest,
        root_fallback=tmp_path / "root",
        category="bottle",
        resize=(6, 8),
        load_masks=False,
        split_policy=ManifestSplitPolicy(seed=1, test_normal_fraction=0.0),
    )

    assert any("g1_a.png" in p for p in split.test_paths)
    assert any("g1_b.png" in p for p in split.test_paths)
    assert split.test_labels.tolist().count(1) == 1


def test_explicit_test_requires_label(tmp_path: Path) -> None:
    mdir = tmp_path / "m"
    manifest = mdir / "manifest.jsonl"
    _write_rgb(mdir / "x.png")
    _write_jsonl(manifest, [{"image_path": "x.png", "category": "bottle", "split": "test"}])

    with pytest.raises(ValueError, match=r"split='test' requires an explicit label"):
        load_manifest_benchmark_split(
            manifest_path=manifest,
            root_fallback=tmp_path / "root",
            category="bottle",
            resize=(6, 8),
            load_masks=False,
        )


def test_missing_anomaly_masks_disables_pixel_metrics(tmp_path: Path) -> None:
    mdir = tmp_path / "m"
    manifest = mdir / "manifest.jsonl"

    _write_rgb(mdir / "n.png")
    _write_rgb(mdir / "a.png")

    _write_jsonl(
        manifest,
        [
            {"image_path": "n.png", "category": "bottle"},
            {"image_path": "a.png", "category": "bottle", "label": 1},
        ],
    )

    split = load_manifest_benchmark_split(
        manifest_path=manifest,
        root_fallback=tmp_path / "root",
        category="bottle",
        resize=(6, 8),
        load_masks=True,
        split_policy=ManifestSplitPolicy(seed=1, test_normal_fraction=0.5),
    )

    assert split.test_masks is None
    assert split.pixel_skip_reason is not None


def test_masks_loaded_when_complete(tmp_path: Path) -> None:
    mdir = tmp_path / "m"
    manifest = mdir / "manifest.jsonl"

    _write_rgb(mdir / "n.png")
    _write_rgb(mdir / "a.png")
    _write_mask(mdir / "a_mask.png", value=255)

    _write_jsonl(
        manifest,
        [
            {"image_path": "n.png", "category": "bottle"},
            {"image_path": "a.png", "category": "bottle", "label": 1, "mask_path": "a_mask.png"},
        ],
    )

    split = load_manifest_benchmark_split(
        manifest_path=manifest,
        root_fallback=tmp_path / "root",
        category="bottle",
        resize=(6, 8),
        load_masks=True,
        split_policy=ManifestSplitPolicy(seed=1, test_normal_fraction=0.5),
    )

    assert split.test_masks is not None
    assert split.test_masks.shape == (len(split.test_paths), 6, 8)
    assert np.isin(split.test_masks, [0, 1]).all()
