from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=color).save(path)


def test_synthesis_end_to_end_manifest_pipeline(tmp_path: Path) -> None:
    from pyimgano.datasets.manifest import load_manifest_benchmark_split
    from pyimgano.models.registry import create_model
    from pyimgano.synthesize_cli import synthesize_dataset

    in_dir = tmp_path / "normals"
    _write_rgb(in_dir / "n0.png", color=(10, 20, 30))
    _write_rgb(in_dir / "n1.png", color=(30, 40, 50))
    _write_rgb(in_dir / "n2.png", color=(60, 70, 80))

    out_root = tmp_path / "synth_root"
    records = synthesize_dataset(
        in_dir=in_dir,
        out_root=out_root,
        category="demo",
        preset="scratch",
        seed=0,
        n_train=3,
        n_test_normal=2,
        n_test_anomaly=2,
    )
    assert records
    assert any(
        int(r.get("label", 0)) == 1 and "mask_path" in r for r in records
    ), "expected synthesized anomaly records to include mask_path when include_masks=True"

    split = load_manifest_benchmark_split(
        manifest_path=out_root / "manifest.jsonl",
        root_fallback=out_root,
        category="demo",
        resize=(64, 64),
        load_masks=True,
    )
    assert split.train_paths
    assert split.test_paths
    assert split.test_labels.shape[0] == len(split.test_paths)
    assert split.test_masks is not None
    assert split.test_masks.shape == (len(split.test_paths), 64, 64)

    det = create_model(
        "vision_iforest",
        contamination=0.25,
        feature_extractor={"name": "structural", "kwargs": {"max_size": 128}},
        random_state=0,
    )
    det.fit(split.train_paths)
    scores = det.decision_function(split.test_paths)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    assert scores.shape[0] == len(split.test_paths)


def test_synthesis_end_to_end_manifest_pipeline_with_preset_mix_meta(tmp_path: Path) -> None:
    from pyimgano.datasets.manifest import load_manifest_benchmark_split
    from pyimgano.synthesize_cli import synthesize_dataset

    in_dir = tmp_path / "normals"
    _write_rgb(in_dir / "n0.png", color=(10, 20, 30))
    _write_rgb(in_dir / "n1.png", color=(30, 40, 50))
    _write_rgb(in_dir / "n2.png", color=(60, 70, 80))

    out_root = tmp_path / "synth_root"
    synthesize_dataset(
        in_dir=in_dir,
        out_root=out_root,
        category="demo",
        preset="scratch",  # ignored when presets is provided
        presets=["scratch", "stain", "tape"],
        seed=0,
        n_train=3,
        n_test_normal=2,
        n_test_anomaly=2,
    )

    split = load_manifest_benchmark_split(
        manifest_path=out_root / "manifest.jsonl",
        root_fallback=out_root,
        category="demo",
        resize=(64, 64),
        load_masks=False,
    )
    assert split.test_meta is not None
    # Only anomalies are guaranteed to have meta (normals may be None).
    for p, lab, meta in zip(split.test_paths, split.test_labels.tolist(), split.test_meta):
        if int(lab) == 1:
            assert meta is not None
            assert meta.get("preset") in {"scratch", "stain", "tape"}
            assert meta.get("preset_mixture") == ["scratch", "stain", "tape"]
