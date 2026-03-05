from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), color=color).save(path)


def test_e2e_synthetic_dataset_plus_embedding_plus_core_pipeline(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    """End-to-end smoke: synthesize -> manifest split -> embeddings -> core detector."""

    import torch.hub

    def _blocked(*_args, **_kwargs):  # noqa: ANN001, ANN201
        raise AssertionError("Torchvision weight download is forbidden in unit tests.")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", _blocked, raising=True)

    # 1) Create a tiny normal image folder.
    in_dir = tmp_path / "in"
    for i in range(6):
        v = 20 + i * 10
        _write_rgb(in_dir / f"{i}.png", color=(v, v, v))

    # 2) Synthesize a dataset + manifest.
    out_root = tmp_path / "out"
    manifest = out_root / "manifest.jsonl"

    from pyimgano.synthesize_cli import synthesize_dataset

    synthesize_dataset(
        in_dir=in_dir,
        out_root=out_root,
        category="synthetic",
        preset="scratch",
        blend="alpha",
        alpha=0.9,
        seed=0,
        n_train=4,
        n_test_normal=1,
        n_test_anomaly=1,
        manifest_path=manifest,
        absolute_paths=True,
    )
    assert manifest.exists()

    # 3) Load split and run embedding+core pipeline.
    from pyimgano.datasets.manifest import load_manifest_benchmark_split

    split = load_manifest_benchmark_split(
        manifest_path=manifest,
        root_fallback=out_root,
        category="synthetic",
        resize=(64, 64),
        load_masks=False,
    )

    assert split.train_paths
    assert split.test_paths

    from pyimgano.models import create_model

    det = create_model(
        "vision_embedding_core",
        embedding_extractor="torchvision_backbone",
        embedding_kwargs={
            "backbone": "resnet18",
            "pretrained": False,
            "device": "cpu",
            "batch_size": 2,
            "image_size": 224,
        },
        core_detector="core_ecod",
        contamination=0.25,
    )

    det.fit(split.train_paths)
    scores = np.asarray(det.decision_function(split.test_paths), dtype=np.float64).reshape(-1)

    assert scores.shape == (len(split.test_paths),)
    assert np.all(np.isfinite(scores))
