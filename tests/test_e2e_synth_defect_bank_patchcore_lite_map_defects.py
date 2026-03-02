from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _write_rgb(path: Path, *, seed: int) -> None:
    rng = np.random.default_rng(int(seed))
    arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_defect_bank(bank_dir: Path) -> None:
    pytest.importorskip("cv2")
    import cv2

    bank_dir.mkdir(parents=True, exist_ok=True)
    defect = np.zeros((28, 28, 3), dtype=np.uint8)
    defect[:, :] = (20, 20, 220)
    mask = np.zeros((28, 28), dtype=np.uint8)
    cv2.rectangle(mask, (6, 6), (22, 22), color=255, thickness=-1)
    cv2.imwrite(str(bank_dir / "sq.png"), defect)
    cv2.imwrite(str(bank_dir / "sq_mask.png"), mask)


class _DummyGridMeanEmbedder:
    def __init__(self, *, grid: int = 8) -> None:
        self.grid = int(grid)

    def embed(self, image):  # noqa: ANN001, ANN201 - test stub
        import cv2

        if isinstance(image, (str, Path)):
            arr = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if arr is None:
                raise ValueError(f"Failed to read image: {image}")
            arr = np.asarray(arr, dtype=np.uint8)
        else:
            arr = np.asarray(image, dtype=np.uint8)

        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected (H,W,3) image, got {arr.shape}")

        h, w = int(arr.shape[0]), int(arr.shape[1])
        g = int(self.grid)
        ph = max(1, h // g)
        pw = max(1, w // g)
        gh = max(1, h // ph)
        gw = max(1, w // pw)

        patches: list[np.ndarray] = []
        for iy in range(gh):
            for ix in range(gw):
                y0 = iy * ph
                x0 = ix * pw
                crop = arr[y0 : y0 + ph, x0 : x0 + pw, :].astype(np.float32)
                emb = crop.mean(axis=(0, 1)) / 255.0
                patches.append(emb.reshape(1, 3))

        patch_embeddings = np.concatenate(patches, axis=0).astype(np.float32, copy=False)
        return patch_embeddings, (gh, gw), (h, w)


def test_e2e_defect_bank_synth_then_patchcore_lite_map_defects_export(tmp_path: Path) -> None:
    pytest.importorskip("cv2")

    from pyimgano.datasets.manifest import load_manifest_benchmark_split
    from pyimgano.defects.extract import extract_defects_from_anomaly_map
    from pyimgano.models.registry import create_model
    from pyimgano.synthesize_cli import synthesize_dataset

    in_dir = tmp_path / "normals"
    for i in range(6):
        _write_rgb(in_dir / f"{i}.png", seed=10 + i)

    bank_dir = tmp_path / "bank"
    _write_defect_bank(bank_dir)

    out_root = tmp_path / "out"
    synthesize_dataset(
        in_dir=in_dir,
        out_root=out_root,
        category="demo",
        defect_bank_dir=bank_dir,
        blend="alpha",
        alpha=0.9,
        seed=0,
        n_train=4,
        n_test_normal=2,
        n_test_anomaly=2,
        num_defects=1,
        severity_range=(1.0, 1.0),
        manifest_path=out_root / "manifest.jsonl",
        absolute_paths=True,
    )

    split = load_manifest_benchmark_split(
        manifest_path=out_root / "manifest.jsonl",
        root_fallback=out_root,
        category="demo",
        resize=(64, 64),
        load_masks=True,
    )
    assert split.test_masks is not None
    assert len(split.train_paths) > 0
    assert len(split.test_paths) > 0

    det = create_model(
        "vision_patchcore_lite_map",
        embedder=_DummyGridMeanEmbedder(grid=8),
        contamination=0.2,
        n_neighbors=1,
        coreset_sampling_ratio=1.0,
        aggregation_method="topk_mean",
        aggregation_topk=0.05,
    )
    det.fit(split.train_paths)

    found_anomaly = False
    for i, (p, lab) in enumerate(zip(split.test_paths, split.test_labels.tolist())):
        if int(lab) != 1:
            continue
        found_anomaly = True

        amap = np.asarray(det.get_anomaly_map(p), dtype=np.float32)
        assert amap.shape == (64, 64)
        assert np.all(np.isfinite(amap))

        defects = extract_defects_from_anomaly_map(
            amap,
            pixel_threshold=float(np.quantile(amap, 0.98)),
            roi_xyxy_norm=None,
            mask_space="full",
            open_ksize=0,
            close_ksize=0,
            fill_holes=False,
            min_area=8,
            max_regions=None,
        )
        pred = np.asarray(defects["mask"], dtype=np.uint8)
        gt = np.asarray(split.test_masks[i], dtype=np.uint8)
        assert pred.shape == gt.shape == (64, 64)

        inter = int(np.sum((pred > 0) & (gt > 0)))
        assert inter > 0

    assert found_anomaly, "expected at least one anomaly sample in the synthesized test split"

