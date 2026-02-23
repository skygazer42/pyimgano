from __future__ import annotations

from pathlib import Path


def list_dataset_categories(*, dataset: str, root: str) -> list[str]:
    """Return categories for a dataset name.

    This normalizes minor differences across loaders and prefers on-disk
    categories when they are available.
    """

    def _list_root_dirs(path: str) -> list[str]:
        root_path = Path(path)
        if not root_path.exists():
            return []
        return sorted(p.name for p in root_path.iterdir() if p.is_dir())

    from pyimgano.datasets.benchmarks import (
        BTADDataset,
        MVTecAD2Dataset,
        MVTecDataset,
        MVTecLOCODataset,
        VisADataset,
    )

    ds = str(dataset).lower()
    if ds in ("mvtec", "mvtec_ad"):
        on_disk = set(_list_root_dirs(root))
        known = set(MVTecDataset.CATEGORIES)
        found = sorted(on_disk & known)
        return found or list(MVTecDataset.CATEGORIES)
    if ds == "mvtec_loco":
        on_disk = set(_list_root_dirs(root))
        known = set(MVTecLOCODataset.CATEGORIES)
        found = sorted(on_disk & known)
        return found or list(MVTecLOCODataset.CATEGORIES)
    if ds == "mvtec_ad2":
        return list(MVTecAD2Dataset.list_categories(root))
    if ds == "visa":
        return list(VisADataset.list_categories(root))
    if ds == "btad":
        on_disk = set(_list_root_dirs(root))
        known = set(BTADDataset.CATEGORIES)
        found = sorted(on_disk & known)
        return found or list(BTADDataset.CATEGORIES)
    if ds == "custom":
        return ["custom"]

    raise ValueError(
        f"Unknown dataset: {dataset!r}. "
        "Choose from: mvtec, mvtec_ad, mvtec_loco, mvtec_ad2, visa, btad, custom."
    )

