from __future__ import annotations

"""Dataset → JSONL manifest converters (paths-first).

These converters are intended for industrial workflows where a stable manifest is
the interchange format between:
- dataset conversion / curation
- synthesis / augmentation
- evaluation / benchmarking
- inference / defects export
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pyimgano.utils.path_normalize import normalize_path


SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _iter_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    out: list[Path] = []
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            out.append(p)
    return out


def _relpath(path: Path, *, base: Path, absolute: bool) -> str:
    if absolute:
        return normalize_path(path.resolve())
    try:
        rel = os.path.relpath(str(path.resolve()), start=str(base.resolve()))
        return normalize_path(rel)
    except Exception:
        return normalize_path(str(path))


def convert_custom_layout_to_manifest(
    *,
    root: str | Path,
    out_path: str | Path,
    category: str = "custom",
    absolute_paths: bool = False,
    include_masks: bool = False,
) -> list[dict[str, Any]]:
    """Convert the built-in `custom` dataset layout into a JSONL manifest.

    Expected directory structure:

    root/
      train/normal/*.png|*.jpg|*.jpeg|*.bmp
      test/normal/*.png|*.jpg|*.jpeg|*.bmp
      test/anomaly/*.png|*.jpg|*.jpeg|*.bmp
      ground_truth/anomaly/<stem>_mask.png   (optional)
    """

    root_path = Path(root)
    out_file = Path(out_path)
    out_dir = out_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cat = str(category).strip() or "custom"

    train_dir = root_path / "train" / "normal"
    test_normal_dir = root_path / "test" / "normal"
    test_anomaly_dir = root_path / "test" / "anomaly"
    gt_anomaly_dir = root_path / "ground_truth" / "anomaly"

    records: list[dict[str, Any]] = []

    for p in _iter_images(train_dir):
        records.append(
            {
                "image_path": _relpath(p, base=out_dir, absolute=absolute_paths),
                "category": cat,
                "split": "train",
            }
        )

    for p in _iter_images(test_normal_dir):
        records.append(
            {
                "image_path": _relpath(p, base=out_dir, absolute=absolute_paths),
                "category": cat,
                "split": "test",
                "label": 0,
            }
        )

    for p in _iter_images(test_anomaly_dir):
        rec: dict[str, Any] = {
            "image_path": _relpath(p, base=out_dir, absolute=absolute_paths),
            "category": cat,
            "split": "test",
            "label": 1,
        }
        if include_masks:
            mask_candidate = gt_anomaly_dir / f"{p.stem}_mask.png"
            if mask_candidate.exists():
                rec["mask_path"] = _relpath(mask_candidate, base=out_dir, absolute=absolute_paths)
        records.append(rec)

    # Stable output order for reproducibility.
    records.sort(key=lambda r: (str(r.get("split", "")), str(r.get("image_path", ""))))

    with out_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records


ConverterFn = Callable[..., list[dict[str, Any]]]


@dataclass(frozen=True)
class DatasetConverter:
    name: str
    description: str
    requires_category: bool
    convert: ConverterFn


def _build_registry() -> dict[str, DatasetConverter]:
    from pyimgano.datasets.mvtec_ad2 import convert_mvtec_ad2_to_manifest
    from pyimgano.datasets.rad import convert_rad_to_manifest
    from pyimgano.datasets.real_iad import convert_real_iad_to_manifest

    items = [
        DatasetConverter(
            name="custom",
            description="Built-in custom layout (train/normal + test/{normal,anomaly} + optional ground_truth masks).",
            requires_category=False,
            convert=convert_custom_layout_to_manifest,
        ),
        DatasetConverter(
            name="mvtec_ad2",
            description="MVTec AD 2 category converter (paths-first, public split layout).",
            requires_category=True,
            convert=convert_mvtec_ad2_to_manifest,
        ),
        DatasetConverter(
            name="real_iad",
            description="Real-IAD-like converter with best-effort layout recognition (study-friendly).",
            requires_category=False,
            convert=convert_real_iad_to_manifest,
        ),
        DatasetConverter(
            name="rad",
            description="RAD-like converter with multi-view metadata in meta (best-effort).",
            requires_category=False,
            convert=convert_rad_to_manifest,
        ),
    ]
    return {c.name: c for c in items}


_REGISTRY = _build_registry()


def list_dataset_converters() -> list[DatasetConverter]:
    return [c for _name, c in sorted(_REGISTRY.items(), key=lambda kv: kv[0])]


def get_dataset_converter(name: str) -> DatasetConverter:
    key = str(name).strip().lower()
    if key in _REGISTRY:
        return _REGISTRY[key]
    available = ", ".join(sorted(_REGISTRY)) or "<empty>"
    raise KeyError(f"Unknown converter: {name!r}. Available: {available}")


def convert_dataset_to_manifest(
    *,
    dataset: str,
    root: str | Path,
    out_path: str | Path,
    category: str | None = None,
    absolute_paths: bool = False,
    include_masks: bool = False,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """High-level helper: resolve a converter and run it."""

    conv = get_dataset_converter(dataset)
    if conv.requires_category:
        cat = str(category).strip() if category is not None else ""
        if not cat:
            raise ValueError(f"dataset={dataset!r} requires --category")
    else:
        cat = str(category).strip() if category is not None else ""
        if not cat:
            cat = str(dataset).strip() or "custom"

    # Converters are responsible for validating their own extra kwargs.
    return conv.convert(
        root=root,
        out_path=out_path,
        category=cat,
        absolute_paths=bool(absolute_paths),
        include_masks=bool(include_masks),
        **dict(kwargs),
    )


__all__ = [
    "DatasetConverter",
    "SUPPORTED_EXTENSIONS",
    "convert_custom_layout_to_manifest",
    "convert_dataset_to_manifest",
    "get_dataset_converter",
    "list_dataset_converters",
]
