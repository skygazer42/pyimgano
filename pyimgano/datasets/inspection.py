from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

from pyimgano.datasets.converters import (
    convert_dataset_to_manifest,
    get_dataset_converter,
    list_dataset_converters,
)
from pyimgano.datasets.manifest_tools import manifest_stats
from pyimgano.datasets.manifest_validate import validate_manifest_file


def _candidate_payload(
    *,
    name: str,
    confidence: float,
    reasons: list[str],
    category_candidates: list[str] | None = None,
    requires_category: bool = False,
    manifest_path: str | None = None,
) -> dict[str, Any]:
    return {
        "name": str(name),
        "confidence": float(confidence),
        "reasons": [str(item) for item in reasons],
        "category_candidates": sorted(str(item) for item in (category_candidates or [])),
        "requires_category": bool(requires_category),
        "manifest_path": (str(manifest_path) if manifest_path is not None else None),
    }


def _iter_dirs(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return [p for p in sorted(root.iterdir()) if p.is_dir()]


def _dir_has_any(root: Path, relative_paths: list[str]) -> bool:
    return any((root / rel_path).exists() for rel_path in relative_paths)


def _looks_like_custom_layout(root: Path) -> bool:
    return (root / "train" / "normal").is_dir() and (
        (root / "test" / "normal").is_dir() or (root / "test" / "anomaly").is_dir()
    )


def _looks_like_mvtec_ad2_category_dir(root: Path) -> bool:
    return (
        (root / "train" / "good").is_dir()
        and (root / "validation" / "good").is_dir()
        and ((root / "test_public" / "good").is_dir() or (root / "test_private" / "good").is_dir())
    )


def _looks_like_btad_category_dir(root: Path) -> bool:
    return (root / "train" / "ok").is_dir() and (
        (root / "test" / "ok").is_dir() or (root / "test" / "ko").is_dir()
    )


def _looks_like_visa_category_dir(root: Path) -> bool:
    train_dir = root / "train"
    test_dir = root / "test"
    return (
        train_dir.is_dir()
        and test_dir.is_dir()
        and _dir_has_any(
            root,
            [
                "train/good",
                "train/ok",
                "train/normal",
                "test/good",
                "test/ok",
                "test/normal",
                "test/bad",
                "test/ko",
                "test/anomaly",
            ],
        )
    )


def _find_category_dirs(root: Path, matcher: Callable[[Path], bool]) -> list[str]:
    categories: list[str] = []
    for child in _iter_dirs(root):
        if matcher(child):
            categories.append(child.name)
    return sorted(categories)


def _detect_manifest_candidate(target: Path) -> dict[str, Any] | None:
    if target.is_file() and target.suffix.lower() == ".jsonl":
        return _candidate_payload(
            name="manifest",
            confidence=1.0,
            reasons=["input is a .jsonl manifest file"],
            manifest_path=str(target),
        )

    manifest_file = target / "manifest.jsonl"
    if manifest_file.is_file():
        return _candidate_payload(
            name="manifest",
            confidence=0.99,
            reasons=["directory contains manifest.jsonl"],
            manifest_path=str(manifest_file),
        )
    return None


def detect_dataset_layout(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Dataset path not found: {target}")

    candidates: list[dict[str, Any]] = []

    manifest_candidate = _detect_manifest_candidate(target)
    if manifest_candidate is not None:
        candidates.append(manifest_candidate)

    if target.is_dir():
        if _looks_like_custom_layout(target):
            candidates.append(
                _candidate_payload(
                    name="custom",
                    confidence=0.95,
                    reasons=["matched train/normal + test/{normal,anomaly} layout"],
                    requires_category=False,
                )
            )

        mvtec_categories = _find_category_dirs(target, _looks_like_mvtec_ad2_category_dir)
        if mvtec_categories:
            candidates.append(
                _candidate_payload(
                    name="mvtec_ad2",
                    confidence=0.88,
                    reasons=["matched MVTec AD 2 category layout under root"],
                    category_candidates=mvtec_categories,
                    requires_category=True,
                )
            )

        btad_categories = _find_category_dirs(target, _looks_like_btad_category_dir)
        if btad_categories:
            candidates.append(
                _candidate_payload(
                    name="btad",
                    confidence=0.86,
                    reasons=["matched BTAD category layout under root"],
                    category_candidates=btad_categories,
                    requires_category=True,
                )
            )

        visa_root = target / "visa_pytorch" if (target / "visa_pytorch").is_dir() else target
        visa_categories = _find_category_dirs(visa_root, _looks_like_visa_category_dir)
        if visa_categories:
            candidates.append(
                _candidate_payload(
                    name="visa",
                    confidence=0.84,
                    reasons=["matched VisA-style category layout under root"],
                    category_candidates=visa_categories,
                    requires_category=True,
                )
            )

    candidates.sort(key=lambda item: (-float(item["confidence"]), str(item["name"])))
    detected = candidates[0]["name"] if candidates else None

    return {
        "path": str(target),
        "path_kind": ("file" if target.is_file() else "directory"),
        "detected": detected,
        "candidates": candidates,
    }


def _resolve_detection_or_dataset(
    *,
    root: str | Path,
    dataset: str,
) -> tuple[str, dict[str, Any] | None]:
    dataset_name = str(dataset).strip().lower()
    if dataset_name and dataset_name != "auto":
        get_dataset_converter(dataset_name)
        return dataset_name, None

    detected = detect_dataset_layout(root)
    resolved = detected.get("detected", None)
    if not isinstance(resolved, str) or not resolved.strip():
        raise ValueError(
            "Could not auto-detect dataset layout. Pass --dataset explicitly to import or lint."
        )
    if str(resolved) == "manifest":
        return "manifest", detected
    get_dataset_converter(str(resolved))
    return str(resolved), detected


def _resolve_category(
    *,
    dataset: str,
    detection: dict[str, Any] | None,
    category: str | None,
) -> str | None:
    explicit = str(category).strip() if category is not None else ""
    if explicit:
        return explicit

    converter = get_dataset_converter(dataset)
    if not bool(converter.requires_category):
        return None

    if not isinstance(detection, dict):
        raise ValueError(f"dataset={dataset!r} requires --category")

    candidates = detection.get("candidates", [])
    if isinstance(candidates, list):
        for item in candidates:
            if not isinstance(item, dict):
                continue
            if str(item.get("name", "")) != str(dataset):
                continue
            cats = item.get("category_candidates", [])
            if isinstance(cats, list) and len(cats) == 1:
                return str(cats[0])

    raise ValueError(f"dataset={dataset!r} requires --category")


def import_dataset_to_manifest_payload(
    *,
    root: str | Path,
    out_path: str | Path,
    dataset: str = "auto",
    category: str | None = None,
    absolute_paths: bool = False,
    include_masks: bool = False,
) -> dict[str, Any]:
    dataset_name, detection = _resolve_detection_or_dataset(root=root, dataset=dataset)
    if dataset_name == "manifest":
        raise ValueError(
            "dataset='manifest' does not support import; pass a directory dataset layout."
        )

    resolved_category = _resolve_category(
        dataset=dataset_name,
        detection=detection,
        category=category,
    )

    records = convert_dataset_to_manifest(
        dataset=dataset_name,
        root=root,
        out_path=out_path,
        category=resolved_category,
        absolute_paths=bool(absolute_paths),
        include_masks=bool(include_masks),
    )
    return {
        "dataset": str(dataset_name),
        "category": (str(resolved_category) if resolved_category is not None else None),
        "out_path": str(Path(out_path)),
        "record_count": int(len(records)),
    }


def lint_dataset_target(
    *,
    target: str | Path,
    dataset: str = "auto",
    category: str | None = None,
    root_fallback: str | Path | None = None,
) -> dict[str, Any]:
    target_path = Path(target)
    detection = (
        detect_dataset_layout(target_path) if str(dataset).strip().lower() == "auto" else None
    )
    dataset_name = str(dataset).strip().lower()
    if not dataset_name or dataset_name == "auto":
        dataset_name = str(detection.get("detected")) if isinstance(detection, dict) else ""

    if dataset_name == "manifest":
        manifest_path = target_path
        if target_path.is_dir():
            manifest_path = Path(str(detection["candidates"][0]["manifest_path"]))
        manifest_root = Path(root_fallback) if root_fallback is not None else manifest_path.parent
        validation = validate_manifest_file(
            manifest_path=manifest_path,
            root_fallback=manifest_root,
            check_files=True,
            category=category,
        )
        stats = manifest_stats(manifest_path=manifest_path, root_fallback=manifest_root)
        return {
            "target": str(target_path),
            "dataset": "manifest",
            "category": (str(category) if category is not None else None),
            "source_kind": "manifest",
            "manifest_path": str(manifest_path),
            "ok": bool(validation.ok),
            "validation": validation.to_jsonable(),
            "stats": stats,
        }

    if not dataset_name:
        raise ValueError("Could not detect dataset layout for lint. Pass --dataset explicitly.")

    resolved_category = _resolve_category(
        dataset=dataset_name,
        detection=detection,
        category=category,
    )
    with tempfile.TemporaryDirectory(prefix="pyimgano-datasets-lint-") as tmp_dir:
        manifest_path = Path(tmp_dir) / "manifest.jsonl"
        convert_dataset_to_manifest(
            dataset=dataset_name,
            root=target_path,
            out_path=manifest_path,
            category=resolved_category,
            absolute_paths=False,
            include_masks=True,
        )
        validation = validate_manifest_file(
            manifest_path=manifest_path,
            root_fallback=target_path,
            check_files=True,
            category=resolved_category,
        )
        stats = manifest_stats(manifest_path=manifest_path, root_fallback=target_path)

    return {
        "target": str(target_path),
        "dataset": str(dataset_name),
        "category": (str(resolved_category) if resolved_category is not None else None),
        "source_kind": "converted_temp",
        "manifest_path": None,
        "ok": bool(validation.ok),
        "validation": validation.to_jsonable(),
        "stats": stats,
    }


__all__ = [
    "detect_dataset_layout",
    "import_dataset_to_manifest_payload",
    "lint_dataset_target",
    "list_dataset_converters",
]
