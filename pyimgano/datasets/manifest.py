from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ManifestSplitPolicy:
    """Controls how missing `split` fields are auto-assigned.

    Notes
    -----
    - v1 is intentionally narrow: industrial benchmarks usually assume that
      anomalies belong in test, while normals are split between train/test.
    - Splits are deterministic given the same manifest + seed.
    """

    mode: str = "benchmark"
    scope: str = "category"
    seed: int = 0
    test_normal_fraction: float = 0.2


@dataclass(frozen=True)
class ManifestRecord:
    image_path: str
    category: str
    split: str | None = None
    label: int | None = None
    mask_path: str | None = None
    group_id: str | None = None
    meta: Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, lineno: int) -> "ManifestRecord":
        def _require_str(key: str) -> str:
            value = raw.get(key, None)
            if value is None:
                raise ValueError(f"Manifest line {lineno}: missing required field {key!r}.")
            s = str(value).strip()
            if not s:
                raise ValueError(f"Manifest line {lineno}: field {key!r} must be non-empty.")
            return s

        image_path = _require_str("image_path")
        category = _require_str("category")

        split_raw = raw.get("split", None)
        split = None if split_raw is None else str(split_raw).strip().lower()
        if split is not None:
            if split not in ("train", "val", "test"):
                raise ValueError(
                    f"Manifest line {lineno}: invalid split {split!r}. "
                    "Supported: train, val, test."
                )

        label_raw = raw.get("label", None)
        label = None
        if label_raw is not None:
            try:
                label_int = int(label_raw)
            except Exception as exc:  # noqa: BLE001 - validation boundary
                raise ValueError(
                    f"Manifest line {lineno}: label must be 0/1, got {label_raw!r}."
                ) from exc
            if label_int not in (0, 1):
                raise ValueError(f"Manifest line {lineno}: label must be 0/1, got {label_int!r}.")
            label = label_int

        mask_path_raw = raw.get("mask_path", None)
        mask_path = None if mask_path_raw is None else str(mask_path_raw).strip()
        if mask_path is not None and not mask_path:
            mask_path = None

        group_id_raw = raw.get("group_id", None)
        group_id = None if group_id_raw is None else str(group_id_raw).strip()
        if group_id is not None and not group_id:
            group_id = None

        meta_raw = raw.get("meta", None)
        meta: Mapping[str, Any] | None
        if meta_raw is None:
            meta = None
        elif isinstance(meta_raw, Mapping):
            meta = meta_raw
        else:
            raise ValueError(
                f"Manifest line {lineno}: meta must be an object/dict, got {type(meta_raw).__name__}."
            )

        # Explicit-split validation rules.
        if split == "test" and label is None:
            raise ValueError(f"Manifest line {lineno}: split='test' requires an explicit label.")
        if split in ("train", "val") and label == 1:
            raise ValueError(
                f"Manifest line {lineno}: split={split!r} cannot have label=1 (anomaly)."
            )

        return cls(
            image_path=image_path,
            category=category,
            split=split,
            label=label,
            mask_path=mask_path,
            group_id=group_id,
            meta=meta,
        )


def iter_manifest_records(manifest_path: str | Path) -> Iterator[ManifestRecord]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            if text.startswith("#"):
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Manifest line {i}: invalid JSON ({exc.msg}).") from exc
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"Manifest line {i}: expected a JSON object, got {type(raw).__name__}."
                )
            yield ManifestRecord.from_mapping(raw, lineno=i)


def list_manifest_categories(manifest_path: str | Path) -> list[str]:
    cats: set[str] = set()
    for rec in iter_manifest_records(manifest_path):
        cats.add(str(rec.category))
    return sorted(cats)


def _seed_for_category(seed: int, category: str) -> int:
    payload = f"{int(seed)}:{category}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**63 - 1)


def _resolve_existing_path(
    raw_value: str,
    *,
    manifest_path: Path,
    root_fallback: Path | None,
) -> Path:
    raw = str(raw_value)
    p = Path(raw)
    if p.is_absolute():
        if p.exists():
            return p
        raise FileNotFoundError(f"Path not found: {p}")

    cand1 = (manifest_path.parent / p).resolve()
    if cand1.exists():
        return cand1

    if root_fallback is not None:
        cand2 = (root_fallback / p).resolve()
        if cand2.exists():
            return cand2

    raise FileNotFoundError(
        "Path not found. Tried: "
        f"{cand1} (manifest-dir) and {((root_fallback / p).resolve() if root_fallback else '<no root>')} (root)."
    )


@dataclass(frozen=True)
class ManifestBenchmarkSplit:
    train_paths: list[str]
    calibration_paths: list[str]
    test_paths: list[str]
    test_labels: np.ndarray
    test_masks: np.ndarray | None
    pixel_skip_reason: str | None = None
    test_meta: list[Mapping[str, Any] | None] | None = None


def load_manifest_benchmark_split(
    *,
    manifest_path: str | Path,
    root_fallback: str | Path | None,
    category: str,
    resize: tuple[int, int] = (256, 256),
    load_masks: bool = True,
    split_policy: ManifestSplitPolicy | None = None,
) -> ManifestBenchmarkSplit:
    """Load a benchmark-style split from a JSONL manifest.

    Parameters
    ----------
    manifest_path:
        JSONL file path.
    root_fallback:
        Used only for resolving relative `image_path`/`mask_path` values when the
        file does not exist relative to the manifest directory.
    category:
        Category name.
    resize:
        Target (H, W) for masks (and for consistency with benchmark loaders).
    load_masks:
        When false, masks are not loaded and pixel metrics can’t be computed.
    split_policy:
        Auto split policy used when records are missing `split`.
    """

    mp = Path(manifest_path)
    root_path = None if root_fallback is None else Path(root_fallback)
    policy = split_policy or ManifestSplitPolicy()

    cat = str(category)
    records: list[ManifestRecord] = [r for r in iter_manifest_records(mp) if str(r.category) == cat]
    if not records:
        raise ValueError(f"Manifest contains no records for category={cat!r}.")

    if policy.scope != "category":
        raise ValueError(
            "Only split_policy.scope='category' is supported for v1. " f"Got: {policy.scope!r}."
        )
    if policy.mode != "benchmark":
        raise ValueError(
            "Only split_policy.mode='benchmark' is supported for v1. " f"Got: {policy.mode!r}."
        )
    frac = float(policy.test_normal_fraction)
    if not (0.0 <= frac <= 1.0):
        raise ValueError(
            f"split_policy.test_normal_fraction must be in [0,1]. Got: {policy.test_normal_fraction!r}."
        )

    # Group records (group-aware split when group_id present).
    group_to_indices: dict[str, list[int]] = {}
    for idx, rec in enumerate(records):
        g = rec.group_id if rec.group_id is not None else f"__ungrouped__{idx}"
        group_to_indices.setdefault(str(g), []).append(int(idx))

    group_ids = list(group_to_indices.keys())

    # Determine fixed group split, and list normal candidate groups.
    fixed_train: set[str] = set()
    fixed_val: set[str] = set()
    fixed_test: set[str] = set()
    normal_candidate: list[str] = []

    for gid in group_ids:
        idxs = group_to_indices[gid]
        splits = {records[i].split for i in idxs if records[i].split is not None}
        if len(splits) > 1:
            raise ValueError(
                f"Manifest group_id={gid!r} has conflicting explicit splits: {sorted(splits)}."
            )
        explicit_split = next(iter(splits), None)

        # Any anomaly label in the group forces test.
        has_anomaly = any(records[i].label == 1 for i in idxs if records[i].label is not None)
        if explicit_split in ("train", "val") and has_anomaly:
            raise ValueError(
                f"Manifest group_id={gid!r} has split={explicit_split!r} but contains label=1 records."
            )

        if explicit_split == "train":
            fixed_train.add(gid)
            continue
        if explicit_split == "val":
            fixed_val.add(gid)
            continue
        if explicit_split == "test":
            fixed_test.add(gid)
            continue

        # No explicit split: apply benchmark policy.
        if has_anomaly:
            fixed_test.add(gid)
        else:
            normal_candidate.append(gid)

    # Select a fraction of normal candidate groups into test.
    normal_candidate_counts = [len(group_to_indices[gid]) for gid in normal_candidate]
    total_normals = int(sum(normal_candidate_counts))
    target_test_normals = int(math.ceil(total_normals * frac))
    if 0.0 < frac < 1.0 and total_normals >= 2:
        target_test_normals = max(1, min(total_normals - 1, target_test_normals))
    target_test_normals = max(0, min(total_normals, target_test_normals))
    # If the split would produce an empty train set (no explicit train items),
    # bias toward keeping at least one normal sample for training.
    if total_normals > 0 and not fixed_train and target_test_normals >= total_normals:
        target_test_normals = max(0, total_normals - 1)

    selected_test_normal: set[str] = set()
    if target_test_normals > 0 and normal_candidate:
        rng = np.random.default_rng(_seed_for_category(int(policy.seed), cat))
        order = list(range(len(normal_candidate)))
        rng.shuffle(order)
        running = 0
        for j in order:
            gid = normal_candidate[j]
            if gid in selected_test_normal:
                continue
            selected_test_normal.add(gid)
            running += int(len(group_to_indices[gid]))
            if running >= target_test_normals:
                break

    # Assemble outputs preserving original record order.
    train_paths: list[str] = []
    cal_paths: list[str] = []
    test_paths: list[str] = []
    test_labels: list[int] = []
    test_mask_paths: list[str | None] = []
    test_meta: list[Mapping[str, Any] | None] = []
    missing_anomaly_mask = False

    for idx, rec in enumerate(records):
        gid = rec.group_id if rec.group_id is not None else f"__ungrouped__{idx}"
        gid = str(gid)

        if gid in fixed_val:
            cal_paths.append(
                str(
                    _resolve_existing_path(
                        rec.image_path, manifest_path=mp, root_fallback=root_path
                    )
                )
            )
            continue

        in_test = gid in fixed_test or gid in selected_test_normal
        if not in_test and gid in fixed_train:
            train_paths.append(
                str(
                    _resolve_existing_path(
                        rec.image_path, manifest_path=mp, root_fallback=root_path
                    )
                )
            )
            continue
        if not in_test and gid not in fixed_test and gid not in selected_test_normal:
            # Unspecified normal group: default to train.
            train_paths.append(
                str(
                    _resolve_existing_path(
                        rec.image_path, manifest_path=mp, root_fallback=root_path
                    )
                )
            )
            continue

        # Test record (explicit or assigned).
        label = int(rec.label) if rec.label is not None else 0
        test_paths.append(
            str(_resolve_existing_path(rec.image_path, manifest_path=mp, root_fallback=root_path))
        )
        test_labels.append(label)
        if rec.meta is None:
            test_meta.append(None)
        else:
            test_meta.append(dict(rec.meta))

        if not load_masks:
            test_mask_paths.append(None)
            continue

        if label == 1:
            if rec.mask_path is None:
                missing_anomaly_mask = True
                test_mask_paths.append(None)
            else:
                try:
                    resolved = _resolve_existing_path(
                        rec.mask_path, manifest_path=mp, root_fallback=root_path
                    )
                except FileNotFoundError:
                    missing_anomaly_mask = True
                    test_mask_paths.append(None)
                else:
                    test_mask_paths.append(str(resolved))
        else:
            # Normal: masks optional; treat missing as background.
            if rec.mask_path is None:
                test_mask_paths.append(None)
            else:
                try:
                    resolved = _resolve_existing_path(
                        rec.mask_path, manifest_path=mp, root_fallback=root_path
                    )
                except FileNotFoundError:
                    test_mask_paths.append(None)
                else:
                    test_mask_paths.append(str(resolved))

    if not train_paths:
        raise ValueError(f"Manifest category={cat!r} produced an empty train split.")
    if not test_paths:
        raise ValueError(f"Manifest category={cat!r} produced an empty test split.")

    masks_arr: np.ndarray | None = None
    pixel_skip_reason: str | None = None
    meta_out: list[Mapping[str, Any] | None] | None = None
    if load_masks:
        if any(p is not None for p in test_mask_paths):
            if any(
                lab == 1 for lab, p in zip(test_labels, test_mask_paths) if p is None and lab == 1
            ):
                missing_anomaly_mask = True
        if missing_anomaly_mask:
            pixel_skip_reason = (
                "Missing mask_path (or missing mask files) for anomaly test samples."
            )
            masks_arr = None
        elif any(p is not None for p in test_mask_paths):
            masks_arr = _load_masks_or_zeros(test_mask_paths, resize=resize)

    if any(m is not None for m in test_meta):
        meta_out = test_meta

    return ManifestBenchmarkSplit(
        train_paths=train_paths,
        calibration_paths=cal_paths,
        test_paths=test_paths,
        test_labels=np.asarray(test_labels, dtype=np.int64),
        test_masks=masks_arr,
        pixel_skip_reason=pixel_skip_reason,
        test_meta=meta_out,
    )


def _load_masks_or_zeros(
    mask_paths: Sequence[str | None], *, resize: tuple[int, int]
) -> np.ndarray:
    import cv2

    h, w = int(resize[0]), int(resize[1])
    out: list[np.ndarray] = []
    for p in mask_paths:
        if p is None:
            out.append(np.zeros((h, w), dtype=np.uint8))
            continue
        arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            out.append(np.zeros((h, w), dtype=np.uint8))
            continue
        if arr.shape != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
        out.append((arr > 127).astype(np.uint8))
    return np.stack(out, axis=0)
