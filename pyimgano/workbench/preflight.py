from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from pyimgano.workbench.config import WorkbenchConfig


IssueSeverity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class PreflightIssue:
    code: str
    severity: IssueSeverity
    message: str
    context: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PreflightReport:
    dataset: str
    category: str
    summary: dict[str, Any]
    issues: list[PreflightIssue]


def run_preflight(*, config: WorkbenchConfig) -> PreflightReport:
    """Run best-effort dataset validation and return a JSON-friendly report."""

    dataset = str(config.dataset.name)
    category = str(config.dataset.category)

    issues: list[PreflightIssue] = []
    summary: dict[str, Any] = {}

    ds = dataset.lower()
    if ds == "manifest":
        summary = _preflight_manifest(config=config, issues=issues)
    else:
        summary = _preflight_non_manifest(config=config, issues=issues)

    return PreflightReport(dataset=dataset, category=category, summary=summary, issues=issues)


def _issue(
    code: str,
    severity: IssueSeverity,
    message: str,
    *,
    context: Mapping[str, Any] | None = None,
) -> PreflightIssue:
    return PreflightIssue(code=str(code), severity=severity, message=str(message), context=context)


def _preflight_manifest(*, config: WorkbenchConfig, issues: list[PreflightIssue]) -> dict[str, Any]:
    if str(config.dataset.input_mode) != "paths":
        issues.append(
            _issue(
                "MANIFEST_UNSUPPORTED_INPUT_MODE",
                "error",
                "dataset.name='manifest' supports only dataset.input_mode='paths'.",
                context={"input_mode": str(config.dataset.input_mode)},
            )
        )

    mp_raw = config.dataset.manifest_path
    if mp_raw is None:
        issues.append(
            _issue(
                "MANIFEST_PATH_MISSING",
                "error",
                "dataset.manifest_path is required when dataset.name='manifest'.",
            )
        )
        return {"manifest": {"ok": False}}

    mp = Path(str(mp_raw))
    if not mp.exists():
        issues.append(
            _issue(
                "MANIFEST_NOT_FOUND",
                "error",
                "Manifest file not found.",
                context={"manifest_path": str(mp)},
            )
        )
        return {"manifest_path": str(mp), "manifest": {"ok": False}}
    if not mp.is_file():
        issues.append(
            _issue(
                "MANIFEST_NOT_A_FILE",
                "error",
                "Manifest path must be a file.",
                context={"manifest_path": str(mp)},
            )
        )
        return {"manifest_path": str(mp), "manifest": {"ok": False}}
    try:
        with mp.open("r", encoding="utf-8") as f:
            f.read(1)
    except Exception as exc:  # noqa: BLE001 - validation boundary
        issues.append(
            _issue(
                "MANIFEST_NOT_READABLE",
                "error",
                "Manifest file is not readable.",
                context={"manifest_path": str(mp), "error": str(exc)},
            )
        )
        return {"manifest_path": str(mp), "manifest": {"ok": False}}

    root_fallback = Path(str(config.dataset.root)) if config.dataset.root is not None else None
    if root_fallback is not None and not root_fallback.exists():
        issues.append(
            _issue(
                "DATASET_ROOT_MISSING",
                "warning",
                "dataset.root does not exist; root fallback will not be used for resolving relative paths.",
                context={"root": str(root_fallback)},
            )
        )

    policy = _manifest_split_policy_from_config(config=config)

    records, raw_categories = _load_manifest_records_best_effort(
        manifest_path=mp, issues=issues
    )
    if not records:
        issues.append(
            _issue(
                "MANIFEST_EMPTY",
                "error",
                "Manifest contains no valid records.",
                context={"manifest_path": str(mp)},
            )
        )
        return {"manifest_path": str(mp), "manifest": {"ok": False}}

    requested_category = str(config.dataset.category)
    categories: list[str]
    if requested_category.lower() == "all":
        categories = sorted(raw_categories)
    else:
        categories = [requested_category]
        if requested_category not in raw_categories:
            issues.append(
                _issue(
                    "MANIFEST_CATEGORY_EMPTY",
                    "error",
                    "Manifest contains no records for the requested category.",
                    context={
                        "category": str(requested_category),
                        "available_categories": sorted(raw_categories),
                    },
                )
            )

    per_category: dict[str, Any] = {}
    for cat in categories:
        cat_records = [r for r in records if str(r.category) == str(cat)]
        per_category[str(cat)] = _preflight_manifest_category(
            category=str(cat),
            records=cat_records,
            manifest_path=mp,
            root_fallback=root_fallback,
            policy=policy,
            issues=issues,
        )

    out: dict[str, Any] = {
        "manifest_path": str(mp),
        "root_fallback": (str(root_fallback) if root_fallback is not None else None),
        "split_policy": {
            "mode": str(policy.mode),
            "scope": str(policy.scope),
            "seed": int(policy.seed),
            "test_normal_fraction": float(policy.test_normal_fraction),
        },
        "categories": categories,
        "per_category": per_category if requested_category.lower() == "all" else None,
    }
    if requested_category.lower() != "all" and categories:
        out.update(per_category.get(str(categories[0]), {}))
    out["manifest"] = {"ok": True}
    return out


def _manifest_split_policy_from_config(*, config: WorkbenchConfig):
    from pyimgano.datasets.manifest import ManifestSplitPolicy

    sp = config.dataset.split_policy
    seed = (
        int(sp.seed)
        if sp.seed is not None
        else (int(config.seed) if config.seed is not None else 0)
    )
    return ManifestSplitPolicy(
        mode=str(sp.mode),
        scope=str(sp.scope),
        seed=seed,
        test_normal_fraction=float(sp.test_normal_fraction),
    )


def _load_manifest_records_best_effort(*, manifest_path: Path, issues: list[PreflightIssue]):
    from pyimgano.datasets.manifest import ManifestRecord

    records: list[ManifestRecord] = []
    raw_categories: set[str] = set()

    with manifest_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            if text.startswith("#"):
                continue

            try:
                raw = _parse_manifest_json(text, lineno=lineno)
            except Exception as exc:  # noqa: BLE001 - preflight boundary
                issues.append(
                    _issue(
                        "MANIFEST_INVALID_JSON",
                        "error",
                        "Invalid JSON line in manifest.",
                        context={"lineno": int(lineno), "error": str(exc)},
                    )
                )
                continue

            try:
                rec = ManifestRecord.from_mapping(raw, lineno=lineno)
            except Exception as exc:  # noqa: BLE001 - validation boundary
                issues.append(
                    _issue(
                        "MANIFEST_INVALID_RECORD",
                        "error",
                        "Invalid manifest record.",
                        context={"lineno": int(lineno), "error": str(exc)},
                    )
                )
                continue

            records.append(rec)
            raw_categories.add(str(rec.category))

    return records, raw_categories


def _parse_manifest_json(text: str, *, lineno: int) -> Mapping[str, Any]:
    import json

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest line {lineno}: invalid JSON ({exc.msg}).") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"Manifest line {lineno}: expected JSON object, got {type(raw).__name__}."
        )
    return raw


def _resolve_manifest_path_best_effort(
    raw_value: str,
    *,
    manifest_path: Path,
    root_fallback: Path | None,
) -> tuple[str, bool, str]:
    raw = str(raw_value)
    p = Path(raw)
    if p.is_absolute():
        resolved = p.resolve()
        return str(resolved), bool(resolved.exists()), "absolute"

    cand1 = (manifest_path.parent / p).resolve()
    if cand1.exists():
        return str(cand1), True, "manifest_dir"

    if root_fallback is not None:
        cand2 = (root_fallback / p).resolve()
        if cand2.exists():
            return str(cand2), True, "root_fallback"

    return str(cand1), False, "manifest_dir"


def _preflight_manifest_category(
    *,
    category: str,
    records: Sequence[Any],
    manifest_path: Path,
    root_fallback: Path | None,
    policy: Any,
    issues: list[PreflightIssue],
) -> dict[str, Any]:
    from pyimgano.datasets.manifest import ManifestRecord, _seed_for_category

    recs: list[ManifestRecord] = [r for r in records if isinstance(r, ManifestRecord)]
    counts_by_split = {"train": 0, "val": 0, "test": 0, "unspecified": 0}
    explicit_test_labels = {"normal": 0, "anomaly": 0}
    for r in recs:
        if r.split is None:
            counts_by_split["unspecified"] += 1
        else:
            counts_by_split[str(r.split)] += 1
        if r.split == "test":
            if int(r.label or 0) == 1:
                explicit_test_labels["anomaly"] += 1
            else:
                explicit_test_labels["normal"] += 1

    # File existence checks + duplicates.
    image_seen: dict[str, int] = {}
    for r in recs:
        resolved, exists, source = _resolve_manifest_path_best_effort(
            r.image_path, manifest_path=manifest_path, root_fallback=root_fallback
        )
        image_seen[resolved] = int(image_seen.get(resolved, 0)) + 1
        if not exists:
            issues.append(
                _issue(
                    "MANIFEST_MISSING_IMAGE",
                    "error",
                    "Manifest image_path does not exist.",
                    context={
                        "category": str(category),
                        "image_path": str(r.image_path),
                        "resolved": str(resolved),
                        "resolution": str(source),
                    },
                )
            )

        if r.mask_path is not None:
            mask_resolved, mask_exists, mask_source = _resolve_manifest_path_best_effort(
                r.mask_path, manifest_path=manifest_path, root_fallback=root_fallback
            )
            if not mask_exists:
                issues.append(
                    _issue(
                        "MANIFEST_MISSING_MASK",
                        "warning",
                        "Manifest mask_path does not exist.",
                        context={
                            "category": str(category),
                            "mask_path": str(r.mask_path),
                            "resolved": str(mask_resolved),
                            "resolution": str(mask_source),
                        },
                    )
                )

    for path, count in sorted(image_seen.items()):
        if int(count) > 1:
            issues.append(
                _issue(
                    "MANIFEST_DUPLICATE_IMAGE",
                    "warning",
                    "Duplicate image_path detected within category.",
                    context={"category": str(category), "resolved": str(path), "count": int(count)},
                )
            )

    # Group leakage / split consistency checks.
    group_to_indices: dict[str, list[int]] = {}
    for idx, r in enumerate(recs):
        gid = r.group_id if r.group_id is not None else f"__ungrouped__{idx}"
        group_to_indices.setdefault(str(gid), []).append(int(idx))

    group_conflict = False
    for gid, idxs in group_to_indices.items():
        splits = {recs[i].split for i in idxs if recs[i].split is not None}
        if len(splits) > 1:
            group_conflict = True
            issues.append(
                _issue(
                    "MANIFEST_GROUP_SPLIT_CONFLICT",
                    "error",
                    "Conflicting explicit split values within a group_id.",
                    context={"category": str(category), "group_id": str(gid), "splits": sorted(splits)},
                )
            )
            continue

        explicit_split = next(iter(splits), None)
        has_anomaly = any(recs[i].label == 1 for i in idxs if recs[i].label is not None)
        if explicit_split in ("train", "val") and has_anomaly:
            issues.append(
                _issue(
                    "MANIFEST_GROUP_ANOMALY_IN_TRAIN",
                    "error",
                    "Group contains anomalies but is explicitly assigned to train/val.",
                    context={"category": str(category), "group_id": str(gid), "split": str(explicit_split)},
                )
            )

    assigned_counts: dict[str, int] | None = None
    mask_coverage: dict[str, Any] | None = None
    pixel_metrics: dict[str, Any] | None = None

    if not group_conflict:
        fixed_train: set[str] = set()
        fixed_val: set[str] = set()
        fixed_test: set[str] = set()
        normal_candidate: list[str] = []
        for gid, idxs in group_to_indices.items():
            splits = {recs[i].split for i in idxs if recs[i].split is not None}
            explicit_split = next(iter(splits), None)
            has_anomaly = any(recs[i].label == 1 for i in idxs if recs[i].label is not None)

            if explicit_split == "train":
                fixed_train.add(gid)
                continue
            if explicit_split == "val":
                fixed_val.add(gid)
                continue
            if explicit_split == "test":
                fixed_test.add(gid)
                continue

            if has_anomaly:
                fixed_test.add(gid)
            else:
                normal_candidate.append(gid)

        frac = float(getattr(policy, "test_normal_fraction", 0.2))
        total_normals = int(sum(len(group_to_indices[g]) for g in normal_candidate))
        target_test_normals = int(np.ceil(total_normals * frac))
        if 0.0 < frac < 1.0 and total_normals >= 2:
            target_test_normals = max(1, min(total_normals - 1, target_test_normals))
        target_test_normals = max(0, min(total_normals, target_test_normals))
        if total_normals > 0 and not fixed_train and target_test_normals >= total_normals:
            target_test_normals = max(0, total_normals - 1)

        selected_test_normal: set[str] = set()
        if target_test_normals > 0 and normal_candidate:
            seed = int(getattr(policy, "seed", 0))
            rng = np.random.default_rng(_seed_for_category(seed, str(category)))
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

        assigned_counts = {"train": 0, "val": 0, "test": 0, "calibration": 0}
        anomaly_test_total = 0
        anomaly_test_with_mask_path = 0
        anomaly_test_mask_exists = 0
        any_mask_path = False

        for idx, r in enumerate(recs):
            gid = r.group_id if r.group_id is not None else f"__ungrouped__{idx}"
            gid = str(gid)

            if gid in fixed_val:
                assigned_counts["val"] += 1
                continue

            in_test = gid in fixed_test or gid in selected_test_normal
            if not in_test:
                assigned_counts["train"] += 1
                continue

            assigned_counts["test"] += 1
            label = int(r.label) if r.label is not None else 0
            if label == 1:
                anomaly_test_total += 1
                if r.mask_path is not None:
                    any_mask_path = True
                    anomaly_test_with_mask_path += 1
                    _resolved, exists, _src = _resolve_manifest_path_best_effort(
                        r.mask_path, manifest_path=manifest_path, root_fallback=root_fallback
                    )
                    if exists:
                        anomaly_test_mask_exists += 1
                else:
                    any_mask_path = True

        assigned_counts["calibration"] = (
            int(assigned_counts["val"]) if int(assigned_counts["val"]) > 0 else int(assigned_counts["train"])
        )

        mask_coverage = {
            "anomaly_test_total": int(anomaly_test_total),
            "anomaly_test_with_mask_path": int(anomaly_test_with_mask_path),
            "anomaly_test_mask_exists": int(anomaly_test_mask_exists),
            "fraction_with_mask_path": (
                float(anomaly_test_with_mask_path / anomaly_test_total) if anomaly_test_total else None
            ),
            "fraction_mask_exists": (
                float(anomaly_test_mask_exists / anomaly_test_total) if anomaly_test_total else None
            ),
        }

        pixel_enabled = True
        pixel_reason = None
        if not any_mask_path:
            pixel_enabled = False
            pixel_reason = "No mask_path entries found."
        elif anomaly_test_total and anomaly_test_mask_exists != anomaly_test_total:
            pixel_enabled = False
            pixel_reason = "Missing masks for anomaly test samples."
        pixel_metrics = {"enabled": bool(pixel_enabled), "reason": pixel_reason}

    return {
        "counts": {
            "total": int(len(recs)),
            "explicit_by_split": counts_by_split,
            "explicit_test_labels": explicit_test_labels,
        },
        "assigned_counts": assigned_counts,
        "mask_coverage": mask_coverage,
        "pixel_metrics": pixel_metrics,
    }


def _preflight_non_manifest(*, config: WorkbenchConfig, issues: list[PreflightIssue]) -> dict[str, Any]:
    from pyimgano.datasets.catalog import list_dataset_categories

    dataset = str(config.dataset.name)
    root = Path(str(config.dataset.root))
    category = str(config.dataset.category)

    if not root.exists():
        issues.append(
            _issue(
                "DATASET_ROOT_MISSING",
                "error",
                "Dataset root does not exist.",
                context={"dataset": dataset, "root": str(root)},
            )
        )
        return {"dataset_root": str(root), "ok": False}

    if dataset.lower() == "custom":
        try:
            from pyimgano.utils.datasets import CustomDataset

            CustomDataset(root=str(root), load_masks=True).validate_structure()
        except Exception as exc:  # noqa: BLE001 - validation boundary
            issues.append(
                _issue(
                    "CUSTOM_DATASET_INVALID_STRUCTURE",
                    "error",
                    "Custom dataset layout validation failed.",
                    context={"root": str(root), "error": str(exc)},
                )
            )

    try:
        categories = list_dataset_categories(
            dataset=dataset,
            root=str(root),
            manifest_path=(
                str(config.dataset.manifest_path) if config.dataset.manifest_path else None
            ),
        )
    except Exception as exc:  # noqa: BLE001 - validation boundary
        issues.append(
            _issue(
                "DATASET_CATEGORY_LIST_FAILED",
                "error",
                "Unable to list dataset categories.",
                context={"dataset": dataset, "root": str(root), "error": str(exc)},
            )
        )
        return {"dataset_root": str(root), "ok": False}

    if category.lower() != "all":
        if str(category) not in set(categories):
            issues.append(
                _issue(
                    "DATASET_CATEGORY_EMPTY",
                    "error",
                    "Requested category not found in dataset.",
                    context={
                        "dataset": dataset,
                        "root": str(root),
                        "category": str(category),
                        "available_categories": categories,
                    },
                )
            )

    return {"dataset_root": str(root), "categories": categories, "ok": True}
