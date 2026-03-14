from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np


def _group_id_for_record(record: Any, idx: int) -> str:
    gid = record.group_id if record.group_id is not None else f"__ungrouped__{idx}"
    return str(gid)


def _build_group_to_indices(records: Sequence[Any]) -> dict[str, list[int]]:
    group_to_indices: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        gid = _group_id_for_record(record, int(idx))
        group_to_indices.setdefault(gid, []).append(int(idx))
    return group_to_indices


def _group_split_conflicts(
    *,
    category: str,
    records: Sequence[Any],
    group_to_indices: Mapping[str, list[int]],
    issues: list[Any],
    issue_builder: Callable[..., Any],
) -> bool:
    group_conflict = False
    for gid, idxs in group_to_indices.items():
        splits = {records[i].split for i in idxs if records[i].split is not None}
        if len(splits) > 1:
            group_conflict = True
            issues.append(
                issue_builder(
                    "MANIFEST_GROUP_SPLIT_CONFLICT",
                    "error",
                    "Conflicting explicit split values within a group_id.",
                    context={
                        "category": str(category),
                        "group_id": str(gid),
                        "splits": sorted(splits),
                    },
                )
            )
            continue

        explicit_split = next(iter(splits), None)
        has_anomaly = any(records[i].label == 1 for i in idxs if records[i].label is not None)
        if explicit_split in ("train", "val") and has_anomaly:
            issues.append(
                issue_builder(
                    "MANIFEST_GROUP_ANOMALY_IN_TRAIN",
                    "error",
                    "Group contains anomalies but is explicitly assigned to train/val.",
                    context={
                        "category": str(category),
                        "group_id": str(gid),
                        "split": str(explicit_split),
                    },
                )
            )

    return bool(group_conflict)


def _classify_groups(
    *,
    records: Sequence[Any],
    group_to_indices: Mapping[str, list[int]],
) -> tuple[set[str], set[str], set[str], list[str]]:
    fixed_train: set[str] = set()
    fixed_val: set[str] = set()
    fixed_test: set[str] = set()
    normal_candidate: list[str] = []

    for gid, idxs in group_to_indices.items():
        splits = {records[i].split for i in idxs if records[i].split is not None}
        explicit_split = next(iter(splits), None)
        has_anomaly = any(records[i].label == 1 for i in idxs if records[i].label is not None)

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

    return fixed_train, fixed_val, fixed_test, normal_candidate


def _compute_target_test_normals(*, frac: float, total_normals: int, fixed_train: set[str]) -> int:
    target = int(np.ceil(total_normals * float(frac)))
    if 0.0 < frac < 1.0 and total_normals >= 2:
        target = max(1, min(total_normals - 1, target))
    target = max(0, min(total_normals, target))
    if total_normals > 0 and not fixed_train and target >= total_normals:
        target = max(0, total_normals - 1)
    return int(target)


def _select_test_normal_groups(
    *,
    category: str,
    policy: Any,
    group_to_indices: Mapping[str, list[int]],
    fixed_train: set[str],
    normal_candidate: list[str],
) -> set[str]:
    from pyimgano.datasets.manifest import _seed_for_category

    frac = float(getattr(policy, "test_normal_fraction", 0.2))
    total_normals = int(sum(len(group_to_indices[g]) for g in normal_candidate))
    target_test_normals = _compute_target_test_normals(
        frac=frac,
        total_normals=total_normals,
        fixed_train=fixed_train,
    )

    selected: set[str] = set()
    if target_test_normals <= 0 or not normal_candidate:
        return selected

    seed = int(getattr(policy, "seed", 0))
    rng = np.random.default_rng(_seed_for_category(seed, str(category)))
    order = list(range(len(normal_candidate)))
    rng.shuffle(order)

    running = 0
    for j in order:
        gid = normal_candidate[j]
        if gid in selected:
            continue
        selected.add(gid)
        running += int(len(group_to_indices[gid]))
        if running >= target_test_normals:
            break

    return selected


def _assignment_summary(
    *,
    records: Sequence[Any],
    mask_exists_by_index: Mapping[int, bool],
    fixed_val: set[str],
    fixed_test: set[str],
    selected_test_normal: set[str],
) -> tuple[dict[str, int], dict[str, Any], dict[str, Any]]:
    assigned_counts = {"train": 0, "val": 0, "test": 0, "calibration": 0}
    anomaly_test_total = 0
    anomaly_test_with_mask_path = 0
    anomaly_test_mask_exists = 0
    any_mask_path = False

    for idx, record in enumerate(records):
        gid = _group_id_for_record(record, int(idx))

        if gid in fixed_val:
            assigned_counts["val"] += 1
            continue

        in_test = gid in fixed_test or gid in selected_test_normal
        if not in_test:
            assigned_counts["train"] += 1
            continue

        assigned_counts["test"] += 1
        label = int(record.label) if record.label is not None else 0
        if label != 1:
            continue

        anomaly_test_total += 1
        if record.mask_path is not None:
            any_mask_path = True
            anomaly_test_with_mask_path += 1
            if bool(mask_exists_by_index.get(int(idx), False)):
                anomaly_test_mask_exists += 1
        else:
            any_mask_path = True

    assigned_counts["calibration"] = (
        int(assigned_counts["val"])
        if int(assigned_counts["val"]) > 0
        else int(assigned_counts["train"])
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

    return assigned_counts, mask_coverage, pixel_metrics


def analyze_manifest_category_assignment(
    *,
    category: str,
    records: Sequence[Any],
    policy: Any,
    mask_exists_by_index: dict[int, bool],
    issues: list[Any],
    issue_builder: Callable[..., Any],
) -> dict[str, Any]:
    group_to_indices = _build_group_to_indices(records)
    if _group_split_conflicts(
        category=str(category),
        records=records,
        group_to_indices=group_to_indices,
        issues=issues,
        issue_builder=issue_builder,
    ):
        return {
            "assigned_counts": None,
            "mask_coverage": None,
            "pixel_metrics": None,
        }

    fixed_train, fixed_val, fixed_test, normal_candidate = _classify_groups(
        records=records,
        group_to_indices=group_to_indices,
    )
    selected_test_normal = _select_test_normal_groups(
        category=str(category),
        policy=policy,
        group_to_indices=group_to_indices,
        fixed_train=fixed_train,
        normal_candidate=normal_candidate,
    )
    assigned_counts, mask_coverage, pixel_metrics = _assignment_summary(
        records=records,
        mask_exists_by_index=mask_exists_by_index,
        fixed_val=fixed_val,
        fixed_test=fixed_test,
        selected_test_normal=selected_test_normal,
    )

    return {
        "assigned_counts": assigned_counts,
        "mask_coverage": mask_coverage,
        "pixel_metrics": pixel_metrics,
    }


__all__ = ["analyze_manifest_category_assignment"]
