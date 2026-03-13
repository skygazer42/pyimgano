from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np


def analyze_manifest_category_assignment(
    *,
    category: str,
    records: Sequence[Any],
    policy: Any,
    mask_exists_by_index: dict[int, bool],
    issues: list[Any],
    issue_builder: Callable[..., Any],
) -> dict[str, Any]:
    from pyimgano.datasets.manifest import _seed_for_category

    group_to_indices: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        gid = record.group_id if record.group_id is not None else f"__ungrouped__{idx}"
        group_to_indices.setdefault(str(gid), []).append(int(idx))

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

    if group_conflict:
        return {
            "assigned_counts": None,
            "mask_coverage": None,
            "pixel_metrics": None,
        }

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

    for idx, record in enumerate(records):
        gid = record.group_id if record.group_id is not None else f"__ungrouped__{idx}"
        gid = str(gid)

        if gid in fixed_val:
            assigned_counts["val"] += 1
            continue

        in_test = gid in fixed_test or gid in selected_test_normal
        if not in_test:
            assigned_counts["train"] += 1
            continue

        assigned_counts["test"] += 1
        label = int(record.label) if record.label is not None else 0
        if label == 1:
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

    return {
        "assigned_counts": assigned_counts,
        "mask_coverage": mask_coverage,
        "pixel_metrics": pixel_metrics,
    }


__all__ = ["analyze_manifest_category_assignment"]
