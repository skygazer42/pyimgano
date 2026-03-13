from __future__ import annotations

from typing import Any, Sequence


def summarize_manifest_category_records(*, records: Sequence[Any]) -> dict[str, Any]:
    from pyimgano.datasets.manifest import ManifestRecord

    recs: list[ManifestRecord] = [record for record in records if isinstance(record, ManifestRecord)]
    counts_by_split = {"train": 0, "val": 0, "test": 0, "unspecified": 0}
    explicit_test_labels = {"normal": 0, "anomaly": 0}
    for record in recs:
        if record.split is None:
            counts_by_split["unspecified"] += 1
        else:
            counts_by_split[str(record.split)] += 1
        if record.split == "test":
            if int(record.label or 0) == 1:
                explicit_test_labels["anomaly"] += 1
            else:
                explicit_test_labels["normal"] += 1

    return {
        "records": recs,
        "counts": {
            "total": int(len(recs)),
            "explicit_by_split": counts_by_split,
            "explicit_test_labels": explicit_test_labels,
        },
    }


__all__ = ["summarize_manifest_category_records"]
