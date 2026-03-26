from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from pyimgano.reporting.robustness_summary import (
    build_robustness_trust_summary,
    summarize_robustness_protocol,
)

_CSV_COLUMNS = [
    "condition",
    "severity",
    "latency_ms_per_image",
    "auroc",
    "average_precision",
    "pixel_auroc",
    "pixel_average_precision",
    "aupro",
    "pixel_segf1",
    "drop_auroc",
    "drop_average_precision",
    "drop_pixel_auroc",
    "drop_pixel_average_precision",
    "drop_aupro",
    "drop_pixel_segf1",
]

_METRIC_COLUMNS = [
    "auroc",
    "average_precision",
    "pixel_auroc",
    "pixel_average_precision",
    "aupro",
    "pixel_segf1",
]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_metric(results: Mapping[str, Any], key: str) -> Any:
    if key in results:
        return results.get(key)
    pixel_metrics = results.get("pixel_metrics", None)
    if isinstance(pixel_metrics, Mapping):
        return pixel_metrics.get(key)
    return None


def _build_metric_row(results: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(results, Mapping):
        return dict.fromkeys(_METRIC_COLUMNS)
    return {key: _safe_metric(results, key) for key in _METRIC_COLUMNS}


def _safe_drop(clean_value: Any, condition_value: Any) -> float | None:
    if not isinstance(clean_value, (int, float)) or not isinstance(condition_value, (int, float)):
        return None
    return round(float(clean_value) - float(condition_value), 12)


def _clean_drop_row(clean_metric_row: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        f"drop_{key}": (0.0 if isinstance(value, (int, float)) else None)
        for key, value in clean_metric_row.items()
    }


def _severity_label(severity_name: object) -> str:
    severity_value = str(severity_name)
    if severity_value.startswith("severity_"):
        return severity_value.split("_", 1)[1]
    return severity_value


def _build_clean_condition_row(clean: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    clean_metric_row = _build_metric_row(clean.get("results", None))
    row = {
        "condition": "clean",
        "severity": "",
        "latency_ms_per_image": clean.get("latency_ms_per_image"),
        **clean_metric_row,
        **_clean_drop_row(clean_metric_row),
    }
    return row, clean_metric_row


def _build_corruption_condition_rows(
    corruptions: Mapping[str, Any],
    *,
    clean_metric_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition_name, by_severity in corruptions.items():
        if not isinstance(by_severity, Mapping):
            continue
        for severity_name, cond_payload in by_severity.items():
            if not isinstance(cond_payload, Mapping):
                continue
            metric_row = _build_metric_row(cond_payload.get("results", None))
            drop_row = {
                f"drop_{key}": _safe_drop(clean_metric_row.get(key), value)
                for key, value in metric_row.items()
            }
            rows.append(
                {
                    "condition": str(condition_name),
                    "severity": _severity_label(severity_name),
                    "latency_ms_per_image": cond_payload.get("latency_ms_per_image"),
                    **metric_row,
                    **drop_row,
                }
            )
    return rows


def _flatten_condition_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    robustness = payload.get("robustness", None)
    if not isinstance(robustness, Mapping):
        return []

    rows: list[dict[str, Any]] = []
    clean_metric_row: dict[str, Any] = dict.fromkeys(_METRIC_COLUMNS)
    clean = robustness.get("clean", None)
    if isinstance(clean, Mapping):
        clean_row, clean_metric_row = _build_clean_condition_row(clean)
        rows.append(clean_row)

    corruptions = robustness.get("corruptions", None)
    if isinstance(corruptions, Mapping):
        rows.extend(
            _build_corruption_condition_rows(
                corruptions,
                clean_metric_row=clean_metric_row,
            )
        )
    return rows


def export_robustness_tables(
    payload: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _flatten_condition_rows(payload)
    csv_path = out_dir / "robustness_conditions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in _CSV_COLUMNS})
    audit_refs = {"robustness_conditions_csv": csv_path.name}
    audit_digests = {"robustness_conditions_csv": _file_sha256(csv_path)}

    summary_payload = {
        "dataset": payload.get("dataset"),
        "category": payload.get("category"),
        "model": payload.get("model"),
        "robustness_summary": payload.get("robustness_summary"),
        "robustness_protocol": (
            payload.get("robustness_protocol")
            if isinstance(payload.get("robustness_protocol"), Mapping)
            else summarize_robustness_protocol(dict(payload.get("robustness", {}) or {}))
        ),
        "trust_summary": build_robustness_trust_summary(
            report=(
                dict(payload.get("robustness", {}))
                if isinstance(payload.get("robustness"), Mapping)
                else {}
            ),
            robustness_summary=(
                dict(payload.get("robustness_summary", {}))
                if isinstance(payload.get("robustness_summary"), Mapping)
                else None
            ),
            robustness_protocol=(
                dict(payload.get("robustness_protocol", {}))
                if isinstance(payload.get("robustness_protocol"), Mapping)
                else None
            ),
            audit_refs=audit_refs,
            audit_digests=audit_digests,
        ),
        "audit_refs": audit_refs,
        "audit_digests": audit_digests,
        "condition_count": len(rows),
    }
    summary_path = out_dir / "robustness_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "conditions_csv": str(csv_path),
        "summary_json": str(summary_path),
    }


__all__ = ["export_robustness_tables"]
