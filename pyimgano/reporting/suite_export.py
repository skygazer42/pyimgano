from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from pyimgano.reporting.evaluation_contract import build_evaluation_contract


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, str)):
        return str(value)
    try:
        f = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(f):
        return ""
    # Compact and stable enough for quick leaderboard inspection.
    return f"{f:.6g}"


def _safe_float(value: Any) -> float | None:
    try:
        f = float(value)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return float(f)


def _rank_rows(rows: list[Mapping[str, Any]], *, key: str) -> list[Mapping[str, Any]]:
    # Sort descending by metric value when available.
    def _sort_key(r: Mapping[str, Any]) -> tuple[int, float]:
        v = _safe_float(r.get(key))
        if v is None:
            return (1, 0.0)
        return (0, -float(v))

    return sorted(rows, key=_sort_key)


def _best_by_baseline(rows: Sequence[Mapping[str, Any]], *, key: str) -> list[Mapping[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        base = row.get("base_name", None)
        if base is None:
            base = row.get("name", None)
        base_name = str(base) if base is not None else ""
        groups.setdefault(base_name, []).append(row)

    best: list[Mapping[str, Any]] = []
    for base_name in sorted(groups.keys()):
        candidates = groups[base_name]

        def _cand_key(r: Mapping[str, Any]) -> tuple[int, float]:
            v = _safe_float(r.get(key))
            if v is None:
                return (0, float("-inf"))
            return (1, float(v))

        best_row = max(candidates, key=_cand_key)
        best.append(best_row)

    return _rank_rows(best, key=str(key))


_LEADERBOARD_COLUMNS = [
    "name",
    "base_name",
    "variant",
    "model",
    "optional",
    "auroc",
    "average_precision",
    "pixel_auroc",
    "pixel_average_precision",
    "aupro",
    "pixel_segf1",
    "run_dir",
]


def _collect_metric_names(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        for key in (
            "auroc",
            "average_precision",
            "pixel_auroc",
            "pixel_average_precision",
            "aupro",
            "pixel_segf1",
        ):
            if _safe_float(row.get(key)) is not None:
                names.add(str(key))
    return sorted(names)


def _build_publication_artifact_quality(
    *,
    written: Mapping[str, str],
    benchmark_config: Mapping[str, Any] | None,
    environment_fingerprint_sha256: Any,
    split_fingerprint: Any,
) -> dict[str, Any]:
    missing_required: list[str] = []
    if "leaderboard_metadata_json" not in written:
        missing_required.append("leaderboard_metadata_json")
    if not any(key.startswith("leaderboard_") and key != "leaderboard_metadata_json" for key in written):
        missing_required.append("leaderboard_table")
    if not isinstance(benchmark_config, Mapping):
        missing_required.append("benchmark_config")
    if not isinstance(environment_fingerprint_sha256, str) or not environment_fingerprint_sha256:
        missing_required.append("environment_fingerprint_sha256")
    split_sha256 = None
    if isinstance(split_fingerprint, Mapping):
        raw_sha256 = split_fingerprint.get("sha256", None)
        if isinstance(raw_sha256, str) and raw_sha256:
            split_sha256 = raw_sha256
    if split_sha256 is None:
        missing_required.append("split_fingerprint.sha256")

    return {
        "required_files_present": len(missing_required) == 0,
        "missing_required": missing_required,
        "has_official_benchmark_config": bool(
            isinstance(benchmark_config, Mapping) and benchmark_config.get("official")
        ),
        "has_environment_fingerprint": bool(
            isinstance(environment_fingerprint_sha256, str) and environment_fingerprint_sha256
        ),
        "has_split_fingerprint": bool(split_sha256),
    }


def _write_csv(path: Path, *, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _format_cell(row.get(k)) for k in columns})


def _write_markdown_table(
    path: Path, *, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for row in rows:
        cells = [_format_cell(row.get(k)) for k in cols]
        # Avoid breaking tables with embedded newlines/pipes.
        safe = [c.replace("\n", " ").replace("|", "\\|") for c in cells]
        lines.append("| " + " | ".join(safe) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_suite_tables(
    payload: Mapping[str, Any],
    output_dir: str | Path,
    *,
    formats: Sequence[str],
    best_metric: str = "auroc",
) -> dict[str, str]:
    """Export suite leaderboard/skipped tables to disk.

    This is intentionally lightweight and uses only the aggregated `payload`
    produced by `run_baseline_suite`.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        rows = []
    rows_norm: list[Mapping[str, Any]] = [r for r in rows if isinstance(r, Mapping)]
    best_by_baseline = _best_by_baseline(rows_norm, key=str(best_metric))

    skipped_payload = payload.get("skipped", {})
    skipped_rows: list[dict[str, Any]] = []
    if isinstance(skipped_payload, Mapping):
        for name, item in skipped_payload.items():
            if isinstance(item, Mapping):
                skipped_rows.append(
                    {
                        "name": str(name),
                        "status": str(item.get("status", "")),
                        "reason": str(item.get("reason", "")),
                    }
                )

    written: dict[str, str] = {}
    fmts = {str(f).strip().lower() for f in formats if str(f).strip()}

    if "csv" in fmts:
        leaderboard_csv = out_dir / "leaderboard.csv"
        skipped_csv = out_dir / "skipped.csv"
        best_csv = out_dir / "best_by_baseline.csv"
        _write_csv(leaderboard_csv, rows=rows_norm, columns=_LEADERBOARD_COLUMNS)
        _write_csv(best_csv, rows=best_by_baseline, columns=_LEADERBOARD_COLUMNS)
        _write_csv(skipped_csv, rows=skipped_rows, columns=["name", "status", "reason"])
        written["leaderboard_csv"] = str(leaderboard_csv)
        written["best_by_baseline_csv"] = str(best_csv)
        written["skipped_csv"] = str(skipped_csv)

    if "md" in fmts or "markdown" in fmts:
        leaderboard_md = out_dir / "leaderboard.md"
        skipped_md = out_dir / "skipped.md"
        best_md = out_dir / "best_by_baseline.md"
        _write_markdown_table(leaderboard_md, rows=rows_norm, columns=_LEADERBOARD_COLUMNS)
        _write_markdown_table(best_md, rows=best_by_baseline, columns=_LEADERBOARD_COLUMNS)
        _write_markdown_table(skipped_md, rows=skipped_rows, columns=["name", "status", "reason"])
        written["leaderboard_md"] = str(leaderboard_md)
        written["best_by_baseline_md"] = str(best_md)
        written["skipped_md"] = str(skipped_md)

    benchmark_config = payload.get("benchmark_config")
    citation = None
    if isinstance(benchmark_config, Mapping) and bool(benchmark_config.get("official")):
        citation = {
            "project": "pyimgano",
            "benchmark_config_source": benchmark_config.get("source"),
            "benchmark_config_sha256": benchmark_config.get("sha256"),
            "publication_guide": "docs/BENCHMARK_PUBLICATION.md",
        }

    artifact_quality = _build_publication_artifact_quality(
        written=written,
        benchmark_config=(benchmark_config if isinstance(benchmark_config, Mapping) else None),
        environment_fingerprint_sha256=payload.get("environment_fingerprint_sha256"),
        split_fingerprint=payload.get("split_fingerprint"),
    )
    evaluation_contract = build_evaluation_contract(
        metric_names=_collect_metric_names(rows_norm),
        primary_metric="auroc",
        ranking_metric=str(best_metric),
        pixel_metrics_enabled=(
            any(_safe_float(row.get("pixel_auroc")) is not None for row in rows_norm)
            or any(_safe_float(row.get("aupro")) is not None for row in rows_norm)
            or any(_safe_float(row.get("pixel_segf1")) is not None for row in rows_norm)
        ),
        comparability_hints=(
            dict(payload.get("evaluation_contract", {}).get("comparability_hints", {}))
            if isinstance(payload.get("evaluation_contract"), Mapping)
            and isinstance(payload.get("evaluation_contract", {}).get("comparability_hints"), Mapping)
            else None
        ),
    )

    metadata = {
        "suite": payload.get("suite"),
        "dataset": payload.get("dataset"),
        "category": payload.get("category"),
        "row_count": len(rows_norm),
        "benchmark_config": benchmark_config,
        "evaluation_contract": evaluation_contract,
        "environment_fingerprint_sha256": payload.get("environment_fingerprint_sha256"),
        "split_fingerprint": payload.get("split_fingerprint"),
        "citation": citation,
        "artifact_quality": artifact_quality,
        "publication_ready": bool(
            artifact_quality["required_files_present"]
            and artifact_quality["has_official_benchmark_config"]
        ),
        "exported_files": dict(written),
    }
    metadata_path = out_dir / "leaderboard_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    written["leaderboard_metadata_json"] = str(metadata_path)
    metadata["exported_files"] = dict(written)
    metadata["artifact_quality"] = _build_publication_artifact_quality(
        written=written,
        benchmark_config=(benchmark_config if isinstance(benchmark_config, Mapping) else None),
        environment_fingerprint_sha256=payload.get("environment_fingerprint_sha256"),
        split_fingerprint=payload.get("split_fingerprint"),
    )
    metadata["publication_ready"] = bool(
        metadata["artifact_quality"]["required_files_present"]
        and metadata["artifact_quality"]["has_official_benchmark_config"]
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return written


__all__ = [
    "export_suite_tables",
]
