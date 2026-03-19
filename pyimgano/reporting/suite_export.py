from __future__ import annotations

import csv
import hashlib
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


def _nonempty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if text else None


def _split_fingerprint_sha256(split_fingerprint: Any) -> str | None:
    if not isinstance(split_fingerprint, Mapping):
        return None
    return _nonempty_str(split_fingerprint.get("sha256"))


def _load_run_report(run_dir: Any) -> Mapping[str, Any] | None:
    path_text = _nonempty_str(run_dir)
    if path_text is None:
        return None

    report_path = Path(path_text) / "report.json"
    if not report_path.is_file():
        return None

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    return payload if isinstance(payload, Mapping) else None


def _resolve_split_fingerprint(
    *, payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    split_fingerprint = payload.get("split_fingerprint")
    if isinstance(split_fingerprint, Mapping):
        return dict(split_fingerprint)

    for row in rows:
        report = _load_run_report(row.get("run_dir"))
        if report is None:
            continue
        candidate = report.get("split_fingerprint")
        if isinstance(candidate, Mapping):
            return dict(candidate)

    return None


def _append_missing_required(missing_required: list[str], item: str) -> None:
    if item not in missing_required:
        missing_required.append(item)


def _build_publication_audit_refs(
    *,
    out_dir: Path,
    benchmark_config: Mapping[str, Any] | None,
) -> dict[str, str]:
    refs: dict[str, str] = {}
    for key, filename in (
        ("report_json", "report.json"),
        ("config_json", "config.json"),
        ("environment_json", "environment.json"),
    ):
        path = out_dir / filename
        if path.is_file():
            refs[key] = filename

    benchmark_source = (
        _nonempty_str(benchmark_config.get("source"))
        if isinstance(benchmark_config, Mapping)
        else None
    )
    if benchmark_source is not None:
        refs["benchmark_config_source"] = benchmark_source

    return refs


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_publication_audit_digests(
    *,
    out_dir: Path,
) -> dict[str, str]:
    digests: dict[str, str] = {}
    for key, filename in (
        ("report_json", "report.json"),
        ("config_json", "config.json"),
        ("environment_json", "environment.json"),
    ):
        path = out_dir / filename
        if path.is_file():
            digests[key] = _file_sha256(path)
    return digests


def _build_exported_file_digests(
    *,
    written: Mapping[str, str],
) -> dict[str, str]:
    digests: dict[str, str] = {}
    for key, raw_path in written.items():
        if str(key) == "leaderboard_metadata_json":
            continue
        path = Path(raw_path)
        if path.is_file():
            digests[str(key)] = _file_sha256(path)
    return digests


def _build_publication_artifact_quality(
    *,
    written: Mapping[str, str],
    benchmark_config: Mapping[str, Any] | None,
    environment_fingerprint_sha256: Any,
    split_fingerprint: Any,
    evaluation_contract: Mapping[str, Any] | None,
    citation: Mapping[str, Any] | None,
    audit_refs: Mapping[str, Any] | None,
    audit_digests: Mapping[str, Any] | None,
    exported_file_digests: Mapping[str, Any] | None,
) -> dict[str, Any]:
    missing_required: list[str] = []
    if "leaderboard_metadata_json" not in written:
        _append_missing_required(missing_required, "leaderboard_metadata_json")
    if not any(key.startswith("leaderboard_") and key != "leaderboard_metadata_json" for key in written):
        _append_missing_required(missing_required, "leaderboard_table")
    required_exported_digest_keys = [
        str(key) for key in written if str(key) != "leaderboard_metadata_json"
    ]

    benchmark_source = None
    benchmark_sha256 = None
    benchmark_trust_summary = None
    if not isinstance(benchmark_config, Mapping):
        _append_missing_required(missing_required, "benchmark_config")
    else:
        benchmark_source = _nonempty_str(benchmark_config.get("source"))
        benchmark_sha256 = _nonempty_str(benchmark_config.get("sha256"))
        if benchmark_source is None:
            _append_missing_required(missing_required, "benchmark_config.source")
        if benchmark_sha256 is None:
            _append_missing_required(missing_required, "benchmark_config.sha256")
        raw_trust_summary = benchmark_config.get("trust_summary")
        if isinstance(raw_trust_summary, Mapping):
            benchmark_trust_summary = raw_trust_summary

    environment_sha256 = _nonempty_str(environment_fingerprint_sha256)
    if environment_sha256 is None:
        _append_missing_required(missing_required, "environment_fingerprint_sha256")

    split_sha256 = _split_fingerprint_sha256(split_fingerprint)
    if split_sha256 is None:
        _append_missing_required(missing_required, "split_fingerprint.sha256")

    has_evaluation_contract = isinstance(evaluation_contract, Mapping)
    if not has_evaluation_contract:
        _append_missing_required(missing_required, "evaluation_contract")

    has_benchmark_citation = isinstance(citation, Mapping)
    if not has_benchmark_citation:
        _append_missing_required(missing_required, "citation")

    has_run_artifact_refs = True
    if not isinstance(audit_refs, Mapping):
        has_run_artifact_refs = False
    else:
        for key in ("report_json", "config_json", "environment_json"):
            if _nonempty_str(audit_refs.get(key)) is None:
                _append_missing_required(missing_required, f"audit_refs.{key}")
                has_run_artifact_refs = False

    has_run_artifact_digests = True
    if not isinstance(audit_digests, Mapping):
        has_run_artifact_digests = False
    else:
        for key in ("report_json", "config_json", "environment_json"):
            if _nonempty_str(audit_digests.get(key)) is None:
                _append_missing_required(missing_required, f"audit_digests.{key}")
                has_run_artifact_digests = False

    has_exported_file_digests = bool(required_exported_digest_keys)
    if not isinstance(exported_file_digests, Mapping):
        has_exported_file_digests = False
        for key in required_exported_digest_keys:
            _append_missing_required(missing_required, f"exported_file_digests.{key}")
    else:
        for key in required_exported_digest_keys:
            if _nonempty_str(exported_file_digests.get(key)) is None:
                _append_missing_required(missing_required, f"exported_file_digests.{key}")
                has_exported_file_digests = False

    has_official_benchmark_config = bool(
        isinstance(benchmark_config, Mapping) and benchmark_config.get("official")
    )
    has_benchmark_provenance = bool(
        has_official_benchmark_config and benchmark_source and benchmark_sha256
    )

    payload = {
        "required_files_present": len(missing_required) == 0,
        "missing_required": missing_required,
        "has_official_benchmark_config": has_official_benchmark_config,
        "has_environment_fingerprint": bool(environment_sha256),
        "has_split_fingerprint": bool(split_sha256),
        "has_evaluation_contract": bool(has_evaluation_contract),
        "has_benchmark_citation": bool(has_benchmark_citation),
        "has_benchmark_provenance": bool(has_benchmark_provenance),
        "has_run_artifact_refs": bool(has_run_artifact_refs),
        "has_run_artifact_digests": bool(has_run_artifact_digests),
        "has_exported_file_digests": bool(has_exported_file_digests),
    }
    if isinstance(benchmark_trust_summary, Mapping):
        payload["has_trust_signaled_benchmark_config"] = (
            str(benchmark_trust_summary.get("status", "")) == "trust-signaled"
        )
    return payload


def _publication_ready(artifact_quality: Mapping[str, Any]) -> bool:
    return bool(
        artifact_quality.get("required_files_present")
        and artifact_quality.get("has_official_benchmark_config")
        and artifact_quality.get("has_evaluation_contract")
        and artifact_quality.get("has_benchmark_citation")
        and artifact_quality.get("has_benchmark_provenance")
        and artifact_quality.get("has_run_artifact_refs")
        and artifact_quality.get("has_run_artifact_digests")
        and artifact_quality.get("has_exported_file_digests")
    )


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
    split_fingerprint = _resolve_split_fingerprint(payload=payload, rows=rows_norm)
    citation = None
    if isinstance(benchmark_config, Mapping) and bool(benchmark_config.get("official")):
        citation = {
            "project": "pyimgano",
            "benchmark_config_source": benchmark_config.get("source"),
            "benchmark_config_sha256": benchmark_config.get("sha256"),
            "publication_guide": "docs/BENCHMARK_PUBLICATION.md",
        }
    audit_refs = _build_publication_audit_refs(
        out_dir=out_dir,
        benchmark_config=(benchmark_config if isinstance(benchmark_config, Mapping) else None),
    )
    audit_digests = _build_publication_audit_digests(out_dir=out_dir)
    exported_file_digests = _build_exported_file_digests(written=written)

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
    artifact_quality = _build_publication_artifact_quality(
        written=written,
        benchmark_config=(benchmark_config if isinstance(benchmark_config, Mapping) else None),
        environment_fingerprint_sha256=payload.get("environment_fingerprint_sha256"),
        split_fingerprint=split_fingerprint,
        evaluation_contract=evaluation_contract,
        citation=(citation if isinstance(citation, Mapping) else None),
        audit_refs=audit_refs,
        audit_digests=audit_digests,
        exported_file_digests=exported_file_digests,
    )

    metadata = {
        "suite": payload.get("suite"),
        "dataset": payload.get("dataset"),
        "category": payload.get("category"),
        "row_count": len(rows_norm),
        "benchmark_config": benchmark_config,
        "evaluation_contract": evaluation_contract,
        "environment_fingerprint_sha256": payload.get("environment_fingerprint_sha256"),
        "split_fingerprint": split_fingerprint,
        "citation": citation,
        "audit_refs": audit_refs,
        "audit_digests": audit_digests,
        "exported_file_digests": exported_file_digests,
        "artifact_quality": artifact_quality,
        "publication_ready": _publication_ready(artifact_quality),
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
        split_fingerprint=split_fingerprint,
        evaluation_contract=evaluation_contract,
        citation=(citation if isinstance(citation, Mapping) else None),
        audit_refs=audit_refs,
        audit_digests=audit_digests,
        exported_file_digests=exported_file_digests,
    )
    metadata["publication_ready"] = _publication_ready(metadata["artifact_quality"])
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return written


__all__ = [
    "export_suite_tables",
]
