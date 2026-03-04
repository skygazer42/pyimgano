from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


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


def _write_csv(path: Path, *, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _format_cell(row.get(k)) for k in columns})


def _write_markdown_table(path: Path, *, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
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

    return written


__all__ = [
    "export_suite_tables",
]
