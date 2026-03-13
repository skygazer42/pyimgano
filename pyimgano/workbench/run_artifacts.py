from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pyimgano.workbench.config import WorkbenchConfig


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing {name}: {path}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {name}: {path}. Original error: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a JSON object/dict, got {type(data).__name__}: {path}")
    return dict(data)


def load_workbench_config_from_run(run_dir: str | Path) -> WorkbenchConfig:
    base = Path(run_dir)
    if not base.exists():
        raise FileNotFoundError(f"Run directory not found: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"--from-run must point to a directory, got: {base}")

    config_path = base / "config.json"
    raw = _load_json_object(config_path, name="config.json")

    inner = raw.get("config", raw)
    if not isinstance(inner, Mapping):
        raise ValueError(f"config.json must contain a dict config, got: {type(inner).__name__}")

    try:
        return WorkbenchConfig.from_dict(inner)
    except Exception as exc:  # noqa: BLE001 - boundary for user configs
        raise ValueError(f"Failed to parse workbench config from {config_path}: {exc}") from exc


def load_report_from_run(run_dir: str | Path) -> dict[str, Any]:
    base = Path(run_dir)
    report_path = base / "report.json"
    return _load_json_object(report_path, name="report.json")


def select_category_report(
    report: Mapping[str, Any],
    *,
    category: str | None,
) -> tuple[str | None, Mapping[str, Any]]:
    per_category = report.get("per_category", None)
    if isinstance(per_category, Mapping):
        categories = sorted(str(k) for k in per_category.keys())
        chosen = category
        if chosen is None:
            if len(categories) == 1:
                chosen = categories[0]
            else:
                preview = ", ".join(categories[:8])
                suffix = "" if len(categories) <= 8 else ", ..."
                raise ValueError(
                    "Run report contains multiple categories; please specify --from-run-category.\n"
                    f"Available: {preview}{suffix}"
                )

        if chosen not in per_category:
            preview = ", ".join(categories[:8])
            suffix = "" if len(categories) <= 8 else ", ..."
            raise ValueError(
                f"Category {chosen!r} not found in run report.\n" f"Available: {preview}{suffix}"
            )

        payload = per_category[chosen]
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"per_category[{chosen!r}] must be a JSON object/dict, got {type(payload).__name__}"
            )
        return str(chosen), payload

    cat = report.get("category", None)
    return (str(cat) if cat is not None else None), report


def extract_threshold(report_payload: Mapping[str, Any]) -> float | None:
    raw = report_payload.get("threshold", None)
    if raw is None:
        return None
    try:
        val = float(raw)
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    return float(val)


def resolve_checkpoint_path(
    run_dir: str | Path,
    report_payload: Mapping[str, Any],
) -> Path | None:
    ckpt = report_payload.get("checkpoint", None)
    if not isinstance(ckpt, Mapping):
        return None
    raw = ckpt.get("path", None)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    p = Path(text)
    if not p.is_absolute():
        p = Path(run_dir) / p
    return p


__all__ = [
    "extract_threshold",
    "load_report_from_run",
    "load_workbench_config_from_run",
    "resolve_checkpoint_path",
    "select_category_report",
]
