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
                f"Category {chosen!r} not found in run report.\n"
                f"Available: {preview}{suffix}"
            )

        payload = per_category[chosen]
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"per_category[{chosen!r}] must be a JSON object/dict, got {type(payload).__name__}"
            )
        return str(chosen), payload

    # Single-category report payloads: return the report itself.
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


def load_checkpoint_into_detector(detector: Any, checkpoint_path: str | Path) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    load_fn = getattr(detector, "load_checkpoint", None)
    if callable(load_fn):
        try:
            load_fn(str(path))
        except TypeError:
            load_fn(path)
        return

    load_fn = getattr(detector, "load", None)
    if callable(load_fn):
        try:
            load_fn(str(path))
        except TypeError:
            load_fn(path)
        return

    model = getattr(detector, "model", None)
    if model is not None and callable(getattr(model, "load_state_dict", None)):
        try:
            import torch
        except Exception as exc:  # pragma: no cover - dependency boundary
            raise ImportError(
                "torch is required to load checkpoints via `detector.model.load_state_dict()`.\n"
                "Install it via:\n"
                "  pip install torch"
            ) from exc

        state = torch.load(path, map_location="cpu")
        if isinstance(state, Mapping):
            # Support both state_dict-only and {model_state_dict: ...} payloads.
            if "model_state_dict" in state and isinstance(state["model_state_dict"], Mapping):
                state = state["model_state_dict"]
            elif "state_dict" in state and isinstance(state["state_dict"], Mapping):
                state = state["state_dict"]

        try:
            model.load_state_dict(state)
        except Exception as exc:
            raise ValueError(f"Failed to load checkpoint into detector.model: {exc}") from exc
        return

    raise NotImplementedError(
        "Unable to load checkpoint into detector. Expected one of:\n"
        "- `detector.load_checkpoint(path)`\n"
        "- `detector.load(path)`\n"
        "- `detector.model.load_state_dict(...)` (torch)\n"
    )

