from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_infer_config(path: str | Path) -> dict[str, Any]:
    """Load an exported infer-config JSON payload from disk.

    This is intended for `pyimgano-infer --infer-config artifacts/infer_config.json`.
    """

    cfg_path = Path(path)
    try:
        text = cfg_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Infer config not found: {cfg_path}") from exc
    except Exception as exc:  # noqa: BLE001 - boundary
        raise ValueError(f"Failed to read infer config: {cfg_path}. Original error: {exc}") from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in infer config: {cfg_path}. Original error: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"Infer config must be a JSON object/dict, got {type(payload).__name__}: {cfg_path}"
        )
    return dict(payload)


def select_infer_category(
    payload: Mapping[str, Any],
    *,
    category: str | None,
) -> dict[str, Any]:
    """Select a single category payload from an infer-config.

    When `payload["per_category"]` exists, it is expected to contain per-category
    report payloads from workbench runs. This function:

    - chooses a category (or errors if ambiguous)
    - propagates per-category `threshold`/`checkpoint` into the returned payload
    - removes `per_category` to avoid ambiguity downstream
    """

    per_category = payload.get("per_category", None)
    if not isinstance(per_category, Mapping):
        return dict(payload)

    categories = sorted(str(k) for k in per_category.keys())
    chosen = category

    if chosen is None:
        cat_field = payload.get("category", None)
        if cat_field is not None and str(cat_field).lower() != "all":
            chosen = str(cat_field)
        elif len(categories) == 1:
            chosen = categories[0]
        else:
            preview = ", ".join(categories[:8])
            suffix = "" if len(categories) <= 8 else ", ..."
            raise ValueError(
                "Infer-config contains multiple categories; please specify --infer-category.\n"
                f"Available: {preview}{suffix}"
            )

    if chosen not in per_category:
        preview = ", ".join(categories[:8])
        suffix = "" if len(categories) <= 8 else ", ..."
        raise ValueError(
            f"Category {chosen!r} not found in infer-config.\nAvailable: {preview}{suffix}"
        )

    cat_payload = per_category.get(chosen, None)
    if not isinstance(cat_payload, Mapping):
        raise ValueError(
            f"per_category[{chosen!r}] must be a JSON object/dict, got {type(cat_payload).__name__}"
        )

    out = dict(payload)
    out.pop("per_category", None)
    out["category"] = str(chosen)
    for key in ("threshold", "threshold_provenance", "checkpoint"):
        if key in cat_payload:
            out[key] = cat_payload.get(key)
    return out


def resolve_infer_checkpoint_path(
    payload: Mapping[str, Any],
    *,
    config_path: str | Path,
) -> Path | None:
    """Resolve the trained checkpoint path referenced by an infer-config.

    Resolution rules (first match wins):

    1) Absolute paths are used as-is.
    2) Relative to the infer-config file directory.
    3) Relative to the infer-config parent directory (common when the file lives in
       `<run_dir>/artifacts/` while checkpoints live in `<run_dir>/checkpoints/`).
    4) Relative to `payload["from_run"]` when present.
    """

    ckpt = payload.get("checkpoint", None)
    if not isinstance(ckpt, Mapping):
        return None
    raw = ckpt.get("path", None)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    p = Path(text)
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    cfg_path = Path(config_path)
    base = cfg_path.parent
    candidates: list[Path] = [
        (base / p).resolve(),
        (base.parent / p).resolve(),
    ]

    from_run = payload.get("from_run", None)
    if from_run is not None:
        try:
            candidates.append((Path(str(from_run)) / p).resolve())
        except Exception:
            pass

    for cand in candidates:
        if cand.exists():
            return cand

    tried = "\n".join(f"- {c}" for c in candidates)
    raise FileNotFoundError(
        "Checkpoint not found for infer-config.\n"
        f"checkpoint.path={text!r}\n"
        "Tried:\n"
        f"{tried}"
    )
