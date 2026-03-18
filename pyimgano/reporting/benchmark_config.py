from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from pyimgano.reporting.evaluation_contract import build_evaluation_contract


def _official_benchmark_config_dir(directory: str | Path | None = None) -> Path:
    if directory is not None:
        return Path(directory)
    return Path(__file__).resolve().parents[2] / "benchmarks" / "configs"


def _nonempty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _resolve_config_path(raw: str) -> Path:
    path = Path(raw)
    candidates: list[Path] = [path]
    if not path.is_absolute():
        official_dir = _official_benchmark_config_dir()
        candidates.append(official_dir / raw)
        if path.name != raw:
            candidates.append(official_dir / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Benchmark config not found: {path}")


def _load_config_spec(spec: str | Path) -> tuple[Any, str, str]:
    raw = str(spec).strip()
    if not raw:
        raise ValueError("Benchmark config spec must not be empty.")

    if raw.startswith("@"):
        raw = raw[1:].strip()

    if raw.startswith("{") or raw.startswith("["):
        try:
            return json.loads(raw), "<inline>", "inline"
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Failed to parse inline benchmark config JSON.") from exc

    path = _resolve_config_path(raw)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to parse benchmark config JSON: {path}") from exc
    return payload, str(path), "file"


def validate_benchmark_config_payload(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["Benchmark config must be a JSON object."]

    errors: list[str] = []
    dataset = payload.get("dataset", None)
    suite = payload.get("suite", None)
    model = payload.get("model", None)
    if dataset is None:
        errors.append("Benchmark config requires key: dataset")
    if suite is None and model is None:
        errors.append("Benchmark config requires either key: suite or key: model")

    resize = payload.get("resize", None)
    if resize is not None:
        if not isinstance(resize, list) or len(resize) != 2:
            errors.append("Benchmark config resize must be a 2-item list [H, W].")

    dataset_name = str(dataset).lower() if dataset is not None else ""
    if dataset_name == "manifest":
        if payload.get("manifest_path", None) in (None, ""):
            errors.append("Manifest benchmark config requires key: manifest_path")
        if payload.get("root", None) in (None, ""):
            errors.append("Manifest benchmark config requires key: root")
    elif dataset is not None and payload.get("root", None) in (None, ""):
        errors.append("Benchmark config requires key: root")

    if suite is not None and payload.get("category", None) in (None, "") and dataset_name != "custom":
        errors.append("Suite benchmark config requires key: category for non-custom datasets")

    return errors


def _build_benchmark_config_trust_summary(
    *,
    payload: Any,
    source: str,
    kind: str,
    official: bool,
    sha256: str,
    evaluation_contract: Mapping[str, Any] | None,
    errors: list[str],
) -> dict[str, Any]:
    dataset = payload.get("dataset") if isinstance(payload, Mapping) else None
    suite = payload.get("suite") if isinstance(payload, Mapping) else None
    model = payload.get("model") if isinstance(payload, Mapping) else None
    trust_signals = {
        "is_official": bool(official),
        "has_source_path": bool(kind == "file" and _nonempty_str(source) not in (None, "<inline>")),
        "has_sha256": bool(_nonempty_str(sha256)),
        "has_dataset": _nonempty_str(dataset) is not None,
        "has_suite_or_model": (_nonempty_str(suite) is not None or _nonempty_str(model) is not None),
        "has_evaluation_contract": isinstance(evaluation_contract, Mapping),
    }

    degraded_by: list[str] = []
    if errors:
        degraded_by.append("invalid_config")
    if not trust_signals["is_official"]:
        degraded_by.append("not_official")
    if not trust_signals["has_source_path"]:
        degraded_by.append("missing_source_path")
    if not trust_signals["has_sha256"]:
        degraded_by.append("missing_sha256")
    if not trust_signals["has_dataset"]:
        degraded_by.append("missing_dataset")
    if not trust_signals["has_suite_or_model"]:
        degraded_by.append("missing_suite_or_model")
    if not trust_signals["has_evaluation_contract"]:
        degraded_by.append("missing_evaluation_contract")

    if errors:
        status = "broken"
    elif not degraded_by:
        status = "trust-signaled"
    else:
        status = "partial"

    audit_refs: dict[str, str] = {}
    if kind == "file" and _nonempty_str(source) is not None:
        audit_refs["benchmark_config_source"] = str(source)

    return {
        "status": status,
        "trust_signals": trust_signals,
        "degraded_by": degraded_by,
        "audit_refs": audit_refs,
    }


def describe_benchmark_config(spec: str | Path) -> dict[str, Any]:
    payload, source, kind = _load_config_spec(spec)
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    name = Path(source).name if kind == "file" else ""
    dataset = payload.get("dataset") if isinstance(payload, dict) else None
    suite = payload.get("suite") if isinstance(payload, dict) else None
    evaluation_contract = build_evaluation_contract(
        metric_names=(
            payload.get("metrics", [])
            if isinstance(payload, dict) and isinstance(payload.get("metrics"), list)
            else [
                "auroc",
                "average_precision",
                "pixel_auroc",
                "pixel_average_precision",
                "aupro",
                "pixel_segf1",
            ]
        ),
        primary_metric=(
            str(payload.get("primary_metric"))
            if isinstance(payload, dict) and payload.get("primary_metric") is not None
            else "auroc"
        ),
        ranking_metric=(
            str(payload.get("ranking_metric"))
            if isinstance(payload, dict) and payload.get("ranking_metric") is not None
            else "auroc"
        ),
    )
    sha256 = hashlib.sha256(canonical).hexdigest()
    errors = validate_benchmark_config_payload(payload)
    official = bool(name.startswith("official_") and name.endswith(".json"))
    return {
        "name": name,
        "source": source,
        "kind": kind,
        "official": official,
        "dataset": dataset,
        "suite": suite,
        "evaluation_contract": evaluation_contract,
        "payload": payload,
        "sha256": sha256,
        "errors": errors,
        "trust_summary": _build_benchmark_config_trust_summary(
            payload=payload,
            source=source,
            kind=kind,
            official=official,
            sha256=sha256,
            evaluation_contract=evaluation_contract,
            errors=errors,
        ),
    }


def list_official_benchmark_configs(directory: str | Path | None = None) -> list[dict[str, Any]]:
    config_dir = _official_benchmark_config_dir(directory)
    return [
        describe_benchmark_config(path)
        for path in sorted(config_dir.glob("official_*.json"))
    ]


def load_benchmark_config_spec(spec: str | Path) -> Any:
    payload, _source, _kind = _load_config_spec(spec)
    return payload


def load_and_validate_benchmark_config(spec: str | Path) -> dict[str, Any]:
    payload, _source, _kind = _load_config_spec(spec)
    errors = validate_benchmark_config_payload(payload)
    if errors:
        raise ValueError("Invalid benchmark config:\n- " + "\n- ".join(errors))
    if not isinstance(payload, dict):
        raise ValueError("Benchmark config must be a JSON object.")
    return payload


__all__ = [
    "describe_benchmark_config",
    "list_official_benchmark_configs",
    "load_benchmark_config_spec",
    "load_and_validate_benchmark_config",
    "validate_benchmark_config_payload",
]
