from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pyimgano.utils.security import FileHasher
from pyimgano.weights.manifest import validate_weights_manifest_file


@dataclass(frozen=True)
class ModelCardReport:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    normalized: dict[str, Any]
    assets: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "normalized": dict(self.normalized),
            "assets": dict(self.assets),
        }


def default_model_card_template() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_name": "example_model_name",
        "summary": {
            "purpose": "Brief summary of the deployment objective.",
            "intended_inputs": "Describe expected image modality, shape, and domain.",
            "output_contract": "image-level + pixel-level",
        },
        "weights": {
            "path": "checkpoints/example_model.pt",
            "manifest_entry": "example_checkpoint",
            "sha256": "replace-with-sha256-if-known",
            "source": "internal training run or upstream checkpoint source",
            "license": "internal-or-upstream-license",
        },
        "training": {
            "dataset": "dataset name or manifest id",
            "split_policy": "document split / calibration policy",
            "preprocessing": "summarize preprocessing and normalization",
            "threshold_strategy": "fixed | normal_pixel_quantile | infer_config",
        },
        "evaluation": {
            "datasets": ["dataset/category names evaluated"],
            "key_metrics": {"auroc": 0.0},
        },
        "deployment": {
            "runtime": "torch",
            "cache_locations": ["TORCH_HOME=/models/torch_cache"],
            "expected_throughput": "document throughput or latency target",
        },
        "limitations": {
            "known_failure_modes": ["lighting shift", "alignment drift"],
            "sensitivity": ["camera changes", "ROI mismatch"],
        },
    }


def _as_dict(obj: Any) -> dict[str, Any] | None:
    return dict(obj) if isinstance(obj, Mapping) else None


def _as_list(obj: Any) -> list[Any] | None:
    return obj if isinstance(obj, list) else None


def _nonempty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _normalize_string_list(value: Any, *, field_name: str, warnings: list[str]) -> list[str]:
    items = _as_list(value)
    if items is None:
        warnings.append(f"{field_name} should be a list of strings.")
        return []
    out: list[str] = []
    for item in items:
        text = _nonempty_str(item)
        if text is None:
            warnings.append(f"{field_name} contains an empty item.")
            continue
        out.append(text)
    return out


def _resolve_path(raw: str, *, base_dir: Path | None) -> Path:
    p = Path(str(raw))
    if p.is_absolute():
        return p
    if base_dir is None:
        return p
    return (base_dir / p).resolve()


def _empty_manifest_asset() -> dict[str, Any]:
    return {
        "path": None,
        "resolved_path": None,
        "ok": None,
        "matched_entry": None,
        "entry_path": None,
        "entry_resolved_path": None,
    }


def _empty_weights_asset() -> dict[str, Any]:
    return {
        "path": None,
        "resolved_path": None,
        "exists": None,
        "sha256_match": None,
    }


def _manifest_runtimes(entry: Mapping[str, Any]) -> tuple[str, ...]:
    out: list[str] = []
    runtime = _nonempty_str(entry.get("runtime", None))
    if runtime is not None:
        out.append(runtime)
    runtimes = _as_list(entry.get("runtimes", None))
    if runtimes is not None:
        for item in runtimes:
            text = _nonempty_str(item)
            if text is not None:
                out.append(text)
    return tuple(dict.fromkeys(out))


def _append_manifest_report_issues(
    manifest_report: Any,
    *,
    errors: list[str],
    warnings: list[str],
) -> int:
    for warning in manifest_report.warnings:
        warnings.append(f"weights_manifest: {warning}")
    for error in manifest_report.errors:
        errors.append(f"weights_manifest: {error}")
    return len(errors)


def _explicit_manifest_entry_match(
    manifest_report: Any,
    *,
    explicit_entry: str,
) -> dict[str, Any] | None:
    for entry in manifest_report.entries:
        if _nonempty_str(entry.get("name", None)) == explicit_entry:
            return dict(entry)
    return None


def _path_manifest_entry_matches(
    manifest_report: Any,
    *,
    model_resolved_path: str,
) -> list[dict[str, Any]]:
    return [
        dict(entry)
        for entry in manifest_report.entries
        if _nonempty_str(entry.get("resolved_path", None)) == model_resolved_path
    ]


def _sha_manifest_entry_matches(
    manifest_report: Any,
    *,
    model_sha: str,
) -> list[dict[str, Any]]:
    return [
        dict(entry)
        for entry in manifest_report.entries
        if _nonempty_str(entry.get("sha256", None)) is not None
        and _nonempty_str(entry.get("sha256", None)).lower() == model_sha.lower()
    ]


def _resolve_manifest_entry_match(
    report: ModelCardReport,
    *,
    manifest_report: Any,
    assets: Mapping[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any] | None:
    weights_norm = _as_dict(report.normalized.get("weights", {})) or {}
    explicit_entry = _nonempty_str(weights_norm.get("manifest_entry", None))
    if explicit_entry is not None:
        matched_entry = _explicit_manifest_entry_match(
            manifest_report,
            explicit_entry=explicit_entry,
        )
        if matched_entry is None:
            errors.append(f"Model card references unknown manifest entry: {explicit_entry!r}")
        return matched_entry

    model_resolved_path = _nonempty_str(dict(assets.get("weights", {})).get("resolved_path", None))
    if model_resolved_path is not None:
        path_matches = _path_manifest_entry_matches(
            manifest_report,
            model_resolved_path=model_resolved_path,
        )
        if len(path_matches) == 1:
            return path_matches[0]
        if len(path_matches) > 1:
            errors.append(
                "Model card weights asset matches multiple manifest entries; "
                "set weights.manifest_entry explicitly."
            )
            return None

    model_sha = _nonempty_str(weights_norm.get("sha256", None))
    if model_sha is not None:
        sha_matches = _sha_manifest_entry_matches(
            manifest_report,
            model_sha=model_sha,
        )
        if len(sha_matches) == 1:
            warnings.append(
                "Matched model card to weights manifest by sha256; "
                "set weights.manifest_entry for an explicit link."
            )
            return sha_matches[0]
        if len(sha_matches) > 1:
            errors.append(
                "Model card sha256 matches multiple manifest entries; "
                "set weights.manifest_entry explicitly."
            )
            return None

    if len(manifest_report.entries) > 0:
        errors.append("No weights manifest entry matches the model card weights asset.")
    return None


def _apply_matched_manifest_entry_checks(
    report: ModelCardReport,
    *,
    matched_entry: dict[str, Any],
    manifest_asset: dict[str, Any],
    manifest_base: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    manifest_asset["matched_entry"] = _nonempty_str(matched_entry.get("name", None))
    manifest_asset["entry_path"] = _nonempty_str(matched_entry.get("path", None))
    manifest_asset["entry_resolved_path"] = _nonempty_str(matched_entry.get("resolved_path", None))

    model_resolved = _nonempty_str(dict(report.assets.get("weights", {})).get("resolved_path", None))
    manifest_entry_path = _nonempty_str(matched_entry.get("path", None))
    manifest_entry_resolved = (
        str(_resolve_path(manifest_entry_path, base_dir=manifest_base))
        if manifest_entry_path is not None
        else None
    )
    if manifest_entry_resolved is not None:
        manifest_asset["entry_resolved_path"] = manifest_entry_resolved

    if model_resolved is not None and manifest_entry_resolved is not None and model_resolved != manifest_entry_resolved:
        errors.append(
            "Model card weights.path does not resolve to the same asset as "
            f"manifest entry {manifest_asset['matched_entry']!r}: "
            f"{model_resolved} != {manifest_entry_resolved}"
        )

    model_sha = _nonempty_str(dict(report.normalized.get("weights", {})).get("sha256", None))
    manifest_sha = _nonempty_str(matched_entry.get("sha256", None))
    if model_sha is not None and manifest_sha is not None:
        if model_sha.lower() != manifest_sha.lower():
            errors.append(
                "Model card weights.sha256 does not match the linked manifest entry: "
                f"card={model_sha} manifest={manifest_sha}"
            )
    elif model_sha is None and manifest_sha is not None:
        warnings.append(
            "Model card weights.sha256 is missing while the linked manifest entry "
            "declares one."
        )
    elif model_sha is not None and manifest_sha is None:
        warnings.append(
            "Linked manifest entry is missing sha256 while the model card declares one."
        )

    for field in ("source", "license"):
        card_value = _nonempty_str(dict(report.normalized.get("weights", {})).get(field, None))
        manifest_value = _nonempty_str(matched_entry.get(field, None))
        if card_value is not None and manifest_value is not None and card_value != manifest_value:
            warnings.append(
                f"Model card weights.{field} differs from linked manifest entry: "
                f"card={card_value!r} manifest={manifest_value!r}"
            )

    deployment = _as_dict(report.normalized.get("deployment", {})) or {}
    deployment_runtime = _nonempty_str(deployment.get("runtime", None))
    supported_runtimes = _manifest_runtimes(matched_entry)
    if deployment_runtime is not None and supported_runtimes and deployment_runtime not in supported_runtimes:
        warnings.append(
            "Model card deployment.runtime is not listed by the linked manifest "
            f"entry: runtime={deployment_runtime!r} supported={list(supported_runtimes)!r}"
        )


def _cross_check_model_card_manifest(
    report: ModelCardReport,
    *,
    manifest_path: str | Path,
    manifest_base_dir: str | Path | None = None,
    check_files: bool = False,
    check_hashes: bool = False,
) -> ModelCardReport:
    errors = list(report.errors)
    warnings = list(report.warnings)
    assets = {
        "weights": dict(report.assets.get("weights", {})),
        "manifest": _empty_manifest_asset(),
    }

    manifest_report = validate_weights_manifest_file(
        manifest_path=manifest_path,
        base_dir=manifest_base_dir,
        check_files=bool(check_files),
        check_hashes=bool(check_hashes),
    )

    manifest_path_obj = Path(manifest_path)
    manifest_base = (
        Path(manifest_base_dir).resolve()
        if manifest_base_dir is not None
        else manifest_path_obj.resolve().parent
    )
    manifest_asset = assets["manifest"]
    manifest_asset["path"] = str(manifest_path_obj)
    manifest_asset["resolved_path"] = str(manifest_path_obj.resolve())

    manifest_error_count_before = _append_manifest_report_issues(
        manifest_report,
        errors=errors,
        warnings=warnings,
    )
    matched_entry = _resolve_manifest_entry_match(
        report,
        manifest_report=manifest_report,
        assets=assets,
        errors=errors,
        warnings=warnings,
    )
    if matched_entry is not None:
        _apply_matched_manifest_entry_checks(
            report,
            matched_entry=matched_entry,
            manifest_asset=manifest_asset,
            manifest_base=manifest_base,
            errors=errors,
            warnings=warnings,
        )

    manifest_asset["ok"] = len(errors) == manifest_error_count_before

    return ModelCardReport(
        ok=(len(errors) == 0),
        errors=tuple(errors),
        warnings=tuple(warnings),
        normalized=dict(report.normalized),
        assets=assets,
    )


def _normalize_schema_version(
    payload: Mapping[str, Any],
    *,
    errors: list[str],
    warnings: list[str],
) -> int:
    schema_version = payload.get("schema_version", None)
    if schema_version is None:
        warnings.append("Missing top-level key: schema_version (recommended).")
        return 1
    try:
        normalized_schema_version = int(schema_version)
    except Exception:
        errors.append(f"schema_version must be an int, got {schema_version!r}")
        return 1
    if normalized_schema_version != 1:
        warnings.append(
            f"Unknown schema_version={normalized_schema_version}; validation is best-effort."
        )
    return normalized_schema_version


def _validate_summary_section(
    payload: Mapping[str, Any],
    *,
    errors: list[str],
) -> dict[str, Any]:
    summary = _as_dict(payload.get("summary", None))
    if summary is None:
        errors.append("Missing required key: summary (expected object).")
        return {}
    summary_norm = dict(summary)
    for key in ("purpose", "intended_inputs", "output_contract"):
        if _nonempty_str(summary.get(key, None)) is None:
            errors.append(f"Missing required key: summary.{key}")
    return summary_norm


def _validate_weights_section(
    payload: Mapping[str, Any],
    *,
    base: Path | None,
    check_files: bool,
    check_hashes: bool,
    errors: list[str],
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    weights = _as_dict(payload.get("weights", None))
    if weights is None:
        errors.append("Missing required key: weights (expected object).")
        return {}, _empty_weights_asset()

    weights_norm = dict(weights)
    weights_asset = _empty_weights_asset()
    for key in ("path", "source", "license"):
        if _nonempty_str(weights.get(key, None)) is None:
            errors.append(f"Missing required key: weights.{key}")

    sha = _nonempty_str(weights.get("sha256", None))
    if sha is not None and len(sha) != 64:
        warnings.append("weights.sha256 should be a 64-character SHA256 hex digest.")

    raw_path = _nonempty_str(weights.get("path", None))
    if raw_path is None:
        return weights_norm, weights_asset

    resolved_path = _resolve_path(raw_path, base_dir=base)
    weights_asset["path"] = raw_path
    weights_asset["resolved_path"] = str(resolved_path)
    exists = resolved_path.exists()
    weights_asset["exists"] = bool(exists)
    if check_files and not exists:
        errors.append(f"Missing weights file referenced by model card: {resolved_path}")
    if check_hashes:
        if sha is None:
            warnings.append("weights.sha256 is missing; hash check was skipped.")
        elif exists:
            actual = FileHasher.compute_hash(str(resolved_path), algorithm="sha256")
            matched = actual.lower() == sha.lower()
            weights_asset["sha256_match"] = bool(matched)
            if not matched:
                errors.append(
                    "SHA256 mismatch for model card weights asset: "
                    f"expected={sha} actual={actual}"
                )
    return weights_norm, weights_asset


def _validate_deployment_section(
    payload: Mapping[str, Any],
    *,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    deployment = _as_dict(payload.get("deployment", None))
    if deployment is None:
        errors.append("Missing required key: deployment (expected object).")
        errors.append("Missing required key: deployment.runtime")
        return {}
    deployment_norm = dict(deployment)
    if _nonempty_str(deployment.get("runtime", None)) is None:
        errors.append("Missing required key: deployment.runtime")
    cache_locations = deployment.get("cache_locations", None)
    if cache_locations is not None:
        deployment_norm["cache_locations"] = _normalize_string_list(
            cache_locations,
            field_name="deployment.cache_locations",
            warnings=warnings,
        )
    return deployment_norm


def _normalized_optional_sections(
    payload: Mapping[str, Any],
    *,
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    training = _as_dict(payload.get("training", {})) or {}
    evaluation = _as_dict(payload.get("evaluation", {})) or {}
    limitations = _as_dict(payload.get("limitations", {})) or {}
    if "datasets" in evaluation:
        evaluation["datasets"] = _normalize_string_list(
            evaluation.get("datasets"),
            field_name="evaluation.datasets",
            warnings=warnings,
        )
    for key in ("known_failure_modes", "sensitivity"):
        if key in limitations:
            limitations[key] = _normalize_string_list(
                limitations.get(key),
                field_name=f"limitations.{key}",
                warnings=warnings,
            )
    return training, evaluation, limitations


def validate_model_card(
    payload: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
    check_files: bool = False,
    check_hashes: bool = False,
) -> ModelCardReport:
    errors: list[str] = []
    warnings: list[str] = []
    base = Path(base_dir).resolve() if base_dir is not None else None

    normalized_schema_version = _normalize_schema_version(
        payload,
        errors=errors,
        warnings=warnings,
    )
    model_name = _nonempty_str(payload.get("model_name", None))
    if model_name is None:
        errors.append("Missing required key: model_name")

    summary_norm = _validate_summary_section(
        payload,
        errors=errors,
    )
    weights_norm, weights_asset = _validate_weights_section(
        payload,
        base=base,
        check_files=bool(check_files),
        check_hashes=bool(check_hashes),
        errors=errors,
        warnings=warnings,
    )
    deployment_norm = _validate_deployment_section(
        payload,
        errors=errors,
        warnings=warnings,
    )
    training, evaluation, limitations = _normalized_optional_sections(
        payload,
        warnings=warnings,
    )

    normalized = {
        "schema_version": int(normalized_schema_version),
        "model_name": model_name,
        "summary": summary_norm,
        "weights": weights_norm,
        "training": training,
        "evaluation": evaluation,
        "deployment": deployment_norm,
        "limitations": limitations,
    }
    return ModelCardReport(
        ok=(len(errors) == 0),
        errors=tuple(errors),
        warnings=tuple(warnings),
        normalized=normalized,
        assets={"weights": weights_asset},
    )


def load_model_card_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model card not found: {p}")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to parse model card JSON: {p}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"Model card must be a JSON object (dict): {p}")
    return obj


def validate_model_card_file(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    manifest_base_dir: str | Path | None = None,
    check_files: bool = False,
    check_hashes: bool = False,
) -> ModelCardReport:
    payload = load_model_card_file(path)
    base = Path(base_dir) if base_dir is not None else Path(path).resolve().parent
    report = validate_model_card(
        payload,
        base_dir=base,
        check_files=bool(check_files),
        check_hashes=bool(check_hashes),
    )
    if manifest_path is None:
        return report
    return _cross_check_model_card_manifest(
        report,
        manifest_path=manifest_path,
        manifest_base_dir=manifest_base_dir,
        check_files=bool(check_files),
        check_hashes=bool(check_hashes),
    )


__all__ = [
    "ModelCardReport",
    "default_model_card_template",
    "load_model_card_file",
    "validate_model_card",
    "validate_model_card_file",
]
