from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class ExportServiceError(RuntimeError):
    """Raised when a persisted run cannot produce the requested artifact transaction."""


@dataclass(frozen=True)
class RunExportSource:
    run_dir: Path
    category: str
    config: Any
    report: Mapping[str, Any]
    category_report: Mapping[str, Any]
    policy: Mapping[str, Any]
    infer_context: Any
    checkpoint_contract: Any

    @property
    def model_name(self) -> str:
        return str(self.infer_context.model_name)

    @property
    def model_kwargs(self) -> dict[str, Any]:
        from pyimgano.services.model_options import resolve_model_options

        user_kwargs = dict(self.infer_context.base_user_kwargs)
        for key in ("checkpoint", "checkpoint_path"):
            user_kwargs.pop(key, None)
        auto_kwargs: dict[str, Any] = {
            "device": str(self.infer_context.device),
            "contamination": float(self.infer_context.contamination),
            "pretrained": bool(self.infer_context.pretrained),
        }
        seed = getattr(self.config, "seed", None)
        if seed is not None:
            auto_kwargs["random_seed"] = int(seed)
            auto_kwargs["random_state"] = int(seed)
        return dict(
            resolve_model_options(
                model_name=self.model_name,
                preset=(
                    str(self.infer_context.preset)
                    if self.infer_context.preset is not None
                    else None
                ),
                user_kwargs=user_kwargs,
                auto_kwargs=auto_kwargs,
                checkpoint_path=None,
            )
        )


def _normalize_verification_level(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in {"reference_parity", "end_to_end"}:
        raise ValueError(
            "verification_level must be reference_parity or end_to_end; "
            "schema-v1 artifacts do not support an unverified mode."
        )
    return normalized


def _normalize_formats(values: Sequence[str]) -> tuple[str, ...]:
    from pyimgano.exporting import ArtifactFormat

    normalized = tuple(str(ArtifactFormat(str(value).strip().lower())) for value in values)
    if not normalized:
        raise ValueError("At least one artifact format is required.")
    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate artifact formats are not allowed.")
    return normalized


def _policy_without_external_reconstruction_paths(
    payload: Mapping[str, Any],
    *,
    model_name: str,
    category: str | None,
    model_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the artifact-local operational policy from a legacy infer-config.

    The legacy schema remains readable, but a new artifact must not depend on its
    source run or checkpoint paths.  A small model identity mirror is retained for
    native/composite conflict detection.
    """

    # Keep only operational policy fields. Audit/source-run material remains in the
    # export verification report and must never become runtime reconstruction input.
    out: dict[str, Any] = {
        "schema_family": "pyimgano-artifact-policy",
        "schema_version": 1,
    }
    for key in ("threshold", "postprocess", "defects", "prediction", "adaptation"):
        if key in payload:
            out[key] = deepcopy(payload[key])

    constructor_kwargs = dict(model_kwargs)
    for key in ("checkpoint", "checkpoint_path"):
        constructor_kwargs.pop(key, None)
    model: dict[str, Any] = {
        "registry_name": str(model_name),
        "constructor_kwargs": constructor_kwargs,
    }
    if category is not None:
        model["category"] = str(category)
    out["model"] = model

    postprocess = out.get("postprocess")
    if not isinstance(postprocess, Mapping):
        postprocess = {}
    else:
        postprocess = dict(postprocess)
    image_threshold = postprocess.get("image_threshold")
    if not isinstance(image_threshold, Mapping):
        image_threshold = {}
    else:
        image_threshold = dict(image_threshold)
    if image_threshold.get("threshold") is None and out.get("threshold") is not None:
        image_threshold["threshold"] = float(out["threshold"])
    image_threshold.setdefault("threshold", None)
    image_threshold.setdefault("score_order", "higher_is_more_anomalous")
    postprocess["image_threshold"] = image_threshold
    out["postprocess"] = postprocess
    return out


def prepare_run_export_source(
    run_dir: str | Path,
    *,
    category: str | None = None,
) -> RunExportSource:
    import pyimgano.services.infer_context_service as infer_context_service
    import pyimgano.services.workbench_run_service as workbench_run_service
    from pyimgano.exporting import CheckpointContract
    from pyimgano.inference.config import select_infer_category
    from pyimgano.services.workbench_service import build_infer_config_payload

    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory not found: {root}")

    config = workbench_run_service.load_workbench_config_from_run(root)
    report = workbench_run_service.load_report_from_run(root)
    selected_name, category_report = workbench_run_service.select_category_report(
        report,
        category=(str(category) if category is not None else None),
    )
    selected = str(
        selected_name
        or category_report.get("category")
        or getattr(config.dataset, "category", "default")
    )

    infer_context = infer_context_service.prepare_from_run_context(
        infer_context_service.FromRunInferContextRequest(
            run_dir=str(root),
            from_run_category=selected,
            pretrained=False,
        )
    )
    if infer_context.trained_checkpoint_path is None:
        raise ExportServiceError(
            f"Run category {selected!r} has no persisted trained checkpoint; "
            "trained artifact export cannot recreate or refit the detector."
        )

    infer_payload = build_infer_config_payload(config=config, report=report)
    infer_payload = select_infer_category(infer_payload, category=selected)
    checkpoint_meta = category_report.get("checkpoint")
    checkpoint_contract = CheckpointContract.from_mapping(
        checkpoint_meta if isinstance(checkpoint_meta, Mapping) else None
    )
    source = RunExportSource(
        run_dir=root,
        category=selected,
        config=config,
        report=dict(report),
        category_report=dict(category_report),
        policy={},
        infer_context=infer_context,
        checkpoint_contract=checkpoint_contract,
    )
    policy = _policy_without_external_reconstruction_paths(
        infer_payload,
        model_name=source.model_name,
        category=selected,
        model_kwargs=source.model_kwargs,
    )
    return RunExportSource(
        run_dir=source.run_dir,
        category=source.category,
        config=source.config,
        report=source.report,
        category_report=source.category_report,
        policy=policy,
        infer_context=source.infer_context,
        checkpoint_contract=source.checkpoint_contract,
    )


def _capability_failure(format_name: str, capability: Any) -> dict[str, Any]:
    payload = capability.to_dict() if hasattr(capability, "to_dict") else {}
    return {
        "format": str(format_name),
        "reason": str(payload.get("reason_code") or "unsupported"),
        "remediation": payload.get("remediation"),
        "capability": payload,
    }


def _static_capabilities(
    source: RunExportSource,
    formats: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from pyimgano.exporting import ArtifactFormat, get_export_capability

    context = {
        "run_dir": str(source.run_dir),
        "category": source.category,
        "model_name": source.model_name,
        "model_kwargs": source.model_kwargs,
        "checkpoint_contract": source.checkpoint_contract.to_dict(),
    }
    supported: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for format_name in formats:
        capability = get_export_capability(
            source.model_name,
            ArtifactFormat(format_name),
            context=context,
        )
        if bool(getattr(capability, "supported", False)):
            supported[format_name] = capability
        else:
            failures.append(_capability_failure(format_name, capability))
    return supported, failures


def _run_export_categories(
    run_dir: str | Path,
    *,
    category: str | None,
) -> tuple[str | None, ...]:
    """Resolve an explicit category or all categories from a persisted report."""

    import pyimgano.services.workbench_run_service as workbench_run_service
    from pyimgano.artifacts import category_slug

    if category is not None:
        names: tuple[str | None, ...] = (str(category),)
    else:
        report = workbench_run_service.load_report_from_run(Path(run_dir).resolve())
        per_category = report.get("per_category")
        if isinstance(per_category, Mapping):
            if not per_category:
                raise ExportServiceError("Run report per_category mapping is empty.")
            names = tuple(sorted((str(key) for key in per_category), key=str.casefold))
        else:
            names = (None,)

    normalized_names: dict[str, str] = {}
    normalized_slugs: dict[str, str] = {}
    for raw_name in names:
        if raw_name is None:
            continue
        slug = category_slug(raw_name)
        category_key = raw_name.casefold()
        previous_name = normalized_names.setdefault(category_key, raw_name)
        if previous_name != raw_name:
            raise ExportServiceError(
                "Run categories collide after Unicode case normalization: "
                f"{previous_name!r} and {raw_name!r}."
            )
        previous_slug = normalized_slugs.setdefault(slug.casefold(), raw_name)
        if previous_slug != raw_name:
            raise ExportServiceError(
                f"Run categories have the same cross-platform slug: "
                f"{previous_slug!r} and {raw_name!r}."
            )
    return names


def prepare_run_export_sources(
    run_dir: str | Path,
    *,
    category: str | None = None,
) -> tuple[RunExportSource, ...]:
    """Normalize a single- or multi-category run without loading checkpoints."""

    return tuple(
        prepare_run_export_source(run_dir, category=selected)
        for selected in _run_export_categories(run_dir, category=category)
    )


def preflight_train_export(
    *, config: Any, formats: Sequence[str], strict: bool = True
) -> dict[str, Any]:
    """Reject statically unsupported train-time export before expensive training.

    Conditional capabilities are allowed here because checkpoint completeness and
    concrete fitted components are only knowable after training.  They are resolved
    again by :func:`export_from_run` before checkpoint loading.
    """

    from pyimgano.exporting import ArtifactFormat, ExportStatus, get_export_capability

    normalized = _normalize_formats(tuple(formats))
    model_name = str(config.model.name)
    context = {
        "phase": "pre_training",
        "model_name": model_name,
        "model_kwargs": dict(getattr(config.model, "model_kwargs", {}) or {}),
    }
    failures: list[dict[str, Any]] = []
    capabilities: list[dict[str, Any]] = []
    for value in normalized:
        capability = get_export_capability(
            model_name,
            ArtifactFormat(value),
            context=context,
        )
        payload = capability.to_dict()
        capabilities.append(payload)
        if capability.status is ExportStatus.UNSUPPORTED:
            failures.append(_capability_failure(value, capability))
    if failures and strict:
        details = ", ".join(f"{row['format']}:{row['reason']}" for row in failures)
        raise ExportServiceError(f"Train-time artifact export is unsupported: {details}")
    return {"model": model_name, "capabilities": capabilities, "failures": failures}


def _load_restored_detector(source: RunExportSource, *, trust_checkpoint: bool) -> Any:
    import pyimgano.services.infer_load_service as infer_load_service

    if bool(source.checkpoint_contract.requires_trust) and not bool(trust_checkpoint):
        raise ExportServiceError(
            "Checkpoint is explicitly marked executable/trust-required. "
            "Re-run with trust_checkpoint=True only after verifying provenance."
        )
    if not bool(source.checkpoint_contract.strict_exportable):
        raise ExportServiceError(
            "Checkpoint does not carry complete adapter/codec round-trip evidence; "
            f"completeness={source.checkpoint_contract.completeness}. "
            "Legacy or unknown checkpoints cannot be promoted by loading them."
        )

    loaded = infer_load_service.load_config_backed_infer_detector(
        infer_load_service.ConfigBackedInferLoadRequest(
            context=source.infer_context,
            seed=(
                int(source.config.seed)
                if getattr(source.config, "seed", None) is not None
                else None
            ),
            trust_checkpoint=bool(trust_checkpoint),
        )
    )
    return loaded.detector


def _native_result_payload(result: Any, *, format_name: str) -> dict[str, Any]:
    root = Path(getattr(result, "artifact_root"))
    manifest_path = Path(getattr(result, "manifest_path"))
    manifest = (
        dict(getattr(result, "manifest"))
        if isinstance(getattr(result, "manifest", None), Mapping)
        else {}
    )
    runtime = manifest.get("runtime")
    runtime = dict(runtime) if isinstance(runtime, Mapping) else {}
    return {
        "format": str(format_name),
        "backend": runtime.get("backend"),
        "path": str(root),
        "manifest": str(manifest_path),
        "artifact_id": manifest.get("artifact_id"),
    }


def _export_one(
    *,
    detector: Any,
    source: RunExportSource,
    format_name: str,
    target: Path,
    verification_level: str,
) -> dict[str, Any]:
    from pyimgano.exporting import (
        ArtifactFormat,
        NativeExportContext,
        export_native,
        get_export_adapter,
    )

    adapter = get_export_adapter(source.model_name)
    build_output_contract = getattr(adapter, "build_output_contract", None)
    output_contract = (
        dict(build_output_contract())
        if callable(build_output_contract)
        else {
            "score": {
                "name": "score",
                "transform": "identity",
                "score_order": "higher_is_more_anomalous",
            }
        }
    )
    context = NativeExportContext(
        model_name=source.model_name,
        model_kwargs=source.model_kwargs,
        policy=source.policy,
        checkpoint_contract=source.checkpoint_contract,
        category=source.category,
        verification={
            "level": str(verification_level),
            "source": "persisted_run",
            "mandatory": True,
        },
        output_contract=output_contract,
    )

    artifact_format = ArtifactFormat(format_name)
    if artifact_format is ArtifactFormat.NATIVE:
        result = export_native(
            detector,
            context=context,
            out=target,
            adapter=adapter,
            overwrite=False,
        )
        return _native_result_payload(result, format_name=format_name)

    export_method = getattr(adapter, "export_artifact", None)
    if not callable(export_method):
        raise ExportServiceError(
            f"Adapter {getattr(adapter, 'adapter_id', source.model_name)!r} declares "
            f"{format_name} support but does not implement export_artifact()."
        )
    result = export_method(
        detector,
        format=artifact_format,
        context=context,
        out=target,
        overwrite=False,
    )
    return _native_result_payload(result, format_name=format_name)


def _atomic_publish(staging: Path, destination: Path, *, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Artifact output already exists: {destination}")

    backup: Path | None = None
    try:
        if destination.exists():
            backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
            os.replace(destination, backup)
        os.replace(staging, destination)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    finally:
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def export_from_run(
    *,
    run_dir: str | Path,
    formats: Sequence[str] = ("native",),
    out_dir: str | Path | None = None,
    category: str | None = None,
    verification_level: str = "reference_parity",
    strict: bool = True,
    trust_checkpoint: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export fitted artifacts from a persisted run through one transactional path."""

    normalized_formats = _normalize_formats(tuple(formats))
    verification = _normalize_verification_level(verification_level)
    sources = prepare_run_export_sources(run_dir, category=category)

    supported_by_category: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for source in sources:
        supported, source_failures = _static_capabilities(source, normalized_formats)
        supported_by_category[source.category] = supported
        failures.extend({"category": source.category, **failure} for failure in source_failures)
    if failures and strict:
        details = ", ".join(
            f"{item['category']}/{item['format']}:{item['reason']}" for item in failures
        )
        raise ExportServiceError(
            "Requested export transaction is unsupported before checkpoint loading: " + details
        )
    selected_by_category = {
        source.category: tuple(
            format_name
            for format_name in normalized_formats
            if format_name in supported_by_category[source.category]
        )
        for source in sources
    }
    if not any(selected_by_category.values()):
        return {
            "status": "failed",
            "run_dir": str(sources[0].run_dir),
            "category": sources[0].category if len(sources) == 1 else None,
            "categories": [source.category for source in sources],
            "artifacts": [],
            "failures": failures,
        }

    destination = (
        Path(out_dir).resolve()
        if out_dir is not None
        else (sources[0].run_dir / "artifacts" / "exported").resolve()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Artifact output already exists: {destination}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=str(destination.parent))
    )

    artifacts: list[dict[str, Any]] = []
    try:
        from pyimgano.artifacts import category_slug, write_export_index

        for source in sources:
            selected_formats = selected_by_category[source.category]
            if not selected_formats:
                continue
            try:
                detector = _load_restored_detector(
                    source,
                    trust_checkpoint=trust_checkpoint,
                )
            except Exception as exc:  # noqa: BLE001 - category checkpoint boundary
                for format_name in selected_formats:
                    failures.append(
                        {
                            "category": source.category,
                            "format": format_name,
                            "reason": str(exc),
                        }
                    )
                if strict:
                    raise
                continue

            for format_name in selected_formats:
                target = staging / category_slug(source.category) / format_name
                try:
                    payload = _export_one(
                        detector=detector,
                        source=source,
                        format_name=format_name,
                        target=target,
                        verification_level=verification,
                    )
                    payload["category"] = source.category
                    payload["slug"] = category_slug(source.category)
                    payload["path"] = str(destination / Path(payload["path"]).relative_to(staging))
                    payload["manifest"] = str(
                        destination / Path(payload["manifest"]).relative_to(staging)
                    )
                    artifacts.append(payload)
                except Exception as exc:  # noqa: BLE001 - format boundary
                    failures.append(
                        {
                            "category": source.category,
                            "format": format_name,
                            "reason": str(exc),
                        }
                    )
                    if strict:
                        raise

        expected_artifacts = len(sources) * len(normalized_formats)
        if strict and len(artifacts) != expected_artifacts:
            raise ExportServiceError("Strict export transaction did not produce every format.")
        if not artifacts:
            return {
                "status": "failed",
                "run_dir": str(sources[0].run_dir),
                "category": sources[0].category if len(sources) == 1 else None,
                "categories": [source.category for source in sources],
                "artifacts": [],
                "failures": failures,
            }

        index_entries = [
            {
                "category": item["category"],
                "slug": item["slug"],
                "format": item["format"],
                "backend": item["backend"],
                "artifact": Path(item["path"]).relative_to(destination).as_posix(),
                "manifest": Path(item["manifest"]).relative_to(destination).as_posix(),
                "artifact_id": item["artifact_id"],
            }
            for item in artifacts
        ]
        write_export_index(staging / "export_index.json", index_entries)
        _atomic_publish(staging, destination, overwrite=bool(overwrite))
        status = "partial" if failures else "ok"
        return {
            "status": status,
            "run_dir": str(sources[0].run_dir),
            "category": sources[0].category if len(sources) == 1 else None,
            "categories": [source.category for source in sources],
            "output_dir": str(destination),
            "artifacts": artifacts,
            "failures": failures,
        }
    except Exception as exc:
        if isinstance(exc, ExportServiceError):
            raise
        raise ExportServiceError(f"Artifact export failed: {exc}") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging)


__all__ = [
    "ExportServiceError",
    "RunExportSource",
    "export_from_run",
    "preflight_train_export",
    "prepare_run_export_source",
    "prepare_run_export_sources",
]
