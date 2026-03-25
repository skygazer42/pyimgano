from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from pyimgano.inference.validate_infer_config import validate_infer_config_file
from pyimgano.models.registry import list_models, model_info
from pyimgano.reporting.report import save_jsonl_records, save_run_report
from pyimgano.reporting.run_acceptance import evaluate_acceptance
from pyimgano.services.discovery_service import list_dataset_categories_payload
from pyimgano.services.train_service import TrainRunRequest, run_train_request
from pyimgano.services.benchmark_service import BenchmarkRunRequest, run_benchmark_request
from pyimgano.weights.bundle_audit import evaluate_bundle_weights_audit


@dataclass(frozen=True)
class DatasetEvaluationTarget:
    name: str
    root: str
    manifest_path: str | None = None
    categories: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ArtifactAuditConfig:
    enabled: bool = False
    train_config_path: str | None = None
    required_quality: str = "audited"
    export_infer_config: bool = True
    export_deploy_bundle: bool = True


@dataclass(frozen=True)
class EvaluationHarnessRequest:
    output_dir: str
    datasets: tuple[DatasetEvaluationTarget, ...]
    model_names: tuple[str, ...] | None = None
    exclude_model_names: tuple[str, ...] | None = None
    checkpoint_paths: dict[str, str] | None = None
    device: str = "cpu"
    pretrained: bool = False
    save_run: bool = True
    per_image_jsonl: bool = True
    continue_on_error: bool = True
    artifact_audit: ArtifactAuditConfig = field(default_factory=ArtifactAuditConfig)


def _as_tuple_str(values: Sequence[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    out = [str(item).strip() for item in values if str(item).strip()]
    return tuple(out) or None


def load_evaluation_harness_request(path: str | Path) -> EvaluationHarnessRequest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    datasets = tuple(
        DatasetEvaluationTarget(
            name=str(item["name"]),
            root=str(item["root"]),
            manifest_path=(
                str(item["manifest_path"]) if item.get("manifest_path") is not None else None
            ),
            categories=_as_tuple_str(item.get("categories")),
        )
        for item in raw.get("datasets", ())
    )
    if not datasets:
        raise ValueError("evaluation harness config requires a non-empty datasets list.")

    artifact_audit_raw = dict(raw.get("artifact_audit", {}))
    artifact_audit = ArtifactAuditConfig(
        enabled=bool(artifact_audit_raw.get("enabled", False)),
        train_config_path=(
            str(artifact_audit_raw["train_config_path"])
            if artifact_audit_raw.get("train_config_path") is not None
            else None
        ),
        required_quality=str(artifact_audit_raw.get("required_quality", "audited")),
        export_infer_config=bool(artifact_audit_raw.get("export_infer_config", True)),
        export_deploy_bundle=bool(artifact_audit_raw.get("export_deploy_bundle", True)),
    )

    runtime = dict(raw.get("runtime", {}))
    models = dict(raw.get("models", {}))
    checkpoint_paths = raw.get("checkpoint_paths", None)
    if checkpoint_paths is not None and not isinstance(checkpoint_paths, Mapping):
        raise ValueError("checkpoint_paths must be an object mapping model name to path.")

    return EvaluationHarnessRequest(
        output_dir=str(raw.get("output_dir", Path(path).resolve().parent / "evaluation_out")),
        datasets=datasets,
        model_names=_as_tuple_str(models.get("include")),
        exclude_model_names=_as_tuple_str(models.get("exclude")),
        checkpoint_paths=(
            {str(name): str(value) for name, value in dict(checkpoint_paths).items()}
            if checkpoint_paths is not None
            else None
        ),
        device=str(runtime.get("device", "cpu")),
        pretrained=bool(runtime.get("pretrained", False)),
        save_run=bool(runtime.get("save_run", True)),
        per_image_jsonl=bool(runtime.get("per_image_jsonl", True)),
        continue_on_error=bool(runtime.get("continue_on_error", True)),
        artifact_audit=artifact_audit,
    )


def collect_model_inventory(
    names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    selected = tuple(names) if names is not None else tuple(list_models())
    excluded = {str(name) for name in (exclude_names or ())}

    rows: list[dict[str, Any]] = []
    for name in selected:
        if str(name) in excluded:
            continue
        info = dict(model_info(str(name)))
        rows.append(
            {
                "name": str(info["name"]),
                "tags": list(info.get("tags", [])),
                "supports_pixel_map": bool(info.get("supports_pixel_map", False)),
                "requires_checkpoint": bool(info.get("requires_checkpoint", False)),
                "supports_save_load": bool(info.get("supports_save_load", False)),
            }
        )
    return rows


def collect_dataset_inventory(
    targets: Sequence[DatasetEvaluationTarget],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        manifest_path = (
            str(target.manifest_path) if target.manifest_path is not None else None
        )
        if target.categories is not None:
            categories = list(target.categories)
        else:
            categories = list_dataset_categories_payload(
                dataset=str(target.name),
                root=str(target.root),
                manifest_path=manifest_path,
            )
        rows.append(
            {
                "dataset": str(target.name),
                "root": str(target.root),
                "manifest_path": (
                    str(target.manifest_path) if target.manifest_path is not None else None
                ),
                "categories": [str(item) for item in categories],
            }
        )
    return rows


def build_evaluation_matrix(
    *,
    model_inventory: Sequence[Mapping[str, Any]],
    dataset_inventory: Sequence[Mapping[str, Any]],
    artifact_audit_enabled: bool,
    checkpoint_paths: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    provided_checkpoints = {str(k): str(v) for k, v in dict(checkpoint_paths or {}).items()}
    matrix: list[dict[str, Any]] = []

    for model_row in model_inventory:
        model_name = str(model_row["name"])
        requires_checkpoint = bool(model_row.get("requires_checkpoint", False))
        blocking_reasons: list[str] = []
        if requires_checkpoint and model_name not in provided_checkpoints:
            blocking_reasons.append("external_artifact_required")

        for dataset_row in dataset_inventory:
            for category in dataset_row.get("categories", ()):
                blocked = bool(blocking_reasons)
                matrix.append(
                    {
                        "model": model_name,
                        "dataset": str(dataset_row["dataset"]),
                        "root": str(dataset_row["root"]),
                        "manifest_path": dataset_row.get("manifest_path"),
                        "category": str(category),
                        "status": ("blocked" if blocked else "pending"),
                        "blocking_reasons": list(blocking_reasons),
                        "supports_pixel_map": bool(model_row.get("supports_pixel_map", False)),
                        "requires_checkpoint": requires_checkpoint,
                        "supports_save_load": bool(model_row.get("supports_save_load", False)),
                        "checkpoint_path": provided_checkpoints.get(model_name),
                        "planned_evaluations": {
                            "benchmark": not blocked,
                            "artifact_audit": bool(artifact_audit_enabled and not blocked),
                        },
                    }
                )
    return matrix


def _run_benchmark_for_entry(
    entry: Mapping[str, Any],
    request: EvaluationHarnessRequest,
) -> dict[str, Any]:
    benchmark_payload = run_benchmark_request(
        BenchmarkRunRequest(
            dataset=str(entry["dataset"]),
            root=str(entry["root"]),
            manifest_path=(
                str(entry["manifest_path"]) if entry.get("manifest_path") is not None else None
            ),
            category=str(entry["category"]),
            model=str(entry["model"]),
            device=str(request.device),
            pretrained=bool(request.pretrained),
            save_run=bool(request.save_run),
            per_image_jsonl=bool(request.per_image_jsonl),
            pixel=bool(entry.get("supports_pixel_map", False)),
            checkpoint_path=(
                str(entry["checkpoint_path"]) if entry.get("checkpoint_path") is not None else None
            ),
        )
    )
    payload = dict(benchmark_payload)
    payload.setdefault("status", "ok")
    return payload


def _run_artifact_audit_for_entry(
    entry: Mapping[str, Any],
    request: EvaluationHarnessRequest,
) -> dict[str, Any] | None:
    artifact_audit = request.artifact_audit
    if not bool(artifact_audit.enabled):
        return None
    if artifact_audit.train_config_path is None:
        return None

    report = run_train_request(
        TrainRunRequest(
            config_path=str(artifact_audit.train_config_path),
            dataset_name=str(entry["dataset"]),
            root=str(entry["root"]),
            category=str(entry["category"]),
            model_name=str(entry["model"]),
            device=str(request.device),
            export_infer_config=bool(artifact_audit.export_infer_config),
            export_deploy_bundle=bool(artifact_audit.export_deploy_bundle),
        )
    )
    run_dir_raw = report.get("run_dir", None)
    run_dir = Path(str(run_dir_raw)) if run_dir_raw is not None else None

    infer_valid = None
    infer_config_path = None
    if run_dir is not None:
        candidate = run_dir / "artifacts" / "infer_config.json"
        if candidate.is_file():
            infer_config_path = str(candidate)
            infer_valid = bool(validate_infer_config_file(candidate, check_files=True))

    acceptance = evaluate_acceptance(
        run_dir,
        required_quality=str(artifact_audit.required_quality),
    ) if run_dir is not None else None

    bundle_dir_raw = report.get("deploy_bundle_dir", None)
    bundle_dir = Path(str(bundle_dir_raw)) if bundle_dir_raw is not None else None
    bundle_audit = (
        evaluate_bundle_weights_audit(bundle_dir)
        if bundle_dir is not None and bundle_dir.is_dir()
        else None
    )

    return {
        "status": (
            str(acceptance.get("status"))
            if isinstance(acceptance, Mapping)
            else "not_applicable"
        ),
        "run_dir": (str(run_dir) if run_dir is not None else None),
        "deploy_bundle_dir": (str(bundle_dir) if bundle_dir is not None else None),
        "infer_config_path": infer_config_path,
        "infer_config_valid": infer_valid,
        "acceptance": dict(acceptance) if isinstance(acceptance, Mapping) else None,
        "bundle_audit": dict(bundle_audit) if isinstance(bundle_audit, Mapping) else None,
    }


def _write_dataset_inventory(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    save_run_report(path, {"datasets": list(rows)})


def _append_artifact_row(
    artifact_rows: list[dict[str, Any]],
    *,
    row: Mapping[str, Any],
    audit_payload: Mapping[str, Any] | None,
) -> None:
    if audit_payload is None:
        return
    artifact_rows.append(
        {
            "model": str(row["model"]),
            "dataset": str(row["dataset"]),
            "category": str(row["category"]),
            **dict(audit_payload),
        }
    )


def _execute_matrix_row(
    row: dict[str, Any],
    request: EvaluationHarnessRequest,
    *,
    artifact_rows: list[dict[str, Any]],
) -> str:
    benchmark_payload = _run_benchmark_for_entry(row, request)
    row["benchmark"] = dict(benchmark_payload)
    row["status"] = str(benchmark_payload.get("status", "ok"))

    if bool(row["planned_evaluations"]["artifact_audit"]):
        _append_artifact_row(
            artifact_rows,
            row=row,
            audit_payload=_run_artifact_audit_for_entry(row, request),
        )
    return str(row["status"])


def run_evaluation_harness(request: EvaluationHarnessRequest) -> dict[str, Any]:
    output_dir = Path(str(request.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_rows = collect_model_inventory(
        names=request.model_names,
        exclude_names=request.exclude_model_names,
    )
    dataset_rows = collect_dataset_inventory(request.datasets)
    matrix_rows = build_evaluation_matrix(
        model_inventory=model_rows,
        dataset_inventory=dataset_rows,
        artifact_audit_enabled=bool(request.artifact_audit.enabled),
        checkpoint_paths=request.checkpoint_paths,
    )

    artifact_rows: list[dict[str, Any]] = []
    succeeded = 0
    failed = 0
    blocked = 0

    for row in matrix_rows:
        if str(row["status"]) == "blocked":
            blocked += 1
            continue

        try:
            status = _execute_matrix_row(
                row,
                request,
                artifact_rows=artifact_rows,
            )
            if status == "ok":
                succeeded += 1
            else:
                failed += 1
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            failed += 1
            if not bool(request.continue_on_error):
                raise
            continue

    model_inventory_path = output_dir / "model_inventory.jsonl"
    dataset_inventory_path = output_dir / "dataset_inventory.json"
    evaluation_matrix_path = output_dir / "evaluation_matrix.jsonl"
    artifact_audit_path = output_dir / "artifact_audit.jsonl"

    save_jsonl_records(model_inventory_path, [dict(item) for item in model_rows])
    _write_dataset_inventory(dataset_inventory_path, dataset_rows)
    save_jsonl_records(evaluation_matrix_path, matrix_rows)
    save_jsonl_records(artifact_audit_path, artifact_rows)

    return {
        "output_dir": str(output_dir),
        "model_inventory_path": str(model_inventory_path),
        "dataset_inventory_path": str(dataset_inventory_path),
        "evaluation_matrix_path": str(evaluation_matrix_path),
        "artifact_audit_path": str(artifact_audit_path),
        "entries_total": int(len(matrix_rows)),
        "entries_succeeded": int(succeeded),
        "entries_failed": int(failed),
        "entries_blocked": int(blocked),
        "artifact_audits_attempted": int(len(artifact_rows)),
    }


__all__ = [
    "ArtifactAuditConfig",
    "DatasetEvaluationTarget",
    "EvaluationHarnessRequest",
    "build_evaluation_matrix",
    "collect_dataset_inventory",
    "collect_model_inventory",
    "load_evaluation_harness_request",
    "run_evaluation_harness",
]
