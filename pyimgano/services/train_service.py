from __future__ import annotations

import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pyimgano.config import load_config
from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest
from pyimgano.reporting.report import save_run_report
from pyimgano.services.train_export_helpers import (
    apply_bundle_manifest_metadata as _apply_bundle_manifest_metadata_helper,
)
from pyimgano.services.train_export_helpers import (
    build_optional_calibration_card_payload as _build_optional_calibration_card_payload_helper,
)
from pyimgano.services.train_export_helpers import (
    require_run_dir as _require_run_dir_helper,
)
from pyimgano.services.train_export_helpers import (
    rewrite_bundle_paths as _rewrite_bundle_paths_helper,
)
from pyimgano.services.train_export_helpers import (
    validate_export_request as _validate_export_request_helper,
)
from pyimgano.train_progress import get_active_train_progress_reporter
from pyimgano.workbench.config import WorkbenchConfig

_INFER_CONFIG_FILENAME = "infer_config.json"
_CALIBRATION_CARD_FILENAME = "calibration_card.json"
_OPERATOR_CONTRACT_FILENAME = "operator_contract.json"


@dataclass(frozen=True)
class TrainRunRequest:
    config_path: str
    dataset_name: str | None = None
    root: str | None = None
    category: str | None = None
    model_name: str | None = None
    device: str | None = None
    preprocessing_preset: str | None = None
    export_infer_config: bool = False
    export_deploy_bundle: bool = False


def apply_train_overrides(raw: dict[str, Any], request: TrainRunRequest) -> dict[str, Any]:
    import pyimgano.services.workbench_service as workbench_service

    return workbench_service.apply_workbench_overrides(
        raw,
        dataset_name=request.dataset_name,
        root=request.root,
        category=request.category,
        model_name=request.model_name,
        device=request.device,
        preprocessing_preset=request.preprocessing_preset,
    )


def load_train_config(request: TrainRunRequest) -> WorkbenchConfig:
    raw = load_config(Path(str(request.config_path)))
    raw = apply_train_overrides(raw, request)
    return WorkbenchConfig.from_dict(raw)


def _validate_manifest_dry_run(cfg: WorkbenchConfig) -> None:
    if str(cfg.dataset.name).lower() != "manifest":
        return

    manifest_path_raw = cfg.dataset.manifest_path
    manifest_path = Path(str(manifest_path_raw)) if manifest_path_raw is not None else None
    if manifest_path is None:
        raise ValueError("dataset.manifest_path is required when dataset.name='manifest'.")
    if not manifest_path.exists():
        raise ValueError(f"dataset.manifest_path not found: {manifest_path}")
    if not manifest_path.is_file():
        raise ValueError(f"dataset.manifest_path must be a file: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            handle.read(1)
    except Exception as exc:  # noqa: BLE001 - validation boundary
        raise ValueError(f"dataset.manifest_path not readable: {manifest_path}") from exc


def build_train_dry_run_payload(request: TrainRunRequest) -> dict[str, Any]:
    cfg = load_train_config(request)
    _validate_manifest_dry_run(cfg)
    return {"config": asdict(cfg)}


def run_train_preflight_payload(request: TrainRunRequest) -> dict[str, Any]:
    from pyimgano.workbench.preflight import run_preflight

    cfg = load_train_config(request)
    report = run_preflight(config=cfg)
    return {"preflight": asdict(report)}


def _export_deploy_bundle(*, run_dir: Path, infer_config_payload: dict[str, Any]) -> Path:
    bundle_dir = run_dir / "deploy_bundle"
    if bundle_dir.exists():
        raise FileExistsError(f"deploy bundle already exists: {bundle_dir}")
    bundle_dir.mkdir(parents=True, exist_ok=False)

    infer_src = run_dir / "artifacts" / _INFER_CONFIG_FILENAME
    if not infer_src.exists():
        raise FileNotFoundError(f"{_INFER_CONFIG_FILENAME} not found: {infer_src}")

    for name in ("report.json", "config.json", "environment.json"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, bundle_dir / name)

    calibration_card_src = run_dir / "artifacts" / _CALIBRATION_CARD_FILENAME
    if calibration_card_src.exists():
        shutil.copy2(calibration_card_src, bundle_dir / _CALIBRATION_CARD_FILENAME)

    operator_contract_src = run_dir / "artifacts" / _OPERATOR_CONTRACT_FILENAME
    if operator_contract_src.exists():
        shutil.copy2(operator_contract_src, bundle_dir / _OPERATOR_CONTRACT_FILENAME)

    bundle_payload = deepcopy(infer_config_payload)
    artifact_quality = bundle_payload.get("artifact_quality", None)
    if isinstance(artifact_quality, dict):
        audit_refs = artifact_quality.get("audit_refs", None)
        if isinstance(audit_refs, dict):
            rewritten_audit_refs = dict(audit_refs)
            if (
                "calibration_card" in rewritten_audit_refs
                and (bundle_dir / _CALIBRATION_CARD_FILENAME).is_file()
            ):
                rewritten_audit_refs["calibration_card"] = _CALIBRATION_CARD_FILENAME
            if (
                "operator_contract" in rewritten_audit_refs
                and (bundle_dir / _OPERATOR_CONTRACT_FILENAME).is_file()
            ):
                rewritten_audit_refs["operator_contract"] = _OPERATOR_CONTRACT_FILENAME
            artifact_quality["audit_refs"] = rewritten_audit_refs
        deploy_refs = artifact_quality.get("deploy_refs", None)
        rewritten_deploy_refs = dict(deploy_refs) if isinstance(deploy_refs, dict) else {}
        rewritten_deploy_refs["bundle_manifest"] = "bundle_manifest.json"
        artifact_quality["deploy_refs"] = rewritten_deploy_refs
        artifact_quality["has_deploy_bundle"] = True
        artifact_quality["has_bundle_manifest"] = True
        artifact_quality["required_bundle_artifacts_present"] = False
        artifact_quality["bundle_artifact_roles"] = {}
    bundle_payload = _rewrite_bundle_paths_helper(
        bundle_payload,
        bundle_dir=bundle_dir,
        infer_src=infer_src,
    )

    save_run_report(bundle_dir / _INFER_CONFIG_FILENAME, bundle_payload)
    bundle_manifest = build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir)
    _apply_bundle_manifest_metadata_helper(bundle_payload, bundle_manifest)
    save_run_report(bundle_dir / _INFER_CONFIG_FILENAME, bundle_payload)
    bundle_manifest = build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir)
    save_run_report(bundle_dir / "bundle_manifest.json", bundle_manifest)
    return bundle_dir


def _needs_export_artifacts(request: TrainRunRequest) -> bool:
    return bool(request.export_infer_config) or bool(request.export_deploy_bundle)


def _validate_export_request(cfg: WorkbenchConfig, request: TrainRunRequest) -> None:
    _validate_export_request_helper(cfg, request)


def _require_run_dir(report: dict[str, Any], *, deploy_bundle: bool = False) -> Path:
    return _require_run_dir_helper(report, deploy_bundle=deploy_bundle)


def _write_optional_operator_contract(
    infer_config_payload: dict[str, Any],
    *,
    run_dir: Path,
    reporter: Any,
) -> None:
    operator_contract_payload = infer_config_payload.get("operator_contract", None)
    if not isinstance(operator_contract_payload, dict):
        return
    operator_contract_path = run_dir / "artifacts" / "operator_contract.json"
    save_run_report(operator_contract_path, operator_contract_payload)
    reporter.on_artifact_written(
        kind="operator_contract",
        path=str(operator_contract_path),
    )


def _build_optional_calibration_card_payload(
    report: dict[str, Any],
    infer_config_payload: dict[str, Any],
) -> dict[str, Any] | None:
    return _build_optional_calibration_card_payload_helper(report, infer_config_payload)


def _write_optional_calibration_card(
    calibration_card_payload: dict[str, Any] | None,
    *,
    run_dir: Path,
    reporter: Any,
) -> None:
    if calibration_card_payload is None:
        return
    calibration_card_path = run_dir / "artifacts" / "calibration_card.json"
    save_run_report(calibration_card_path, calibration_card_payload)
    reporter.on_artifact_written(
        kind="calibration_card",
        path=str(calibration_card_path),
    )


def _export_infer_artifacts(
    *,
    cfg: WorkbenchConfig,
    report: dict[str, Any],
    reporter: Any,
) -> tuple[Path, dict[str, Any]]:
    import pyimgano.services.workbench_service as workbench_service

    run_dir = _require_run_dir(report)
    infer_config_path = run_dir / "artifacts" / _INFER_CONFIG_FILENAME
    infer_config_payload = workbench_service.build_infer_config_payload(
        config=cfg,
        report=report,
    )
    save_run_report(infer_config_path, infer_config_payload)
    reporter.on_artifact_written(kind="infer_config", path=str(infer_config_path))
    _write_optional_operator_contract(
        infer_config_payload,
        run_dir=run_dir,
        reporter=reporter,
    )
    _write_optional_calibration_card(
        _build_optional_calibration_card_payload(report, infer_config_payload),
        run_dir=run_dir,
        reporter=reporter,
    )
    return run_dir, infer_config_payload


def run_train_request(request: TrainRunRequest) -> dict[str, Any]:
    import pyimgano.recipes  # noqa: F401
    from pyimgano.recipes.registry import RECIPE_REGISTRY

    reporter = get_active_train_progress_reporter()
    cfg = load_train_config(request)
    _validate_export_request(cfg, request)
    recipe = RECIPE_REGISTRY.get(cfg.recipe)
    report = recipe(cfg)

    infer_config_payload: dict[str, Any] | None = None
    run_dir: Path | None = None
    if _needs_export_artifacts(request):
        run_dir, infer_config_payload = _export_infer_artifacts(
            cfg=cfg,
            report=report,
            reporter=reporter,
        )

    if bool(request.export_deploy_bundle):
        if infer_config_payload is None:
            raise RuntimeError(
                "Internal error: infer-config payload was not built for deploy bundle."
            )
        if run_dir is None:
            run_dir = _require_run_dir(report, deploy_bundle=True)
        bundle_dir = _export_deploy_bundle(
            run_dir=run_dir, infer_config_payload=infer_config_payload
        )
        report = dict(report)
        report["deploy_bundle_dir"] = str(bundle_dir)
        reporter.on_artifact_written(kind="deploy_bundle", path=str(bundle_dir))

    return report


__all__ = [
    "TrainRunRequest",
    "apply_train_overrides",
    "build_train_dry_run_payload",
    "load_train_config",
    "run_train_preflight_payload",
    "run_train_request",
]
