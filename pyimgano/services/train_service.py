from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pyimgano.config import load_config
from pyimgano.reporting.deploy_bundle import (
    build_deploy_bundle_handoff_report,
    build_deploy_bundle_manifest,
)
from pyimgano.reporting.report import save_run_report
from pyimgano.services.train_export_helpers import (
    apply_bundle_manifest_metadata as _apply_bundle_manifest_metadata_helper,
)
from pyimgano.services.train_export_helpers import (
    build_optional_calibration_card_payload as _build_optional_calibration_card_payload_helper,
)
from pyimgano.services.train_export_helpers import (
    copy_deploy_bundle_supporting_files as _copy_deploy_bundle_supporting_files_helper,
)
from pyimgano.services.train_export_helpers import (
    prepare_bundle_infer_config_payload as _prepare_bundle_infer_config_payload_helper,
)
from pyimgano.services.train_export_helpers import require_run_dir as _require_run_dir_helper
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
_HANDOFF_REPORT_FILENAME = "handoff_report.json"


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


def _fallback_preflight_dataset_readiness(report: Any) -> dict[str, Any]:
    issues = getattr(report, "issues", [])
    issue_details = []
    has_error = False
    for item in issues:
        if isinstance(item, dict):
            code = item.get("code")
            message = item.get("message")
            severity = item.get("severity")
        else:
            code = getattr(item, "code", None)
            message = getattr(item, "message", None)
            severity = getattr(item, "severity", None)
        if code is None or message is None:
            continue
        if str(severity) == "error":
            has_error = True
        issue_details.append(
            {
                "code": str(code),
                "message": str(message),
            }
        )
    return {
        "status": ("error" if has_error else "ok"),
        "issue_codes": [str(item["code"]) for item in issue_details],
        "issue_details": issue_details,
    }


def _build_preflight_dataset_readiness(cfg: WorkbenchConfig, report: Any) -> dict[str, Any]:
    try:
        from pyimgano.datasets.inspection import profile_dataset_target

        dataset_name = str(cfg.dataset.name)
        category = str(cfg.dataset.category) if cfg.dataset.category is not None else None
        if dataset_name.lower() == "manifest":
            manifest_path = getattr(cfg.dataset, "manifest_path", None)
            if manifest_path is None:
                raise ValueError("manifest_path missing from config")
            payload = profile_dataset_target(
                target=str(manifest_path),
                dataset="manifest",
                category=category,
                root_fallback=str(cfg.dataset.root) if cfg.dataset.root is not None else None,
            )
        else:
            payload = profile_dataset_target(
                target=str(cfg.dataset.root),
                dataset=dataset_name,
                category=category,
            )
        readiness = payload.get("readiness", None)
        if isinstance(readiness, dict):
            return dict(readiness)
    except Exception:
        pass
    return _fallback_preflight_dataset_readiness(report)


def run_train_preflight_payload(request: TrainRunRequest) -> dict[str, Any]:
    from pyimgano.workbench.preflight import run_preflight

    cfg = load_train_config(request)
    report = run_preflight(config=cfg)
    payload = {"preflight": asdict(report)}
    payload["preflight"]["dataset_readiness"] = _build_preflight_dataset_readiness(cfg, report)
    return payload


def _export_deploy_bundle(*, run_dir: Path, infer_config_payload: dict[str, Any]) -> Path:
    bundle_dir = run_dir / "deploy_bundle"
    if bundle_dir.exists():
        raise FileExistsError(f"deploy bundle already exists: {bundle_dir}")
    bundle_dir.mkdir(parents=True, exist_ok=False)

    infer_src = run_dir / "artifacts" / _INFER_CONFIG_FILENAME
    if not infer_src.exists():
        raise FileNotFoundError(f"{_INFER_CONFIG_FILENAME} not found: {infer_src}")

    _copy_deploy_bundle_supporting_files_helper(
        run_dir=run_dir,
        bundle_dir=bundle_dir,
        calibration_card_filename=_CALIBRATION_CARD_FILENAME,
        operator_contract_filename=_OPERATOR_CONTRACT_FILENAME,
    )

    bundle_payload = _prepare_bundle_infer_config_payload_helper(
        infer_config_payload,
        bundle_dir=bundle_dir,
        calibration_card_filename=_CALIBRATION_CARD_FILENAME,
        operator_contract_filename=_OPERATOR_CONTRACT_FILENAME,
    )
    bundle_payload = _rewrite_bundle_paths_helper(
        bundle_payload,
        bundle_dir=bundle_dir,
        infer_src=infer_src,
    )

    infer_config_bundle_path = bundle_dir / _INFER_CONFIG_FILENAME
    handoff_report_bundle_path = bundle_dir / _HANDOFF_REPORT_FILENAME
    bundle_manifest_path = bundle_dir / "bundle_manifest.json"

    save_run_report(infer_config_bundle_path, bundle_payload)
    handoff_report_payload = build_deploy_bundle_handoff_report(
        bundle_dir=bundle_dir,
        source_run_dir=run_dir,
    )
    save_run_report(
        handoff_report_bundle_path,
        handoff_report_payload,
    )
    initial_bundle_manifest = build_deploy_bundle_manifest(
        bundle_dir=bundle_dir,
        source_run_dir=run_dir,
    )
    _apply_bundle_manifest_metadata_helper(bundle_payload, initial_bundle_manifest)
    save_run_report(infer_config_bundle_path, bundle_payload)
    final_bundle_manifest = build_deploy_bundle_manifest(
        bundle_dir=bundle_dir,
        source_run_dir=run_dir,
    )
    save_run_report(bundle_manifest_path, final_bundle_manifest)
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
