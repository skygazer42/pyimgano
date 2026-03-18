from __future__ import annotations

import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pyimgano.config import load_config
from pyimgano.reporting.calibration_card import build_calibration_card_payload
from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest
from pyimgano.reporting.report import save_run_report
from pyimgano.workbench.config import WorkbenchConfig

_INFER_CONFIG_FILENAME = "infer_config.json"


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

    calibration_card_src = run_dir / "artifacts" / "calibration_card.json"
    if calibration_card_src.exists():
        shutil.copy2(calibration_card_src, bundle_dir / "calibration_card.json")

    def _resolve_path(raw: str) -> Path:
        path = Path(raw)
        if path.is_absolute():
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            return path

        base = infer_src.parent
        candidates: list[Path] = [
            (base / path).resolve(),
            (base.parent / path).resolve(),
        ]

        from_run = infer_config_payload.get("from_run", None)
        if from_run is not None:
            try:
                candidates.append((Path(str(from_run)) / path).resolve())
            except Exception:
                pass

        for candidate in candidates:
            if candidate.exists():
                return candidate

        tried = "\n".join(f"- {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            "Artifact referenced by infer-config not found.\n"
            f"path={raw!r}\n"
            "Tried:\n"
            f"{tried}"
        )

    def _iter_path_slots(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any], str]]:
        out: list[tuple[str, dict[str, Any], str]] = []

        def _add_checkpoint(obj: Any) -> None:
            if not isinstance(obj, dict):
                return
            checkpoint = obj.get("checkpoint", None)
            if isinstance(checkpoint, dict):
                out.append(("trained_checkpoint", checkpoint, "path"))

        def _add_model_checkpoint(obj: Any) -> None:
            if not isinstance(obj, dict):
                return
            model = obj.get("model", None)
            if isinstance(model, dict):
                out.append(("model_checkpoint", model, "checkpoint_path"))

        _add_checkpoint(payload)
        _add_model_checkpoint(payload)

        per_category = payload.get("per_category", None)
        if isinstance(per_category, dict):
            for _category, category_payload in per_category.items():
                _add_checkpoint(category_payload)
                _add_model_checkpoint(category_payload)
        return out

    bundle_payload = deepcopy(infer_config_payload)
    artifact_quality = bundle_payload.get("artifact_quality", None)
    if isinstance(artifact_quality, dict):
        audit_refs = artifact_quality.get("audit_refs", None)
        if isinstance(audit_refs, dict) and (bundle_dir / "calibration_card.json").is_file():
            rewritten_audit_refs = dict(audit_refs)
            if "calibration_card" in rewritten_audit_refs:
                rewritten_audit_refs["calibration_card"] = "calibration_card.json"
            artifact_quality["audit_refs"] = rewritten_audit_refs
        deploy_refs = artifact_quality.get("deploy_refs", None)
        rewritten_deploy_refs = dict(deploy_refs) if isinstance(deploy_refs, dict) else {}
        rewritten_deploy_refs["bundle_manifest"] = "bundle_manifest.json"
        artifact_quality["deploy_refs"] = rewritten_deploy_refs
        artifact_quality["has_deploy_bundle"] = True
        artifact_quality["has_bundle_manifest"] = True
        artifact_quality["required_bundle_artifacts_present"] = False
        artifact_quality["bundle_artifact_roles"] = {}
    used_dst: set[Path] = set()
    bundle_root = bundle_dir.resolve()

    for kind, container, key in _iter_path_slots(bundle_payload):
        raw_any = container.get(key, None)
        if raw_any is None:
            continue
        raw = str(raw_any).strip()
        if not raw:
            continue

        path = Path(raw)
        src = _resolve_path(raw)

        if path.is_absolute():
            prefix = "checkpoints_abs" if kind == "trained_checkpoint" else "artifacts_abs"
            base = Path(prefix)
            dst_rel = base / path.name
            if dst_rel in used_dst:
                stem = path.stem
                suffix = path.suffix
                index = 2
                while True:
                    candidate = base / f"{stem}_{index}{suffix}"
                    if candidate not in used_dst:
                        dst_rel = candidate
                        break
                    index += 1
            container[key] = dst_rel.as_posix()
        else:
            dst_rel = Path(raw)

        used_dst.add(dst_rel)
        dst = (bundle_dir / dst_rel).resolve()
        if bundle_root not in dst.parents and dst != bundle_root:
            raise ValueError(f"infer-config path escapes deploy bundle: {raw!r} -> {dst_rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def _apply_bundle_manifest_metadata(
        payload: dict[str, Any], manifest: dict[str, Any]
    ) -> dict[str, Any]:
        artifact_quality = payload.get("artifact_quality", None)
        if not isinstance(artifact_quality, dict):
            return payload
        artifact_quality["required_bundle_artifacts_present"] = bool(
            manifest.get("required_bundle_artifacts_present", False)
        )
        artifact_roles = manifest.get("artifact_roles", None)
        artifact_quality["bundle_artifact_roles"] = (
            dict(artifact_roles) if isinstance(artifact_roles, dict) else {}
        )
        return payload

    save_run_report(bundle_dir / _INFER_CONFIG_FILENAME, bundle_payload)
    bundle_manifest = build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir)
    bundle_payload = _apply_bundle_manifest_metadata(bundle_payload, bundle_manifest)
    save_run_report(bundle_dir / _INFER_CONFIG_FILENAME, bundle_payload)
    bundle_manifest = build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir)
    save_run_report(bundle_dir / "bundle_manifest.json", bundle_manifest)
    return bundle_dir


def run_train_request(request: TrainRunRequest) -> dict[str, Any]:
    import pyimgano.recipes  # noqa: F401
    import pyimgano.services.workbench_service as workbench_service
    from pyimgano.recipes.registry import RECIPE_REGISTRY

    cfg = load_train_config(request)
    recipe = RECIPE_REGISTRY.get(cfg.recipe)
    report = recipe(cfg)

    infer_config_payload: dict[str, Any] | None = None
    if bool(request.export_infer_config) or bool(request.export_deploy_bundle):
        if (
            bool(request.export_deploy_bundle)
            and bool(cfg.defects.enabled)
            and cfg.defects.pixel_threshold is None
        ):
            raise ValueError(
                "--export-deploy-bundle with defects.enabled=true requires defects.pixel_threshold to be set.\n"
                "Deploy bundles are intended to be self-contained for `pyimgano-infer --infer-config ... --defects`."
            )
        if not bool(cfg.output.save_run):
            raise ValueError(
                "--export-infer-config/--export-deploy-bundle require output.save_run=true."
            )
        run_dir_raw = report.get("run_dir", None)
        if run_dir_raw is None:
            raise ValueError(
                "--export-infer-config/--export-deploy-bundle require recipe output to include run_dir."
            )
        run_dir = Path(str(run_dir_raw))
        infer_config_path = run_dir / "artifacts" / _INFER_CONFIG_FILENAME

        infer_config_payload = workbench_service.build_infer_config_payload(
            config=cfg,
            report=report,
        )
        save_run_report(infer_config_path, infer_config_payload)
        calibration_card_source = dict(report)
        prediction_payload = infer_config_payload.get("prediction", None)
        if isinstance(prediction_payload, dict):
            calibration_card_source["prediction"] = dict(prediction_payload)
        try:
            calibration_card_payload = build_calibration_card_payload(calibration_card_source)
        except ValueError:
            calibration_card_payload = None
        if calibration_card_payload is not None:
            save_run_report(
                run_dir / "artifacts" / "calibration_card.json",
                calibration_card_payload,
            )

    if bool(request.export_deploy_bundle):
        if infer_config_payload is None:
            raise RuntimeError(
                "Internal error: infer-config payload was not built for deploy bundle."
            )
        run_dir_raw = report.get("run_dir", None)
        if run_dir_raw is None:
            raise ValueError("--export-deploy-bundle requires recipe output to include run_dir.")
        run_dir = Path(str(run_dir_raw))
        bundle_dir = _export_deploy_bundle(run_dir=run_dir, infer_config_payload=infer_config_payload)
        report = dict(report)
        report["deploy_bundle_dir"] = str(bundle_dir)

    return report


__all__ = [
    "TrainRunRequest",
    "apply_train_overrides",
    "build_train_dry_run_payload",
    "load_train_config",
    "run_train_preflight_payload",
    "run_train_request",
]
