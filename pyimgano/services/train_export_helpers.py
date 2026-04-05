from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from pyimgano.reporting.calibration_card import build_calibration_card_payload


def require_run_dir(report: dict[str, Any], *, deploy_bundle: bool = False) -> Path:
    run_dir_raw = report.get("run_dir", None)
    if run_dir_raw is None:
        if deploy_bundle:
            raise ValueError("--export-deploy-bundle requires recipe output to include run_dir.")
        raise ValueError(
            "--export-infer-config/--export-deploy-bundle require recipe output to include run_dir."
        )
    return Path(str(run_dir_raw))


def validate_export_request(cfg: Any, request: Any) -> None:
    if not (bool(request.export_infer_config) or bool(request.export_deploy_bundle)):
        return
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


def build_optional_calibration_card_payload(
    report: dict[str, Any],
    infer_config_payload: dict[str, Any],
) -> dict[str, Any] | None:
    calibration_card_source = dict(report)
    prediction_payload = infer_config_payload.get("prediction", None)
    if isinstance(prediction_payload, dict):
        calibration_card_source["prediction"] = dict(prediction_payload)
    try:
        return build_calibration_card_payload(calibration_card_source)
    except ValueError:
        return None


def copy_deploy_bundle_supporting_files(
    *,
    run_dir: Path,
    bundle_dir: Path,
    calibration_card_filename: str,
    operator_contract_filename: str,
) -> None:
    for name in ("report.json", "config.json", "environment.json"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, bundle_dir / name)

    calibration_card_src = run_dir / "artifacts" / str(calibration_card_filename)
    if calibration_card_src.exists():
        shutil.copy2(calibration_card_src, bundle_dir / str(calibration_card_filename))

    operator_contract_src = run_dir / "artifacts" / str(operator_contract_filename)
    if operator_contract_src.exists():
        shutil.copy2(operator_contract_src, bundle_dir / str(operator_contract_filename))


def prepare_bundle_infer_config_payload(
    infer_config_payload: dict[str, Any],
    *,
    bundle_dir: Path,
    calibration_card_filename: str,
    operator_contract_filename: str,
) -> dict[str, Any]:
    bundle_payload = deepcopy(infer_config_payload)
    artifact_quality = bundle_payload.get("artifact_quality", None)
    if not isinstance(artifact_quality, dict):
        return bundle_payload

    audit_refs = artifact_quality.get("audit_refs", None)
    if isinstance(audit_refs, dict):
        rewritten_audit_refs = dict(audit_refs)
        if (
            "calibration_card" in rewritten_audit_refs
            and (bundle_dir / str(calibration_card_filename)).is_file()
        ):
            rewritten_audit_refs["calibration_card"] = str(calibration_card_filename)
        if (
            "operator_contract" in rewritten_audit_refs
            and (bundle_dir / str(operator_contract_filename)).is_file()
        ):
            rewritten_audit_refs["operator_contract"] = str(operator_contract_filename)
        artifact_quality["audit_refs"] = rewritten_audit_refs

    deploy_refs = artifact_quality.get("deploy_refs", None)
    rewritten_deploy_refs = dict(deploy_refs) if isinstance(deploy_refs, dict) else {}
    rewritten_deploy_refs["bundle_manifest"] = "bundle_manifest.json"
    artifact_quality["deploy_refs"] = rewritten_deploy_refs
    artifact_quality["has_deploy_bundle"] = True
    artifact_quality["has_bundle_manifest"] = True
    artifact_quality["required_bundle_artifacts_present"] = False
    artifact_quality["bundle_artifact_roles"] = {}
    return bundle_payload


def rewrite_bundle_paths(
    infer_config_payload: dict[str, Any],
    *,
    bundle_dir: Path,
    infer_src: Path,
) -> dict[str, Any]:
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
            for category_payload in per_category.values():
                _add_checkpoint(category_payload)
                _add_model_checkpoint(category_payload)
        return out

    bundle_payload = deepcopy(infer_config_payload)
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

    return bundle_payload


def apply_bundle_manifest_metadata(payload: dict[str, Any], manifest: dict[str, Any]) -> None:
    artifact_quality = payload.get("artifact_quality", None)
    if not isinstance(artifact_quality, dict):
        return
    artifact_quality["required_bundle_artifacts_present"] = bool(
        manifest.get("required_bundle_artifacts_present", False)
    )
    artifact_roles = manifest.get("artifact_roles", None)
    artifact_quality["bundle_artifact_roles"] = (
        dict(artifact_roles) if isinstance(artifact_roles, dict) else {}
    )


__all__ = [
    "apply_bundle_manifest_metadata",
    "build_optional_calibration_card_payload",
    "copy_deploy_bundle_supporting_files",
    "prepare_bundle_infer_config_payload",
    "require_run_dir",
    "rewrite_bundle_paths",
    "validate_export_request",
]
