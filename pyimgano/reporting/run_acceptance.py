from __future__ import annotations

from pathlib import Path
from typing import Any

from pyimgano.inference.validate_infer_config import validate_infer_config_file
from pyimgano.reporting.publication_quality import evaluate_publication_quality
from pyimgano.reporting.run_quality import evaluate_run_quality
from pyimgano.weights.bundle_audit import evaluate_bundle_weights_audit

_QUALITY_STATUS_RANK = {
    "broken": 0,
    "partial": 1,
    "reproducible": 2,
    "audited": 3,
    "deployable": 4,
}
_REASON_CODE_MAP = {
    "insufficient_quality_status": "BUNDLE_REQUIRED_QUALITY_NOT_MET",
    "missing_infer_config": "BUNDLE_MISSING_INFER_CONFIG",
    "invalid_infer_config": "BUNDLE_INVALID_INFER_CONFIG",
    "bundle_weights_not_ready": "BUNDLE_WEIGHTS_NOT_READY",
}


def _infer_config_validation_payload() -> dict[str, Any]:
    return {
        "selected_source": None,
        "path": None,
        "present": False,
        "valid": None,
        "warnings": [],
        "errors": [],
        "trust_summary": {},
        "resolved_checkpoint_path": None,
        "resolved_model_checkpoint_path": None,
    }


def _select_infer_config_path(run_dir: Path) -> tuple[str | None, Path | None]:
    candidates = (
        ("deploy_bundle", run_dir / "deploy_bundle" / "infer_config.json"),
        ("artifacts", run_dir / "artifacts" / "infer_config.json"),
    )
    for source, path in candidates:
        if path.is_file():
            return source, path
    return None, None


def _evaluate_infer_config_validation(run_dir: Path) -> dict[str, Any]:
    source, path = _select_infer_config_path(run_dir)
    payload = _infer_config_validation_payload()
    payload["selected_source"] = source
    payload["path"] = path.as_posix() if path is not None else None
    payload["present"] = path is not None
    if path is None:
        payload["errors"] = ["No infer_config.json found under deploy_bundle/ or artifacts/."]
        return payload

    try:
        validation = validate_infer_config_file(path, check_files=True)
    except Exception as exc:  # noqa: BLE001 - acceptance boundary
        payload["valid"] = False
        payload["errors"] = [str(exc)]
        return payload

    payload["valid"] = True
    payload["warnings"] = list(validation.warnings)
    payload["trust_summary"] = dict(validation.trust_summary)
    payload["resolved_checkpoint_path"] = (
        str(validation.resolved_checkpoint_path)
        if validation.resolved_checkpoint_path is not None
        else None
    )
    payload["resolved_model_checkpoint_path"] = (
        str(validation.resolved_model_checkpoint_path)
        if validation.resolved_model_checkpoint_path is not None
        else None
    )
    return payload


def _bundle_weights_payload(run_dir: Path, *, check_bundle_hashes: bool) -> dict[str, Any]:
    bundle_dir = run_dir / "deploy_bundle"
    if not bundle_dir.is_dir():
        return {
            "applicable": False,
            "bundle_dir": str(bundle_dir),
            "present": False,
            "valid": None,
            "ready": None,
            "status": "not_applicable",
            "missing_required": [],
            "warnings": [],
            "errors": [],
            "trust_summary": {},
        }

    metadata_present = any(
        (bundle_dir / name).is_file() for name in ("model_card.json", "weights_manifest.json")
    )
    if not metadata_present:
        return {
            "applicable": False,
            "bundle_dir": str(bundle_dir),
            "present": False,
            "valid": None,
            "ready": None,
            "status": "not_applicable",
            "missing_required": [],
            "warnings": [],
            "errors": [],
            "trust_summary": {},
        }

    audit = evaluate_bundle_weights_audit(bundle_dir, check_hashes=bool(check_bundle_hashes))
    return {
        "applicable": True,
        **audit,
    }


def _acceptance_state(
    *,
    quality_status: str,
    quality_rank: int,
    required_rank: int,
    infer_present: bool,
    infer_valid: bool | None,
    bundle_weights_applicable: bool,
    bundle_weights_ready: bool | None,
) -> str:
    if not infer_present or infer_valid is False:
        return "blocked"
    if bundle_weights_applicable and bundle_weights_ready is not True:
        return "blocked"

    intrinsic_state = {
        "reproducible": "draft",
        "audited": "audited",
        "deployable": "deployable",
    }.get(str(quality_status), "blocked")
    if quality_rank < required_rank:
        return "blocked"
    return intrinsic_state


def _reason_codes(blocking_reasons: list[str]) -> list[str]:
    out: list[str] = []
    for reason in blocking_reasons:
        code = _REASON_CODE_MAP.get(str(reason))
        if code is not None and code not in out:
            out.append(code)
    return out


def evaluate_run_acceptance(
    run_dir: str | Path,
    *,
    required_quality: str = "audited",
    check_bundle_hashes: bool = False,
) -> dict[str, Any]:
    root = Path(run_dir)
    if str(required_quality) not in {"reproducible", "audited", "deployable"}:
        raise ValueError("required_quality must be one of: reproducible, audited, deployable")

    quality = evaluate_run_quality(root, check_bundle_hashes=bool(check_bundle_hashes))
    infer_config = _evaluate_infer_config_validation(root)
    bundle_weights = _bundle_weights_payload(root, check_bundle_hashes=bool(check_bundle_hashes))

    quality_rank = int(_QUALITY_STATUS_RANK.get(str(quality.get("status")), -1))
    required_rank = int(_QUALITY_STATUS_RANK[str(required_quality)])

    blocking_reasons: list[str] = []
    if quality_rank < required_rank:
        blocking_reasons.append("insufficient_quality_status")
    if not bool(infer_config.get("present")):
        blocking_reasons.append("missing_infer_config")
    elif infer_config.get("valid") is not True:
        blocking_reasons.append("invalid_infer_config")
    if bool(bundle_weights.get("applicable")) and bundle_weights.get("ready") is not True:
        blocking_reasons.append("bundle_weights_not_ready")

    ready = len(blocking_reasons) == 0
    acceptance_state = _acceptance_state(
        quality_status=str(quality.get("status")),
        quality_rank=quality_rank,
        required_rank=required_rank,
        infer_present=bool(infer_config.get("present")),
        infer_valid=(infer_config.get("valid") if isinstance(infer_config, dict) else None),
        bundle_weights_applicable=bool(bundle_weights.get("applicable")),
        bundle_weights_ready=(
            bundle_weights.get("ready") if isinstance(bundle_weights, dict) else None
        ),
    )
    reason_codes = _reason_codes(blocking_reasons)
    return {
        "run_dir": str(root),
        "status": ("ready" if ready else "partial"),
        "ready": bool(ready),
        "acceptance_state": acceptance_state,
        "reason_codes": reason_codes,
        "required_quality": str(required_quality),
        "quality": quality,
        "infer_config": infer_config,
        "bundle_weights": bundle_weights,
        "blocking_reasons": list(dict.fromkeys(str(item) for item in blocking_reasons)),
    }


_PUBLICATION_MARKER_FILES = ("leaderboard_metadata.json", "leaderboard.csv")


def _looks_like_publication_target(path: Path) -> bool:
    if path.name in _PUBLICATION_MARKER_FILES:
        return True
    if path.is_dir():
        return any((path / name).is_file() for name in _PUBLICATION_MARKER_FILES)
    return False


def _normalize_publication_target(path: Path) -> Path:
    if path.is_file() and path.name == "leaderboard.csv":
        return path.parent
    return path


def _publication_blocking_reasons(publication: dict[str, Any]) -> list[str]:
    blocking_reasons: list[str] = []
    if bool(publication.get("missing_required")):
        blocking_reasons.append("missing_required_exports")
    if bool(publication.get("invalid_declared")):
        blocking_reasons.append("invalid_declared_assets")
    if publication.get("publication_ready") is not True:
        blocking_reasons.append("publication_not_ready")
    return list(dict.fromkeys(blocking_reasons))


def evaluate_acceptance(
    path: str | Path,
    *,
    required_quality: str = "audited",
    check_bundle_hashes: bool = False,
) -> dict[str, Any]:
    root = Path(path)
    if _looks_like_publication_target(root):
        publication_target = _normalize_publication_target(root)
        publication = evaluate_publication_quality(publication_target)
        blocking_reasons = _publication_blocking_reasons(publication)
        ready = len(blocking_reasons) == 0 and str(publication.get("status")) == "ready"
        return {
            "kind": "publication",
            "path": str(root),
            "status": str(publication.get("status")),
            "ready": bool(ready),
            "required_quality": None,
            "blocking_reasons": blocking_reasons,
            "publication": publication,
        }

    run_acceptance = evaluate_run_acceptance(
        root,
        required_quality=required_quality,
        check_bundle_hashes=check_bundle_hashes,
    )
    return {
        "kind": "run",
        "path": str(root),
        **run_acceptance,
    }


__all__ = ["evaluate_acceptance", "evaluate_run_acceptance"]
