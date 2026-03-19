from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from pyimgano.weights.bundle_audit import evaluate_bundle_weights_audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyimgano-weights",
        description=(
            "Utilities for managing local model weights/checkpoints (manifest validation + hashing). "
            "This tool never downloads weights."
        ),
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_hash = sub.add_parser("hash", help="Compute file hash (default: sha256)")
    p_hash.add_argument("path", help="Path to file")
    p_hash.add_argument(
        "--algorithm",
        default="sha256",
        choices=["md5", "sha1", "sha256", "sha512"],
        help="Hash algorithm. Default: sha256",
    )

    p_validate = sub.add_parser("validate", help="Validate a weights manifest JSON file")
    p_validate.add_argument("manifest", help="Path to weights manifest JSON")
    p_validate.add_argument(
        "--base-dir",
        default=None,
        help=(
            "Base directory used to resolve relative entry paths. "
            "Defaults to manifest parent directory."
        ),
    )
    p_validate.add_argument(
        "--check-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check that weight files exist on disk. Default: true",
    )
    p_validate.add_argument(
        "--check-hashes",
        action="store_true",
        default=False,
        help="Verify sha256 for entries that provide it. Default: false",
    )
    p_validate.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON report instead of printing human-readable errors/warnings.",
    )

    p_validate_model_card = sub.add_parser(
        "validate-model-card",
        help="Validate a model card JSON file",
    )
    p_validate_model_card.add_argument("path", help="Path to model card JSON")
    p_validate_model_card.add_argument(
        "--base-dir",
        default=None,
        help=(
            "Base directory used to resolve relative weights.path entries. "
            "Defaults to the model card parent directory."
        ),
    )
    p_validate_model_card.add_argument(
        "--manifest",
        default=None,
        help=(
            "Optional weights manifest JSON to cross-check against the model card. "
            "When provided, validation verifies that the referenced asset matches a "
            "manifest entry."
        ),
    )
    p_validate_model_card.add_argument(
        "--manifest-base-dir",
        default=None,
        help=(
            "Optional base directory used to resolve relative manifest entry paths. "
            "Defaults to the manifest parent directory."
        ),
    )
    p_validate_model_card.add_argument(
        "--check-files",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Check that the model card weights.path exists on disk. Default: false",
    )
    p_validate_model_card.add_argument(
        "--check-hashes",
        action="store_true",
        default=False,
        help="Verify weights.sha256 when the asset exists. Default: false",
    )
    p_validate_model_card.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON report instead of printing human-readable errors/warnings.",
    )

    p_audit_bundle = sub.add_parser(
        "audit-bundle",
        help="Validate deploy-bundle model_card.json + weights_manifest.json together",
    )
    p_audit_bundle.add_argument(
        "bundle_dir",
        help="Path to deploy_bundle directory containing model_card.json / weights_manifest.json",
    )
    p_audit_bundle.add_argument(
        "--check-hashes",
        action="store_true",
        default=False,
        help="Verify sha256 values for bundle-local assets when declared. Default: false",
    )
    p_audit_bundle.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON audit report instead of printing human-readable output.",
    )

    p_template = sub.add_parser(
        "template",
        help="Print a JSON template for a weights manifest or model card",
    )
    template_sub = p_template.add_subparsers(dest="template_kind", required=True)
    template_sub.add_parser("manifest", help="Emit a sample weights manifest JSON")
    template_sub.add_parser("model-card", help="Emit a sample model card JSON")

    return parser


def _weights_manifest_template() -> dict[str, object]:
    return {
        "schema_version": 1,
        "entries": [
            {
                "name": "example_checkpoint",
                "path": "checkpoints/example_model.pt",
                "sha256": "replace-with-sha256-if-known",
                "source": "internal training run or upstream checkpoint source",
                "license": "internal-or-upstream-license",
                "runtime": "torch",
                "models": ["vision_patchcore"],
            }
        ],
    }


def _nonempty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _has_runtime(entry: Mapping[str, Any]) -> bool:
    runtime = _nonempty_str(entry.get("runtime", None))
    if runtime is not None:
        return True
    runtimes = entry.get("runtimes", None)
    if not isinstance(runtimes, list):
        return False
    return any(_nonempty_str(item) is not None for item in runtimes)


def _build_manifest_trust_summary(
    report: Any,
    *,
    manifest_path: str | Path,
    check_files: bool,
    check_hashes: bool,
) -> dict[str, Any]:
    entries = [dict(item) for item in getattr(report, "entries", ())]
    has_entries = len(entries) > 0
    trust_signals = {
        "file_refs_checked": bool(check_files),
        "hashes_checked": bool(check_hashes),
        "has_entries": bool(has_entries),
        "all_entries_have_sha256": bool(
            has_entries and all(_nonempty_str(item.get("sha256", None)) is not None for item in entries)
        ),
        "all_entries_have_source": bool(
            has_entries and all(_nonempty_str(item.get("source", None)) is not None for item in entries)
        ),
        "all_entries_have_license": bool(
            has_entries and all(_nonempty_str(item.get("license", None)) is not None for item in entries)
        ),
        "all_entries_have_runtime": bool(has_entries and all(_has_runtime(item) for item in entries)),
    }
    degraded_by: list[str] = []
    if not trust_signals["has_entries"]:
        degraded_by.append("missing_entries")
    if not trust_signals["all_entries_have_sha256"]:
        degraded_by.append("missing_sha256")
    if not trust_signals["all_entries_have_source"]:
        degraded_by.append("missing_source")
    if not trust_signals["all_entries_have_license"]:
        degraded_by.append("missing_license")
    if not trust_signals["all_entries_have_runtime"]:
        degraded_by.append("missing_runtime")

    if not bool(getattr(report, "ok", False)):
        status = "broken"
    elif not degraded_by:
        status = "trust-signaled"
    else:
        status = "partial"

    return {
        "status": status,
        "trust_signals": trust_signals,
        "degraded_by": degraded_by,
        "audit_refs": {
            "weights_manifest_json": str(Path(str(manifest_path))),
        },
    }


def _build_model_card_trust_summary(
    report: Any,
    *,
    model_card_path: str | Path,
    manifest_path: str | Path | None,
    check_files: bool,
    check_hashes: bool,
) -> dict[str, Any]:
    normalized = getattr(report, "normalized", {})
    if not isinstance(normalized, dict):
        normalized = {}
    weights = normalized.get("weights", {})
    if not isinstance(weights, dict):
        weights = {}
    deployment = normalized.get("deployment", {})
    if not isinstance(deployment, dict):
        deployment = {}
    assets = getattr(report, "assets", {})
    if not isinstance(assets, dict):
        assets = {}
    manifest_asset = assets.get("manifest", {})
    if not isinstance(manifest_asset, dict):
        manifest_asset = {}

    trust_signals = {
        "file_refs_checked": bool(check_files),
        "hashes_checked": bool(check_hashes),
        "has_weights_sha256": _nonempty_str(weights.get("sha256", None)) is not None,
        "has_weights_source": _nonempty_str(weights.get("source", None)) is not None,
        "has_weights_license": _nonempty_str(weights.get("license", None)) is not None,
        "has_deployment_runtime": _nonempty_str(deployment.get("runtime", None)) is not None,
        "has_manifest_link": (
            _nonempty_str(weights.get("manifest_entry", None)) is not None
            or _nonempty_str(manifest_asset.get("matched_entry", None)) is not None
        ),
        "has_cross_checked_manifest": bool(
            _nonempty_str(manifest_asset.get("matched_entry", None)) is not None
            and manifest_asset.get("ok", None) is True
        ),
    }
    degraded_by: list[str] = []
    if not trust_signals["has_weights_sha256"]:
        degraded_by.append("missing_weights_sha256")
    if not trust_signals["has_weights_source"]:
        degraded_by.append("missing_weights_source")
    if not trust_signals["has_weights_license"]:
        degraded_by.append("missing_weights_license")
    if not trust_signals["has_deployment_runtime"]:
        degraded_by.append("missing_deployment_runtime")
    if not trust_signals["has_manifest_link"]:
        degraded_by.append("missing_manifest_link")
    if not trust_signals["has_cross_checked_manifest"]:
        degraded_by.append("missing_cross_checked_manifest")

    if not bool(getattr(report, "ok", False)):
        status = "broken"
    elif not degraded_by:
        status = "trust-signaled"
    else:
        status = "partial"

    audit_refs = {
        "model_card_json": str(Path(str(model_card_path))),
    }
    if manifest_path is not None:
        audit_refs["weights_manifest_json"] = str(Path(str(manifest_path)))

    return {
        "status": status,
        "trust_signals": trust_signals,
        "degraded_by": degraded_by,
        "audit_refs": audit_refs,
    }


def _emit_trust_summary(summary: Mapping[str, Any]) -> None:
    print(f"trust_status={summary.get('status')}")
    trust_signals = summary.get("trust_signals", {})
    if isinstance(trust_signals, Mapping):
        for key, value in trust_signals.items():
            print(f"trust_signal.{key}={value}")
    for item in summary.get("degraded_by", []):
        print(f"degraded_by={item}")
    audit_refs = summary.get("audit_refs", {})
    if isinstance(audit_refs, Mapping):
        for key, value in audit_refs.items():
            print(f"audit_ref.{key}={value}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if str(args.cmd) == "hash":
            from pyimgano.utils.security import FileHasher

            p = Path(str(args.path))
            if not p.exists():
                raise FileNotFoundError(f"File not found: {p}")
            digest = FileHasher.compute_hash(str(p), algorithm=str(args.algorithm))
            print(digest)
            return 0

        if str(args.cmd) == "validate":
            from pyimgano.weights.manifest import validate_weights_manifest_file

            report = validate_weights_manifest_file(
                manifest_path=str(args.manifest),
                base_dir=(str(args.base_dir) if args.base_dir is not None else None),
                check_files=bool(args.check_files),
                check_hashes=bool(args.check_hashes),
            )
            trust_summary = _build_manifest_trust_summary(
                report,
                manifest_path=str(args.manifest),
                check_files=bool(args.check_files),
                check_hashes=bool(args.check_hashes),
            )
            if bool(args.json):
                payload = report.to_jsonable()
                payload["trust_summary"] = trust_summary
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _emit_trust_summary(trust_summary)
                for w in report.warnings:
                    print(f"warning: {w}")
                for e in report.errors:
                    print(f"error: {e}")
            return 0 if report.ok else 1

        if str(args.cmd) == "validate-model-card":
            from pyimgano.weights.model_card import validate_model_card_file

            report = validate_model_card_file(
                str(args.path),
                base_dir=(str(args.base_dir) if args.base_dir is not None else None),
                manifest_path=(str(args.manifest) if args.manifest is not None else None),
                manifest_base_dir=(
                    str(args.manifest_base_dir) if args.manifest_base_dir is not None else None
                ),
                check_files=bool(args.check_files),
                check_hashes=bool(args.check_hashes),
            )
            trust_summary = _build_model_card_trust_summary(
                report,
                model_card_path=str(args.path),
                manifest_path=(str(args.manifest) if args.manifest is not None else None),
                check_files=bool(args.check_files),
                check_hashes=bool(args.check_hashes),
            )
            if bool(args.json):
                payload = report.to_jsonable()
                payload["trust_summary"] = trust_summary
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _emit_trust_summary(trust_summary)
                for w in report.warnings:
                    print(f"warning: {w}")
                for e in report.errors:
                    print(f"error: {e}")
            return 0 if report.ok else 1

        if str(args.cmd) == "audit-bundle":
            audit = evaluate_bundle_weights_audit(
                str(args.bundle_dir),
                check_hashes=bool(args.check_hashes),
            )
            if bool(args.json):
                print(json.dumps(audit, indent=2, sort_keys=True))
            else:
                print(
                    f"status={audit.get('status')} ready={str(bool(audit.get('ready'))).lower()} "
                    f"bundle_dir={audit.get('bundle_dir')}"
                )
                _emit_trust_summary(dict(audit.get("trust_summary", {})))
                for item in audit.get("missing_required", []):
                    print(f"missing_required={item}")
                for item in audit.get("warnings", []):
                    print(f"warning: {item}")
                for item in audit.get("errors", []):
                    print(f"error: {item}")
            return 0 if bool(audit.get("ready")) else 1

        if str(args.cmd) == "template":
            if str(args.template_kind) == "manifest":
                print(json.dumps(_weights_manifest_template(), indent=2, sort_keys=True))
                return 0
            if str(args.template_kind) == "model-card":
                from pyimgano.weights.model_card import default_model_card_template

                print(json.dumps(default_model_card_template(), indent=2, sort_keys=True))
                return 0

        raise RuntimeError(f"Unhandled cmd: {args.cmd!r}")
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        import sys

        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
