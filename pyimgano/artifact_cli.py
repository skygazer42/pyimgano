from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pyimgano.artifacts.importers import import_onnx


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyimgano-artifact",
        description="Import, inspect, validate, and safely rebind executable artifacts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import", help="Import a third-party runtime model.")
    p_import.add_argument("--format", choices=("onnx",), required=True)
    p_import.add_argument("--model", required=True)
    p_import.add_argument("--contract", required=True)
    p_import.add_argument("--policy", default=None)
    p_import.add_argument("--out", required=True)
    p_import.add_argument("--overwrite", action="store_true")
    p_import.add_argument("--json", action="store_true")

    p_bind = sub.add_parser("bind-policy", help="Clone an artifact with a validated policy.")
    p_bind.add_argument("--artifact", required=True)
    p_bind.add_argument("--policy", required=True)
    p_bind.add_argument("--out", required=True)
    p_bind.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help=(
            "Permit the mandatory probe to load integrity-verified executable "
            "serialization such as TorchScript or a legacy checkpoint."
        ),
    )
    p_bind.add_argument("--json", action="store_true")

    for name in ("inspect", "validate"):
        command = sub.add_parser(name, help=f"{name.title()} an artifact manifest.")
        command.add_argument("artifact")
        command.add_argument("--json", action="store_true")
    return parser


def _emit(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    for key in ("artifact_root", "artifact_id", "runtime_id", "policy_id", "verification_level"):
        if payload.get(key) is not None:
            print(f"{key}={payload[key]}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "import":
            payload = import_onnx(
                str(args.model),
                contract=str(args.contract),
                policy=(str(args.policy) if args.policy is not None else None),
                out=str(args.out),
                overwrite=bool(args.overwrite),
            )
            _emit(dict(payload), json_output=bool(args.json))
            return 0

        if args.command == "bind-policy":
            from pyimgano.artifacts import bind_policy

            result = bind_policy(
                str(args.artifact),
                str(args.policy),
                out=str(args.out),
                trust_checkpoint=bool(args.trust_checkpoint),
            )
            from pyimgano.artifacts import load_artifact_manifest

            manifest = load_artifact_manifest(result)
            payload = {
                "artifact_root": str(result),
                "artifact_id": manifest.get("artifact_id"),
                "runtime_id": manifest.get("runtime_id"),
                "policy_id": manifest.get("policy_id"),
                "verification_level": dict(manifest.get("verification", {})).get("level"),
            }
            _emit(payload, json_output=bool(args.json))
            return 0

        from pyimgano.artifacts import load_artifact_manifest

        artifact_path = Path(str(args.artifact))
        manifest = load_artifact_manifest(artifact_path)
        artifact_root = artifact_path if artifact_path.is_dir() else artifact_path.parent
        if args.command == "validate":
            from pyimgano.artifacts import verify_artifact_files

            verify_artifact_files(artifact_root, manifest)
            payload = {
                "status": "ok",
                "artifact_root": str(artifact_root.resolve()),
                "artifact_id": manifest.get("artifact_id"),
            }
        else:
            payload = dict(manifest)
        _emit(payload, json_output=bool(args.json))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(str(exc), file=sys.stderr)
        return 2


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
