from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyimgano-datasets")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List available dataset→manifest converters")
    p_list.add_argument("--json", action="store_true", help="Emit JSON payload to stdout")

    p_info = sub.add_parser("info", help="Show details for one converter")
    p_info.add_argument("name", help="Converter name (e.g. custom, mvtec_ad2)")
    p_info.add_argument("--json", action="store_true", help="Emit JSON payload to stdout")

    return parser


def _converter_to_jsonable(conv) -> dict[str, Any]:  # noqa: ANN001 - CLI boundary
    return {
        "name": str(conv.name),
        "description": str(conv.description),
        "requires_category": bool(conv.requires_category),
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        from pyimgano.datasets.converters import get_dataset_converter, list_dataset_converters

        if args.cmd == "list":
            payload = [_converter_to_jsonable(c) for c in list_dataset_converters()]
            if bool(args.json):
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                for c in list_dataset_converters():
                    req = " (category required)" if bool(c.requires_category) else ""
                    print(f"- {c.name}{req}: {c.description}")
            return 0

        if args.cmd == "info":
            c = get_dataset_converter(str(args.name))
            payload = _converter_to_jsonable(c)
            if bool(args.json):
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                req = "yes" if bool(c.requires_category) else "no"
                print(f"name: {c.name}")
                print(f"requires_category: {req}")
                print(f"description: {c.description}")
            return 0

        raise ValueError(f"Unknown command: {args.cmd!r}")
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

