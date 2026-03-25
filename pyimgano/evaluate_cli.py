from __future__ import annotations

import argparse

import pyimgano.cli_output as cli_output
import pyimgano.services.evaluation_harness_service as harness_service


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyimgano-evaluate")
    parser.add_argument("--config", required=True, help="Path to evaluation harness JSON config.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        request = harness_service.load_evaluation_harness_request(str(args.config))
        payload = harness_service.run_evaluation_harness(request)
        if bool(args.json):
            return cli_output.emit_json(payload)

        print(f"output_dir={payload['output_dir']}")
        print(f"entries_total={payload['entries_total']}")
        print(f"entries_succeeded={payload['entries_succeeded']}")
        print(f"entries_failed={payload['entries_failed']}")
        print(f"entries_blocked={payload['entries_blocked']}")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        cli_output.print_cli_error(exc)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
