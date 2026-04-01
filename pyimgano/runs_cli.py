from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import pyimgano.cli_output as cli_output
import pyimgano.runs_cli_rendering as runs_cli_rendering
from pyimgano.reporting.publication_quality import evaluate_publication_quality
from pyimgano.reporting.run_acceptance import evaluate_acceptance
from pyimgano.reporting.run_index import (
    compare_run_summaries,
    latest_run_summary,
    list_run_summaries,
)
from pyimgano.reporting.run_index_helpers import (
    comparability_gate_status as _comparability_gate_status_helper,
)
from pyimgano.reporting.run_index_helpers import (
    compare_blocking_flags as _compare_blocking_flags_helper,
)
from pyimgano.reporting.run_index_helpers import (
    comparison_trust_gate as _comparison_trust_gate_helper,
)
from pyimgano.reporting.run_index_helpers import (
    comparison_trust_reason as _comparison_trust_reason_helper,
)
from pyimgano.reporting.run_index_helpers import (
    format_metric_value as _format_metric_value_helper,
)
from pyimgano.reporting.run_quality import evaluate_run_quality

_QUALITY_STATUS_RANK = {
    "broken": 0,
    "partial": 1,
    "reproducible": 2,
    "audited": 3,
    "deployable": 4,
}

_CANDIDATE_COMPARABILITY_GATES_ORDER = (
    "split",
    "environment",
    "target",
    "target_dataset",
    "target_category",
    "robustness_protocol",
    "operator_contract",
    "bundle_operator_contract",
)
_JSON_OUTPUT_HELP = "Emit JSON output."
_MISSING_REQUIRED_PREFIX = "missing_required="


def _format_metric_value(value: object) -> str | None:
    return _format_metric_value_helper(value)


def _comparability_gate_status(summary: dict[str, object]) -> str:
    return _comparability_gate_status_helper(summary)


def _compare_blocking_flags(
    *,
    total_regressions: int,
    split_summary: dict[str, object],
    environment_summary: dict[str, object],
    target_summary: dict[str, object],
    robustness_protocol_summary: dict[str, object],
    operator_contract_summary: dict[str, object],
    bundle_operator_contract_summary: dict[str, object],
) -> list[str]:
    return _compare_blocking_flags_helper(
        total_regressions=total_regressions,
        split_summary=split_summary,
        environment_summary=environment_summary,
        target_summary=target_summary,
        robustness_protocol_summary=robustness_protocol_summary,
        operator_contract_summary=operator_contract_summary,
        bundle_operator_contract_summary=bundle_operator_contract_summary,
    )


def _comparison_trust_gate(trust_status: object) -> str:
    gate = _comparison_trust_gate_helper(trust_status)
    return str(gate) if gate is not None else "limited"


def _comparison_trust_reason(
    *,
    trust_status: object,
    quality_status: object,
    status_reasons: list[object],
    degraded_by: list[object],
) -> str | None:
    return _comparison_trust_reason_helper(
        trust_status=trust_status,
        quality_status=quality_status,
        status_reasons=status_reasons,
        degraded_by=degraded_by,
    )


def _resolve_operator_contract_status(
    *,
    run: dict[str, object],
    trust_summary: dict[str, object],
) -> str:
    run_level = run.get("operator_contract_status", None)
    if isinstance(run_level, str) and run_level:
        return str(run_level)

    trust_signals = trust_summary.get("trust_signals", {})
    signal_map = dict(trust_signals) if isinstance(trust_signals, dict) else {}
    has_contract = bool(signal_map.get("has_operator_contract"))
    if not has_contract:
        return "missing"
    return (
        "consistent" if bool(signal_map.get("has_operator_contract_consistent")) else "mismatched"
    )


def _resolve_bundle_operator_contract_status(
    *,
    run: dict[str, object],
    trust_summary: dict[str, object],
) -> str:
    run_level = run.get("bundle_operator_contract_status", None)
    if isinstance(run_level, str) and run_level:
        return str(run_level)

    trust_signals = trust_summary.get("trust_signals", {})
    signal_map = dict(trust_signals) if isinstance(trust_signals, dict) else {}
    has_contract = bool(signal_map.get("has_bundle_operator_contract"))
    if not has_contract:
        return "missing"
    return (
        "consistent"
        if bool(signal_map.get("has_bundle_operator_contract_consistent"))
        else "mismatched"
    )


def _format_run_brief(run: dict[str, object]) -> str:
    return runs_cli_rendering.format_run_brief_line(run)


def _format_compare_run_brief(
    run: dict[str, object],
    *,
    primary_metric_name: str | None = None,
    primary_metric_row: dict[str, object] | None = None,
) -> str:
    return runs_cli_rendering.format_compare_run_brief_line(
        run,
        primary_metric_name=primary_metric_name,
        primary_metric_row=primary_metric_row,
    )


def _format_candidate_comparability_gates(gates: dict[str, object]) -> str:
    parts: list[str] = []
    for key in _CANDIDATE_COMPARABILITY_GATES_ORDER:
        value = gates.get(key, None)
        if isinstance(value, str) and value:
            parts.append(f"{key}:{value}")
    return ",".join(parts) if parts else "none"


def _format_candidate_incompatibility_digest(entry: dict[str, object]) -> str:
    verdict = entry.get("verdict", None)
    verdict_text = str(verdict) if isinstance(verdict, str) and verdict else "pass"
    incompatible = entry.get("incompatible_gates", [])
    blocking = entry.get("blocking_reasons", [])
    incompatible_items = (
        [str(item) for item in incompatible if str(item)] if isinstance(incompatible, list) else []
    )
    blocking_items = (
        [str(item) for item in blocking if str(item)] if isinstance(blocking, list) else []
    )
    incompatible_text = ",".join(incompatible_items) if incompatible_items else "none"
    blocking_text = ",".join(blocking_items) if blocking_items else "none"
    return (
        f"verdict:{verdict_text}|"
        f"incompatible_gates:{incompatible_text}|"
        f"blocking_reasons:{blocking_text}"
    )


def _contract_incompat_summary_line(
    *,
    label: str,
    summary_payload: Mapping[str, object],
) -> str:
    return (
        f"{label}: "
        f"checked={summary_payload.get('checked')} "
        f"matched={summary_payload.get('matched_runs', 0)} "
        f"mismatched={summary_payload.get('mismatched_runs', 0)} "
        f"missing={summary_payload.get('missing_runs', 0)}"
    )


def _contract_incompat_detail_line(
    *,
    label: str,
    run_dir_name: str,
    status: str,
    mismatch_reason: object,
) -> str:
    if isinstance(mismatch_reason, str) and mismatch_reason:
        return f"{label}_incompat.{run_dir_name}={status}:{mismatch_reason}"
    return f"{label}_incompat.{run_dir_name}={status}"


def _iter_contract_incompat_rows(
    comparison_payload: Mapping[str, object],
) -> list[Mapping[str, object]]:
    rows = comparison_payload.get("comparisons", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _emit_contract_incompat_details(
    *,
    label: str,
    comparison_payload: dict[str, object],
    summary_payload: dict[str, object],
) -> None:
    print(_contract_incompat_summary_line(label=label, summary_payload=summary_payload))
    baseline_sha256 = comparison_payload.get("baseline_contract_sha256", None)
    if isinstance(baseline_sha256, str) and baseline_sha256:
        print(f"{label}_baseline.sha256={baseline_sha256}")
    for row in _iter_contract_incompat_rows(comparison_payload):
        status = str(row.get("status"))
        if status not in {"mismatched", "missing"}:
            continue
        run_dir_name = row.get("run_dir_name", None)
        if not isinstance(run_dir_name, str) or not run_dir_name:
            continue
        mismatch_reason = row.get("mismatch_reason", None)
        print(
            _contract_incompat_detail_line(
                label=label,
                run_dir_name=run_dir_name,
                status=status,
                mismatch_reason=mismatch_reason,
            )
        )
        candidate_sha256 = row.get("contract_sha256", None)
        if isinstance(candidate_sha256, str) and candidate_sha256:
            print(f"{label}_sha256.{run_dir_name}={candidate_sha256}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyimgano-runs",
        description="Inspect and compare saved pyimgano benchmark/workbench runs.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List saved top-level runs under a root directory.")
    p_list.add_argument("--root", default="runs", help="Root directory to scan. Default: runs")
    p_list.add_argument("--limit", type=int, default=None, help="Optional max number of runs.")
    p_list.add_argument(
        "--kind",
        choices=["benchmark", "suite", "workbench", "robustness"],
        default=None,
        help="Optional run kind filter.",
    )
    p_list.add_argument("--dataset", default=None, help="Optional dataset filter.")
    p_list.add_argument(
        "--min-quality",
        choices=["reproducible", "audited", "deployable"],
        default=None,
        help="Optional minimum artifact quality filter.",
    )
    p_list.add_argument(
        "--query",
        default=None,
        help="Optional substring filter over run name, model/suite name, and category.",
    )
    p_list.add_argument(
        "--same-split-as",
        default=None,
        help="Optional run directory whose split fingerprint must match.",
    )
    p_list.add_argument(
        "--same-environment-as",
        default=None,
        help="Optional run directory whose environment fingerprint must match.",
    )
    p_list.add_argument(
        "--same-target-as",
        default=None,
        help="Optional run directory whose dataset/category target must match.",
    )
    p_list.add_argument(
        "--same-robustness-protocol-as",
        default=None,
        help="Optional run directory whose robustness protocol must match.",
    )
    p_list.add_argument("--json", action="store_true", help=_JSON_OUTPUT_HELP)

    p_latest = sub.add_parser("latest", help="Show the latest saved top-level run under a root.")
    p_latest.add_argument("--root", default="runs", help="Root directory to scan. Default: runs")
    p_latest.add_argument(
        "--kind",
        choices=["benchmark", "suite", "workbench", "robustness"],
        default=None,
        help="Optional run kind filter.",
    )
    p_latest.add_argument("--dataset", default=None, help="Optional dataset filter.")
    p_latest.add_argument(
        "--min-quality",
        choices=["reproducible", "audited", "deployable"],
        default=None,
        help="Optional minimum artifact quality filter.",
    )
    p_latest.add_argument(
        "--query",
        default=None,
        help="Optional substring filter over run name, model/suite name, and category.",
    )
    p_latest.add_argument(
        "--same-split-as",
        default=None,
        help="Optional run directory whose split fingerprint must match.",
    )
    p_latest.add_argument(
        "--same-environment-as",
        default=None,
        help="Optional run directory whose environment fingerprint must match.",
    )
    p_latest.add_argument(
        "--same-target-as",
        default=None,
        help="Optional run directory whose dataset/category target must match.",
    )
    p_latest.add_argument(
        "--same-robustness-protocol-as",
        default=None,
        help="Optional run directory whose robustness protocol must match.",
    )
    p_latest.add_argument("--json", action="store_true", help=_JSON_OUTPUT_HELP)

    p_compare = sub.add_parser("compare", help="Compare one or more saved runs.")
    p_compare.add_argument("run_dirs", nargs="+", help="Run directories to compare.")
    p_compare.add_argument(
        "--baseline",
        default=None,
        help="Optional baseline run directory used to compute deltas/regressions.",
    )
    p_compare.add_argument(
        "--metric",
        default=None,
        help="Optional single metric to focus on (e.g. auroc, aupro).",
    )
    p_compare.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Return exit code 1 when regressions exceed --max-regressions.",
    )
    p_compare.add_argument(
        "--max-regressions",
        type=int,
        default=0,
        help="Allowed regression count when --fail-on-regression is set. Default: 0",
    )
    p_compare.add_argument(
        "--require-same-split",
        action="store_true",
        help="Return exit code 1 unless every non-baseline run matches the baseline split fingerprint.",
    )
    p_compare.add_argument(
        "--require-same-target",
        action="store_true",
        help="Return exit code 1 unless every non-baseline run matches the baseline dataset/category.",
    )
    p_compare.add_argument(
        "--require-same-environment",
        action="store_true",
        help="Return exit code 1 unless every non-baseline run matches the baseline environment fingerprint.",
    )
    p_compare.add_argument(
        "--require-same-robustness-protocol",
        action="store_true",
        help=(
            "Return exit code 1 unless every non-baseline run matches the baseline robustness "
            "protocol (corruption mode, conditions, severities, input mode, resize)."
        ),
    )
    p_compare.add_argument(
        "--require-same-operator-contract",
        action="store_true",
        help="Return exit code 1 unless every non-baseline run matches the baseline operator contract.",
    )
    p_compare.add_argument(
        "--require-same-bundle-operator-contract",
        action="store_true",
        help=(
            "Return exit code 1 unless every non-baseline run matches the baseline deploy bundle "
            "operator contract."
        ),
    )
    p_compare.add_argument("--json", action="store_true", help=_JSON_OUTPUT_HELP)

    p_quality = sub.add_parser("quality", help="Inspect artifact completeness for a saved run.")
    p_quality.add_argument("run_dir", help="Run directory to inspect.")
    p_quality.add_argument(
        "--check-bundle-hashes",
        action="store_true",
        help="Also verify deploy bundle manifest hashes when present.",
    )
    p_quality.add_argument(
        "--require-status",
        choices=["reproducible", "audited", "deployable"],
        default=None,
        help="Return exit code 1 unless the run quality reaches at least this status.",
    )
    p_quality.add_argument("--json", action="store_true", help=_JSON_OUTPUT_HELP)

    p_acceptance = sub.add_parser(
        "acceptance",
        help="Run the aggregated handoff/release gate for a saved run or suite export.",
    )
    p_acceptance.add_argument(
        "path",
        help="Run directory, suite export directory, or leaderboard_metadata.json path.",
    )
    p_acceptance.add_argument(
        "--check-bundle-hashes",
        action="store_true",
        help="Also verify deploy bundle manifest/weights hashes when present for run directories.",
    )
    p_acceptance.add_argument(
        "--require-status",
        choices=["reproducible", "audited", "deployable"],
        default="audited",
        help="Minimum run quality required before the run acceptance gate can pass. Ignored for suite exports. Default: audited",
    )
    p_acceptance.add_argument("--json", action="store_true", help=_JSON_OUTPUT_HELP)

    p_publication = sub.add_parser(
        "publication",
        help="Inspect suite publication readiness for an export directory.",
    )
    p_publication.add_argument(
        "path", help="Suite export directory or leaderboard_metadata.json path."
    )
    p_publication.add_argument("--json", action="store_true", help=_JSON_OUTPUT_HELP)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        if str(args.cmd) == "list":
            payload = {
                "runs": list_run_summaries(
                    args.root,
                    limit=args.limit,
                    kind=args.kind,
                    dataset=args.dataset,
                    min_quality=args.min_quality,
                    query=args.query,
                    same_split_as=args.same_split_as,
                    same_environment_as=args.same_environment_as,
                    same_target_as=args.same_target_as,
                    same_robustness_protocol_as=args.same_robustness_protocol_as,
                )
            }
            if bool(args.json):
                return cli_output.emit_jsonable(payload, indent=None)

            for item in payload["runs"]:
                print(_format_run_brief(item))
            return 0

        if str(args.cmd) == "latest":
            run = latest_run_summary(
                args.root,
                kind=args.kind,
                dataset=args.dataset,
                min_quality=args.min_quality,
                query=args.query,
                same_split_as=args.same_split_as,
                same_environment_as=args.same_environment_as,
                same_target_as=args.same_target_as,
                same_robustness_protocol_as=args.same_robustness_protocol_as,
            )
            payload = {"run": run}
            if bool(args.json):
                return cli_output.emit_jsonable(payload, indent=None)
            if run is None:
                print("No runs found.")
                return 0
            print(_format_run_brief(run))
            return 0

        if str(args.cmd) == "compare":
            if bool(args.require_same_split) and args.baseline is None:
                raise ValueError("--require-same-split requires --baseline.")
            if bool(args.require_same_target) and args.baseline is None:
                raise ValueError("--require-same-target requires --baseline.")
            if bool(args.require_same_environment) and args.baseline is None:
                raise ValueError("--require-same-environment requires --baseline.")
            if bool(args.require_same_robustness_protocol) and args.baseline is None:
                raise ValueError("--require-same-robustness-protocol requires --baseline.")
            payload = compare_run_summaries(
                args.run_dirs,
                baseline_run_dir=args.baseline,
                metric=args.metric,
            )
            if bool(args.json):
                rc = cli_output.emit_jsonable(payload, indent=None)
                split_summary = dict(payload.get("split_comparison", {}).get("summary", {}))
                environment_summary = dict(
                    payload.get("environment_comparison", {}).get("summary", {})
                )
                target_summary = dict(payload.get("target_comparison", {}).get("summary", {}))
                robustness_protocol_summary = dict(
                    payload.get("robustness_protocol_comparison", {}).get("summary", {})
                )
                operator_contract_summary = dict(
                    payload.get("operator_contract_comparison", {}).get("summary", {})
                )
                bundle_operator_contract_summary = dict(
                    payload.get("bundle_operator_contract_comparison", {}).get("summary", {})
                )
                if bool(args.require_same_split) and (
                    not bool(split_summary.get("checked"))
                    or int(split_summary.get("incompatible_runs", 0)) > 0
                ):
                    return 1
                if bool(args.require_same_environment) and (
                    not bool(environment_summary.get("checked"))
                    or int(environment_summary.get("incompatible_runs", 0)) > 0
                ):
                    return 1
                if bool(args.require_same_target) and (
                    not bool(target_summary.get("checked"))
                    or int(target_summary.get("incompatible_runs", 0)) > 0
                ):
                    return 1
                if bool(args.require_same_robustness_protocol) and (
                    not bool(robustness_protocol_summary.get("checked"))
                    or int(robustness_protocol_summary.get("incompatible_runs", 0)) > 0
                ):
                    return 1
                if bool(args.require_same_operator_contract) and (
                    not bool(operator_contract_summary.get("checked"))
                    or int(operator_contract_summary.get("incompatible_runs", 0)) > 0
                ):
                    return 1
                if bool(args.require_same_bundle_operator_contract) and (
                    not bool(bundle_operator_contract_summary.get("checked"))
                    or int(bundle_operator_contract_summary.get("incompatible_runs", 0)) > 0
                ):
                    return 1
                if bool(args.fail_on_regression) and int(
                    payload["summary"]["total_regressions"]
                ) > int(args.max_regressions):
                    return 1
                return rc

            evaluation_contract = dict(payload.get("evaluation_contract", {}))
            primary_metric_name = (
                str(evaluation_contract.get("primary_metric"))
                if evaluation_contract.get("primary_metric") is not None
                else None
            )
            primary_metric_rows_by_name: dict[str, dict[str, object]] = {}
            if isinstance(primary_metric_name, str) and primary_metric_name:
                primary_metric_info = dict(payload.get("metrics", {}).get(primary_metric_name, {}))
                for row in primary_metric_info.get("comparisons", []):
                    if not isinstance(row, dict):
                        continue
                    run_dir_name = row.get("run_dir_name", None)
                    if isinstance(run_dir_name, str) and run_dir_name:
                        primary_metric_rows_by_name[run_dir_name] = row
            for run in payload["runs"]:
                run_name = run.get("run_dir_name", None)
                metric_row = (
                    primary_metric_rows_by_name.get(run_name) if isinstance(run_name, str) else None
                )
                print(
                    _format_compare_run_brief(
                        run,
                        primary_metric_name=primary_metric_name,
                        primary_metric_row=metric_row,
                    )
                )
            for name, info in sorted(payload["metrics"].items()):
                print(f"{name}: min={info['min']:.6g} max={info['max']:.6g}")
                if args.baseline is not None:
                    print(
                        f"  baseline={info.get('baseline')} "
                        f"direction={info.get('direction')} "
                        f"regressions={info.get('regression_count', 0)}"
                    )
            if args.baseline is not None:
                evaluation_contract = dict(payload.get("evaluation_contract", {}))
                primary_metric = evaluation_contract.get("primary_metric", None)
                metrics_payload = dict(payload.get("metrics", {}))
                if isinstance(primary_metric, str) and primary_metric in metrics_payload:
                    primary_metric_info = dict(metrics_payload.get(primary_metric, {}))
                    print(f"comparison_primary_metric={primary_metric}")
                    primary_metric_direction = primary_metric_info.get("direction", None)
                    if isinstance(primary_metric_direction, str) and primary_metric_direction:
                        print("comparison_primary_metric_direction=" f"{primary_metric_direction}")
                    primary_metric_baseline = _format_metric_value(
                        primary_metric_info.get("baseline", None)
                    )
                    if primary_metric_baseline is not None:
                        print("comparison_primary_metric_baseline=" f"{primary_metric_baseline}")
                    print(
                        "comparison_primary_metric_total_regressions="
                        f"{payload.get('summary', {}).get('total_regressions', 0)}"
                    )
                    for row in primary_metric_info.get("comparisons", []):
                        if not isinstance(row, dict):
                            continue
                        if str(row.get("status")) == "baseline":
                            continue
                        run_dir_name = row.get("run_dir_name", None)
                        if not isinstance(run_dir_name, str) or not run_dir_name:
                            continue
                        print("primary_metric_status." f"{run_dir_name}={row.get('status')}")
                        delta = _format_metric_value(row.get("delta_vs_baseline", None))
                        if delta is not None:
                            print(f"primary_metric_delta.{run_dir_name}={delta}")

                baseline_run = dict(payload.get("baseline_run", {}))
                baseline_quality = dict(baseline_run.get("artifact_quality", {}))
                baseline_trust = dict(baseline_quality.get("trust_summary", {}))
                baseline_quality_status = baseline_quality.get("status", None)
                if isinstance(baseline_quality_status, str) and baseline_quality_status:
                    print(f"baseline_quality={baseline_quality_status}")
                baseline_trust_status = baseline_trust.get("status", None)
                baseline_reasons = list(baseline_trust.get("status_reasons", []))
                baseline_degraded_by = list(baseline_trust.get("degraded_by", []))
                baseline_audit_refs = dict(baseline_trust.get("audit_refs", {}))
                if isinstance(baseline_trust_status, str) and baseline_trust_status:
                    print(f"baseline_trust={baseline_trust_status}")
                    print(
                        "comparison_trust_gate=" f"{_comparison_trust_gate(baseline_trust_status)}"
                    )
                    print(f"comparison_trust_status={baseline_trust_status}")
                    trust_reason = _comparison_trust_reason(
                        trust_status=baseline_trust_status,
                        quality_status=baseline_quality_status,
                        status_reasons=baseline_reasons,
                        degraded_by=baseline_degraded_by,
                    )
                    if isinstance(trust_reason, str) and trust_reason:
                        print(f"comparison_trust_reason={trust_reason}")
                summary_payload = dict(payload.get("summary", {}))
                operator_contract_status = summary_payload.get("operator_contract_status", None)
                if isinstance(operator_contract_status, str) and operator_contract_status:
                    print(f"comparison_operator_contract_status={operator_contract_status}")
                print(
                    "comparison_operator_contract_consistent="
                    f"{str(bool(summary_payload.get('operator_contract_consistent', False))).lower()}"
                )
                bundle_operator_contract_status = summary_payload.get(
                    "bundle_operator_contract_status",
                    None,
                )
                if (
                    isinstance(bundle_operator_contract_status, str)
                    and bundle_operator_contract_status
                ):
                    print(
                        "comparison_bundle_operator_contract_status="
                        f"{bundle_operator_contract_status}"
                    )
                print(
                    "comparison_bundle_operator_contract_consistent="
                    f"{str(bool(summary_payload.get('bundle_operator_contract_consistent', False))).lower()}"
                )
                print(
                    "comparison_bundle_operator_contract_digests_valid="
                    f"{str(bool(summary_payload.get('bundle_operator_contract_digests_valid', False))).lower()}"
                )
                if baseline_degraded_by:
                    print(
                        "comparison_trust_degraded_by="
                        + ",".join(str(item) for item in baseline_degraded_by)
                    )
                if baseline_reasons:
                    print(f"baseline_reason={baseline_reasons[0]}")
                for key, value in baseline_audit_refs.items():
                    print(f"comparison_trust_ref.{key}={value}")

                split_summary = dict(payload.get("split_comparison", {}).get("summary", {}))
                environment_summary = dict(
                    payload.get("environment_comparison", {}).get("summary", {})
                )
                target_summary = dict(payload.get("target_comparison", {}).get("summary", {}))
                robustness_protocol_summary = dict(
                    payload.get("robustness_protocol_comparison", {}).get("summary", {})
                )
                operator_contract_summary = dict(
                    payload.get("operator_contract_comparison", {}).get("summary", {})
                )
                bundle_operator_contract_summary = dict(
                    payload.get("bundle_operator_contract_comparison", {}).get("summary", {})
                )
                print(
                    "comparability_gates: "
                    f"split={_comparability_gate_status(split_summary)} "
                    f"environment={_comparability_gate_status(environment_summary)} "
                    f"target={_comparability_gate_status(target_summary)} "
                    "robustness_protocol="
                    f"{_comparability_gate_status(robustness_protocol_summary)} "
                    "operator_contract="
                    f"{_comparability_gate_status(operator_contract_summary)} "
                    "bundle_operator_contract="
                    f"{_comparability_gate_status(bundle_operator_contract_summary)}"
                )
                total_regressions = int(payload.get("summary", {}).get("total_regressions", 0) or 0)
                regression_gate = "clean" if total_regressions == 0 else "regressed"
                print(f"comparison_regression_gate={regression_gate}")
                blocking_flags = _compare_blocking_flags(
                    total_regressions=total_regressions,
                    split_summary=split_summary,
                    environment_summary=environment_summary,
                    target_summary=target_summary,
                    robustness_protocol_summary=robustness_protocol_summary,
                    operator_contract_summary=operator_contract_summary,
                    bundle_operator_contract_summary=bundle_operator_contract_summary,
                )
                print(
                    "comparison_blocking_flags="
                    + (",".join(blocking_flags) if blocking_flags else "none")
                )
                print(f"comparison_verdict={'pass' if not blocking_flags else 'blocked'}")
                summary = dict(payload.get("summary", {}))
                candidate_names = sorted(
                    {
                        str(name)
                        for name in (
                            list(dict(summary.get("candidate_verdicts", {})).keys())
                            + list(dict(summary.get("candidate_blocking_reasons", {})).keys())
                            + list(dict(summary.get("candidate_comparability_gates", {})).keys())
                        )
                        if str(name)
                    }
                )
                candidate_verdicts = dict(summary.get("candidate_verdicts", {}))
                candidate_blocking_reasons = dict(summary.get("candidate_blocking_reasons", {}))
                candidate_comparability_gates = dict(
                    summary.get("candidate_comparability_gates", {})
                )
                candidate_bundle_digest_statuses = dict(
                    summary.get("candidate_bundle_operator_contract_digest_statuses", {})
                )
                candidate_incompatibility_digest = dict(
                    summary.get("candidate_incompatibility_digest", {})
                )
                for run_name in candidate_names:
                    verdict = candidate_verdicts.get(run_name, None)
                    if isinstance(verdict, str) and verdict:
                        print(f"candidate_verdict.{run_name}={verdict}")
                    reasons = candidate_blocking_reasons.get(run_name, [])
                    reason_tokens = (
                        {str(item) for item in reasons if str(item)}
                        if isinstance(reasons, list)
                        else set()
                    )
                    if isinstance(reasons, list) and reasons:
                        print(
                            "candidate_blocking_reasons."
                            f"{run_name}=" + ",".join(str(item) for item in reasons)
                        )
                    else:
                        print(f"candidate_blocking_reasons.{run_name}=none")
                    gates = candidate_comparability_gates.get(run_name, {})
                    if isinstance(gates, dict):
                        print(
                            "candidate_comparability_gates."
                            f"{run_name}={_format_candidate_comparability_gates(gates)}"
                        )
                    digest_entry = candidate_incompatibility_digest.get(run_name, None)
                    if not isinstance(digest_entry, dict):
                        gate_map = gates if isinstance(gates, dict) else {}
                        incompatible_gates: list[str] = []
                        for gate_name in _CANDIDATE_COMPARABILITY_GATES_ORDER:
                            gate_status = gate_map.get(gate_name, None)
                            gate_status_text = (
                                str(gate_status).strip().lower()
                                if isinstance(gate_status, str)
                                else ""
                            )
                            if gate_status_text in {"missing", "mismatched"}:
                                incompatible_gates.append(f"{gate_name}:{gate_status_text}")
                        digest_entry = {
                            "verdict": verdict if isinstance(verdict, str) and verdict else "pass",
                            "incompatible_gates": incompatible_gates,
                            "blocking_reasons": reasons if isinstance(reasons, list) else [],
                        }
                    print(
                        "candidate_incompatibility_digest."
                        f"{run_name}={_format_candidate_incompatibility_digest(digest_entry)}"
                    )
                    bundle_digest_status = candidate_bundle_digest_statuses.get(run_name, None)
                    if isinstance(bundle_digest_status, str) and bundle_digest_status:
                        print(
                            "candidate_bundle_operator_contract_digest_status."
                            f"{run_name}={bundle_digest_status}"
                        )
                    if "operator_contract_bundle:digest_mismatch" in reason_tokens:
                        print("bundle_operator_contract_integrity." f"{run_name}=digest_mismatch")
                print(
                    "split: "
                    f"checked={split_summary.get('checked')} "
                    f"matched={split_summary.get('matched_runs', 0)} "
                    f"mismatched={split_summary.get('mismatched_runs', 0)} "
                    f"missing={split_summary.get('missing_runs', 0)}"
                )
                split_baseline_sha256 = payload.get("split_comparison", {}).get(
                    "baseline_split_fingerprint_sha256", None
                )
                if isinstance(split_baseline_sha256, str) and split_baseline_sha256:
                    print(f"split_baseline.sha256={split_baseline_sha256}")
                for row in payload.get("split_comparison", {}).get("comparisons", []):
                    if not isinstance(row, dict):
                        continue
                    status = str(row.get("status"))
                    if status not in {"mismatched", "missing"}:
                        continue
                    run_dir_name = row.get("run_dir_name", None)
                    if isinstance(run_dir_name, str) and run_dir_name:
                        print(f"split_incompat.{run_dir_name}={status}")
                print(
                    "environment: "
                    f"checked={environment_summary.get('checked')} "
                    f"matched={environment_summary.get('matched_runs', 0)} "
                    f"mismatched={environment_summary.get('mismatched_runs', 0)} "
                    f"missing={environment_summary.get('missing_runs', 0)}"
                )
                environment_baseline_sha256 = payload.get("environment_comparison", {}).get(
                    "baseline_environment_fingerprint_sha256", None
                )
                if isinstance(environment_baseline_sha256, str) and environment_baseline_sha256:
                    print(
                        "environment_baseline.fingerprint_sha256=" f"{environment_baseline_sha256}"
                    )
                for row in payload.get("environment_comparison", {}).get("comparisons", []):
                    if not isinstance(row, dict):
                        continue
                    status = str(row.get("status"))
                    if status not in {"mismatched", "missing"}:
                        continue
                    run_dir_name = row.get("run_dir_name", None)
                    if isinstance(run_dir_name, str) and run_dir_name:
                        print(f"environment_incompat.{run_dir_name}={status}")
                print(
                    "target: "
                    f"checked={target_summary.get('checked')} "
                    f"matched={target_summary.get('matched_runs', 0)} "
                    f"mismatched={target_summary.get('mismatched_runs', 0)} "
                    f"missing={target_summary.get('missing_runs', 0)}"
                )
                target_baseline = payload.get("target_comparison", {}).get("baseline", {})
                if isinstance(target_baseline, dict):
                    baseline_dataset = target_baseline.get("dataset", None)
                    if isinstance(baseline_dataset, str) and baseline_dataset:
                        print(f"target_baseline.dataset={baseline_dataset}")
                    baseline_category = target_baseline.get("category", None)
                    if isinstance(baseline_category, str) and baseline_category:
                        print(f"target_baseline.category={baseline_category}")
                for row in payload.get("target_comparison", {}).get("comparisons", []):
                    if not isinstance(row, dict):
                        continue
                    status = str(row.get("status"))
                    if status not in {"mismatched", "missing"}:
                        continue
                    run_dir_name = row.get("run_dir_name", None)
                    if not isinstance(run_dir_name, str) or not run_dir_name:
                        continue
                    dataset_status = str(row.get("dataset_status", "unchecked"))
                    category_status = str(row.get("category_status", "unchecked"))
                    print(
                        "target_incompat."
                        f"{run_dir_name}=dataset:{dataset_status},category:{category_status}"
                    )
                print(
                    "robustness_protocol: "
                    f"checked={robustness_protocol_summary.get('checked')} "
                    f"matched={robustness_protocol_summary.get('matched_runs', 0)} "
                    f"mismatched={robustness_protocol_summary.get('mismatched_runs', 0)} "
                    f"missing={robustness_protocol_summary.get('missing_runs', 0)}"
                )
                robustness_protocol_baseline = payload.get(
                    "robustness_protocol_comparison", {}
                ).get("baseline", {})
                if isinstance(robustness_protocol_baseline, dict):
                    corruption_mode = robustness_protocol_baseline.get("corruption_mode", None)
                    if isinstance(corruption_mode, str) and corruption_mode:
                        print(f"robustness_protocol_baseline.corruption_mode={corruption_mode}")
                    conditions = robustness_protocol_baseline.get("conditions", None)
                    if isinstance(conditions, list) and conditions:
                        print(
                            "robustness_protocol_baseline.conditions="
                            + ",".join(str(item) for item in conditions)
                        )
                    severities = robustness_protocol_baseline.get("severities", None)
                    if isinstance(severities, list) and severities:
                        print(
                            "robustness_protocol_baseline.severities="
                            + ",".join(str(item) for item in severities)
                        )
                    input_mode = robustness_protocol_baseline.get("input_mode", None)
                    if isinstance(input_mode, str) and input_mode:
                        print(f"robustness_protocol_baseline.input_mode={input_mode}")
                    resize = robustness_protocol_baseline.get("resize", None)
                    if isinstance(resize, list) and len(resize) == 2:
                        print(
                            "robustness_protocol_baseline.resize="
                            + ",".join(str(item) for item in resize)
                        )
                for row in payload.get("robustness_protocol_comparison", {}).get("comparisons", []):
                    if not isinstance(row, dict):
                        continue
                    if str(row.get("status")) != "mismatched":
                        continue
                    mismatch_fields = row.get("mismatch_fields", None)
                    if not isinstance(mismatch_fields, list) or not mismatch_fields:
                        continue
                    run_dir_name = row.get("run_dir_name", None)
                    if isinstance(run_dir_name, str) and run_dir_name:
                        print(
                            "robustness_protocol_mismatch."
                            f"{run_dir_name}={','.join(str(item) for item in mismatch_fields)}"
                        )
                _emit_contract_incompat_details(
                    label="operator_contract",
                    comparison_payload=dict(payload.get("operator_contract_comparison", {})),
                    summary_payload=operator_contract_summary,
                )
                _emit_contract_incompat_details(
                    label="bundle_operator_contract",
                    comparison_payload=dict(payload.get("bundle_operator_contract_comparison", {})),
                    summary_payload=bundle_operator_contract_summary,
                )
            if bool(args.require_same_split):
                split_summary = dict(payload.get("split_comparison", {}).get("summary", {}))
                if (
                    not bool(split_summary.get("checked"))
                    or int(split_summary.get("incompatible_runs", 0)) > 0
                ):
                    return 1
            if bool(args.require_same_environment):
                environment_summary = dict(
                    payload.get("environment_comparison", {}).get("summary", {})
                )
                if (
                    not bool(environment_summary.get("checked"))
                    or int(environment_summary.get("incompatible_runs", 0)) > 0
                ):
                    return 1
            if bool(args.require_same_target):
                target_summary = dict(payload.get("target_comparison", {}).get("summary", {}))
                if (
                    not bool(target_summary.get("checked"))
                    or int(target_summary.get("incompatible_runs", 0)) > 0
                ):
                    return 1
            if bool(args.require_same_robustness_protocol):
                robustness_protocol_summary = dict(
                    payload.get("robustness_protocol_comparison", {}).get("summary", {})
                )
                if (
                    not bool(robustness_protocol_summary.get("checked"))
                    or int(robustness_protocol_summary.get("incompatible_runs", 0)) > 0
                ):
                    return 1
            if bool(args.require_same_operator_contract):
                operator_contract_summary = dict(
                    payload.get("operator_contract_comparison", {}).get("summary", {})
                )
                if (
                    not bool(operator_contract_summary.get("checked"))
                    or int(operator_contract_summary.get("incompatible_runs", 0)) > 0
                ):
                    return 1
            if bool(args.require_same_bundle_operator_contract):
                bundle_operator_contract_summary = dict(
                    payload.get("bundle_operator_contract_comparison", {}).get("summary", {})
                )
                if (
                    not bool(bundle_operator_contract_summary.get("checked"))
                    or int(bundle_operator_contract_summary.get("incompatible_runs", 0)) > 0
                ):
                    return 1
            if bool(args.fail_on_regression) and int(payload["summary"]["total_regressions"]) > int(
                args.max_regressions
            ):
                return 1
            return 0

        if str(args.cmd) == "quality":
            quality = evaluate_run_quality(
                args.run_dir,
                check_bundle_hashes=bool(args.check_bundle_hashes),
            )
            payload = {"run_dir": str(args.run_dir), "quality": quality}
            meets_required_status = True
            if args.require_status is not None:
                current_rank = int(_QUALITY_STATUS_RANK.get(str(quality.get("status")), -1))
                required_rank = int(_QUALITY_STATUS_RANK[str(args.require_status)])
                meets_required_status = current_rank >= required_rank
            if bool(args.json):
                rc = cli_output.emit_jsonable(payload, indent=None)
                if not bool(meets_required_status):
                    return 1
                if str(quality.get("status")) in {"partial", "broken"}:
                    return 1
                return rc

            print(
                runs_cli_rendering.format_quality_summary_line(
                    run_name=Path(str(args.run_dir)).name,
                    quality=quality,
                )
            )
            warnings = list(quality.get("warnings", []))
            for item in warnings:
                print(f"warning={item}")
            trust_summary = dict(quality.get("trust_summary", {}))
            trust_signals = dict(trust_summary.get("trust_signals", {}))
            for key, value in trust_signals.items():
                print(f"trust_signal.{key}={value}")
            raw_dataset_readiness = quality.get("dataset_readiness", {})
            dataset_readiness = (
                dict(raw_dataset_readiness) if isinstance(raw_dataset_readiness, dict) else {}
            )
            if dataset_readiness:
                print(f"dataset_readiness_status={dataset_readiness.get('status')}")
                issue_codes = list(dataset_readiness.get("issue_codes", []))
                if issue_codes:
                    print("dataset_issue_codes=" + ",".join(str(item) for item in issue_codes))
            for item in trust_summary.get("status_reasons", []):
                print(f"reason={item}")
            for item in trust_summary.get("degraded_by", []):
                print(f"degraded_by={item}")
            audit_refs = dict(trust_summary.get("audit_refs", {}))
            for key, value in audit_refs.items():
                print(f"ref={key}:{value}")
            missing_required = list(quality.get("missing_required", []))
            if missing_required:
                print(_MISSING_REQUIRED_PREFIX + ", ".join(str(item) for item in missing_required))
                return 1
            if not bool(meets_required_status):
                return 1
            return 0

        if str(args.cmd) == "acceptance":
            acceptance = evaluate_acceptance(
                args.path,
                required_quality=str(args.require_status),
                check_bundle_hashes=bool(args.check_bundle_hashes),
            )
            payload = {"path": str(args.path), "acceptance": acceptance}
            if str(acceptance.get("kind")) == "run":
                payload["run_dir"] = str(acceptance.get("run_dir"))
            if bool(args.json):
                rc = cli_output.emit_jsonable(payload, indent=None)
                if not bool(acceptance.get("ready")):
                    return 1
                return rc

            if str(acceptance.get("kind")) == "publication":
                publication = dict(acceptance.get("publication", {}))
                print(
                    f"{Path(str(args.path)).name}: kind=publication "
                    f"status={acceptance.get('status')} "
                    f"publication_ready={publication.get('publication_ready')}"
                )
                for item in acceptance.get("blocking_reasons", []):
                    print(f"blocking_reason={item}")
                trust_signals = dict(publication.get("trust_signals", {}))
                for key, value in trust_signals.items():
                    print(f"trust_signal.{key}={value}")
                dataset_readiness = dict(publication.get("dataset_readiness", {}))
                if dataset_readiness:
                    print(f"dataset_readiness_status={dataset_readiness.get('status')}")
                    issue_codes = list(dataset_readiness.get("issue_codes", []))
                    if issue_codes:
                        print("dataset_issue_codes=" + ",".join(str(item) for item in issue_codes))
                audit_refs = dict(publication.get("audit_refs", {}))
                for key, value in audit_refs.items():
                    print(f"audit_ref.{key}={value}")
                missing_required = list(publication.get("missing_required", []))
                if missing_required:
                    print(
                        _MISSING_REQUIRED_PREFIX + ", ".join(str(item) for item in missing_required)
                    )
                invalid_declared = list(publication.get("invalid_declared", []))
                if invalid_declared:
                    print("invalid_declared=" + ", ".join(str(item) for item in invalid_declared))
                return 0 if bool(acceptance.get("ready")) else 1

            infer_cfg = dict(acceptance.get("infer_config", {}))
            bundle_weights = dict(acceptance.get("bundle_weights", {}))
            print(
                runs_cli_rendering.format_acceptance_run_summary_line(
                    run_name=Path(str(args.path)).name,
                    acceptance=acceptance,
                )
            )
            for item in acceptance.get("blocking_reasons", []):
                print(f"blocking_reason={item}")
            dataset_readiness = dict(dict(acceptance.get("quality", {})).get("dataset_readiness", {}))
            if dataset_readiness:
                print(f"dataset_readiness_status={dataset_readiness.get('status')}")
                issue_codes = list(dataset_readiness.get("issue_codes", []))
                if issue_codes:
                    print("dataset_issue_codes=" + ",".join(str(item) for item in issue_codes))
            for item in infer_cfg.get("warnings", []):
                print(f"infer_warning={item}")
            for item in infer_cfg.get("errors", []):
                print(f"infer_error={item}")
            if bool(bundle_weights.get("applicable")):
                for item in bundle_weights.get("missing_required", []):
                    print(f"bundle_missing_required={item}")
                for item in bundle_weights.get("warnings", []):
                    print(f"bundle_warning={item}")
                for item in bundle_weights.get("errors", []):
                    print(f"bundle_error={item}")
            return 0 if bool(acceptance.get("ready")) else 1

        if str(args.cmd) == "publication":
            publication = evaluate_publication_quality(args.path)
            payload = {"path": str(args.path), "publication": publication}
            if bool(args.json):
                rc = cli_output.emit_jsonable(payload, indent=None)
                if str(publication.get("status")) != "ready":
                    return 1
                return rc

            print(
                runs_cli_rendering.format_publication_summary_line(
                    path_name=Path(str(args.path)).name,
                    publication=publication,
                )
            )
            trust_signals = dict(publication.get("trust_signals", {}))
            for key, value in trust_signals.items():
                print(f"trust_signal.{key}={value}")
            dataset_readiness = dict(publication.get("dataset_readiness", {}))
            if dataset_readiness:
                print(f"dataset_readiness_status={dataset_readiness.get('status')}")
                issue_codes = list(dataset_readiness.get("issue_codes", []))
                if issue_codes:
                    print("dataset_issue_codes=" + ",".join(str(item) for item in issue_codes))
            audit_refs = dict(publication.get("audit_refs", {}))
            for key, value in audit_refs.items():
                print(f"audit_ref.{key}={value}")
            missing_required = list(publication.get("missing_required", []))
            if missing_required:
                print(_MISSING_REQUIRED_PREFIX + ", ".join(str(item) for item in missing_required))
                return 1
            return 0

        raise RuntimeError(f"Unhandled cmd: {args.cmd!r}")
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        cli_output.print_cli_error(exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
