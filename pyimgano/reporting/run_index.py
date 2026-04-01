from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pyimgano.reporting.evaluation_contract import build_evaluation_contract
from pyimgano.reporting.run_index_helpers import (
    build_trust_comparison,
)
from pyimgano.reporting.run_index_helpers import (
    bundle_operator_contract_status_from_trust_summary as _bundle_operator_contract_status_from_trust_summary_helper,
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
    operator_contract_status_from_trust_summary as _operator_contract_status_from_trust_summary_helper,
)
from pyimgano.reporting.run_quality import evaluate_run_quality

_REPORT_JSON = "report.json"
_ENVIRONMENT_JSON = "environment.json"
from pyimgano.reporting.robustness_summary import (
    build_robustness_trust_summary,
    summarize_robustness_protocol,
    summarize_robustness_report,
)

_HIGHER_IS_BETTER_METRICS = {
    "auroc",
    "average_precision",
    "pixel_auroc",
    "pixel_average_precision",
    "aupro",
    "pixel_segf1",
    "clean_auroc",
    "clean_average_precision",
    "clean_pixel_auroc",
    "clean_pixel_average_precision",
    "clean_aupro",
    "clean_pixel_segf1",
    "mean_corruption_auroc",
    "mean_corruption_average_precision",
    "mean_corruption_pixel_auroc",
    "mean_corruption_pixel_average_precision",
    "mean_corruption_aupro",
    "mean_corruption_pixel_segf1",
    "worst_corruption_auroc",
    "worst_corruption_average_precision",
    "worst_corruption_pixel_auroc",
    "worst_corruption_pixel_average_precision",
    "worst_corruption_aupro",
    "worst_corruption_pixel_segf1",
}

_LOWER_IS_BETTER_METRICS = {
    "clean_latency_ms_per_image",
    "mean_corruption_drop_auroc",
    "mean_corruption_drop_average_precision",
    "mean_corruption_drop_pixel_auroc",
    "mean_corruption_drop_pixel_average_precision",
    "mean_corruption_drop_aupro",
    "mean_corruption_drop_pixel_segf1",
    "mean_corruption_latency_ms_per_image",
    "mean_corruption_latency_ratio",
    "worst_corruption_drop_auroc",
    "worst_corruption_drop_average_precision",
    "worst_corruption_drop_pixel_auroc",
    "worst_corruption_drop_pixel_average_precision",
    "worst_corruption_drop_aupro",
    "worst_corruption_drop_pixel_segf1",
    "worst_corruption_latency_ms_per_image",
    "worst_corruption_latency_ratio",
}

_QUALITY_STATUS_RANK = {
    "broken": 0,
    "partial": 1,
    "reproducible": 2,
    "audited": 3,
    "deployable": 4,
}


def _comparison_trust_gate(trust_status: object) -> str | None:
    return _comparison_trust_gate_helper(trust_status)


def _operator_contract_status_from_trust_summary(
    trust_summary: Mapping[str, Any]
) -> tuple[str, bool]:
    return _operator_contract_status_from_trust_summary_helper(trust_summary)


def _bundle_operator_contract_status_from_trust_summary(
    trust_summary: Mapping[str, Any],
) -> tuple[str, bool]:
    return _bundle_operator_contract_status_from_trust_summary_helper(trust_summary)


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


def _build_trust_comparison(
    baseline_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return build_trust_comparison(baseline_summary)


def _comparability_gate_status(summary: Mapping[str, Any]) -> str:
    return _comparability_gate_status_helper(summary)


def _compare_blocking_flags(
    *,
    total_regressions: int,
    split_summary: Mapping[str, Any],
    environment_summary: Mapping[str, Any],
    target_summary: Mapping[str, Any],
    robustness_protocol_summary: Mapping[str, Any],
    operator_contract_summary: Mapping[str, Any],
    bundle_operator_contract_summary: Mapping[str, Any],
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


def _candidate_run_dir_names(
    runs: list[dict[str, Any]],
    *,
    baseline_path_str: str | None,
) -> list[str]:
    names: list[str] = []
    for run in runs:
        run_path = str(Path(str(run.get("run_dir"))).resolve())
        if baseline_path_str is not None and run_path == baseline_path_str:
            continue
        run_dir_name = run.get("run_dir_name", None)
        if isinstance(run_dir_name, str) and run_dir_name and run_dir_name not in names:
            names.append(run_dir_name)
    return names


def _append_candidate_reason(
    reasons_by_name: dict[str, list[str]],
    *,
    run_dir_name: object,
    reason: str,
) -> None:
    if not isinstance(run_dir_name, str) or not run_dir_name:
        return
    if run_dir_name not in reasons_by_name:
        reasons_by_name[run_dir_name] = []
    if reason not in reasons_by_name[run_dir_name]:
        reasons_by_name[run_dir_name].append(reason)


def _run_trust_degraded_by(run: Mapping[str, Any]) -> set[str]:
    artifact_quality = run.get("artifact_quality", None)
    if not isinstance(artifact_quality, Mapping):
        return set()
    trust_summary = artifact_quality.get("trust_summary", None)
    if not isinstance(trust_summary, Mapping):
        return set()
    degraded_by_raw = trust_summary.get("degraded_by", None)
    if not isinstance(degraded_by_raw, list):
        return set()
    return {str(item) for item in degraded_by_raw if str(item)}


def _run_trust_signals(run: Mapping[str, Any]) -> dict[str, Any]:
    artifact_quality = run.get("artifact_quality", None)
    if not isinstance(artifact_quality, Mapping):
        return {}
    trust_summary = artifact_quality.get("trust_summary", None)
    if not isinstance(trust_summary, Mapping):
        return {}
    signals_raw = trust_summary.get("trust_signals", None)
    return dict(signals_raw) if isinstance(signals_raw, Mapping) else {}


def _run_bundle_operator_contract_digest_status(run: Mapping[str, Any]) -> str:
    degraded_by = _run_trust_degraded_by(run)
    if "operator_contract_bundle_digest_mismatch" in degraded_by:
        return "invalid"

    bundle_status = str(run.get("bundle_operator_contract_status", "") or "").strip().lower()
    if bundle_status == "missing":
        return "missing"

    trust_signals = _run_trust_signals(run)
    if not bool(trust_signals.get("has_bundle_operator_contract", False)):
        return "missing"
    if "has_bundle_operator_contract_digests_valid" in trust_signals:
        return (
            "valid"
            if bool(trust_signals.get("has_bundle_operator_contract_digests_valid"))
            else "invalid"
        )
    return "unknown"


_CANDIDATE_COMPARABILITY_GATE_ORDER = (
    "split",
    "environment",
    "target",
    "target_dataset",
    "target_category",
    "robustness_protocol",
    "operator_contract",
    "bundle_operator_contract",
)


def _candidate_comparability_defaults() -> dict[str, str]:
    return {gate_name: "unchecked" for gate_name in _CANDIDATE_COMPARABILITY_GATE_ORDER}


def _candidate_summary_state(
    candidate_names: list[str],
) -> tuple[dict[str, list[str]], dict[str, dict[str, str]], dict[str, str]]:
    return (
        {name: [] for name in candidate_names},
        {name: _candidate_comparability_defaults() for name in candidate_names},
        {name: "unavailable" for name in candidate_names},
    )


def _set_candidate_gate(
    candidate_gates: Mapping[str, dict[str, str]],
    *,
    run_dir_name: object,
    gate_name: str,
    status: str,
) -> None:
    if not isinstance(run_dir_name, str) or not run_dir_name or not status:
        return
    if run_dir_name not in candidate_gates:
        return
    candidate_gates[run_dir_name][gate_name] = status


def _apply_simple_candidate_comparison(
    *,
    reasons_by_name: dict[str, list[str]],
    candidate_gates: dict[str, dict[str, str]],
    comparison: Mapping[str, Any],
    gate_name: str,
    reason_prefix: str,
) -> None:
    for row in comparison.get("comparisons", []):
        if not isinstance(row, Mapping):
            continue
        run_dir_name = row.get("run_dir_name", None)
        status = str(row.get("status", ""))
        _set_candidate_gate(
            candidate_gates,
            run_dir_name=run_dir_name,
            gate_name=gate_name,
            status=status,
        )
        if status in {"mismatched", "missing"}:
            _append_candidate_reason(
                reasons_by_name,
                run_dir_name=run_dir_name,
                reason=f"{reason_prefix}:{status}",
            )


def _apply_target_candidate_comparison(
    *,
    reasons_by_name: dict[str, list[str]],
    candidate_gates: dict[str, dict[str, str]],
    target_comparison: Mapping[str, Any],
) -> None:
    for row in target_comparison.get("comparisons", []):
        if not isinstance(row, Mapping):
            continue
        run_dir_name = row.get("run_dir_name", None)
        row_status = str(row.get("status", ""))
        dataset_status = str(row.get("dataset_status", ""))
        category_status = str(row.get("category_status", ""))
        _set_candidate_gate(
            candidate_gates,
            run_dir_name=run_dir_name,
            gate_name="target",
            status=row_status,
        )
        _set_candidate_gate(
            candidate_gates,
            run_dir_name=run_dir_name,
            gate_name="target_dataset",
            status=dataset_status,
        )
        _set_candidate_gate(
            candidate_gates,
            run_dir_name=run_dir_name,
            gate_name="target_category",
            status=category_status,
        )
        if dataset_status in {"mismatched", "missing"}:
            _append_candidate_reason(
                reasons_by_name,
                run_dir_name=run_dir_name,
                reason=f"target.dataset:{dataset_status}",
            )
        if category_status in {"mismatched", "missing"}:
            _append_candidate_reason(
                reasons_by_name,
                run_dir_name=run_dir_name,
                reason=f"target.category:{category_status}",
            )


def _robustness_protocol_mismatch_reasons(row: Mapping[str, Any]) -> list[str]:
    status = str(row.get("status", ""))
    if status == "missing":
        return ["robustness_protocol:missing"]
    if status != "mismatched":
        return []
    mismatch_fields = row.get("mismatch_fields", None)
    if not isinstance(mismatch_fields, list) or not mismatch_fields:
        return ["robustness_protocol:mismatched"]
    return [
        f"robustness_protocol.{field_name}:mismatched"
        for field_name in (str(field) for field in mismatch_fields)
        if field_name
    ]


def _apply_robustness_protocol_candidate_comparison(
    *,
    reasons_by_name: dict[str, list[str]],
    candidate_gates: dict[str, dict[str, str]],
    robustness_protocol_comparison: Mapping[str, Any],
) -> None:
    for row in robustness_protocol_comparison.get("comparisons", []):
        if not isinstance(row, Mapping):
            continue
        run_dir_name = row.get("run_dir_name", None)
        status = str(row.get("status", ""))
        _set_candidate_gate(
            candidate_gates,
            run_dir_name=run_dir_name,
            gate_name="robustness_protocol",
            status=status,
        )
        for reason in _robustness_protocol_mismatch_reasons(row):
            _append_candidate_reason(
                reasons_by_name,
                run_dir_name=run_dir_name,
                reason=reason,
            )


def _iter_candidate_runs(
    runs: list[dict[str, Any]],
    *,
    baseline_path_str: str,
) -> Iterable[dict[str, Any]]:
    for run in runs:
        if _is_baseline_run(run, baseline_path_str):
            continue
        yield run


def _baseline_contract_payload(
    *,
    baseline_path_str: str,
    contract_status: str | None,
    bundle: bool,
) -> dict[str, Any] | None:
    if str(contract_status or "").strip().lower() != "consistent":
        return None
    return _load_operator_contract_payload(baseline_path_str, bundle=bundle)


def _apply_operator_contract_candidates(
    *,
    reasons_by_name: dict[str, list[str]],
    candidate_gates: dict[str, dict[str, str]],
    runs: list[dict[str, Any]],
    baseline_path_str: str,
    baseline_operator_contract_status: str | None,
    baseline_operator_contract_payload: Mapping[str, Any] | None,
) -> None:
    if str(baseline_operator_contract_status or "").strip().lower() != "consistent":
        return
    for run in _iter_candidate_runs(runs, baseline_path_str=baseline_path_str):
        run_dir_name = run.get("run_dir_name", None)
        status = str(run.get("operator_contract_status", "missing") or "missing").strip().lower()
        _set_candidate_gate(
            candidate_gates,
            run_dir_name=run_dir_name,
            gate_name="operator_contract",
            status=status,
        )
        if status in {"missing", "mismatched"}:
            _append_candidate_reason(
                reasons_by_name,
                run_dir_name=run_dir_name,
                reason=f"operator_contract:{status}",
            )
            continue
        if status != "consistent" or not isinstance(baseline_operator_contract_payload, Mapping):
            continue
        candidate_payload = _load_operator_contract_payload(run.get("run_dir", None), bundle=False)
        if isinstance(candidate_payload, Mapping) and dict(candidate_payload) != dict(
            baseline_operator_contract_payload
        ):
            _set_candidate_gate(
                candidate_gates,
                run_dir_name=run_dir_name,
                gate_name="operator_contract",
                status="mismatched",
            )
            _append_candidate_reason(
                reasons_by_name,
                run_dir_name=run_dir_name,
                reason="operator_contract:baseline_mismatch",
            )
            continue
        _set_candidate_gate(
            candidate_gates,
            run_dir_name=run_dir_name,
            gate_name="operator_contract",
            status="matched",
        )


def _set_bundle_operator_contract_candidate_state(
    candidate_gates: dict[str, dict[str, str]],
    candidate_bundle_digest_statuses: dict[str, str],
    *,
    run_dir_name: object,
    status: str,
    run: Mapping[str, Any],
) -> None:
    _set_candidate_gate(
        candidate_gates,
        run_dir_name=run_dir_name,
        gate_name="bundle_operator_contract",
        status=status,
    )
    if isinstance(run_dir_name, str) and run_dir_name in candidate_bundle_digest_statuses:
        candidate_bundle_digest_statuses[
            run_dir_name
        ] = _run_bundle_operator_contract_digest_status(run)


def _append_bundle_operator_contract_status_reasons(
    reasons_by_name: dict[str, list[str]],
    *,
    run_dir_name: object,
    status: str,
    run: Mapping[str, Any],
) -> bool:
    if status not in {"missing", "mismatched"}:
        return False
    _append_candidate_reason(
        reasons_by_name,
        run_dir_name=run_dir_name,
        reason=f"operator_contract_bundle:{status}",
    )
    if (
        status == "mismatched"
        and "operator_contract_bundle_digest_mismatch" in _run_trust_degraded_by(run)
    ):
        _append_candidate_reason(
            reasons_by_name,
            run_dir_name=run_dir_name,
            reason="operator_contract_bundle:digest_mismatch",
        )
    return True


def _bundle_operator_contract_payload_matches(
    run: Mapping[str, Any],
    *,
    baseline_bundle_operator_contract_payload: Mapping[str, Any],
) -> bool:
    candidate_payload = _load_operator_contract_payload(run.get("run_dir", None), bundle=True)
    return not (
        isinstance(candidate_payload, Mapping)
        and dict(candidate_payload) != dict(baseline_bundle_operator_contract_payload)
    )


def _apply_bundle_operator_contract_candidates(
    *,
    reasons_by_name: dict[str, list[str]],
    candidate_gates: dict[str, dict[str, str]],
    candidate_bundle_digest_statuses: dict[str, str],
    runs: list[dict[str, Any]],
    baseline_path_str: str,
    baseline_bundle_operator_contract_status: str | None,
    baseline_bundle_operator_contract_payload: Mapping[str, Any] | None,
) -> None:
    if str(baseline_bundle_operator_contract_status or "").strip().lower() != "consistent":
        return
    for run in _iter_candidate_runs(runs, baseline_path_str=baseline_path_str):
        run_dir_name = run.get("run_dir_name", None)
        status = (
            str(run.get("bundle_operator_contract_status", "missing") or "missing").strip().lower()
        )
        _set_bundle_operator_contract_candidate_state(
            candidate_gates,
            candidate_bundle_digest_statuses,
            run_dir_name=run_dir_name,
            status=status,
            run=run,
        )
        if _append_bundle_operator_contract_status_reasons(
            reasons_by_name,
            run_dir_name=run_dir_name,
            status=status,
            run=run,
        ):
            continue
        if status != "consistent" or not isinstance(
            baseline_bundle_operator_contract_payload, Mapping
        ):
            continue
        if not _bundle_operator_contract_payload_matches(
            run,
            baseline_bundle_operator_contract_payload=baseline_bundle_operator_contract_payload,
        ):
            _set_candidate_gate(
                candidate_gates,
                run_dir_name=run_dir_name,
                gate_name="bundle_operator_contract",
                status="mismatched",
            )
            _append_candidate_reason(
                reasons_by_name,
                run_dir_name=run_dir_name,
                reason="operator_contract_bundle:baseline_mismatch",
            )
            continue
        _set_candidate_gate(
            candidate_gates,
            run_dir_name=run_dir_name,
            gate_name="bundle_operator_contract",
            status="matched",
        )


def _build_candidate_incompatibility_digest(
    *,
    candidate_verdicts: Mapping[str, Any],
    candidate_blocking_reasons: Mapping[str, Any],
    candidate_comparability_gates: Mapping[str, Any],
    candidate_dataset_readiness: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    candidate_names = sorted(
        {
            str(name)
            for name in (
                list(candidate_verdicts.keys())
                + list(candidate_blocking_reasons.keys())
                + list(candidate_comparability_gates.keys())
            )
            if str(name)
        }
    )
    digest: dict[str, dict[str, Any]] = {}
    for name in candidate_names:
        verdict = candidate_verdicts.get(name, None)
        if not isinstance(verdict, str) or not verdict:
            verdict = "pass"
        reasons_raw = candidate_blocking_reasons.get(name, [])
        reasons = [str(reason) for reason in reasons_raw if str(reason)]
        gate_states = candidate_comparability_gates.get(name, None)
        gate_map = dict(gate_states) if isinstance(gate_states, Mapping) else {}
        incompatible_gates: list[str] = []
        for gate_name in _CANDIDATE_COMPARABILITY_GATE_ORDER:
            status = gate_map.get(gate_name, None)
            status_text = str(status).strip().lower() if status is not None else ""
            if status_text in {"missing", "mismatched"}:
                incompatible_gates.append(f"{gate_name}:{status_text}")
        digest_entry: dict[str, Any] = {
            "verdict": verdict,
            "incompatible_gates": incompatible_gates,
            "blocking_reasons": reasons,
        }
        readiness = (
            candidate_dataset_readiness.get(name, None)
            if isinstance(candidate_dataset_readiness, Mapping)
            else None
        )
        if isinstance(readiness, Mapping):
            readiness_status = readiness.get("status", None)
            if isinstance(readiness_status, str) and readiness_status:
                digest_entry["dataset_readiness_status"] = readiness_status
            issue_codes = readiness.get("issue_codes", None)
            if isinstance(issue_codes, list):
                digest_entry["dataset_issue_codes"] = [
                    str(item) for item in issue_codes if str(item)
                ]
        digest[name] = digest_entry
    return digest


def _dataset_readiness_payload(run: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(run, Mapping):
        return None
    raw = run.get("dataset_readiness", None)
    if isinstance(raw, Mapping):
        readiness = dict(raw)
    else:
        readiness = {}
        status = run.get("dataset_readiness_status", None)
        if isinstance(status, str) and status:
            readiness["status"] = status
        issue_codes = run.get("dataset_issue_codes", None)
        if isinstance(issue_codes, list):
            readiness["issue_codes"] = [str(item) for item in issue_codes if str(item)]
    if not readiness:
        return None
    issue_codes = readiness.get("issue_codes", None)
    readiness["issue_codes"] = [str(item) for item in issue_codes if str(item)] if isinstance(
        issue_codes, list
    ) else []
    issue_details = readiness.get("issue_details", None)
    readiness["issue_details"] = list(issue_details) if isinstance(issue_details, list) else []
    status = readiness.get("status", None)
    if (
        not isinstance(status, str)
        or not status
    ) and not readiness["issue_codes"] and not readiness["issue_details"]:
        return None
    return readiness


def _candidate_dataset_readiness_by_name(
    runs: Sequence[Mapping[str, Any]],
    *,
    baseline_path_str: str | None,
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for run in runs:
        if _is_baseline_run(run, baseline_path_str):
            continue
        run_dir_name = run.get("run_dir_name", None)
        if not isinstance(run_dir_name, str) or not run_dir_name:
            continue
        readiness = _dataset_readiness_payload(run)
        if isinstance(readiness, dict):
            payload[run_dir_name] = readiness
    return payload


def _load_operator_contract_payload(
    run_dir: object,
    *,
    bundle: bool,
) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    root = Path(str(run_dir))
    rel_path = (
        "deploy_bundle/operator_contract.json"
        if bool(bundle)
        else "artifacts/operator_contract.json"
    )
    contract_path = root / rel_path
    if not contract_path.is_file():
        return None
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    return dict(payload)


def _build_candidate_blocking_summary(
    *,
    runs: list[dict[str, Any]],
    baseline_path_str: str | None,
    baseline_operator_contract_status: str | None,
    baseline_bundle_operator_contract_status: str | None,
    primary_metric_info: Mapping[str, Any],
    split_comparison: Mapping[str, Any],
    environment_comparison: Mapping[str, Any],
    target_comparison: Mapping[str, Any],
    robustness_protocol_comparison: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if baseline_path_str is None:
        return {
            "candidate_verdicts": {},
            "candidate_blocking_reasons": {},
            "candidate_comparability_gates": {},
            "candidate_bundle_operator_contract_digest_statuses": {},
        }

    candidate_names = _candidate_run_dir_names(runs, baseline_path_str=baseline_path_str)
    reasons_by_name, candidate_gates, candidate_bundle_digest_statuses = _candidate_summary_state(
        candidate_names
    )
    baseline_operator_contract_payload = _baseline_contract_payload(
        baseline_path_str=baseline_path_str,
        contract_status=baseline_operator_contract_status,
        bundle=False,
    )
    baseline_bundle_operator_contract_payload = _baseline_contract_payload(
        baseline_path_str=baseline_path_str,
        contract_status=baseline_bundle_operator_contract_status,
        bundle=True,
    )

    for row in primary_metric_info.get("comparisons", []):
        if not isinstance(row, Mapping):
            continue
        status = str(row.get("status", ""))
        if status in {"regressed", "missing"}:
            _append_candidate_reason(
                reasons_by_name,
                run_dir_name=row.get("run_dir_name", None),
                reason=f"primary_metric:{status}",
            )

    _apply_simple_candidate_comparison(
        reasons_by_name=reasons_by_name,
        candidate_gates=candidate_gates,
        comparison=split_comparison,
        gate_name="split",
        reason_prefix="split",
    )
    _apply_simple_candidate_comparison(
        reasons_by_name=reasons_by_name,
        candidate_gates=candidate_gates,
        comparison=environment_comparison,
        gate_name="environment",
        reason_prefix="environment",
    )
    _apply_target_candidate_comparison(
        reasons_by_name=reasons_by_name,
        candidate_gates=candidate_gates,
        target_comparison=target_comparison,
    )
    _apply_robustness_protocol_candidate_comparison(
        reasons_by_name=reasons_by_name,
        candidate_gates=candidate_gates,
        robustness_protocol_comparison=robustness_protocol_comparison,
    )
    _apply_operator_contract_candidates(
        reasons_by_name=reasons_by_name,
        candidate_gates=candidate_gates,
        runs=runs,
        baseline_path_str=baseline_path_str,
        baseline_operator_contract_status=baseline_operator_contract_status,
        baseline_operator_contract_payload=baseline_operator_contract_payload,
    )
    _apply_bundle_operator_contract_candidates(
        reasons_by_name=reasons_by_name,
        candidate_gates=candidate_gates,
        candidate_bundle_digest_statuses=candidate_bundle_digest_statuses,
        runs=runs,
        baseline_path_str=baseline_path_str,
        baseline_bundle_operator_contract_status=baseline_bundle_operator_contract_status,
        baseline_bundle_operator_contract_payload=baseline_bundle_operator_contract_payload,
    )

    candidate_verdicts = {
        name: ("blocked" if reasons_by_name.get(name, []) else "pass") for name in candidate_names
    }
    return {
        "candidate_verdicts": candidate_verdicts,
        "candidate_blocking_reasons": reasons_by_name,
        "candidate_comparability_gates": candidate_gates,
        "candidate_bundle_operator_contract_digest_statuses": candidate_bundle_digest_statuses,
    }


def _load_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in: {path}")
    return payload


def _contract_payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _numeric_metrics_from_mapping(
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        value = payload.get(key, None)
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
    return out


def _row_metric_maxima(rows: Any, keys: Sequence[str]) -> dict[str, float]:
    if not isinstance(rows, list):
        return {}
    metrics: dict[str, float] = {}
    for key in keys:
        vals = [
            float(item[key])
            for item in rows
            if isinstance(item, dict) and isinstance(item.get(key), (int, float))
        ]
        if vals:
            metrics[str(key)] = max(vals)
    return metrics


def _extract_metrics(report: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    robustness_summary = report.get("robustness_summary", None)
    if isinstance(robustness_summary, Mapping):
        metrics.update(
            {
                str(key): float(value)
                for key, value in robustness_summary.items()
                if isinstance(value, (int, float))
            }
        )

    results = report.get("results", None)
    if isinstance(results, Mapping):
        metrics.update(_numeric_metrics_from_mapping(results, ("auroc", "average_precision")))
        pixel = results.get("pixel_metrics", None)
        if isinstance(pixel, Mapping):
            metrics.update(
                _numeric_metrics_from_mapping(
                    pixel,
                    ("pixel_auroc", "pixel_average_precision", "aupro", "pixel_segf1"),
                )
            )

    metrics.update(
        _row_metric_maxima(
            report.get("rows", None),
            ("auroc", "average_precision", "pixel_auroc", "aupro", "pixel_segf1"),
        )
    )

    summary = report.get("summary", None)
    if isinstance(summary, dict):
        by_auroc = summary.get("by_auroc", None)
        if isinstance(by_auroc, list) and by_auroc:
            first = by_auroc[0]
            if isinstance(first, dict) and isinstance(first.get("auroc"), (int, float)):
                metrics.setdefault("auroc", float(first["auroc"]))

    robustness = report.get("robustness", None)
    if isinstance(robustness, dict):
        for key, value in summarize_robustness_report(robustness).items():
            metrics.setdefault(str(key), float(value))
    return metrics


def _extract_robustness_payload(report: Mapping[str, Any]) -> dict[str, Any] | None:
    robustness = report.get("robustness", None)
    if isinstance(robustness, Mapping):
        return dict(robustness)
    return None


def _extract_robustness_protocol(report: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = report.get("robustness_protocol", None)
    if isinstance(raw, Mapping):
        return dict(raw)

    robustness = _extract_robustness_payload(report)
    if robustness is None:
        return None
    return summarize_robustness_protocol(robustness)


def _extract_robustness_trust(
    report: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any] | None:
    raw = report.get("robustness_trust", None)
    robustness = _extract_robustness_payload(report)
    if robustness is None:
        if isinstance(raw, Mapping):
            return dict(raw)
        return None

    robustness_summary = report.get("robustness_summary", None)
    robustness_protocol = _extract_robustness_protocol(report)
    raw_audit_refs = raw.get("audit_refs", None) if isinstance(raw, Mapping) else None
    raw_audit_digests = raw.get("audit_digests", None) if isinstance(raw, Mapping) else None
    return build_robustness_trust_summary(
        report=robustness,
        robustness_summary=(
            dict(robustness_summary) if isinstance(robustness_summary, Mapping) else None
        ),
        robustness_protocol=robustness_protocol,
        audit_refs=(dict(raw_audit_refs) if isinstance(raw_audit_refs, Mapping) else None),
        audit_digests=(dict(raw_audit_digests) if isinstance(raw_audit_digests, Mapping) else None),
        audit_root=root,
    )


def _extract_split_fingerprint_sha256(report: dict[str, Any]) -> str | None:
    split_fingerprint = report.get("split_fingerprint", None)
    if not isinstance(split_fingerprint, dict):
        return None
    value = split_fingerprint.get("sha256", None)
    return str(value) if isinstance(value, str) and value else None


def _extract_environment_fingerprint_sha256(run: Mapping[str, Any]) -> str | None:
    value = run.get("environment_fingerprint_sha256", None)
    return str(value) if isinstance(value, str) and value else None


def _extract_target_signature(run: Mapping[str, Any]) -> dict[str, str | None] | None:
    dataset = run.get("dataset", None)
    category = run.get("category", None)
    signature = {
        "dataset": (str(dataset) if isinstance(dataset, str) and dataset else None),
        "category": (str(category) if isinstance(category, str) and category else None),
    }
    if all(value is None for value in signature.values()):
        return None
    return signature


def _matches_target_signature(run: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    run_signature = _extract_target_signature(run)
    baseline_signature = _extract_target_signature(baseline)
    if run_signature is None or baseline_signature is None:
        return False
    for field in ("dataset", "category"):
        expected = baseline_signature.get(field, None)
        if expected is None:
            continue
        if run_signature.get(field, None) != expected:
            return False
    return True


def _resolve_report_path(run_dir: str | Path) -> Path:
    path = Path(run_dir)
    if path.is_dir():
        return path / _REPORT_JSON
    return path


def _load_report_for_run_dir(run_dir: str | Path) -> dict[str, Any]:
    return _load_json_dict(_resolve_report_path(run_dir))


def _report_pixel_metrics_enabled(report: Mapping[str, Any]) -> bool | None:
    dataset_summary = report.get("dataset_summary", None)
    if isinstance(dataset_summary, Mapping):
        pixel_metrics = dataset_summary.get("pixel_metrics", None)
        if isinstance(pixel_metrics, Mapping) and isinstance(pixel_metrics.get("enabled"), bool):
            return bool(pixel_metrics["enabled"])

    pixel_metrics_status = report.get("pixel_metrics_status", None)
    if isinstance(pixel_metrics_status, Mapping) and isinstance(
        pixel_metrics_status.get("enabled"),
        bool,
    ):
        return bool(pixel_metrics_status["enabled"])

    results = report.get("results", None)
    if isinstance(results, Mapping) and isinstance(results.get("pixel_metrics"), Mapping):
        return True
    return None


def _default_primary_metric(
    report: Mapping[str, Any],
    *,
    metrics: Mapping[str, float],
) -> str:
    robustness = _extract_robustness_payload(report)
    if robustness is not None or isinstance(report.get("robustness_summary"), Mapping):
        for name in (
            "worst_corruption_auroc",
            "mean_corruption_auroc",
            "clean_auroc",
            "worst_corruption_average_precision",
            "mean_corruption_average_precision",
            "clean_average_precision",
        ):
            if isinstance(metrics.get(name), (int, float)):
                return str(name)
    if isinstance(metrics.get("auroc"), (int, float)):
        return "auroc"
    if metrics:
        return str(sorted(metrics.keys())[0])
    return "auroc"


def _extract_evaluation_contract(
    report: Mapping[str, Any],
    *,
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    raw = report.get("evaluation_contract", None)
    if isinstance(raw, Mapping):
        raw_directions = raw.get("metric_directions", None)
        raw_metric_names = list(metrics.keys())
        if isinstance(raw_directions, Mapping):
            raw_metric_names.extend(str(name) for name in raw_directions.keys())
        return build_evaluation_contract(
            metric_names=raw_metric_names,
            primary_metric=(str(raw.get("primary_metric")) if raw.get("primary_metric") else None),
            ranking_metric=(str(raw.get("ranking_metric")) if raw.get("ranking_metric") else None),
            pixel_metrics_enabled=(
                bool(raw.get("pixel_metrics_enabled"))
                if raw.get("pixel_metrics_enabled") is not None
                else _report_pixel_metrics_enabled(report)
            ),
            comparability_hints=(
                dict(raw.get("comparability_hints", {}))
                if isinstance(raw.get("comparability_hints"), Mapping)
                else None
            ),
        )

    return build_evaluation_contract(
        metric_names=metrics.keys(),
        primary_metric=_default_primary_metric(report, metrics=metrics),
        ranking_metric=_default_primary_metric(report, metrics=metrics),
        pixel_metrics_enabled=_report_pixel_metrics_enabled(report),
        comparability_hints=(
            dict(_extract_robustness_protocol(report).get("comparability_hints", {}))
            if isinstance(_extract_robustness_protocol(report), Mapping)
            and isinstance(_extract_robustness_protocol(report).get("comparability_hints"), Mapping)
            else None
        ),
    )


def summarize_run_dir(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    report_path = root / _REPORT_JSON
    report = _load_json_dict(report_path)
    metrics = _extract_metrics(report)

    env_path = root / _ENVIRONMENT_JSON
    env = _load_json_dict(env_path) if env_path.exists() else {}

    kind = "workbench"
    model_or_suite = report.get("recipe", None)
    if "robustness" in report:
        kind = "robustness"
        model_or_suite = report.get("model", None)
    elif "suite" in report:
        kind = "suite"
        model_or_suite = report.get("suite", None)
    elif "model" in report:
        kind = "benchmark"
        model_or_suite = report.get("model", None)

    robustness_protocol = _extract_robustness_protocol(report)
    robustness_trust = _extract_robustness_trust(report, root=root)

    artifact_quality = evaluate_run_quality(root)
    trust_summary = (
        dict(artifact_quality.get("trust_summary", {}))
        if isinstance(artifact_quality, Mapping)
        else {}
    )
    operator_contract_status, _ = _operator_contract_status_from_trust_summary(trust_summary)
    bundle_operator_contract_status, _ = _bundle_operator_contract_status_from_trust_summary(
        trust_summary
    )

    payload = {
        "run_dir": str(root),
        "run_dir_name": root.name,
        "kind": kind,
        "timestamp_utc": report.get("timestamp_utc", None),
        "dataset": report.get("dataset", None),
        "category": report.get("category", None),
        "model_or_suite": model_or_suite,
        "input_mode": report.get("input_mode", None),
        "resize": report.get("resize", None),
        "environment_fingerprint_sha256": env.get("fingerprint_sha256", None),
        "split_fingerprint_sha256": _extract_split_fingerprint_sha256(report),
        "artifact_quality": artifact_quality,
        "dataset_readiness": (
            None
            if not isinstance(artifact_quality.get("dataset_readiness"), Mapping)
            else dict(artifact_quality.get("dataset_readiness", {}))
        ),
        "dataset_readiness_status": (
            None
            if not isinstance(artifact_quality.get("dataset_readiness"), Mapping)
            else dict(artifact_quality.get("dataset_readiness", {})).get("status")
        ),
        "dataset_issue_codes": (
            []
            if not isinstance(artifact_quality.get("dataset_readiness"), Mapping)
            else list(dict(artifact_quality.get("dataset_readiness", {})).get("issue_codes", []))
        ),
        "operator_contract_status": str(operator_contract_status),
        "bundle_operator_contract_status": str(bundle_operator_contract_status),
        "metrics": metrics,
        "evaluation_contract": _extract_evaluation_contract(report, metrics=metrics),
    }
    if robustness_protocol is not None:
        payload["robustness_protocol"] = robustness_protocol
    if robustness_trust is not None:
        payload["robustness_trust"] = robustness_trust
    return payload


def _normalize_protocol_conditions(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    items = [str(item) for item in value if str(item)]
    return items if items else []


def _normalize_protocol_severities(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except Exception:
            return None
    return out


def _normalize_resize(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return [int(value[0]), int(value[1])]
    except Exception:
        return None


def _extract_robustness_protocol_signature(run: Mapping[str, Any]) -> dict[str, Any] | None:
    protocol = run.get("robustness_protocol", None)
    if not isinstance(protocol, Mapping):
        return None
    return {
        "corruption_mode": (
            str(protocol.get("corruption_mode"))
            if protocol.get("corruption_mode") is not None
            else None
        ),
        "conditions": _normalize_protocol_conditions(protocol.get("conditions")),
        "severities": _normalize_protocol_severities(protocol.get("severities")),
        "input_mode": (str(run.get("input_mode")) if run.get("input_mode") is not None else None),
        "resize": _normalize_resize(run.get("resize")),
    }


def _compare_robustness_protocol_fields(
    signature: Mapping[str, Any],
    baseline_signature: Mapping[str, Any] | None,
) -> tuple[list[str], list[str]]:
    missing_fields: list[str] = []
    mismatch_fields: list[str] = []
    for field in ("conditions", "corruption_mode", "input_mode", "resize", "severities"):
        baseline_value = (
            baseline_signature.get(field) if isinstance(baseline_signature, Mapping) else None
        )
        if baseline_value is None:
            continue
        run_value = signature.get(field)
        if run_value is None:
            missing_fields.append(str(field))
        elif run_value != baseline_value:
            mismatch_fields.append(str(field))
    return missing_fields, mismatch_fields


def _robustness_protocol_status(
    *,
    missing_fields: list[str],
    mismatch_fields: list[str],
) -> str:
    if mismatch_fields:
        return "mismatched"
    if missing_fields:
        return "missing"
    return "matched"


def _robustness_protocol_row(
    run: Mapping[str, Any],
    *,
    baseline_signature: Mapping[str, Any] | None,
    checked: bool,
    baseline_path_str: str | None,
) -> tuple[dict[str, Any], str]:
    signature = _extract_robustness_protocol_signature(run)
    row: dict[str, Any] = {
        "run_dir": run.get("run_dir"),
        "run_dir_name": run.get("run_dir_name"),
        "corruption_mode": (signature or {}).get("corruption_mode"),
        "conditions": (signature or {}).get("conditions"),
        "severities": (signature or {}).get("severities"),
        "input_mode": (signature or {}).get("input_mode"),
        "resize": (signature or {}).get("resize"),
        "status": "unchecked",
        "mismatch_fields": [],
    }
    if _is_baseline_run(run, baseline_path_str):
        row["status"] = "baseline"
        return row, "baseline"
    if not checked:
        return row, "unchecked"
    if signature is None:
        row["status"] = "missing"
        return row, "missing"

    missing_fields, mismatch_fields = _compare_robustness_protocol_fields(
        signature,
        baseline_signature,
    )
    row["mismatch_fields"] = list(dict.fromkeys(missing_fields + mismatch_fields))
    row["status"] = _robustness_protocol_status(
        missing_fields=missing_fields,
        mismatch_fields=mismatch_fields,
    )
    return row, str(row["status"])


def _build_robustness_protocol_comparison(
    runs: list[dict[str, Any]],
    *,
    baseline_summary: dict[str, Any] | None,
    baseline_path_str: str | None,
) -> dict[str, Any]:
    baseline_signature = (
        _extract_robustness_protocol_signature(baseline_summary)
        if isinstance(baseline_summary, Mapping)
        else None
    )
    checked = baseline_signature is not None
    comparisons: list[dict[str, Any]] = []
    matched_runs = 0
    mismatched_runs = 0
    missing_runs = 0

    for run in runs:
        row, status = _robustness_protocol_row(
            run,
            baseline_signature=baseline_signature,
            checked=checked,
            baseline_path_str=baseline_path_str,
        )
        matched_runs, mismatched_runs, missing_runs = _bump_counts(
            status,
            matched_runs=matched_runs,
            mismatched_runs=mismatched_runs,
            missing_runs=missing_runs,
        )
        comparisons.append(row)

    return {
        "baseline": dict(baseline_signature) if isinstance(baseline_signature, Mapping) else None,
        "comparisons": comparisons,
        "summary": {
            "checked": bool(checked),
            "matched_runs": int(matched_runs),
            "mismatched_runs": int(mismatched_runs),
            "missing_runs": int(missing_runs),
            "incompatible_runs": int(mismatched_runs + missing_runs),
        },
    }


def _resolve_contract_status(run: Mapping[str, Any], *, key: str) -> str:
    status = run.get(key, None)
    if isinstance(status, str) and status:
        return str(status).strip().lower()
    return "missing"


def _operator_contract_row(
    run: Mapping[str, Any],
    *,
    baseline_payload: Mapping[str, Any] | None,
    baseline_contract_sha256: str | None,
    baseline_path_str: str | None,
    checked: bool,
    status_key: str,
    bundle: bool,
) -> tuple[dict[str, Any], str]:
    run_status = _resolve_contract_status(run, key=status_key)
    row: dict[str, Any] = {
        "run_dir": run.get("run_dir"),
        "run_dir_name": run.get("run_dir_name"),
        "contract_status": run_status,
        "contract_sha256": None,
        "status": "unchecked",
    }
    if _is_baseline_run(run, baseline_path_str):
        row["status"] = "baseline"
        row["contract_sha256"] = baseline_contract_sha256
        return row, "baseline"
    if not checked:
        return row, "unchecked"
    if run_status == "missing":
        row["status"] = "missing"
        return row, "missing"
    if run_status != "consistent":
        row["status"] = "mismatched"
        row["mismatch_reason"] = f"candidate_{run_status}"
        candidate_payload = _load_operator_contract_payload(run.get("run_dir", None), bundle=bundle)
        if isinstance(candidate_payload, Mapping):
            row["contract_sha256"] = _contract_payload_sha256(candidate_payload)
        return row, "mismatched"

    candidate_payload = _load_operator_contract_payload(run.get("run_dir", None), bundle=bundle)
    if not isinstance(candidate_payload, Mapping):
        row["status"] = "missing"
        return row, "missing"

    candidate_contract_sha256 = _contract_payload_sha256(candidate_payload)
    row["contract_sha256"] = candidate_contract_sha256
    if dict(candidate_payload) != dict(baseline_payload):
        row["status"] = "mismatched"
        row["mismatch_reason"] = "baseline_mismatch"
        return row, "mismatched"
    row["status"] = "matched"
    return row, "matched"


def _build_operator_contract_comparison(
    runs: list[dict[str, Any]],
    *,
    baseline_summary: dict[str, Any] | None,
    baseline_path_str: str | None,
    status_key: str,
    bundle: bool,
) -> dict[str, Any]:
    baseline_status = (
        _resolve_contract_status(baseline_summary, key=status_key)
        if isinstance(baseline_summary, Mapping)
        else "missing"
    )
    baseline_payload = (
        _load_operator_contract_payload(baseline_path_str, bundle=bundle)
        if baseline_status == "consistent"
        else None
    )
    checked = baseline_status == "consistent" and isinstance(baseline_payload, Mapping)
    baseline_contract_sha256 = (
        _contract_payload_sha256(baseline_payload)
        if checked and isinstance(baseline_payload, Mapping)
        else None
    )
    comparisons: list[dict[str, Any]] = []
    matched_runs = 0
    mismatched_runs = 0
    missing_runs = 0

    for run in runs:
        row, status = _operator_contract_row(
            run,
            baseline_payload=baseline_payload,
            baseline_contract_sha256=baseline_contract_sha256,
            baseline_path_str=baseline_path_str,
            checked=checked,
            status_key=status_key,
            bundle=bundle,
        )
        matched_runs, mismatched_runs, missing_runs = _bump_counts(
            status,
            matched_runs=matched_runs,
            mismatched_runs=mismatched_runs,
            missing_runs=missing_runs,
        )
        comparisons.append(row)

    return {
        "baseline_status": baseline_status,
        "baseline_contract_sha256": baseline_contract_sha256,
        "comparisons": comparisons,
        "summary": {
            "checked": bool(checked),
            "matched_runs": int(matched_runs),
            "mismatched_runs": int(mismatched_runs),
            "missing_runs": int(missing_runs),
            "incompatible_runs": int(mismatched_runs + missing_runs),
        },
    }


def _apply_items_filter(
    items: list[dict[str, Any]],
    *,
    value: str | None,
    item_value: callable,
) -> list[dict[str, Any]]:
    if value is None:
        return items
    needle = str(value).strip().lower()
    return [item for item in items if str(item_value(item)).lower() == needle]


def _filter_same_split(
    items: list[dict[str, Any]],
    *,
    same_split_as: str | Path | None,
) -> list[dict[str, Any]]:
    if same_split_as is None:
        return items
    target_report = _load_report_for_run_dir(same_split_as)
    target_sha256 = _extract_split_fingerprint_sha256(target_report)
    if target_sha256 is None:
        return []
    return [
        item
        for item in items
        if str(item.get("split_fingerprint_sha256", "")) == str(target_sha256)
    ]


def _filter_same_environment(
    items: list[dict[str, Any]],
    *,
    same_environment_as: str | Path | None,
) -> list[dict[str, Any]]:
    if same_environment_as is None:
        return items
    target_summary = summarize_run_dir(same_environment_as)
    target_fingerprint = _extract_environment_fingerprint_sha256(target_summary)
    if target_fingerprint is None:
        return []
    return [
        item
        for item in items
        if _extract_environment_fingerprint_sha256(item) == target_fingerprint
    ]


def _filter_same_target(
    items: list[dict[str, Any]],
    *,
    same_target_as: str | Path | None,
) -> list[dict[str, Any]]:
    if same_target_as is None:
        return items
    target_summary = summarize_run_dir(same_target_as)
    target_signature = _extract_target_signature(target_summary)
    if target_signature is None:
        return []
    return [item for item in items if _matches_target_signature(item, target_signature)]


def _filter_same_robustness_protocol(
    items: list[dict[str, Any]],
    *,
    same_robustness_protocol_as: str | Path | None,
) -> list[dict[str, Any]]:
    if same_robustness_protocol_as is None:
        return items
    target_summary = summarize_run_dir(same_robustness_protocol_as)
    target_signature = _extract_robustness_protocol_signature(target_summary)
    if target_signature is None:
        return []
    return [
        item for item in items if _extract_robustness_protocol_signature(item) == target_signature
    ]


def _load_run_summaries(root: Path) -> list[dict[str, Any]]:
    return [
        summarize_run_dir(path.parent)
        for path in sorted(root.rglob(_REPORT_JSON))
        if _is_top_level_report(path, root)
    ]


def _filter_query(items: list[dict[str, Any]], *, query: str | None) -> list[dict[str, Any]]:
    if query is None:
        return items
    needle = str(query).strip().lower()
    if not needle:
        return items
    return [
        item
        for item in items
        if needle in str(item.get("run_dir_name", "")).lower()
        or needle in str(item.get("model_or_suite", "")).lower()
        or needle in str(item.get("category", "")).lower()
    ]


def _filter_min_quality(
    items: list[dict[str, Any]],
    *,
    min_quality: str | None,
) -> list[dict[str, Any]]:
    if min_quality is None:
        return items
    minimum_rank = int(_QUALITY_STATUS_RANK.get(str(min_quality), -1))
    return [
        item
        for item in items
        if int(
            _QUALITY_STATUS_RANK.get(
                str(dict(item.get("artifact_quality", {})).get("status", "")),
                -1,
            )
        )
        >= minimum_rank
    ]


def _compare_metric_names(
    runs: Sequence[Mapping[str, Any]],
    *,
    baseline_summary: Mapping[str, Any] | None,
    metric: str | None,
) -> list[str]:
    metric_names = sorted(
        {
            key
            for run in ([baseline_summary] if baseline_summary is not None else []) + list(runs)
            for key, value in dict(run.get("metrics", {})).items()
            if isinstance(value, (int, float))
        }
    )
    if metric is not None:
        metric_names = [name for name in metric_names if name == str(metric)]
    return metric_names


def _metric_comparison_row(
    run: Mapping[str, Any],
    *,
    baseline_value: float | None,
    baseline_path_str: str | None,
    name: str,
    direction: str,
) -> tuple[dict[str, Any], int]:
    value = run.get("metrics", {}).get(name, None)
    row: dict[str, Any] = {
        "run_dir": run.get("run_dir"),
        "run_dir_name": run.get("run_dir_name"),
        "value": (float(value) if isinstance(value, (int, float)) else None),
        "delta_vs_baseline": None,
        "status": "missing",
    }
    if _is_baseline_run(run, baseline_path_str):
        row["status"] = "baseline"
        row["delta_vs_baseline"] = 0.0
        return row, 0
    if not isinstance(value, (int, float)) or baseline_value is None:
        return row, 0

    delta = round(float(value) - float(baseline_value), 12)
    row["delta_vs_baseline"] = delta
    if direction == "higher_is_better":
        if delta < 0.0:
            row["status"] = "regressed"
            return row, 1
        if delta > 0.0:
            row["status"] = "improved"
        else:
            row["status"] = "unchanged"
        return row, 0

    if delta > 0.0:
        row["status"] = "regressed"
        return row, 1
    if delta < 0.0:
        row["status"] = "improved"
    else:
        row["status"] = "unchanged"
    return row, 0


def _metric_info(
    name: str,
    *,
    runs: Sequence[Mapping[str, Any]],
    baseline_summary: Mapping[str, Any] | None,
    baseline_path_str: str | None,
) -> tuple[dict[str, Any] | None, int]:
    values = [
        float(run["metrics"][name])
        for run in runs
        if isinstance(run.get("metrics", {}).get(name), (int, float))
    ]
    if not values:
        return None, 0

    info: dict[str, Any] = {
        "values": values,
        "min": min(values),
        "max": max(values),
    }
    direction = _metric_direction(name)
    info["direction"] = direction

    if baseline_summary is None:
        return info, 0

    baseline_value_raw = baseline_summary.get("metrics", {}).get(name, None)
    baseline_value = (
        float(baseline_value_raw) if isinstance(baseline_value_raw, (int, float)) else None
    )
    info["baseline"] = baseline_value
    comparisons: list[dict[str, Any]] = []
    regressions = 0
    for run in runs:
        row, row_regressions = _metric_comparison_row(
            run,
            baseline_value=baseline_value,
            baseline_path_str=baseline_path_str,
            name=name,
            direction=direction,
        )
        comparisons.append(row)
        regressions += int(row_regressions)
    info["comparisons"] = comparisons
    info["regression_count"] = int(regressions)
    return info, regressions


def _comparison_blocks(
    runs: list[dict[str, Any]],
    *,
    baseline_summary: dict[str, Any] | None,
    baseline_path_str: str | None,
) -> dict[str, Any]:
    return {
        "split_comparison": _build_split_comparison(
            runs,
            baseline_summary=baseline_summary,
            baseline_path_str=baseline_path_str,
        ),
        "environment_comparison": _build_environment_comparison(
            runs,
            baseline_summary=baseline_summary,
            baseline_path_str=baseline_path_str,
        ),
        "target_comparison": _build_target_comparison(
            runs,
            baseline_summary=baseline_summary,
            baseline_path_str=baseline_path_str,
        ),
        "robustness_protocol_comparison": _build_robustness_protocol_comparison(
            runs,
            baseline_summary=baseline_summary,
            baseline_path_str=baseline_path_str,
        ),
        "operator_contract_comparison": _build_operator_contract_comparison(
            runs,
            baseline_summary=baseline_summary,
            baseline_path_str=baseline_path_str,
            status_key="operator_contract_status",
            bundle=False,
        ),
        "bundle_operator_contract_comparison": _build_operator_contract_comparison(
            runs,
            baseline_summary=baseline_summary,
            baseline_path_str=baseline_path_str,
            status_key="bundle_operator_contract_status",
            bundle=True,
        ),
    }


def _primary_metric_summary_fields(
    summary: dict[str, Any],
    *,
    baseline_summary: Mapping[str, Any] | None,
    runs: Sequence[Mapping[str, Any]],
    baseline_path_str: str | None,
    evaluation_contract: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
    trust_comparison: Mapping[str, Any],
    operator_contract_comparison: Mapping[str, Any],
    bundle_operator_contract_comparison: Mapping[str, Any],
    candidate_blocking_summary: Mapping[str, Any],
) -> None:
    primary_metric_name, primary_metric_info = _primary_metric_details(
        evaluation_contract,
        metrics,
    )
    summary["primary_metric"] = (
        primary_metric_name
        if isinstance(primary_metric_name, str)
        and primary_metric_name
        and bool(primary_metric_info)
        else None
    )
    summary["primary_metric_direction"] = (
        primary_metric_info.get("direction")
        if isinstance(primary_metric_info.get("direction"), str)
        else None
    )
    summary["primary_metric_baseline"] = (
        float(primary_metric_info["baseline"])
        if isinstance(primary_metric_info.get("baseline"), (int, float))
        else None
    )
    summary["primary_metric_total_regressions"] = (
        int(primary_metric_info.get("regression_count", 0)) if bool(primary_metric_info) else None
    )
    primary_metric_statuses, primary_metric_deltas = _primary_metric_candidate_maps(
        primary_metric_info
    )
    summary["primary_metric_statuses"] = primary_metric_statuses
    summary["primary_metric_deltas"] = primary_metric_deltas
    _apply_trust_contract_summary_fields(
        summary,
        trust_comparison=trust_comparison,
        operator_contract_comparison=operator_contract_comparison,
        bundle_operator_contract_comparison=bundle_operator_contract_comparison,
    )
    _apply_candidate_blocking_summary_fields(
        summary,
        candidate_blocking_summary=candidate_blocking_summary,
    )
    baseline_dataset_readiness = _dataset_readiness_payload(baseline_summary)
    candidate_dataset_readiness = _candidate_dataset_readiness_by_name(
        runs,
        baseline_path_str=baseline_path_str,
    )
    summary["baseline_dataset_readiness"] = baseline_dataset_readiness
    summary["candidate_dataset_readiness"] = candidate_dataset_readiness
    summary["candidate_incompatibility_digest"] = _build_candidate_incompatibility_digest(
        candidate_verdicts=summary["candidate_verdicts"],
        candidate_blocking_reasons=summary["candidate_blocking_reasons"],
        candidate_comparability_gates=summary["candidate_comparability_gates"],
        candidate_dataset_readiness=candidate_dataset_readiness,
    )


def _primary_metric_details(
    evaluation_contract: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
) -> tuple[str | None, dict[str, Any]]:
    primary_metric_name = evaluation_contract.get("primary_metric", None)
    if not isinstance(primary_metric_name, str) or primary_metric_name not in metrics:
        return None, {}
    return primary_metric_name, dict(metrics.get(primary_metric_name, {}))


def _primary_metric_candidate_maps(
    primary_metric_info: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, float]]:
    primary_metric_statuses: dict[str, str] = {}
    primary_metric_deltas: dict[str, float] = {}
    for row in primary_metric_info.get("comparisons", []):
        if not isinstance(row, Mapping) or str(row.get("status")) == "baseline":
            continue
        run_dir_name = row.get("run_dir_name", None)
        if not isinstance(run_dir_name, str) or not run_dir_name:
            continue
        status = row.get("status", None)
        if isinstance(status, str) and status:
            primary_metric_statuses[run_dir_name] = status
        delta = row.get("delta_vs_baseline", None)
        if isinstance(delta, (int, float)):
            primary_metric_deltas[run_dir_name] = float(delta)
    return primary_metric_statuses, primary_metric_deltas


def _apply_trust_contract_summary_fields(
    summary: dict[str, Any],
    *,
    trust_comparison: Mapping[str, Any],
    operator_contract_comparison: Mapping[str, Any],
    bundle_operator_contract_comparison: Mapping[str, Any],
) -> None:
    summary["trust_checked"] = bool(trust_comparison.get("checked"))
    summary["trust_gate"] = trust_comparison.get("gate", None)
    summary["trust_status"] = trust_comparison.get("status", None)
    summary["trust_reason"] = trust_comparison.get("reason", None)
    summary["operator_contract_gate"] = _comparability_gate_status(
        dict(operator_contract_comparison.get("summary", {}))
    )
    summary["bundle_operator_contract_gate"] = _comparability_gate_status(
        dict(bundle_operator_contract_comparison.get("summary", {}))
    )
    summary["operator_contract_baseline_sha256"] = operator_contract_comparison.get(
        "baseline_contract_sha256",
        None,
    )
    summary["bundle_operator_contract_baseline_sha256"] = bundle_operator_contract_comparison.get(
        "baseline_contract_sha256",
        None,
    )
    summary["operator_contract_status"] = trust_comparison.get("operator_contract_status", None)
    summary["operator_contract_consistent"] = bool(
        trust_comparison.get("operator_contract_consistent", False)
    )
    summary["bundle_operator_contract_status"] = trust_comparison.get(
        "bundle_operator_contract_status",
        None,
    )
    summary["bundle_operator_contract_consistent"] = bool(
        trust_comparison.get("bundle_operator_contract_consistent", False)
    )
    summary["bundle_operator_contract_digests_valid"] = bool(
        trust_comparison.get("bundle_operator_contract_digests_valid", False)
    )


def _apply_candidate_blocking_summary_fields(
    summary: dict[str, Any],
    *,
    candidate_blocking_summary: Mapping[str, Any],
) -> None:
    summary["candidate_verdicts"] = dict(candidate_blocking_summary["candidate_verdicts"])
    summary["candidate_blocking_reasons"] = dict(
        candidate_blocking_summary["candidate_blocking_reasons"]
    )
    summary["candidate_comparability_gates"] = dict(
        candidate_blocking_summary["candidate_comparability_gates"]
    )
    summary["candidate_bundle_operator_contract_digest_statuses"] = dict(
        candidate_blocking_summary["candidate_bundle_operator_contract_digest_statuses"]
    )


def latest_run_summary(
    root: str | Path,
    *,
    kind: str | None = None,
    dataset: str | None = None,
    query: str | None = None,
    min_quality: str | None = None,
    same_split_as: str | Path | None = None,
    same_environment_as: str | Path | None = None,
    same_target_as: str | Path | None = None,
    same_robustness_protocol_as: str | Path | None = None,
) -> dict[str, Any] | None:
    items = list_run_summaries(
        root,
        limit=1,
        kind=kind,
        dataset=dataset,
        query=query,
        min_quality=min_quality,
        same_split_as=same_split_as,
        same_environment_as=same_environment_as,
        same_target_as=same_target_as,
        same_robustness_protocol_as=same_robustness_protocol_as,
    )
    if not items:
        return None
    return dict(items[0])


def _metric_direction(name: str) -> str:
    if str(name) in _LOWER_IS_BETTER_METRICS:
        return "lower_is_better"
    if str(name) in _HIGHER_IS_BETTER_METRICS:
        return "higher_is_better"
    return "higher_is_better"


def _resolved_run_dir(run: Mapping[str, Any]) -> str:
    return str(Path(str(run.get("run_dir"))).resolve())


def _is_baseline_run(run: Mapping[str, Any], baseline_path_str: str | None) -> bool:
    return baseline_path_str is not None and _resolved_run_dir(run) == baseline_path_str


def _bump_counts(
    status: str,
    *,
    matched_runs: int,
    mismatched_runs: int,
    missing_runs: int,
) -> tuple[int, int, int]:
    if status == "matched":
        matched_runs += 1
    elif status == "mismatched":
        mismatched_runs += 1
    elif status == "missing":
        missing_runs += 1
    return matched_runs, mismatched_runs, missing_runs


def _split_comparison_row(
    run: Mapping[str, Any],
    *,
    baseline_sha256: str | None,
    checked: bool,
    baseline_path_str: str | None,
) -> tuple[dict[str, Any], str]:
    run_sha256 = run.get("split_fingerprint_sha256", None)
    row: dict[str, Any] = {
        "run_dir": run.get("run_dir"),
        "run_dir_name": run.get("run_dir_name"),
        "split_fingerprint_sha256": (
            str(run_sha256) if isinstance(run_sha256, str) and run_sha256 else None
        ),
        "status": "unchecked",
    }
    if _is_baseline_run(run, baseline_path_str):
        row["status"] = "baseline"
        return row, "baseline"
    if not checked:
        return row, "unchecked"
    if row["split_fingerprint_sha256"] is None:
        row["status"] = "missing"
    elif row["split_fingerprint_sha256"] == baseline_sha256:
        row["status"] = "matched"
    else:
        row["status"] = "mismatched"
    return row, str(row["status"])


def _build_split_comparison(
    runs: list[dict[str, Any]],
    *,
    baseline_summary: dict[str, Any] | None,
    baseline_path_str: str | None,
) -> dict[str, Any]:
    baseline_sha256 = None
    if baseline_summary is not None:
        value = baseline_summary.get("split_fingerprint_sha256", None)
        if isinstance(value, str) and value:
            baseline_sha256 = value

    checked = baseline_sha256 is not None
    comparisons: list[dict[str, Any]] = []
    matched_runs = 0
    mismatched_runs = 0
    missing_runs = 0
    for run in runs:
        row, status = _split_comparison_row(
            run,
            baseline_sha256=baseline_sha256,
            checked=checked,
            baseline_path_str=baseline_path_str,
        )
        matched_runs, mismatched_runs, missing_runs = _bump_counts(
            status,
            matched_runs=matched_runs,
            mismatched_runs=mismatched_runs,
            missing_runs=missing_runs,
        )
        comparisons.append(row)

    return {
        "baseline_split_fingerprint_sha256": baseline_sha256,
        "comparisons": comparisons,
        "summary": {
            "checked": bool(checked),
            "matched_runs": int(matched_runs),
            "mismatched_runs": int(mismatched_runs),
            "missing_runs": int(missing_runs),
            "incompatible_runs": int(mismatched_runs + missing_runs),
        },
    }


def _match_status(actual: Any, expected: Any) -> str:
    if expected is None:
        return "unchecked"
    if actual is None:
        return "missing"
    if str(actual) == str(expected):
        return "matched"
    return "mismatched"


def _target_comparison_row(
    run: Mapping[str, Any],
    *,
    baseline_dataset: str | None,
    baseline_category: str | None,
    dataset_checked: bool,
    category_checked: bool,
    checked: bool,
    baseline_path_str: str | None,
) -> tuple[dict[str, Any], str]:
    run_dataset = run.get("dataset", None)
    run_category = run.get("category", None)
    dataset_status = _match_status(run_dataset, baseline_dataset)
    category_status = _match_status(run_category, baseline_category)

    row: dict[str, Any] = {
        "run_dir": run.get("run_dir"),
        "run_dir_name": run.get("run_dir_name"),
        "dataset": (str(run_dataset) if isinstance(run_dataset, str) and run_dataset else None),
        "category": (str(run_category) if isinstance(run_category, str) and run_category else None),
        "dataset_status": dataset_status,
        "category_status": category_status,
        "status": "unchecked",
    }

    if _is_baseline_run(run, baseline_path_str):
        row["status"] = "baseline"
        return row, "baseline"
    if not checked:
        return row, "unchecked"

    statuses = [
        status
        for status, is_checked in (
            (dataset_status, dataset_checked),
            (category_status, category_checked),
        )
        if is_checked
    ]
    if any(status == "mismatched" for status in statuses):
        row["status"] = "mismatched"
    elif any(status == "missing" for status in statuses):
        row["status"] = "missing"
    else:
        row["status"] = "matched"
    return row, str(row["status"])


def _build_target_comparison(
    runs: list[dict[str, Any]],
    *,
    baseline_summary: dict[str, Any] | None,
    baseline_path_str: str | None,
) -> dict[str, Any]:
    baseline_dataset = None
    baseline_category = None
    if baseline_summary is not None:
        dataset_value = baseline_summary.get("dataset", None)
        category_value = baseline_summary.get("category", None)
        baseline_dataset = (
            str(dataset_value) if isinstance(dataset_value, str) and dataset_value else None
        )
        baseline_category = (
            str(category_value) if isinstance(category_value, str) and category_value else None
        )

    dataset_checked = baseline_dataset is not None
    category_checked = baseline_category is not None
    checked = bool(dataset_checked or category_checked)

    comparisons: list[dict[str, Any]] = []
    matched_runs = 0
    mismatched_runs = 0
    missing_runs = 0
    for run in runs:
        row, status = _target_comparison_row(
            run,
            baseline_dataset=baseline_dataset,
            baseline_category=baseline_category,
            dataset_checked=dataset_checked,
            category_checked=category_checked,
            checked=checked,
            baseline_path_str=baseline_path_str,
        )
        matched_runs, mismatched_runs, missing_runs = _bump_counts(
            status,
            matched_runs=matched_runs,
            mismatched_runs=mismatched_runs,
            missing_runs=missing_runs,
        )
        comparisons.append(row)

    return {
        "baseline": {"dataset": baseline_dataset, "category": baseline_category},
        "comparisons": comparisons,
        "summary": {
            "checked": checked,
            "dataset_checked": bool(dataset_checked),
            "category_checked": bool(category_checked),
            "matched_runs": int(matched_runs),
            "mismatched_runs": int(mismatched_runs),
            "missing_runs": int(missing_runs),
            "incompatible_runs": int(mismatched_runs + missing_runs),
        },
    }


def _environment_comparison_row(
    run: Mapping[str, Any],
    *,
    baseline_fingerprint: str | None,
    checked: bool,
    baseline_path_str: str | None,
) -> tuple[dict[str, Any], str]:
    run_fingerprint = run.get("environment_fingerprint_sha256", None)
    row: dict[str, Any] = {
        "run_dir": run.get("run_dir"),
        "run_dir_name": run.get("run_dir_name"),
        "environment_fingerprint_sha256": (
            str(run_fingerprint) if isinstance(run_fingerprint, str) and run_fingerprint else None
        ),
        "status": "unchecked",
    }
    if _is_baseline_run(run, baseline_path_str):
        row["status"] = "baseline"
        return row, "baseline"
    if not checked:
        return row, "unchecked"
    if row["environment_fingerprint_sha256"] is None:
        row["status"] = "missing"
    elif row["environment_fingerprint_sha256"] == baseline_fingerprint:
        row["status"] = "matched"
    else:
        row["status"] = "mismatched"
    return row, str(row["status"])


def _build_environment_comparison(
    runs: list[dict[str, Any]],
    *,
    baseline_summary: dict[str, Any] | None,
    baseline_path_str: str | None,
) -> dict[str, Any]:
    baseline_fingerprint = None
    if baseline_summary is not None:
        value = baseline_summary.get("environment_fingerprint_sha256", None)
        if isinstance(value, str) and value:
            baseline_fingerprint = value

    checked = baseline_fingerprint is not None
    comparisons: list[dict[str, Any]] = []
    matched_runs = 0
    mismatched_runs = 0
    missing_runs = 0
    for run in runs:
        row, status = _environment_comparison_row(
            run,
            baseline_fingerprint=baseline_fingerprint,
            checked=checked,
            baseline_path_str=baseline_path_str,
        )
        matched_runs, mismatched_runs, missing_runs = _bump_counts(
            status,
            matched_runs=matched_runs,
            mismatched_runs=mismatched_runs,
            missing_runs=missing_runs,
        )
        comparisons.append(row)

    return {
        "baseline_environment_fingerprint_sha256": baseline_fingerprint,
        "comparisons": comparisons,
        "summary": {
            "checked": bool(checked),
            "matched_runs": int(matched_runs),
            "mismatched_runs": int(mismatched_runs),
            "missing_runs": int(missing_runs),
            "incompatible_runs": int(mismatched_runs + missing_runs),
        },
    }


def _is_top_level_report(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    banned = {"categories", "models", "variants", "deploy_bundle"}
    return not any(part in banned for part in rel.parts[:-1])


def list_run_summaries(
    root: str | Path,
    *,
    limit: int | None = None,
    kind: str | None = None,
    dataset: str | None = None,
    query: str | None = None,
    min_quality: str | None = None,
    same_split_as: str | Path | None = None,
    same_environment_as: str | Path | None = None,
    same_target_as: str | Path | None = None,
    same_robustness_protocol_as: str | Path | None = None,
) -> list[dict[str, Any]]:
    items = _load_run_summaries(Path(root))
    items = _apply_items_filter(
        items,
        value=kind,
        item_value=lambda item: item.get("kind", ""),
    )
    items = _apply_items_filter(
        items,
        value=dataset,
        item_value=lambda item: item.get("dataset", ""),
    )
    items = _filter_query(items, query=query)
    items = _filter_min_quality(items, min_quality=min_quality)
    items = _filter_same_split(items, same_split_as=same_split_as)
    items = _filter_same_environment(items, same_environment_as=same_environment_as)
    items = _filter_same_target(items, same_target_as=same_target_as)
    items = _filter_same_robustness_protocol(
        items,
        same_robustness_protocol_as=same_robustness_protocol_as,
    )
    items.sort(
        key=lambda item: (
            str(item.get("timestamp_utc") or ""),
            str(item.get("run_dir_name") or ""),
        ),
        reverse=True,
    )
    if limit is not None:
        items = items[: int(limit)]
    return items


def compare_run_summaries(
    run_dirs: Iterable[str | Path],
    *,
    baseline_run_dir: str | Path | None = None,
    metric: str | None = None,
) -> dict[str, Any]:
    run_paths = [Path(path) for path in run_dirs]
    runs = [summarize_run_dir(path) for path in run_paths]
    baseline_summary = summarize_run_dir(baseline_run_dir) if baseline_run_dir is not None else None
    metric_names = _compare_metric_names(
        runs,
        baseline_summary=baseline_summary,
        metric=metric,
    )
    baseline_path_str = (
        str(Path(baseline_run_dir).resolve()) if baseline_run_dir is not None else None
    )
    comparisons = _comparison_blocks(
        runs,
        baseline_summary=baseline_summary,
        baseline_path_str=baseline_path_str,
    )
    split_comparison = comparisons["split_comparison"]
    environment_comparison = comparisons["environment_comparison"]
    target_comparison = comparisons["target_comparison"]
    robustness_protocol_comparison = comparisons["robustness_protocol_comparison"]
    operator_contract_comparison = comparisons["operator_contract_comparison"]
    bundle_operator_contract_comparison = comparisons["bundle_operator_contract_comparison"]
    metrics, total_regressions = _collect_metric_infos(
        metric_names,
        runs=runs,
        baseline_summary=baseline_summary,
        baseline_path_str=baseline_path_str,
    )
    comparison_summaries = _comparison_summary_maps(comparisons)
    baseline_checked = baseline_summary is not None
    summary = _build_compare_summary(
        baseline_checked=bool(baseline_checked),
        total_regressions=total_regressions,
        comparison_summaries=comparison_summaries,
    )
    trust_comparison = _build_trust_comparison(baseline_summary)
    evaluation_contract = _build_summary_evaluation_contract(
        metric_names,
        baseline_summary=baseline_summary,
    )
    primary_metric_name = evaluation_contract.get("primary_metric", None)
    primary_metric_info = (
        dict(metrics.get(primary_metric_name, {}))
        if isinstance(primary_metric_name, str) and primary_metric_name in metrics
        else {}
    )
    candidate_blocking_summary = _build_candidate_blocking_summary(
        runs=runs,
        baseline_path_str=baseline_path_str,
        baseline_operator_contract_status=_baseline_summary_status(
            baseline_summary,
            key="operator_contract_status",
        ),
        baseline_bundle_operator_contract_status=_baseline_summary_status(
            baseline_summary,
            key="bundle_operator_contract_status",
        ),
        primary_metric_info=primary_metric_info,
        split_comparison=split_comparison,
        environment_comparison=environment_comparison,
        target_comparison=target_comparison,
        robustness_protocol_comparison=robustness_protocol_comparison,
    )
    _primary_metric_summary_fields(
        summary,
        baseline_summary=baseline_summary,
        runs=runs,
        baseline_path_str=baseline_path_str,
        evaluation_contract=evaluation_contract,
        metrics=metrics,
        trust_comparison=trust_comparison,
        operator_contract_comparison=operator_contract_comparison,
        bundle_operator_contract_comparison=bundle_operator_contract_comparison,
        candidate_blocking_summary=candidate_blocking_summary,
    )

    return {
        "runs": runs,
        "baseline_run": baseline_summary,
        "trust_comparison": trust_comparison,
        "split_comparison": split_comparison,
        "environment_comparison": environment_comparison,
        "target_comparison": target_comparison,
        "robustness_protocol_comparison": robustness_protocol_comparison,
        "operator_contract_comparison": operator_contract_comparison,
        "bundle_operator_contract_comparison": bundle_operator_contract_comparison,
        "metrics": metrics,
        "evaluation_contract": evaluation_contract,
        "summary": summary,
    }


def _collect_metric_infos(
    metric_names: list[str],
    *,
    runs: list[dict[str, Any]],
    baseline_summary: dict[str, Any] | None,
    baseline_path_str: str | None,
) -> tuple[dict[str, dict[str, Any]], int]:
    metrics: dict[str, dict[str, Any]] = {}
    total_regressions = 0
    for name in metric_names:
        info, regressions = _metric_info(
            name,
            runs=runs,
            baseline_summary=baseline_summary,
            baseline_path_str=baseline_path_str,
        )
        if info is None:
            continue
        metrics[name] = info
        total_regressions += int(regressions)
    return metrics, total_regressions


def _comparison_summary_maps(
    comparisons: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        "split": dict(comparisons["split_comparison"].get("summary", {})),
        "environment": dict(comparisons["environment_comparison"].get("summary", {})),
        "target": dict(comparisons["target_comparison"].get("summary", {})),
        "robustness_protocol": dict(
            comparisons["robustness_protocol_comparison"].get("summary", {})
        ),
        "operator_contract": dict(comparisons["operator_contract_comparison"].get("summary", {})),
        "bundle_operator_contract": dict(
            comparisons["bundle_operator_contract_comparison"].get("summary", {})
        ),
    }


def _build_compare_summary(
    *,
    baseline_checked: bool,
    total_regressions: int,
    comparison_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    blocking_flags = (
        _compare_blocking_flags(
            total_regressions=total_regressions,
            split_summary=comparison_summaries["split"],
            environment_summary=comparison_summaries["environment"],
            target_summary=comparison_summaries["target"],
            robustness_protocol_summary=comparison_summaries["robustness_protocol"],
            operator_contract_summary=comparison_summaries["operator_contract"],
            bundle_operator_contract_summary=comparison_summaries["bundle_operator_contract"],
        )
        if baseline_checked
        else []
    )
    if baseline_checked:
        regression_gate = "clean" if int(total_regressions) == 0 else "regressed"
        verdict = "pass" if not blocking_flags else "blocked"
    else:
        regression_gate = "unchecked"
        verdict = "informational"
    return {
        "baseline_checked": baseline_checked,
        "total_regressions": int(total_regressions),
        "regression_gate": regression_gate,
        "comparability_gates": {
            gate_name: _comparability_gate_status(summary)
            for gate_name, summary in comparison_summaries.items()
        },
        "blocking_flags": blocking_flags,
        "verdict": verdict,
    }


def _baseline_evaluation_contract(
    baseline_summary: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not isinstance(baseline_summary, Mapping):
        return {}
    value = baseline_summary.get("evaluation_contract", {})
    return value if isinstance(value, Mapping) else {}


def _build_summary_evaluation_contract(
    metric_names: list[str],
    *,
    baseline_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    baseline_contract = _baseline_evaluation_contract(baseline_summary)
    comparability_hints = baseline_contract.get("comparability_hints", None)
    return build_evaluation_contract(
        metric_names=metric_names,
        primary_metric=(
            str(baseline_contract.get("primary_metric"))
            if baseline_contract.get("primary_metric") is not None
            else "auroc"
        ),
        ranking_metric=(
            str(baseline_contract.get("ranking_metric"))
            if baseline_contract.get("ranking_metric") is not None
            else "auroc"
        ),
        pixel_metrics_enabled=baseline_contract.get("pixel_metrics_enabled"),
        comparability_hints=(
            dict(comparability_hints) if isinstance(comparability_hints, Mapping) else None
        ),
    )


def _baseline_summary_status(
    baseline_summary: Mapping[str, Any] | None,
    *,
    key: str,
) -> str | None:
    if not isinstance(baseline_summary, Mapping):
        return None
    value = baseline_summary.get(key)
    return str(value) if value is not None else None


__all__ = [
    "compare_run_summaries",
    "latest_run_summary",
    "list_run_summaries",
    "summarize_run_dir",
]
