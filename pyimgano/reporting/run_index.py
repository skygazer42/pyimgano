from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from pyimgano.reporting.evaluation_contract import build_evaluation_contract
from pyimgano.reporting.run_quality import evaluate_run_quality
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
    status_text = str(trust_status) if trust_status is not None else ""
    if not status_text:
        return None
    return "trusted" if status_text == "trust-signaled" else "limited"


def _operator_contract_status_from_trust_summary(trust_summary: Mapping[str, Any]) -> tuple[str, bool]:
    trust_signals = trust_summary.get("trust_signals", None)
    signal_map = dict(trust_signals) if isinstance(trust_signals, Mapping) else {}
    has_contract = bool(signal_map.get("has_operator_contract"))
    is_consistent = bool(signal_map.get("has_operator_contract_consistent"))
    if not has_contract:
        return "missing", False
    return ("consistent" if is_consistent else "mismatched"), bool(is_consistent)


def _bundle_operator_contract_status_from_trust_summary(
    trust_summary: Mapping[str, Any],
) -> tuple[str, bool]:
    trust_signals = trust_summary.get("trust_signals", None)
    signal_map = dict(trust_signals) if isinstance(trust_signals, Mapping) else {}
    has_contract = bool(signal_map.get("has_bundle_operator_contract"))
    is_consistent = bool(signal_map.get("has_bundle_operator_contract_consistent"))
    if not has_contract:
        return "missing", False
    return ("consistent" if is_consistent else "mismatched"), bool(is_consistent)


def _comparison_trust_reason(
    *,
    trust_status: object,
    quality_status: object,
    status_reasons: list[object],
    degraded_by: list[object],
) -> str | None:
    reasons = [str(item) for item in status_reasons if str(item)]
    degradations = [str(item) for item in degraded_by if str(item)]
    if str(trust_status) == "trust-signaled":
        if "calibration_audit_consistent" in reasons:
            return "calibration_audit_consistent"
        if reasons:
            return reasons[0]
    if degradations:
        return degradations[0]
    if str(quality_status) in {"reproducible", "partial"}:
        return "calibration_audit_incomplete"
    if reasons:
        return reasons[0]
    status_text = str(trust_status)
    return status_text if status_text else None


def _build_trust_comparison(
    baseline_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(baseline_summary, Mapping):
        return {
            "checked": False,
            "quality_status": None,
            "gate": None,
            "status": None,
            "reason": None,
            "status_reasons": [],
            "degraded_by": [],
            "audit_refs": {},
        }

    artifact_quality = dict(baseline_summary.get("artifact_quality", {}))
    trust_summary = dict(artifact_quality.get("trust_summary", {}))
    trust_signals = trust_summary.get("trust_signals", None)
    signal_map = dict(trust_signals) if isinstance(trust_signals, Mapping) else {}
    operator_contract_status, operator_contract_consistent = (
        _operator_contract_status_from_trust_summary(trust_summary)
    )
    bundle_operator_contract_status, bundle_operator_contract_consistent = (
        _bundle_operator_contract_status_from_trust_summary(trust_summary)
    )
    baseline_operator_contract_status = baseline_summary.get("operator_contract_status", None)
    if isinstance(baseline_operator_contract_status, str) and baseline_operator_contract_status:
        operator_contract_status = str(baseline_operator_contract_status)
        operator_contract_consistent = operator_contract_status == "consistent"
    baseline_bundle_operator_contract_status = baseline_summary.get(
        "bundle_operator_contract_status",
        None,
    )
    if (
        isinstance(baseline_bundle_operator_contract_status, str)
        and baseline_bundle_operator_contract_status
    ):
        bundle_operator_contract_status = str(baseline_bundle_operator_contract_status)
        bundle_operator_contract_consistent = bundle_operator_contract_status == "consistent"
    quality_status = artifact_quality.get("status", None)
    trust_status = trust_summary.get("status", None)
    status_reasons = list(trust_summary.get("status_reasons", []))
    degraded_by = list(trust_summary.get("degraded_by", []))
    audit_refs = dict(trust_summary.get("audit_refs", {}))
    return {
        "checked": True,
        "quality_status": (
            str(quality_status) if isinstance(quality_status, str) and quality_status else None
        ),
        "gate": _comparison_trust_gate(trust_status),
        "status": (str(trust_status) if isinstance(trust_status, str) and trust_status else None),
        "reason": _comparison_trust_reason(
            trust_status=trust_status,
            quality_status=quality_status,
            status_reasons=status_reasons,
            degraded_by=degraded_by,
        ),
        "status_reasons": [str(item) for item in status_reasons if str(item)],
        "degraded_by": [str(item) for item in degraded_by if str(item)],
        "operator_contract_status": str(operator_contract_status),
        "operator_contract_consistent": bool(operator_contract_consistent),
        "bundle_operator_contract_status": str(bundle_operator_contract_status),
        "bundle_operator_contract_consistent": bool(bundle_operator_contract_consistent),
        "bundle_operator_contract_digests_valid": bool(
            signal_map.get("has_bundle_operator_contract_digests_valid", False)
        ),
        "audit_refs": {
            str(key): str(value)
            for key, value in audit_refs.items()
            if str(key) and str(value)
        },
    }


def _comparability_gate_status(summary: Mapping[str, Any]) -> str:
    if not bool(summary.get("checked")):
        return "unchecked"
    if int(summary.get("incompatible_runs", 0) or 0) > 0:
        return "incompatible"
    return "compatible"


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
    flags: list[str] = []
    if int(total_regressions) > 0:
        flags.append("--fail-on-regression")
    if _comparability_gate_status(split_summary) == "incompatible":
        flags.append("--require-same-split")
    if _comparability_gate_status(environment_summary) == "incompatible":
        flags.append("--require-same-environment")
    if _comparability_gate_status(target_summary) == "incompatible":
        flags.append("--require-same-target")
    if _comparability_gate_status(robustness_protocol_summary) == "incompatible":
        flags.append("--require-same-robustness-protocol")
    if _comparability_gate_status(operator_contract_summary) == "incompatible":
        flags.append("--require-same-operator-contract")
    if _comparability_gate_status(bundle_operator_contract_summary) == "incompatible":
        flags.append("--require-same-bundle-operator-contract")
    return flags


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


def _build_candidate_incompatibility_digest(
    *,
    candidate_verdicts: Mapping[str, Any],
    candidate_blocking_reasons: Mapping[str, Any],
    candidate_comparability_gates: Mapping[str, Any],
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
        reasons = [str(reason) for reason in list(reasons_raw) if str(reason)]
        gate_states = candidate_comparability_gates.get(name, None)
        gate_map = dict(gate_states) if isinstance(gate_states, Mapping) else {}
        incompatible_gates: list[str] = []
        for gate_name in _CANDIDATE_COMPARABILITY_GATE_ORDER:
            status = gate_map.get(gate_name, None)
            status_text = str(status).strip().lower() if status is not None else ""
            if status_text in {"missing", "mismatched"}:
                incompatible_gates.append(f"{gate_name}:{status_text}")
        digest[name] = {
            "verdict": verdict,
            "incompatible_gates": incompatible_gates,
            "blocking_reasons": reasons,
        }
    return digest


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
    reasons_by_name: dict[str, list[str]] = {name: [] for name in candidate_names}
    candidate_gates: dict[str, dict[str, str]] = {
        name: {
            "split": "unchecked",
            "environment": "unchecked",
            "target": "unchecked",
            "target_dataset": "unchecked",
            "target_category": "unchecked",
            "robustness_protocol": "unchecked",
            "operator_contract": "unchecked",
            "bundle_operator_contract": "unchecked",
        }
        for name in candidate_names
    }
    candidate_bundle_digest_statuses: dict[str, str] = {
        name: "unavailable" for name in candidate_names
    }
    baseline_operator_contract_payload = (
        _load_operator_contract_payload(baseline_path_str, bundle=False)
        if str(baseline_operator_contract_status or "").strip().lower() == "consistent"
        else None
    )
    baseline_bundle_operator_contract_payload = (
        _load_operator_contract_payload(baseline_path_str, bundle=True)
        if str(baseline_bundle_operator_contract_status or "").strip().lower() == "consistent"
        else None
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

    for row in split_comparison.get("comparisons", []):
        if not isinstance(row, Mapping):
            continue
        run_dir_name = row.get("run_dir_name", None)
        status = str(row.get("status", ""))
        if isinstance(run_dir_name, str) and run_dir_name in candidate_gates and status:
            candidate_gates[run_dir_name]["split"] = status
        if status in {"mismatched", "missing"}:
            _append_candidate_reason(
                reasons_by_name,
                run_dir_name=run_dir_name,
                reason=f"split:{status}",
            )

    for row in environment_comparison.get("comparisons", []):
        if not isinstance(row, Mapping):
            continue
        run_dir_name = row.get("run_dir_name", None)
        status = str(row.get("status", ""))
        if isinstance(run_dir_name, str) and run_dir_name in candidate_gates and status:
            candidate_gates[run_dir_name]["environment"] = status
        if status in {"mismatched", "missing"}:
            _append_candidate_reason(
                reasons_by_name,
                run_dir_name=run_dir_name,
                reason=f"environment:{status}",
            )

    for row in target_comparison.get("comparisons", []):
        if not isinstance(row, Mapping):
            continue
        run_dir_name = row.get("run_dir_name", None)
        row_status = str(row.get("status", ""))
        dataset_status = str(row.get("dataset_status", ""))
        category_status = str(row.get("category_status", ""))
        if isinstance(run_dir_name, str) and run_dir_name in candidate_gates:
            if row_status:
                candidate_gates[run_dir_name]["target"] = row_status
            if dataset_status:
                candidate_gates[run_dir_name]["target_dataset"] = dataset_status
            if category_status:
                candidate_gates[run_dir_name]["target_category"] = category_status
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

    for row in robustness_protocol_comparison.get("comparisons", []):
        if not isinstance(row, Mapping):
            continue
        run_dir_name = row.get("run_dir_name", None)
        status = str(row.get("status", ""))
        if isinstance(run_dir_name, str) and run_dir_name in candidate_gates and status:
            candidate_gates[run_dir_name]["robustness_protocol"] = status
        if status == "missing":
            _append_candidate_reason(
                reasons_by_name,
                run_dir_name=run_dir_name,
                reason="robustness_protocol:missing",
            )
            continue
        if status != "mismatched":
            continue
        mismatch_fields = row.get("mismatch_fields", None)
        if isinstance(mismatch_fields, list) and mismatch_fields:
            for field in mismatch_fields:
                field_name = str(field)
                if field_name:
                    _append_candidate_reason(
                        reasons_by_name,
                        run_dir_name=run_dir_name,
                        reason=f"robustness_protocol.{field_name}:mismatched",
                    )
        else:
            _append_candidate_reason(
                reasons_by_name,
                run_dir_name=run_dir_name,
                reason="robustness_protocol:mismatched",
            )

    if str(baseline_operator_contract_status or "").strip().lower() == "consistent":
        for run in runs:
            run_path = str(Path(str(run.get("run_dir"))).resolve())
            if baseline_path_str is not None and run_path == baseline_path_str:
                continue
            run_dir_name = run.get("run_dir_name", None)
            status = str(run.get("operator_contract_status", "missing") or "missing").strip().lower()
            if isinstance(run_dir_name, str) and run_dir_name in candidate_gates:
                candidate_gates[run_dir_name]["operator_contract"] = status
            if status in {"missing", "mismatched"}:
                _append_candidate_reason(
                    reasons_by_name,
                    run_dir_name=run_dir_name,
                    reason=f"operator_contract:{status}",
                )
            elif status == "consistent" and isinstance(baseline_operator_contract_payload, Mapping):
                candidate_payload = _load_operator_contract_payload(run.get("run_dir", None), bundle=False)
                if isinstance(candidate_payload, Mapping) and dict(candidate_payload) != dict(
                    baseline_operator_contract_payload
                ):
                    if isinstance(run_dir_name, str) and run_dir_name in candidate_gates:
                        candidate_gates[run_dir_name]["operator_contract"] = "mismatched"
                    _append_candidate_reason(
                        reasons_by_name,
                        run_dir_name=run_dir_name,
                        reason="operator_contract:baseline_mismatch",
                    )
                elif isinstance(run_dir_name, str) and run_dir_name in candidate_gates:
                    candidate_gates[run_dir_name]["operator_contract"] = "matched"

    if str(baseline_bundle_operator_contract_status or "").strip().lower() == "consistent":
        for run in runs:
            run_path = str(Path(str(run.get("run_dir"))).resolve())
            if baseline_path_str is not None and run_path == baseline_path_str:
                continue
            run_dir_name = run.get("run_dir_name", None)
            status = str(run.get("bundle_operator_contract_status", "missing") or "missing").strip().lower()
            if isinstance(run_dir_name, str) and run_dir_name in candidate_gates:
                candidate_gates[run_dir_name]["bundle_operator_contract"] = status
                candidate_bundle_digest_statuses[run_dir_name] = (
                    _run_bundle_operator_contract_digest_status(run)
                )
            if status in {"missing", "mismatched"}:
                _append_candidate_reason(
                    reasons_by_name,
                    run_dir_name=run_dir_name,
                    reason=f"operator_contract_bundle:{status}",
                )
                if status == "mismatched":
                    degraded_by = _run_trust_degraded_by(run)
                    if "operator_contract_bundle_digest_mismatch" in degraded_by:
                        _append_candidate_reason(
                            reasons_by_name,
                            run_dir_name=run_dir_name,
                            reason="operator_contract_bundle:digest_mismatch",
                        )
            elif status == "consistent" and isinstance(
                baseline_bundle_operator_contract_payload,
                Mapping,
            ):
                candidate_payload = _load_operator_contract_payload(run.get("run_dir", None), bundle=True)
                if isinstance(candidate_payload, Mapping) and dict(candidate_payload) != dict(
                    baseline_bundle_operator_contract_payload
                ):
                    if isinstance(run_dir_name, str) and run_dir_name in candidate_gates:
                        candidate_gates[run_dir_name]["bundle_operator_contract"] = "mismatched"
                    _append_candidate_reason(
                        reasons_by_name,
                        run_dir_name=run_dir_name,
                        reason="operator_contract_bundle:baseline_mismatch",
                    )
                elif isinstance(run_dir_name, str) and run_dir_name in candidate_gates:
                    candidate_gates[run_dir_name]["bundle_operator_contract"] = "matched"

    candidate_verdicts = {
        name: ("blocked" if reasons_by_name.get(name, []) else "pass")
        for name in candidate_names
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


def _extract_metrics(report: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    robustness_summary = report.get("robustness_summary", None)
    if isinstance(robustness_summary, dict):
        for key, value in robustness_summary.items():
            if isinstance(value, (int, float)):
                metrics[str(key)] = float(value)

    results = report.get("results", None)
    if isinstance(results, dict):
        for key in ("auroc", "average_precision"):
            value = results.get(key, None)
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
        pixel = results.get("pixel_metrics", None)
        if isinstance(pixel, dict):
            for key in ("pixel_auroc", "pixel_average_precision", "aupro", "pixel_segf1"):
                value = pixel.get(key, None)
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)

    rows = report.get("rows", None)
    if isinstance(rows, list):
        for key in ("auroc", "average_precision", "pixel_auroc", "aupro", "pixel_segf1"):
            vals = [
                float(item[key])
                for item in rows
                if isinstance(item, dict) and isinstance(item.get(key), (int, float))
            ]
            if vals:
                metrics[key] = max(vals)

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


def _extract_robustness_trust(report: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = report.get("robustness_trust", None)
    if isinstance(raw, Mapping):
        return dict(raw)

    robustness = _extract_robustness_payload(report)
    if robustness is None:
        return None

    robustness_summary = report.get("robustness_summary", None)
    robustness_protocol = _extract_robustness_protocol(report)
    return build_robustness_trust_summary(
        report=robustness,
        robustness_summary=(
            dict(robustness_summary) if isinstance(robustness_summary, Mapping) else None
        ),
        robustness_protocol=robustness_protocol,
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
        return path / "report.json"
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
    report_path = root / "report.json"
    report = _load_json_dict(report_path)
    metrics = _extract_metrics(report)

    env_path = root / "environment.json"
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
    robustness_trust = _extract_robustness_trust(report)

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
        run_path = str(Path(str(run.get("run_dir"))).resolve())
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
        is_baseline = baseline_path_str is not None and run_path == baseline_path_str
        if is_baseline:
            row["status"] = "baseline"
        elif checked:
            if signature is None:
                row["status"] = "missing"
                missing_runs += 1
            else:
                missing_fields: list[str] = []
                mismatch_fields: list[str] = []
                for field in ("conditions", "corruption_mode", "input_mode", "resize", "severities"):
                    baseline_value = baseline_signature.get(field)
                    run_value = signature.get(field)
                    if baseline_value is None:
                        continue
                    if run_value is None:
                        missing_fields.append(str(field))
                    elif run_value != baseline_value:
                        mismatch_fields.append(str(field))

                row["mismatch_fields"] = list(dict.fromkeys(missing_fields + mismatch_fields))
                if mismatch_fields:
                    row["status"] = "mismatched"
                    mismatched_runs += 1
                elif missing_fields:
                    row["status"] = "missing"
                    missing_runs += 1
                else:
                    row["status"] = "matched"
                    matched_runs += 1
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
        run_path = str(Path(str(run.get("run_dir"))).resolve())
        run_status = _resolve_contract_status(run, key=status_key)
        row: dict[str, Any] = {
            "run_dir": run.get("run_dir"),
            "run_dir_name": run.get("run_dir_name"),
            "contract_status": run_status,
            "contract_sha256": None,
            "status": "unchecked",
        }
        is_baseline = baseline_path_str is not None and run_path == baseline_path_str
        if is_baseline:
            row["status"] = "baseline"
            row["contract_sha256"] = baseline_contract_sha256
        elif checked:
            if run_status == "missing":
                row["status"] = "missing"
                missing_runs += 1
            elif run_status != "consistent":
                row["status"] = "mismatched"
                mismatched_runs += 1
                row["mismatch_reason"] = f"candidate_{run_status}"
                candidate_payload = _load_operator_contract_payload(run.get("run_dir", None), bundle=bundle)
                if isinstance(candidate_payload, Mapping):
                    row["contract_sha256"] = _contract_payload_sha256(candidate_payload)
            else:
                candidate_payload = _load_operator_contract_payload(run.get("run_dir", None), bundle=bundle)
                if not isinstance(candidate_payload, Mapping):
                    row["status"] = "missing"
                    missing_runs += 1
                else:
                    candidate_contract_sha256 = _contract_payload_sha256(candidate_payload)
                    row["contract_sha256"] = candidate_contract_sha256
                    if dict(candidate_payload) != dict(baseline_payload):
                        row["status"] = "mismatched"
                        mismatched_runs += 1
                        row["mismatch_reason"] = "baseline_mismatch"
                    else:
                        row["status"] = "matched"
                        matched_runs += 1
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
        run_path = str(Path(str(run.get("run_dir"))).resolve())
        run_sha256 = run.get("split_fingerprint_sha256", None)
        row: dict[str, Any] = {
            "run_dir": run.get("run_dir"),
            "run_dir_name": run.get("run_dir_name"),
            "split_fingerprint_sha256": (
                str(run_sha256) if isinstance(run_sha256, str) and run_sha256 else None
            ),
            "status": "unchecked",
        }
        is_baseline = baseline_path_str is not None and run_path == baseline_path_str
        if is_baseline:
            row["status"] = "baseline"
        elif checked:
            if row["split_fingerprint_sha256"] is None:
                row["status"] = "missing"
                missing_runs += 1
            elif row["split_fingerprint_sha256"] == baseline_sha256:
                row["status"] = "matched"
                matched_runs += 1
            else:
                row["status"] = "mismatched"
                mismatched_runs += 1
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
        baseline_dataset = str(dataset_value) if isinstance(dataset_value, str) and dataset_value else None
        baseline_category = str(category_value) if isinstance(category_value, str) and category_value else None

    dataset_checked = baseline_dataset is not None
    category_checked = baseline_category is not None
    checked = bool(dataset_checked or category_checked)

    comparisons: list[dict[str, Any]] = []
    matched_runs = 0
    mismatched_runs = 0
    missing_runs = 0
    for run in runs:
        run_path = str(Path(str(run.get("run_dir"))).resolve())
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

        is_baseline = baseline_path_str is not None and run_path == baseline_path_str
        if is_baseline:
            row["status"] = "baseline"
        elif checked:
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
                mismatched_runs += 1
            elif any(status == "missing" for status in statuses):
                row["status"] = "missing"
                missing_runs += 1
            else:
                row["status"] = "matched"
                matched_runs += 1
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
        run_path = str(Path(str(run.get("run_dir"))).resolve())
        run_fingerprint = run.get("environment_fingerprint_sha256", None)
        row: dict[str, Any] = {
            "run_dir": run.get("run_dir"),
            "run_dir_name": run.get("run_dir_name"),
            "environment_fingerprint_sha256": (
                str(run_fingerprint)
                if isinstance(run_fingerprint, str) and run_fingerprint
                else None
            ),
            "status": "unchecked",
        }
        is_baseline = baseline_path_str is not None and run_path == baseline_path_str
        if is_baseline:
            row["status"] = "baseline"
        elif checked:
            if row["environment_fingerprint_sha256"] is None:
                row["status"] = "missing"
                missing_runs += 1
            elif row["environment_fingerprint_sha256"] == baseline_fingerprint:
                row["status"] = "matched"
                matched_runs += 1
            else:
                row["status"] = "mismatched"
                mismatched_runs += 1
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
    base = Path(root)
    items = [
        summarize_run_dir(path.parent)
        for path in sorted(base.rglob("report.json"))
        if _is_top_level_report(path, base)
    ]
    if kind is not None:
        kind_norm = str(kind).strip().lower()
        items = [item for item in items if str(item.get("kind", "")).lower() == kind_norm]
    if dataset is not None:
        dataset_norm = str(dataset).strip().lower()
        items = [item for item in items if str(item.get("dataset", "")).lower() == dataset_norm]
    if query is not None:
        needle = str(query).strip().lower()
        if needle:
            items = [
                item
                for item in items
                if needle in str(item.get("run_dir_name", "")).lower()
                or needle in str(item.get("model_or_suite", "")).lower()
                or needle in str(item.get("category", "")).lower()
            ]
    if min_quality is not None:
        minimum_rank = int(_QUALITY_STATUS_RANK.get(str(min_quality), -1))
        items = [
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
    if same_split_as is not None:
        target_report = _load_report_for_run_dir(same_split_as)
        target_sha256 = _extract_split_fingerprint_sha256(target_report)
        if target_sha256 is None:
            items = []
        else:
            items = [
                item
                for item in items
                if str(item.get("split_fingerprint_sha256", "")) == str(target_sha256)
            ]
    if same_environment_as is not None:
        target_summary = summarize_run_dir(same_environment_as)
        target_fingerprint = _extract_environment_fingerprint_sha256(target_summary)
        if target_fingerprint is None:
            items = []
        else:
            items = [
                item
                for item in items
                if _extract_environment_fingerprint_sha256(item) == target_fingerprint
            ]
    if same_target_as is not None:
        target_summary = summarize_run_dir(same_target_as)
        target_signature = _extract_target_signature(target_summary)
        if target_signature is None:
            items = []
        else:
            items = [item for item in items if _matches_target_signature(item, target_signature)]
    if same_robustness_protocol_as is not None:
        target_summary = summarize_run_dir(same_robustness_protocol_as)
        target_signature = _extract_robustness_protocol_signature(target_summary)
        if target_signature is None:
            items = []
        else:
            items = [
                item
                for item in items
                if _extract_robustness_protocol_signature(item) == target_signature
            ]
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
    baseline_summary = (
        summarize_run_dir(baseline_run_dir) if baseline_run_dir is not None else None
    )
    metric_names = sorted(
        {
            key
            for run in ([baseline_summary] if baseline_summary is not None else []) + runs
            for key, value in dict(run.get("metrics", {})).items()
            if isinstance(value, (int, float))
        }
    )
    if metric is not None:
        metric_names = [name for name in metric_names if name == str(metric)]

    metrics: dict[str, dict[str, Any]] = {}
    total_regressions = 0
    baseline_path_str = (
        str(Path(baseline_run_dir).resolve()) if baseline_run_dir is not None else None
    )
    split_comparison = _build_split_comparison(
        runs,
        baseline_summary=baseline_summary,
        baseline_path_str=baseline_path_str,
    )
    environment_comparison = _build_environment_comparison(
        runs,
        baseline_summary=baseline_summary,
        baseline_path_str=baseline_path_str,
    )
    target_comparison = _build_target_comparison(
        runs,
        baseline_summary=baseline_summary,
        baseline_path_str=baseline_path_str,
    )
    robustness_protocol_comparison = _build_robustness_protocol_comparison(
        runs,
        baseline_summary=baseline_summary,
        baseline_path_str=baseline_path_str,
    )
    operator_contract_comparison = _build_operator_contract_comparison(
        runs,
        baseline_summary=baseline_summary,
        baseline_path_str=baseline_path_str,
        status_key="operator_contract_status",
        bundle=False,
    )
    bundle_operator_contract_comparison = _build_operator_contract_comparison(
        runs,
        baseline_summary=baseline_summary,
        baseline_path_str=baseline_path_str,
        status_key="bundle_operator_contract_status",
        bundle=True,
    )
    for name in metric_names:
        values = [
            float(run["metrics"][name])
            for run in runs
            if isinstance(run.get("metrics", {}).get(name), (int, float))
        ]
        if values:
            info: dict[str, Any] = {
                "values": values,
                "min": min(values),
                "max": max(values),
            }
            direction = _metric_direction(name)
            info["direction"] = direction

            if baseline_summary is not None:
                baseline_value = baseline_summary.get("metrics", {}).get(name, None)
                info["baseline"] = (
                    float(baseline_value) if isinstance(baseline_value, (int, float)) else None
                )
                comparisons: list[dict[str, Any]] = []
                regressions = 0
                for run in runs:
                    value = run.get("metrics", {}).get(name, None)
                    row: dict[str, Any] = {
                        "run_dir": run.get("run_dir"),
                        "run_dir_name": run.get("run_dir_name"),
                        "value": (float(value) if isinstance(value, (int, float)) else None),
                        "delta_vs_baseline": None,
                        "status": "missing",
                    }

                    run_path = str(Path(str(run.get("run_dir"))).resolve())
                    is_baseline = baseline_path_str is not None and run_path == baseline_path_str
                    if is_baseline:
                        row["status"] = "baseline"
                        row["delta_vs_baseline"] = 0.0
                    elif isinstance(value, (int, float)) and isinstance(baseline_value, (int, float)):
                        delta = round(float(value) - float(baseline_value), 12)
                        row["delta_vs_baseline"] = delta
                        if direction == "higher_is_better":
                            if delta < 0.0:
                                row["status"] = "regressed"
                                regressions += 1
                            elif delta > 0.0:
                                row["status"] = "improved"
                            else:
                                row["status"] = "unchanged"
                        elif direction == "lower_is_better":
                            if delta > 0.0:
                                row["status"] = "regressed"
                                regressions += 1
                            elif delta < 0.0:
                                row["status"] = "improved"
                            else:
                                row["status"] = "unchanged"
                    comparisons.append(row)
                info["comparisons"] = comparisons
                info["regression_count"] = int(regressions)
                total_regressions += int(regressions)

            metrics[name] = info

    split_summary = dict(split_comparison.get("summary", {}))
    environment_summary = dict(environment_comparison.get("summary", {}))
    target_summary = dict(target_comparison.get("summary", {}))
    robustness_protocol_summary = dict(robustness_protocol_comparison.get("summary", {}))
    operator_contract_summary = dict(operator_contract_comparison.get("summary", {}))
    bundle_operator_contract_summary = dict(bundle_operator_contract_comparison.get("summary", {}))
    baseline_checked = baseline_summary is not None
    blocking_flags = (
        _compare_blocking_flags(
            total_regressions=total_regressions,
            split_summary=split_summary,
            environment_summary=environment_summary,
            target_summary=target_summary,
            robustness_protocol_summary=robustness_protocol_summary,
            operator_contract_summary=operator_contract_summary,
            bundle_operator_contract_summary=bundle_operator_contract_summary,
        )
        if bool(baseline_checked)
        else []
    )
    summary = {
        "baseline_checked": bool(baseline_checked),
        "total_regressions": int(total_regressions),
        "regression_gate": (
            ("clean" if int(total_regressions) == 0 else "regressed")
            if bool(baseline_checked)
            else "unchecked"
        ),
        "comparability_gates": {
            "split": _comparability_gate_status(split_summary),
            "environment": _comparability_gate_status(environment_summary),
            "target": _comparability_gate_status(target_summary),
            "robustness_protocol": _comparability_gate_status(robustness_protocol_summary),
            "operator_contract": _comparability_gate_status(operator_contract_summary),
            "bundle_operator_contract": _comparability_gate_status(
                bundle_operator_contract_summary
            ),
        },
        "blocking_flags": blocking_flags,
        "verdict": (
            ("pass" if not blocking_flags else "blocked")
            if bool(baseline_checked)
            else "informational"
        ),
    }
    trust_comparison = _build_trust_comparison(baseline_summary)
    evaluation_contract = build_evaluation_contract(
        metric_names=metric_names,
        primary_metric=(
            str(baseline_summary.get("evaluation_contract", {}).get("primary_metric"))
            if isinstance(baseline_summary, Mapping)
            and isinstance(baseline_summary.get("evaluation_contract"), Mapping)
            and baseline_summary.get("evaluation_contract", {}).get("primary_metric") is not None
            else "auroc"
        ),
        ranking_metric=(
            str(baseline_summary.get("evaluation_contract", {}).get("ranking_metric"))
            if isinstance(baseline_summary, Mapping)
            and isinstance(baseline_summary.get("evaluation_contract"), Mapping)
            and baseline_summary.get("evaluation_contract", {}).get("ranking_metric") is not None
            else "auroc"
        ),
        pixel_metrics_enabled=(
            baseline_summary.get("evaluation_contract", {}).get("pixel_metrics_enabled")
            if isinstance(baseline_summary, Mapping)
            and isinstance(baseline_summary.get("evaluation_contract"), Mapping)
            else None
        ),
        comparability_hints=(
            dict(baseline_summary.get("evaluation_contract", {}).get("comparability_hints", {}))
            if isinstance(baseline_summary, Mapping)
            and isinstance(baseline_summary.get("evaluation_contract"), Mapping)
            and isinstance(
                baseline_summary.get("evaluation_contract", {}).get("comparability_hints"),
                Mapping,
            )
            else None
        ),
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
        baseline_operator_contract_status=(
            str(baseline_summary.get("operator_contract_status"))
            if isinstance(baseline_summary, Mapping)
            and baseline_summary.get("operator_contract_status") is not None
            else None
        ),
        baseline_bundle_operator_contract_status=(
            str(baseline_summary.get("bundle_operator_contract_status"))
            if isinstance(baseline_summary, Mapping)
            and baseline_summary.get("bundle_operator_contract_status") is not None
            else None
        ),
        primary_metric_info=primary_metric_info,
        split_comparison=split_comparison,
        environment_comparison=environment_comparison,
        target_comparison=target_comparison,
        robustness_protocol_comparison=robustness_protocol_comparison,
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
        int(primary_metric_info.get("regression_count", 0))
        if bool(primary_metric_info)
        else None
    )
    primary_metric_statuses: dict[str, str] = {}
    primary_metric_deltas: dict[str, float] = {}
    for row in primary_metric_info.get("comparisons", []):
        if not isinstance(row, Mapping):
            continue
        if str(row.get("status")) == "baseline":
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
    summary["primary_metric_statuses"] = primary_metric_statuses
    summary["primary_metric_deltas"] = primary_metric_deltas
    summary["trust_checked"] = bool(trust_comparison.get("checked"))
    summary["trust_gate"] = trust_comparison.get("gate", None)
    summary["trust_status"] = trust_comparison.get("status", None)
    summary["trust_reason"] = trust_comparison.get("reason", None)
    summary["operator_contract_gate"] = _comparability_gate_status(operator_contract_summary)
    summary["bundle_operator_contract_gate"] = _comparability_gate_status(
        bundle_operator_contract_summary
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
    summary["candidate_incompatibility_digest"] = _build_candidate_incompatibility_digest(
        candidate_verdicts=summary["candidate_verdicts"],
        candidate_blocking_reasons=summary["candidate_blocking_reasons"],
        candidate_comparability_gates=summary["candidate_comparability_gates"],
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


__all__ = [
    "compare_run_summaries",
    "latest_run_summary",
    "list_run_summaries",
    "summarize_run_dir",
]
