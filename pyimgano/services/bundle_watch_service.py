from __future__ import annotations

import hashlib
import hmac
import json
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

from pyimgano.infer_cli_inputs import collect_image_paths
from pyimgano.services.bundle_run_service import BundleInferenceBatchRequest, run_bundle_inference_batch

_WATCH_SCHEMA_VERSION = 1
_WATCH_REASON_CODE_MAP = {
    "watch_input_not_found": "WATCH_INPUT_NOT_FOUND",
    "watch_processing_errors": "WATCH_PROCESSING_ERRORS",
    "watch_batch_blocked": "WATCH_BATCH_BLOCKED",
    "watch_webhook_errors": "WATCH_WEBHOOK_ERRORS",
}


@dataclass(frozen=True)
class BundleWatchRequest:
    bundle_dir: str | Path
    watch_dir: str | Path
    output_dir: str | Path
    poll_seconds: float = 1.0
    settle_seconds: float = 2.0
    once: bool = False
    state_file: str | Path | None = None
    check_hashes: bool = False
    export_masks: bool = False
    export_overlays: bool = False
    export_defects_regions: bool = False
    webhook_url: str | None = None
    webhook_bearer_token: str | None = None
    webhook_signing_secret: str | None = None
    webhook_headers: dict[str, str] | None = None
    webhook_timeout_seconds: float = 5.0
    max_anomaly_rate: float | None = None
    max_reject_rate: float | None = None
    max_error_rate: float | None = None
    min_processed: int | None = None


def _state_file_path(request: BundleWatchRequest, *, output_dir: Path) -> Path:
    if request.state_file is not None:
        return Path(request.state_file)
    return output_dir / "watch_state.json"


def _watch_artifact_paths(request: BundleWatchRequest, *, output_dir: Path) -> dict[str, Path]:
    return {
        "results_jsonl": output_dir / "results.jsonl",
        "watch_report_json": output_dir / "watch_report.json",
        "watch_state_json": _state_file_path(request, output_dir=output_dir),
        "watch_events_jsonl": output_dir / "watch_events.jsonl",
        "defects_regions_jsonl": output_dir / "defects_regions.jsonl",
        "masks_dir": output_dir / "masks",
        "overlays_dir": output_dir / "overlays",
    }


def _default_watch_state(
    request: BundleWatchRequest,
    *,
    state_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": int(_WATCH_SCHEMA_VERSION),
        "bundle_dir": str(Path(request.bundle_dir)),
        "watch_dir": str(Path(request.watch_dir)),
        "state_file": str(state_path),
        "last_poll_at": None,
        "entries": {},
    }


def _load_watch_state(request: BundleWatchRequest, *, output_dir: Path) -> tuple[dict[str, Any], Path]:
    state_path = _state_file_path(request, output_dir=output_dir)
    if not state_path.is_file():
        return _default_watch_state(request, state_path=state_path), state_path

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("watch_state.json must contain a JSON object.")
    payload.setdefault("entries", {})
    payload["state_file"] = str(state_path)
    return payload, state_path


def _write_watch_state(payload: Mapping[str, Any], *, state_path: Path) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl_line(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True))
        handle.write("\n")


def _append_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    start_line = 1
    if path.is_file():
        start_line += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    end_line = start_line + max(len(rows) - 1, 0)
    return start_line, end_line


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("JSONL payload must contain JSON objects.")
        rows.append(payload)
    return rows


def _append_existing_jsonl(src: Path, dst: Path) -> None:
    rows = _load_jsonl_rows(src)
    if not rows:
        return
    _append_jsonl_rows(dst, rows)


def _jsonl_row_by_ref(ref: str) -> dict[str, Any]:
    path_text, sep, line_text = str(ref).rpartition("#L")
    if not sep:
        raise ValueError(f"Invalid result ref: {ref!r}")
    line_no = int(line_text)
    path = Path(path_text)
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if idx != line_no or not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("Referenced JSONL line must contain a JSON object.")
        return payload
    raise ValueError(f"Result ref not found: {ref!r}")


def _watch_record_for_path(*, watch_dir: Path, path: Path) -> dict[str, Any]:
    resolved_path = path.resolve()
    rel_path = resolved_path.relative_to(watch_dir.resolve()).as_posix()
    return {
        "id": str(rel_path),
        "image_path": str(resolved_path),
        "category": None,
        "meta": None,
        "resolved_input_path": str(resolved_path),
    }


def _entry_fingerprint(*, relative_path: str, size_bytes: int, mtime_ns: int) -> str:
    return f"{relative_path}:{int(size_bytes)}:{int(mtime_ns)}"


def _rewrite_watch_results(
    *,
    staging_results_path: Path,
    input_record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = _load_jsonl_rows(staging_results_path)
    rewritten: list[dict[str, Any]] = []
    for row in rows:
        row["id"] = str(input_record["id"])
        row["image_path"] = str(input_record["image_path"])
        row["category"] = input_record.get("category")
        row["meta"] = input_record.get("meta")
        rewritten.append(row)
    return rewritten


def _result_summary(records: list[dict[str, Any]]) -> dict[str, int]:
    normal = 0
    anomalous = 0
    rejected = 0
    error = 0

    for record in records:
        if str(record.get("status", "")) == "error":
            error += 1
            continue
        label = record.get("label")
        if bool(record.get("rejected")):
            rejected += 1
        elif label == 0:
            normal += 1
        elif label == 1:
            anomalous += 1
    return {
        "normal": int(normal),
        "anomalous": int(anomalous),
        "rejected": int(rejected),
        "error": int(error),
    }


def _watch_counts(entries: Mapping[str, Any]) -> dict[str, int]:
    counts = {"processed": 0, "pending": 0, "error": 0}
    for entry in entries.values():
        status = str(dict(entry).get("status", "pending"))
        if status == "processed":
            counts["processed"] += 1
        elif status == "error":
            counts["error"] += 1
        else:
            counts["pending"] += 1
    return counts


def _watch_delivery_counts(entries: Mapping[str, Any]) -> dict[str, int]:
    counts = {"delivered": 0, "pending": 0, "error": 0, "not_requested": 0}
    for entry in entries.values():
        delivery_status = str(dict(entry).get("delivery_status", "not_requested"))
        if delivery_status in counts:
            counts[delivery_status] += 1
    return counts


def _watch_batch_gate_namespace(request: BundleWatchRequest) -> Any:
    return type(
        "_WatchBatchGateArgs",
        (),
        {
            "max_anomaly_rate": request.max_anomaly_rate,
            "max_reject_rate": request.max_reject_rate,
            "max_error_rate": request.max_error_rate,
            "min_processed": request.min_processed,
        },
    )()


def _watch_exit_code(*, status: str) -> int:
    return 0 if str(status) in {"completed", "running"} else 1


def _webhook_enabled(request: BundleWatchRequest) -> bool:
    return bool(str(request.webhook_url or "").strip())


def _default_delivery_status(request: BundleWatchRequest) -> str:
    return "pending" if _webhook_enabled(request) else "not_requested"


def _resolve_webhook_headers(request: BundleWatchRequest) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if request.webhook_bearer_token is not None and str(request.webhook_bearer_token).strip():
        headers["Authorization"] = f"Bearer {str(request.webhook_bearer_token).strip()}"
    for key, value in dict(request.webhook_headers or {}).items():
        headers[str(key)] = str(value)
    return headers


def _build_webhook_body(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sign_webhook_body(*, secret: str, timestamp: str, body: str) -> str:
    return hmac.new(
        str(secret).encode("utf-8"),
        f"{timestamp}.{body}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _resolve_signed_webhook_headers(
    request: BundleWatchRequest,
    *,
    body: str,
    now: float,
) -> dict[str, str]:
    headers = _resolve_webhook_headers(request)
    secret = str(request.webhook_signing_secret or "").strip()
    if not secret:
        return headers
    timestamp = str(int(now))
    headers["X-PyImgAno-Timestamp"] = timestamp
    headers["X-PyImgAno-Signature"] = _sign_webhook_body(
        secret=secret,
        timestamp=timestamp,
        body=body,
    )
    return headers


def _emit_watch_event(
    *,
    artifacts: Mapping[str, Path],
    event: str,
    relative_path: str,
    fingerprint: str,
    now: float,
    status: str | None = None,
    detail: str | None = None,
    last_result_ref: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "event": str(event),
        "relative_path": str(relative_path),
        "fingerprint": str(fingerprint),
        "timestamp": float(now),
    }
    if status is not None:
        payload["status"] = str(status)
    if detail is not None:
        payload["detail"] = str(detail)
    if last_result_ref is not None:
        payload["last_result_ref"] = str(last_result_ref)
    _append_jsonl_line(artifacts["watch_events_jsonl"], payload)


def _build_watch_webhook_payload(
    *,
    request: BundleWatchRequest,
    relative_path: str,
    fingerprint: str,
    result_ref: str,
    delivery_id: str,
    delivery_attempt: int,
) -> dict[str, Any]:
    return {
        "schema_version": int(_WATCH_SCHEMA_VERSION),
        "tool": "pyimgano-bundle",
        "command": "watch",
        "event": "processed",
        "bundle_dir": str(Path(request.bundle_dir)),
        "watch_dir": str(Path(request.watch_dir)),
        "output_dir": str(Path(request.output_dir)),
        "relative_path": str(relative_path),
        "fingerprint": str(fingerprint),
        "delivery_id": str(delivery_id),
        "delivery_attempt": int(delivery_attempt),
        "result_ref": str(result_ref),
        "result": _jsonl_row_by_ref(result_ref),
    }


def _send_watch_webhook(
    payload: dict[str, Any],
    url: str,
    timeout: float,
    headers: dict[str, str],
    body: str,
) -> None:
    req = urllib_request.Request(
        url=str(url),
        data=str(body).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=float(timeout)) as response:
        status = int(getattr(response, "status", response.getcode()))
        if not 200 <= status < 300:
            raise RuntimeError(f"webhook returned HTTP {status}")


def _entry_needs_webhook_delivery(entry: Mapping[str, Any], request: BundleWatchRequest) -> bool:
    if not _webhook_enabled(request):
        return False
    if str(entry.get("status", "")) != "processed":
        return False
    if not str(entry.get("last_result_ref", "")).strip():
        return False
    delivery_status = str(entry.get("delivery_status", "pending")).strip() or "pending"
    return delivery_status != "delivered"


def _resolve_delivery_id(*, entry: dict[str, Any], fingerprint: str, result_ref: str) -> str:
    delivery_id = str(entry.get("delivery_id", "")).strip()
    if delivery_id:
        return delivery_id
    delivery_id = hashlib.sha256(f"{fingerprint}|{result_ref}".encode("utf-8")).hexdigest()
    entry["delivery_id"] = delivery_id
    return delivery_id


def _deliver_entry_webhook(
    *,
    request: BundleWatchRequest,
    entry: dict[str, Any],
    relative_path: str,
    fingerprint: str,
    now: float,
    artifacts: Mapping[str, Path],
    send_webhook_impl: Callable[[dict[str, Any], str, float, dict[str, str], str], None],
) -> tuple[bool, str | None]:
    entry["delivery_attempts"] = int(entry.get("delivery_attempts", 0)) + 1
    delivery_attempt = int(entry["delivery_attempts"])
    delivery_id = _resolve_delivery_id(
        entry=entry,
        fingerprint=fingerprint,
        result_ref=str(entry["last_result_ref"]),
    )
    payload = _build_watch_webhook_payload(
        request=request,
        relative_path=relative_path,
        fingerprint=fingerprint,
        result_ref=str(entry["last_result_ref"]),
        delivery_id=delivery_id,
        delivery_attempt=delivery_attempt,
    )
    body = _build_webhook_body(payload)
    headers = _resolve_signed_webhook_headers(request, body=body, now=now)
    headers["X-PyImgAno-Delivery-Id"] = str(delivery_id)
    headers["X-PyImgAno-Delivery-Attempt"] = str(delivery_attempt)
    try:
        send_webhook_impl(
            payload,
            str(request.webhook_url),
            float(request.webhook_timeout_seconds),
            headers,
            body,
        )
    except Exception as exc:  # noqa: BLE001 - network boundary
        entry["delivery_status"] = "error"
        entry["last_delivery_error"] = f"{type(exc).__name__}: {exc}"
        entry["last_delivery_at"] = now
        _emit_watch_event(
            artifacts=artifacts,
            event="webhook_error",
            relative_path=relative_path,
            fingerprint=fingerprint,
            now=now,
            status="processed",
            detail=str(entry["last_delivery_error"]),
            last_result_ref=str(entry["last_result_ref"]),
        )
        return False, str(entry["last_delivery_error"])

    entry["delivery_status"] = "delivered"
    entry["last_delivery_error"] = None
    entry["last_delivery_at"] = now
    _emit_watch_event(
        artifacts=artifacts,
        event="webhook_delivered",
        relative_path=relative_path,
        fingerprint=fingerprint,
        now=now,
        status="processed",
        last_result_ref=str(entry["last_result_ref"]),
    )
    return True, None


def _build_watch_report(
    *,
    request: BundleWatchRequest,
    bundle_validation: Mapping[str, Any],
    output_dir: Path,
    artifacts: Mapping[str, Path],
    state_path: Path,
    state: Mapping[str, Any],
    processed_now: int,
    batch_verdict: str,
    batch_gate_summary: Mapping[str, Any],
    batch_gate_reason_codes: list[str],
    cycle_errors: int,
    webhook_delivery_count: int,
    webhook_error_count: int,
    status: str,
    reason_codes: list[str],
    last_success_at: float | None,
    last_error_at: float | None,
) -> dict[str, Any]:
    counts = _watch_counts(dict(state.get("entries", {})))
    delivery_counts = _watch_delivery_counts(dict(state.get("entries", {})))
    report = {
        "schema_version": int(_WATCH_SCHEMA_VERSION),
        "tool": "pyimgano-bundle",
        "command": "watch",
        "bundle_dir": str(Path(request.bundle_dir)),
        "watch_dir": str(Path(request.watch_dir)),
        "output_dir": str(output_dir),
        "state_file": str(state_path),
        "status": str(status),
        "ready": bool(status in {"completed", "running"}),
        "exit_code": int(_watch_exit_code(status=str(status))),
        "reason_codes": list(dict.fromkeys(str(item) for item in reason_codes)),
        "processed": int(processed_now),
        "pending": int(counts["pending"]),
        "error": int(counts["error"]),
        "webhook_enabled": bool(_webhook_enabled(request)),
        "webhook_url": (str(request.webhook_url) if _webhook_enabled(request) else None),
        "webhook_delivery_count": int(webhook_delivery_count),
        "webhook_error_count": int(webhook_error_count),
        "delivery_summary": delivery_counts,
        "poll_seconds": float(request.poll_seconds),
        "settle_seconds": float(request.settle_seconds),
        "last_poll_at": state.get("last_poll_at"),
        "last_success_at": last_success_at,
        "last_error_at": last_error_at,
        "bundle_validation": {
            "status": bundle_validation.get("status"),
            "ready": bool(bundle_validation.get("ready", False)),
            "reason_codes": list(bundle_validation.get("reason_codes", [])),
        },
        "batch_verdict": str(batch_verdict),
        "batch_gate_summary": dict(batch_gate_summary),
        "batch_gate_reason_codes": list(dict.fromkeys(str(item) for item in batch_gate_reason_codes)),
        "cycle_error_count": int(cycle_errors),
        "artifacts": {
            "results_jsonl": str(artifacts["results_jsonl"]),
            "watch_report_json": str(artifacts["watch_report_json"]),
            "watch_state_json": str(artifacts["watch_state_json"]),
            "watch_events_jsonl": str(artifacts["watch_events_jsonl"]),
            "defects_regions_jsonl": (
                str(artifacts["defects_regions_jsonl"]) if bool(request.export_defects_regions) else None
            ),
            "masks_dir": str(artifacts["masks_dir"]) if bool(request.export_masks) else None,
            "overlays_dir": (
                str(artifacts["overlays_dir"]) if bool(request.export_overlays) else None
            ),
        },
    }
    return report


def _write_watch_report(report: Mapping[str, Any], *, artifacts: Mapping[str, Path]) -> None:
    artifacts["watch_report_json"].parent.mkdir(parents=True, exist_ok=True)
    artifacts["watch_report_json"].write_text(
        json.dumps(dict(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_bundle_watch_once(
    request: BundleWatchRequest,
    *,
    infer_main_impl: Callable[[list[str]], int] | None = None,
    now_fn: Callable[[], float] | None = None,
    validate_bundle_impl: Callable[..., dict[str, Any]] | None = None,
    batch_gate_evaluator: Callable[..., tuple[dict[str, Any], str, list[str]]] | None = None,
    send_webhook_impl: Callable[[dict[str, Any], str, float, dict[str, str], str], None] | None = None,
) -> dict[str, Any]:
    if now_fn is None:
        now_fn = time.time
    if send_webhook_impl is None:
        send_webhook_impl = _send_watch_webhook

    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _watch_artifact_paths(request, output_dir=output_dir)
    state, state_path = _load_watch_state(request, output_dir=output_dir)
    now = float(now_fn())
    state["last_poll_at"] = now
    entries = dict(state.get("entries", {}))
    state["entries"] = entries

    if validate_bundle_impl is None:
        from pyimgano.bundle_cli import evaluate_bundle

        validate_bundle_impl = evaluate_bundle
    if batch_gate_evaluator is None:
        from pyimgano.bundle_cli import _evaluate_batch_gates as batch_gate_evaluator_fn

        batch_gate_evaluator = batch_gate_evaluator_fn

    bundle_validation = validate_bundle_impl(
        str(request.bundle_dir),
        check_hashes=bool(request.check_hashes),
    )
    runtime_policy = {}
    contract = bundle_validation.get("contract", {})
    if isinstance(contract, Mapping):
        runtime_policy = dict(contract.get("runtime_policy", {}) or {})

    if not bool(bundle_validation.get("ready")):
        report = _build_watch_report(
            request=request,
            bundle_validation=bundle_validation,
            output_dir=output_dir,
            artifacts=artifacts,
            state_path=state_path,
            state=state,
            processed_now=0,
            batch_verdict="not_evaluated",
            batch_gate_summary={},
            batch_gate_reason_codes=[],
            cycle_errors=0,
            webhook_delivery_count=0,
            webhook_error_count=0,
            status="blocked",
            reason_codes=list(bundle_validation.get("reason_codes", [])),
            last_success_at=None,
            last_error_at=None,
        )
        _write_watch_state(state, state_path=state_path)
        _write_watch_report(report, artifacts=artifacts)
        return report

    processed_rows: list[dict[str, Any]] = []
    processed_now = 0
    cycle_errors = 0
    webhook_delivery_count = 0
    webhook_error_count = 0
    last_success_at: float | None = None
    last_error_at: float | None = None

    try:
        input_paths = collect_image_paths(str(request.watch_dir))
    except FileNotFoundError:
        report = _build_watch_report(
            request=request,
            bundle_validation=bundle_validation,
            output_dir=output_dir,
            artifacts=artifacts,
            state_path=state_path,
            state=state,
            processed_now=0,
            batch_verdict="not_evaluated",
            batch_gate_summary={},
            batch_gate_reason_codes=[],
            cycle_errors=0,
            webhook_delivery_count=0,
            webhook_error_count=0,
            status="failed",
            reason_codes=[_WATCH_REASON_CODE_MAP["watch_input_not_found"]],
            last_success_at=None,
            last_error_at=None,
        )
        _write_watch_state(state, state_path=state_path)
        _write_watch_report(report, artifacts=artifacts)
        return report

    watch_root = Path(request.watch_dir)
    for raw_path in input_paths:
        path = Path(raw_path)
        stat = path.stat()
        input_record = _watch_record_for_path(watch_dir=watch_root, path=path)
        rel_path = str(input_record["id"])
        fingerprint = _entry_fingerprint(
            relative_path=rel_path,
            size_bytes=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
        )
        existing = dict(entries.get(rel_path, {}))
        if existing:
            entries[rel_path] = existing
        if existing.get("fingerprint") != fingerprint:
            existing = {
                "relative_path": rel_path,
                "fingerprint": fingerprint,
                "status": "pending",
                "first_seen_at": now,
                "last_attempt_at": None,
                "last_result_ref": None,
                "delivery_status": _default_delivery_status(request),
                "delivery_attempts": 0,
                "last_delivery_error": None,
                "last_delivery_at": None,
                "last_size_bytes": int(stat.st_size),
                "last_mtime_ns": int(stat.st_mtime_ns),
                "last_skip_reason": None,
            }
            entries[rel_path] = existing
            _emit_watch_event(
                artifacts=artifacts,
                event="discovered",
                relative_path=rel_path,
                fingerprint=fingerprint,
                now=now,
                status="pending",
            )

        if _entry_needs_webhook_delivery(existing, request):
            delivered, _error_text = _deliver_entry_webhook(
                request=request,
                entry=existing,
                relative_path=rel_path,
                fingerprint=fingerprint,
                now=now,
                artifacts=artifacts,
                send_webhook_impl=send_webhook_impl,
            )
            if delivered:
                webhook_delivery_count += 1
                last_success_at = now
            else:
                webhook_error_count += 1
                last_error_at = now
            continue

        if str(existing.get("status", "pending")) in {"processed", "error"}:
            if existing.get("last_skip_reason") != "already_processed":
                existing["last_skip_reason"] = "already_processed"
                _emit_watch_event(
                    artifacts=artifacts,
                    event="skipped_already_processed",
                    relative_path=rel_path,
                    fingerprint=fingerprint,
                    now=now,
                    status=str(existing.get("status")),
                )
            continue

        settled_after = max(float(existing.get("first_seen_at", now)), float(stat.st_mtime_ns) / 1_000_000_000.0)
        if now < settled_after + float(request.settle_seconds):
            if existing.get("last_skip_reason") != "unsettled":
                existing["last_skip_reason"] = "unsettled"
                _emit_watch_event(
                    artifacts=artifacts,
                    event="skipped_unsettled",
                    relative_path=rel_path,
                    fingerprint=fingerprint,
                    now=now,
                    status="pending",
                )
            continue

        existing["last_skip_reason"] = None
        existing["last_attempt_at"] = now
        staging_dir = output_dir / ".watch_tmp" / str(abs(hash((rel_path, fingerprint))))
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        staging_results = staging_dir / "results.jsonl"
        staging_regions = staging_dir / "defects_regions.jsonl"

        rc = run_bundle_inference_batch(
            BundleInferenceBatchRequest(
                bundle_dir=request.bundle_dir,
                input_records=[input_record],
                results_jsonl=staging_results,
                defects_enabled=bool(
                    request.export_masks
                    or request.export_overlays
                    or request.export_defects_regions
                ),
                masks_dir=(str(artifacts["masks_dir"]) if bool(request.export_masks) else None),
                overlays_dir=(
                    str(artifacts["overlays_dir"]) if bool(request.export_overlays) else None
                ),
                defects_regions_jsonl=(
                    str(staging_regions) if bool(request.export_defects_regions) else None
                ),
            ),
            infer_main_impl=infer_main_impl,
        )

        if rc != 0:
            cycle_errors += 1
            last_error_at = now
            existing["status"] = "error"
            _emit_watch_event(
                artifacts=artifacts,
                event="error",
                relative_path=rel_path,
                fingerprint=fingerprint,
                now=now,
                status="error",
                detail=f"infer_exit_code={rc}",
            )
            shutil.rmtree(staging_dir, ignore_errors=True)
            continue

        rewritten = _rewrite_watch_results(
            staging_results_path=staging_results,
            input_record=input_record,
        )
        start_line, _end_line = _append_jsonl_rows(artifacts["results_jsonl"], rewritten)
        if bool(request.export_defects_regions):
            _append_existing_jsonl(staging_regions, artifacts["defects_regions_jsonl"])
        processed_rows.extend(rewritten)
        processed_now += len(rewritten)
        last_success_at = now
        existing["status"] = "processed"
        existing["last_result_ref"] = f"{artifacts['results_jsonl']}#L{start_line}"
        existing["delivery_status"] = _default_delivery_status(request)
        _emit_watch_event(
            artifacts=artifacts,
            event="processed",
            relative_path=rel_path,
            fingerprint=fingerprint,
            now=now,
            status="processed",
            last_result_ref=str(existing["last_result_ref"]),
        )
        if _entry_needs_webhook_delivery(existing, request):
            delivered, _error_text = _deliver_entry_webhook(
                request=request,
                entry=existing,
                relative_path=rel_path,
                fingerprint=fingerprint,
                now=now,
                artifacts=artifacts,
                send_webhook_impl=send_webhook_impl,
            )
            if delivered:
                webhook_delivery_count += 1
            else:
                webhook_error_count += 1
                last_error_at = now
        shutil.rmtree(staging_dir, ignore_errors=True)

    batch_gate_summary, batch_verdict, batch_gate_reason_codes = batch_gate_evaluator(
        args=_watch_batch_gate_namespace(request),
        processed=len(processed_rows),
        result_summary=_result_summary(processed_rows),
        evaluated=bool(len(processed_rows) > 0),
        runtime_policy=runtime_policy,
    )

    reason_codes: list[str] = []
    if str(batch_verdict) == "blocked":
        reason_codes.extend(batch_gate_reason_codes)
    elif cycle_errors > 0:
        reason_codes.append(_WATCH_REASON_CODE_MAP["watch_processing_errors"])
    elif webhook_error_count > 0:
        reason_codes.append(_WATCH_REASON_CODE_MAP["watch_webhook_errors"])

    status = "completed"
    if str(batch_verdict) == "blocked":
        status = "blocked"
    elif cycle_errors > 0 or webhook_error_count > 0:
        status = "failed"

    report = _build_watch_report(
        request=request,
        bundle_validation=bundle_validation,
        output_dir=output_dir,
        artifacts=artifacts,
        state_path=state_path,
        state=state,
        processed_now=processed_now,
        batch_verdict=batch_verdict,
        batch_gate_summary=batch_gate_summary,
        batch_gate_reason_codes=batch_gate_reason_codes,
        cycle_errors=cycle_errors,
        webhook_delivery_count=webhook_delivery_count,
        webhook_error_count=webhook_error_count,
        status=status,
        reason_codes=reason_codes,
        last_success_at=last_success_at,
        last_error_at=last_error_at,
    )
    _write_watch_state(state, state_path=state_path)
    _write_watch_report(report, artifacts=artifacts)
    return report


__all__ = [
    "BundleWatchRequest",
    "run_bundle_watch_once",
]
