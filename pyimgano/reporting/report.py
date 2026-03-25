from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyimgano.utils.jsonable import to_jsonable

REPORT_SCHEMA_VERSION = 1


def _resolve_output_path(path: str | Path) -> Path:
    raw = Path(path)
    if ".." in raw.parts:
        raise ValueError("path traversal is not allowed for report outputs")
    if raw.is_absolute():
        return raw.resolve(strict=False)
    return (Path.cwd() / raw).resolve(strict=False)


def stamp_report_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach run-level metadata to report payloads without changing their shape."""

    stamped = dict(payload)
    stamped.setdefault("schema_version", int(REPORT_SCHEMA_VERSION))
    stamped.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
    try:
        from pyimgano import __version__ as pyimgano_version
    except Exception:
        pyimgano_version = None
    stamped.setdefault("pyimgano_version", pyimgano_version)
    return stamped


def save_run_report(path: str | Path, results: dict) -> None:
    """Save a run result dict as JSON (converting numpy types)."""

    out_path = _resolve_output_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = to_jsonable(results)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_jsonl_records(path: str | Path, records: list[dict]) -> None:
    """Save a list of dict records as JSONL (converting numpy types).

    Each item in `records` must be JSON-serializable after converting numpy types.
    """

    out_path = _resolve_output_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            payload = to_jsonable(record)
            f.write(json.dumps(payload, sort_keys=True))
            f.write("\n")
