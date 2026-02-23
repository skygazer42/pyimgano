from __future__ import annotations

import json
from pathlib import Path

from pyimgano.utils.jsonable import to_jsonable


def save_run_report(path: str | Path, results: dict) -> None:
    """Save a run result dict as JSON (converting numpy types)."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = to_jsonable(results)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_jsonl_records(path: str | Path, records: list[dict]) -> None:
    """Save a list of dict records as JSONL (converting numpy types).

    Each item in `records` must be JSON-serializable after converting numpy types.
    """

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            payload = to_jsonable(record)
            f.write(json.dumps(payload, sort_keys=True))
            f.write("\n")
