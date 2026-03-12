from __future__ import annotations

from typing import Any, Iterable

import pyimgano.cli_output as cli_output


def emit_listing(
    items: Iterable[Any],
    *,
    json_output: bool,
    json_payload: Any | None = None,
    sort_keys: bool = True,
    status_code: int = 0,
) -> int:
    materialized = list(items)
    if bool(json_output):
        payload = materialized if json_payload is None else json_payload
        return cli_output.emit_json(
            payload,
            status_code=status_code,
            sort_keys=sort_keys,
        )
    for item in materialized:
        print(item)
    return int(status_code)
