from __future__ import annotations

import json
import sys
from typing import Any, Iterable

from pyimgano.utils.jsonable import to_jsonable


def emit_json(
    payload: Any,
    *,
    status_code: int = 0,
    indent: int | None = 2,
    sort_keys: bool = True,
) -> int:
    print(json.dumps(payload, indent=indent, sort_keys=sort_keys))
    return int(status_code)


def emit_jsonable(
    payload: Any,
    *,
    status_code: int = 0,
    indent: int | None = 2,
    sort_keys: bool = True,
) -> int:
    return emit_json(
        to_jsonable(payload),
        status_code=status_code,
        indent=indent,
        sort_keys=sort_keys,
    )


def print_cli_error(exc: BaseException, *, context_lines: Iterable[str] | None = None) -> None:
    print(f"error: {exc}", file=sys.stderr)
    for line in context_lines or ():
        print(str(line), file=sys.stderr)
