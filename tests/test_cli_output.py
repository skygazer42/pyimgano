from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_emit_json_prints_sorted_indented_payload_and_returns_status(capsys) -> None:
    from pyimgano.cli_output import emit_json

    rc = emit_json({"b": 2, "a": 1}, status_code=7)

    assert rc == 7
    assert capsys.readouterr().out == '{\n  "a": 1,\n  "b": 2\n}\n'


def test_emit_jsonable_converts_non_json_native_values(capsys) -> None:
    from pyimgano.cli_output import emit_jsonable

    rc = emit_jsonable({"value": np.int64(3), "path": Path("/tmp/example")})

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"path": "/tmp/example", "value": 3}


def test_print_cli_error_writes_standard_prefix_and_context_lines(capsys) -> None:
    from pyimgano.cli_output import print_cli_error

    print_cli_error(ValueError("boom"), context_lines=["context: model='vision_ecod'"])

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: boom\ncontext: model='vision_ecod'\n"
