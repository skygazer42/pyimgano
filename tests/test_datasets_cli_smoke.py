from __future__ import annotations

import json


def test_datasets_cli_list_json_smoke(capsys) -> None:
    from pyimgano.datasets_cli import main as datasets_main

    rc = datasets_main(["list", "--json"])
    assert rc == 0

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert isinstance(payload, list)
    names = {str(item.get("name")) for item in payload if isinstance(item, dict)}
    assert "custom" in names
