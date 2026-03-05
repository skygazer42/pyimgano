from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import pyimgano.infer_cli as infer_cli


def _write_png(path: Path) -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


def test_infer_cli_continue_on_error_records_error_lines_and_returns_nonzero(
    tmp_path, monkeypatch
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a_good.png")
    _write_png(input_dir / "b_bad.png")

    out_jsonl = tmp_path / "out.jsonl"

    class _SometimesFails:
        def __init__(self) -> None:
            self.threshold_ = 0.5

        def decision_function(self, X):
            for item in list(X):
                if "bad" in str(item):
                    raise ValueError("boom")
            return np.linspace(0.0, 1.0, num=len(X), dtype=np.float32)

    det = _SometimesFails()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--model",
            "vision_ecod",
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
            "--batch-size",
            "2",
            "--continue-on-error",
        ]
    )
    assert rc == 1

    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert {r.get("status") for r in records} == {"ok", "error"}

    err = next(r for r in records if r.get("status") == "error")
    assert "error" in err
    assert err["error"]["type"] == "ValueError"
    assert "boom" in str(err["error"]["message"])
    assert "input" in err

    ok = next(r for r in records if r.get("status") == "ok")
    assert "score" in ok
    assert isinstance(ok["score"], float)


def test_infer_cli_profile_json_writes_payload(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"
    profile_path = tmp_path / "profile.json"

    class _OK:
        def __init__(self) -> None:
            self.threshold_ = 0.5

        def decision_function(self, X):
            return np.asarray([0.1 for _ in list(X)], dtype=np.float32)

    det = _OK()
    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: det)

    rc = infer_cli.main(
        [
            "--model",
            "vision_ecod",
            "--input",
            str(input_dir),
            "--save-jsonl",
            str(out_jsonl),
            "--profile-json",
            str(profile_path),
        ]
    )
    assert rc == 0

    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    assert payload.get("tool") == "pyimgano-infer"
    counts = payload.get("counts")
    assert isinstance(counts, dict)
    assert counts.get("inputs") == 1
    assert counts.get("processed") == 1
