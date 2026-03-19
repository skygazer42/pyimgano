from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import pyimgano.infer_cli as infer_cli
from pyimgano.inference.api import InferenceResult


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

        def decision_function(self, x):
            for item in x:
                if "bad" in str(item):
                    raise ValueError("boom")
            return np.linspace(0.0, 1.0, num=len(x), dtype=np.float32)

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


def test_infer_cli_continue_on_error_delegates_to_inference_service(
    tmp_path, monkeypatch
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a_good.png")
    _write_png(input_dir / "b_bad.png")

    out_jsonl = tmp_path / "out.jsonl"

    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: object())

    import pyimgano.services.inference_service as inference_service

    calls: list[list[str]] = []

    def fake_run_inference(**kwargs):
        inputs = [str(item) for item in kwargs["inputs"]]
        calls.append(inputs)
        if any("bad" in item for item in inputs):
            raise ValueError("boom")
        return inference_service.InferenceRunResult(
            records=[InferenceResult(score=0.1, label=0, anomaly_map=None) for _ in inputs],
            timing_seconds=0.0,
        )

    monkeypatch.setattr(inference_service, "run_inference", fake_run_inference)

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
    assert any(len(call) == 2 for call in calls)
    assert [str(input_dir / "a_good.png")] in calls
    assert [str(input_dir / "b_bad.png")] in calls


def test_infer_cli_continue_on_error_delegates_to_continue_service(
    tmp_path, monkeypatch
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")
    _write_png(input_dir / "b.png")

    out_jsonl = tmp_path / "out.jsonl"

    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: object())

    import pyimgano.services.infer_continue_service as infer_continue_service

    calls: list[object] = []
    monkeypatch.setattr(
        infer_continue_service,
        "run_continue_on_error_inference",
        lambda request, *, process_ok_result=None, handle_error=None, run_inference_impl=None: (
            calls.append(
                {
                    "request": request,
                    "process_ok_result": process_ok_result,
                    "handle_error": handle_error,
                    "run_inference_impl": run_inference_impl,
                }
            )
            or infer_continue_service.ContinueOnErrorInferResult(
                processed=2,
                errors=1,
                timing_seconds=0.0,
                stop_early=False,
            )
        ),
    )

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
    assert len(calls) == 1
    request = calls[0]["request"]
    assert request.inputs == [str(input_dir / "a.png"), str(input_dir / "b.png")]
    assert request.batch_size == 2
    assert calls[0]["process_ok_result"] is not None
    assert calls[0]["handle_error"] is not None
    assert calls[0]["run_inference_impl"] is not None


def test_infer_cli_validates_save_maps_before_inference_service(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    monkeypatch.setattr(infer_cli, "create_model", lambda name, **kwargs: object())

    import pyimgano.services.inference_service as inference_service

    called = {"run": 0, "iter": 0}

    def fake_run_inference(**kwargs):
        _ = kwargs
        called["run"] += 1
        return inference_service.InferenceRunResult(records=[], timing_seconds=0.0)

    def fake_iter_inference_records(**kwargs):
        _ = kwargs
        called["iter"] += 1
        yield InferenceResult(score=0.1, label=0, anomaly_map=None)

    monkeypatch.setattr(inference_service, "run_inference", fake_run_inference)
    monkeypatch.setattr(inference_service, "iter_inference_records", fake_iter_inference_records)

    rc = infer_cli.main(
        [
            "--model",
            "vision_ecod",
            "--input",
            str(input_dir),
            "--save-maps",
            str(tmp_path / "maps"),
        ]
    )
    assert rc == 2
    assert called == {"run": 0, "iter": 0}


def test_infer_cli_profile_json_writes_payload(tmp_path, monkeypatch) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_png(input_dir / "a.png")

    out_jsonl = tmp_path / "out.jsonl"
    profile_path = tmp_path / "profile.json"

    class _OK:
        def __init__(self) -> None:
            self.threshold_ = 0.5

        def decision_function(self, x):
            return np.asarray([0.1 for _ in x], dtype=np.float32)

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

