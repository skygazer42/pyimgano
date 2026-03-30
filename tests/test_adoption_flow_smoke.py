from __future__ import annotations

import json
from pathlib import Path


def test_adoption_entrypoints_work_together(tmp_path: Path, capsys) -> None:
    from pyimgano.cli import main as benchmark_main
    from pyimgano.demo_cli import main as demo_main
    from pyimgano.doctor_cli import main as doctor_main
    from pyimgano.pyim_cli import main as pyim_main

    rc = doctor_main(["--json", "--recommend-extras", "--for-command", "export-onnx"])
    assert rc == 0
    doctor_payload = json.loads(capsys.readouterr().out)
    assert doctor_payload["extras_recommendation"]["target"] == "export-onnx"

    rc = pyim_main(
        [
            "--list",
            "models",
            "--objective",
            "latency",
            "--selection-profile",
            "cpu-screening",
            "--topk",
            "2",
            "--json",
        ]
    )
    assert rc == 0
    pyim_payload = json.loads(capsys.readouterr().out)
    assert pyim_payload["starter_picks"][0]["name"] == "vision_ecod"

    dataset_root = tmp_path / "demo_dataset"
    out_dir = tmp_path / "demo_suite"
    summary_path = tmp_path / "demo_summary.json"
    rc = demo_main(
        [
            "--smoke",
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(out_dir),
            "--summary-json",
            str(summary_path),
            "--no-pretrained",
        ]
    )
    assert rc == 0
    capsys.readouterr()
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["run_dir"] == str(out_dir)

    rc = benchmark_main(
        [
            "--starter-config-info",
            "official_mvtec_industrial_v4_cpu_offline.json",
            "--json",
        ]
    )
    assert rc == 0
    starter_payload = json.loads(capsys.readouterr().out)
    assert starter_payload["starter"] is True
