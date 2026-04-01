from __future__ import annotations

import json
from pathlib import Path


def test_demo_cli_runs_and_writes_tables(tmp_path: Path, capsys) -> None:
    from pyimgano.demo_cli import main as demo_main

    dataset_root = tmp_path / "demo_dataset"
    out_dir = tmp_path / "suite_out"

    rc = demo_main(
        [
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(out_dir),
            "--suite",
            "industrial-ci",
            "--sweep",
            "industrial-small",
            "--sweep-max-variants",
            "1",
            "--export",
            "csv",
            "--resize",
            "32",
            "32",
            "--limit-train",
            "2",
            "--limit-test",
            "2",
            "--no-pretrained",
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "Suite run dir" in out

    assert (out_dir / "report.json").exists()
    assert (out_dir / "environment.json").exists()

    assert (out_dir / "leaderboard.csv").exists()
    assert (out_dir / "best_by_baseline.csv").exists()
    assert (out_dir / "skipped.csv").exists()


def test_demo_cli_can_run_infer_defects_loop(tmp_path: Path) -> None:
    from pyimgano.demo_cli import main as demo_main

    dataset_root = tmp_path / "demo_dataset"
    out_dir = tmp_path / "suite_out"

    rc = demo_main(
        [
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(out_dir),
            "--suite",
            "industrial-ci",
            "--no-sweep",
            "--export",
            "none",
            "--resize",
            "32",
            "32",
            "--limit-train",
            "2",
            "--limit-test",
            "2",
            "--no-pretrained",
            "--infer-defects",
        ]
    )
    assert rc == 0

    infer_dir = out_dir / "infer"
    assert (infer_dir / "results.jsonl").exists()
    assert (infer_dir / "masks").exists()
    assert (infer_dir / "overlays").exists()
    assert (infer_dir / "regions.jsonl").exists()

    overlays = sorted((infer_dir / "overlays").glob("*.png"))
    assert overlays, "expected at least one overlay artifact"

    regions_lines = (infer_dir / "regions.jsonl").read_text(encoding="utf-8").splitlines()
    assert regions_lines, "expected at least one regions JSONL record"
    r0 = json.loads(regions_lines[0])
    assert isinstance(r0, dict)
    assert "defects" in r0


def test_demo_cli_smoke_can_write_summary_and_next_steps(tmp_path: Path, capsys) -> None:
    from pyimgano.demo_cli import main as demo_main

    dataset_root = tmp_path / "demo_dataset"
    out_dir = tmp_path / "suite_out"
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
            "--emit-next-steps",
            "--no-pretrained",
        ]
    )
    assert rc == 0

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["smoke"] is True
    assert payload["dataset_root"] == str(dataset_root)
    assert payload["run_dir"] == str(out_dir)
    assert isinstance(payload["next_steps"], list)
    assert payload["next_steps"]

    out = capsys.readouterr().out
    assert "Next steps:" in out
    assert "pyimgano-infer" in out


def test_demo_cli_benchmark_scenario_writes_scenario_summary_and_publication_next_step(
    tmp_path: Path,
) -> None:
    from pyimgano.demo_cli import main as demo_main

    dataset_root = tmp_path / "demo_dataset"
    out_dir = tmp_path / "suite_out"
    summary_path = tmp_path / "benchmark_summary.json"

    rc = demo_main(
        [
            "--scenario",
            "benchmark",
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

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["scenario"] == "benchmark"
    assert payload["smoke"] is False
    assert "pyimgano runs publication" in " || ".join(payload["next_steps"])
    assert (out_dir / "leaderboard.csv").exists()
    assert (out_dir / "leaderboard_metadata.json").exists()


def test_demo_cli_infer_defects_scenario_writes_infer_artifacts_and_scenario_next_steps(
    tmp_path: Path,
) -> None:
    from pyimgano.demo_cli import main as demo_main

    dataset_root = tmp_path / "demo_dataset"
    out_dir = tmp_path / "suite_out"
    summary_path = tmp_path / "infer_defects_summary.json"

    rc = demo_main(
        [
            "--scenario",
            "infer-defects",
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

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["scenario"] == "infer-defects"
    assert "pyimgano-infer --from-run" in " || ".join(payload["next_steps"])
    assert (out_dir / "infer" / "results.jsonl").exists()
    assert (out_dir / "infer" / "regions.jsonl").exists()
