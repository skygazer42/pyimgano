import re
from pathlib import Path

from pyimgano.reporting.runs import build_workbench_run_dir_name, build_workbench_run_paths


def test_build_workbench_run_dir_name_includes_components_and_sanitizes():
    name = build_workbench_run_dir_name(
        dataset="my dataset",
        recipe="industrial-adapt",
        model="vision/patchcore",
        category="bottle",
    )

    assert re.match(r"^\d{8}_\d{6}_", name)
    assert "my_dataset" in name
    assert "industrial-adapt" in name
    assert "vision_patchcore" in name
    assert name.endswith("_bottle")


def test_build_workbench_run_paths_shape(tmp_path):
    run_dir = tmp_path / "run"
    paths = build_workbench_run_paths(Path(run_dir))
    assert paths.run_dir == Path(run_dir)
    assert paths.report_json == Path(run_dir) / "report.json"
    assert paths.config_json == Path(run_dir) / "config.json"
    assert paths.environment_json == Path(run_dir) / "environment.json"
    assert paths.categories_dir == Path(run_dir) / "categories"
    assert paths.checkpoints_dir == Path(run_dir) / "checkpoints"
    assert paths.artifacts_dir == Path(run_dir) / "artifacts"
