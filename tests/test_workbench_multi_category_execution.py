from __future__ import annotations

from pathlib import Path

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.multi_category_execution import run_all_workbench_categories


def test_run_all_workbench_categories_orchestrates_category_fanout(
    monkeypatch, tmp_path: Path
) -> None:
    import pyimgano.workbench.multi_category_execution as execution_module

    calls: dict[str, object] = {}

    def _fake_list_workbench_categories(*, config):  # noqa: ANN001 - test seam
        calls["categories"] = str(config.dataset.name)
        return ["bottle", "capsule"]

    def _fake_run_workbench_category(
        *, config, recipe_name, category, run_dir
    ):  # noqa: ANN001 - test seam
        fanout = calls.setdefault("per_category", [])
        fanout.append(
            {
                "recipe_name": str(recipe_name),
                "category": str(category),
                "run_dir": run_dir,
            }
        )
        return {"dataset": str(config.dataset.name), "category": str(category)}

    def _fake_build_workbench_aggregate_report(
        *, config, recipe_name, categories, per_category
    ):  # noqa: ANN001 - test seam
        calls["aggregate"] = {
            "dataset": str(config.dataset.name),
            "recipe_name": str(recipe_name),
            "categories": list(categories),
            "per_category": dict(per_category),
        }
        return {"dataset": "custom", "category": "all", "results": {"mean_auroc": 0.95}}

    monkeypatch.setattr(
        execution_module,
        "list_workbench_categories",
        _fake_list_workbench_categories,
    )
    monkeypatch.setattr(
        execution_module,
        "run_workbench_category",
        _fake_run_workbench_category,
    )
    monkeypatch.setattr(
        execution_module,
        "build_workbench_aggregate_report",
        _fake_build_workbench_aggregate_report,
    )

    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 123,
            "dataset": {
                "name": "custom",
                "root": str(tmp_path / "dataset"),
                "category": "all",
                "resize": [16, 16],
                "input_mode": "paths",
            },
            "model": {"name": "vision_ecod", "device": "cpu", "pretrained": False},
            "output": {"output_dir": str(tmp_path / "run_out"), "save_run": True},
        }
    )

    payload = run_all_workbench_categories(
        config=cfg,
        recipe_name="industrial-adapt",
        run_dir=tmp_path / "run_out",
    )

    assert payload == {"dataset": "custom", "category": "all", "results": {"mean_auroc": 0.95}}
    assert calls["categories"] == "custom"
    assert calls["per_category"] == [
        {
            "recipe_name": "industrial-adapt",
            "category": "bottle",
            "run_dir": tmp_path / "run_out",
        },
        {
            "recipe_name": "industrial-adapt",
            "category": "capsule",
            "run_dir": tmp_path / "run_out",
        },
    ]
    assert calls["aggregate"] == {
        "dataset": "custom",
        "recipe_name": "industrial-adapt",
        "categories": ["bottle", "capsule"],
        "per_category": {
            "bottle": {"dataset": "custom", "category": "bottle"},
            "capsule": {"dataset": "custom", "category": "capsule"},
        },
    }
