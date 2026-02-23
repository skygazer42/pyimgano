from __future__ import annotations

from dataclasses import asdict
from typing import Any

from pyimgano.pipelines.run_benchmark import run_benchmark
from pyimgano.recipes.registry import register_recipe
from pyimgano.reporting.environment import collect_environment
from pyimgano.reporting.report import save_run_report
from pyimgano.reporting.runs import (
    build_workbench_run_dir_name,
    build_workbench_run_paths,
    ensure_run_dir,
)
from pyimgano.workbench.config import WorkbenchConfig


@register_recipe(
    "industrial-adapt",
    tags=("builtin", "adaptation"),
    metadata={
        "description": "Adaptation-first industrial benchmark wrapper (uses pyimgano-benchmark pipeline).",
    },
)
def industrial_adapt(config: WorkbenchConfig) -> dict[str, Any]:
    recipe_name = "industrial-adapt"

    category_for_name: str | None
    if str(config.dataset.category).lower() == "all":
        category_for_name = None
    else:
        category_for_name = str(config.dataset.category)

    run_name = build_workbench_run_dir_name(
        dataset=str(config.dataset.name),
        recipe=recipe_name,
        model=str(config.model.name),
        category=category_for_name,
    )
    run_dir = ensure_run_dir(output_dir=config.output.output_dir, name=run_name)
    paths = build_workbench_run_paths(run_dir)

    model_kwargs = dict(config.model.model_kwargs)
    if config.model.checkpoint_path is not None:
        model_kwargs.setdefault("checkpoint_path", str(config.model.checkpoint_path))

    report = run_benchmark(
        dataset=str(config.dataset.name),
        root=str(config.dataset.root),
        category=str(config.dataset.category),
        model=str(config.model.name),
        input_mode=str(config.dataset.input_mode),
        seed=config.seed,
        device=str(config.model.device),
        preset=config.model.preset,
        pretrained=bool(config.model.pretrained),
        contamination=float(config.model.contamination),
        resize=tuple(config.dataset.resize),
        model_kwargs=model_kwargs,
        limit_train=config.dataset.limit_train,
        limit_test=config.dataset.limit_test,
        save_run=bool(config.output.save_run),
        per_image_jsonl=bool(config.output.per_image_jsonl),
        output_dir=paths.run_dir,
    )

    report_out = dict(report)
    report_out["recipe"] = recipe_name
    report_out["run_dir"] = str(paths.run_dir)

    if bool(config.output.save_run):
        paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Preserve the benchmark pipeline config before overwriting config.json with
        # the workbench config.
        benchmark_config_path = paths.artifacts_dir / "benchmark_config.json"
        if paths.config_json.exists():
            benchmark_config_path.write_text(paths.config_json.read_text(encoding="utf-8"), encoding="utf-8")

        save_run_report(paths.config_json, {"config": asdict(config)})
        save_run_report(paths.environment_json, collect_environment())

    return report_out
