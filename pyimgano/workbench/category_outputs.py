from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pyimgano.reporting.report import save_jsonl_records, save_run_report
from pyimgano.reporting.runs import build_workbench_run_paths
from pyimgano.workbench.maps import save_anomaly_map_npy


@dataclass(frozen=True)
class WorkbenchCategoryOutputs:
    payload: Mapping[str, Any]
    test_inputs: Sequence[Any]
    test_labels: np.ndarray
    scores: np.ndarray
    threshold: float
    maps: Sequence[np.ndarray | None] | None = None
    test_meta: Sequence[Mapping[str, Any] | None] | None = None


def _serialize_input_value(item: Any, *, index: int) -> str:
    if isinstance(item, (str, Path)):
        return str(item)
    return f"numpy[{int(index)}]"


def _save_map_artifacts(
    *,
    run_dir: Path,
    test_inputs: Sequence[Any],
    maps: Sequence[np.ndarray | None],
) -> list[str | None]:
    paths = build_workbench_run_paths(run_dir)
    map_paths: list[str | None] = []

    for i, (item, anomaly_map) in enumerate(zip(test_inputs, maps)):
        if anomaly_map is None:
            map_paths.append(None)
            continue
        saved = save_anomaly_map_npy(
            paths.artifacts_dir,
            index=int(i),
            input_path=_serialize_input_value(item, index=i),
            anomaly_map=np.asarray(anomaly_map, dtype=np.float32),
        )
        try:
            rel = saved.relative_to(paths.run_dir)
            map_paths.append(str(rel))
        except Exception:
            map_paths.append(str(saved))

    return map_paths


def _build_per_image_records(
    *,
    outputs: WorkbenchCategoryOutputs,
    map_paths: Sequence[str | None] | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    y_true = np.asarray(outputs.test_labels).astype(int).tolist()
    pred = (np.asarray(outputs.scores) >= float(outputs.threshold)).astype(int).tolist()
    dataset = str(outputs.payload["dataset"])
    category = str(outputs.payload["category"])

    for i, item in enumerate(outputs.test_inputs):
        record: dict[str, Any] = {
            "index": int(i),
            "dataset": dataset,
            "category": category,
            "input": _serialize_input_value(item, index=i),
            "y_true": int(y_true[i]),
            "score": float(np.asarray(outputs.scores)[i]),
            "threshold": float(outputs.threshold),
            "pred": int(pred[i]),
        }
        if outputs.test_meta is not None:
            meta = outputs.test_meta[i]
            if meta is not None:
                record["meta"] = dict(meta)
        if (
            map_paths is not None
            and i < len(map_paths)
            and map_paths[i] is not None
            and outputs.maps is not None
        ):
            anomaly_map = outputs.maps[i]
            if anomaly_map is not None:
                arr = np.asarray(anomaly_map)
                record["anomaly_map"] = {
                    "path": str(map_paths[i]),
                    "shape": [int(d) for d in arr.shape],
                    "dtype": str(arr.dtype),
                }
        records.append(record)

    return records


def save_workbench_category_outputs(
    *,
    run_dir: str | Path,
    outputs: WorkbenchCategoryOutputs,
    save_maps: bool,
    per_image_jsonl: bool,
) -> None:
    run_path = Path(run_dir)
    paths = build_workbench_run_paths(run_path)
    category = str(outputs.payload["category"])
    cat_dir = paths.categories_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    save_run_report(cat_dir / "report.json", dict(outputs.payload))

    map_paths: list[str | None] | None = None
    if bool(save_maps) and outputs.maps is not None:
        map_paths = _save_map_artifacts(
            run_dir=run_path,
            test_inputs=outputs.test_inputs,
            maps=outputs.maps,
        )

    if bool(per_image_jsonl):
        save_jsonl_records(
            cat_dir / "per_image.jsonl",
            _build_per_image_records(outputs=outputs, map_paths=map_paths),
        )


__all__ = ["WorkbenchCategoryOutputs", "save_workbench_category_outputs"]
