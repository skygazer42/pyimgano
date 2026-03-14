from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pyimgano.workbench.category_outputs import (
    WorkbenchCategoryOutputs,
    save_workbench_category_outputs,
)


def test_workbench_category_outputs_writes_report_maps_and_jsonl(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": "custom",
        "category": "custom",
        "model": "vision_ecod",
        "recipe": "industrial-adapt",
        "threshold": 0.5,
        "results": {"auroc": 1.0},
    }
    outputs = WorkbenchCategoryOutputs(
        payload=payload,
        test_inputs=[tmp_path / "good.png", np.zeros((4, 4, 3), dtype=np.uint8)],
        test_labels=np.asarray([0, 1], dtype=np.int64),
        scores=np.asarray([0.1, 0.9], dtype=np.float32),
        threshold=0.5,
        maps=[
            np.ones((3, 3), dtype=np.float32),
            None,
        ],
        test_meta=[{"frame": 0}, None],
    )

    save_workbench_category_outputs(
        run_dir=run_dir,
        outputs=outputs,
        save_maps=True,
        per_image_jsonl=True,
    )

    report_path = run_dir / "categories" / "custom" / "report.json"
    assert report_path.exists()
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert np.isclose(saved_report["threshold"], 0.5)

    records_path = run_dir / "categories" / "custom" / "per_image.jsonl"
    assert records_path.exists()
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2

    first = records[0]
    assert first["input"].endswith("good.png")
    assert first["meta"] == {"frame": 0}
    assert first["anomaly_map"]["path"].endswith(".npy")
    assert first["anomaly_map"]["shape"] == [3, 3]

    second = records[1]
    assert second["input"] == "numpy[1]"
    assert "anomaly_map" not in second

    map_path = run_dir / first["anomaly_map"]["path"]
    assert map_path.exists()
    loaded_map = np.load(map_path)
    assert np.allclose(loaded_map, np.ones((3, 3), dtype=np.float32))
