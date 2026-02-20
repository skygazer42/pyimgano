import json

import numpy as np

from pyimgano.reporting.report import save_run_report


def test_save_run_report(tmp_path):
    path = tmp_path / "report.json"
    results = {
        "auroc": np.float64(0.9),
        "labels": np.array([0, 1, 0]),
        "nested": {"x": np.int64(1)},
    }
    save_run_report(path, results)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["auroc"] == 0.9
    assert data["labels"] == [0, 1, 0]
    assert data["nested"]["x"] == 1

