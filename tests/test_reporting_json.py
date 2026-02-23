import json

import numpy as np

from pyimgano.reporting.report import save_jsonl_records, save_run_report


def test_save_run_report(tmp_path):
    path = tmp_path / "report.json"
    results = {
        "auroc": np.float64(0.9),
        "flag": np.bool_(False),
        "labels": np.array([0, 1, 0]),
        "nested": {"x": np.int64(1), 2: np.bool_(True), "arr": np.array([1.0, 2.0], dtype=np.float32)},
        "path": tmp_path / "artifact.bin",
    }
    save_run_report(path, results)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["auroc"] == 0.9
    assert data["flag"] is False
    assert data["labels"] == [0, 1, 0]
    assert data["nested"]["x"] == 1
    assert data["nested"]["2"] is True
    assert data["nested"]["arr"] == [1.0, 2.0]
    assert data["path"].endswith("artifact.bin")


def test_save_jsonl_records(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [
        {
            "i": np.int64(1),
            "score": np.float32(0.25),
            "ok": np.bool_(True),
            "path": tmp_path / "x.png",
        },
        {"nested": {"arr": np.asarray([1, 2, 3], dtype=np.int32)}},
    ]

    save_jsonl_records(path, records)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    parsed0 = json.loads(lines[0])
    assert parsed0["i"] == 1
    assert parsed0["score"] == 0.25
    assert parsed0["ok"] is True
    assert parsed0["path"].endswith("x.png")

    parsed1 = json.loads(lines[1])
    assert parsed1["nested"]["arr"] == [1, 2, 3]
