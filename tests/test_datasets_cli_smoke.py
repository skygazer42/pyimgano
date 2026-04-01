from __future__ import annotations

import json
from pathlib import Path


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE"
            "0000000C49444154789C63606060000000040001F61738550000000049454E44AE426082"
        )
    )


def test_datasets_cli_list_json_smoke(capsys) -> None:
    from pyimgano.datasets_cli import main as datasets_main

    rc = datasets_main(["list", "--json"])
    assert rc == 0

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert isinstance(payload, list)
    names = {str(item.get("name")) for item in payload if isinstance(item, dict)}
    assert "custom" in names


def test_datasets_cli_detect_import_and_lint_custom_layout(tmp_path: Path, capsys) -> None:
    from pyimgano.datasets_cli import main as datasets_main

    root = tmp_path / "custom"
    _write_png(root / "train" / "normal" / "train_0.png")
    _write_png(root / "test" / "normal" / "good_0.png")
    _write_png(root / "test" / "anomaly" / "bad_0.png")
    _write_png(root / "ground_truth" / "anomaly" / "bad_0_mask.png")

    rc = datasets_main(["detect", str(root), "--json"])
    assert rc == 0
    detected = json.loads(capsys.readouterr().out)
    assert detected["detected"] == "custom"
    assert detected["path_kind"] == "directory"

    manifest_path = tmp_path / "out" / "manifest.jsonl"
    rc = datasets_main(
        [
            "import",
            "--root",
            str(root),
            "--out",
            str(manifest_path),
            "--dataset",
            "auto",
            "--include-masks",
            "--json",
        ]
    )
    assert rc == 0
    imported = json.loads(capsys.readouterr().out)
    assert imported["dataset"] == "custom"
    assert imported["record_count"] == 3
    assert manifest_path.exists()

    rc = datasets_main(["lint", str(manifest_path), "--json"])
    assert rc == 0
    linted = json.loads(capsys.readouterr().out)
    assert linted["ok"] is True
    assert linted["dataset"] == "manifest"
    assert linted["stats"]["total_records"] == 3
    assert linted["validation"]["record_count"] == 3


def test_datasets_cli_profile_reports_industrial_dataset_profile(tmp_path: Path, capsys) -> None:
    from pyimgano.datasets_cli import main as datasets_main

    root = tmp_path / "custom"
    _write_png(root / "train" / "normal" / "train_0.png")
    _write_png(root / "test" / "normal" / "good_0.png")
    _write_png(root / "test" / "anomaly" / "bad_0.png")
    _write_png(root / "ground_truth" / "anomaly" / "bad_0_mask.png")

    rc = datasets_main(["profile", str(root), "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset"] == "custom"
    assert payload["source_kind"] == "converted_temp"

    profile = payload["dataset_profile"]
    assert profile["total_records"] == 3
    assert profile["train_count"] == 1
    assert profile["test_count"] == 2
    assert profile["test_normal_count"] == 1
    assert profile["test_anomaly_count"] == 1
    assert profile["categories"] == ["custom"]
    assert profile["category_count"] == 1
    assert profile["has_masks"] is True
    assert profile["pixel_metrics_available"] is True
    assert profile["fewshot_risk"] is True
    assert profile["multi_category"] is False

    readiness = payload["evaluation_readiness"]
    assert readiness["ready_for_image_metrics"] is True
    assert readiness["ready_for_pixel_metrics"] is True
    assert payload["readiness"]["status"] == "warning"
    assert payload["readiness"]["issue_codes"] == ["FEWSHOT_TRAIN_SET"]


def test_datasets_cli_lint_text_renders_readiness_issue_codes(tmp_path: Path, capsys) -> None:
    from pyimgano.datasets_cli import main as datasets_main

    root = tmp_path / "custom"
    _write_png(root / "train" / "normal" / "train_0.png")
    _write_png(root / "test" / "normal" / "good_0.png")

    rc = datasets_main(["lint", str(root), "--dataset", "custom"])
    assert rc == 1

    out = capsys.readouterr().out
    assert "readiness_status: error" in out
    assert "issue_codes: MISSING_TEST_ANOMALY, PIXEL_METRICS_UNAVAILABLE, FEWSHOT_TRAIN_SET" in out
