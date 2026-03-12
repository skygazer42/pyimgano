from __future__ import annotations

import json

import numpy as np
import pytest

from pyimgano.robust_cli import main


def test_robust_cli_list_models_delegates_to_discovery_service(monkeypatch, capsys) -> None:
    import pyimgano.services.discovery_service as discovery_service

    monkeypatch.setattr(
        discovery_service,
        "list_discovery_model_names",
        lambda **_kwargs: ["delegated_robust_model"],
    )

    rc = main(["--list-models"])
    assert rc == 0
    assert "delegated_robust_model" in capsys.readouterr().out


def test_robust_cli_list_models_uses_shared_listing_helper(monkeypatch) -> None:
    import pyimgano.robust_cli as robust_cli
    import pyimgano.services.discovery_service as discovery_service

    monkeypatch.setattr(
        discovery_service,
        "list_discovery_model_names",
        lambda **_kwargs: ["delegated_robust_model"],
    )

    calls = []
    monkeypatch.setattr(
        robust_cli,
        "cli_listing",
        type(
            "_StubCliListing",
            (),
            {
                "emit_listing": staticmethod(
                    lambda items, **kwargs: calls.append((list(items), kwargs)) or 71
                )
            },
        ),
        raising=False,
    )

    rc = robust_cli.main(["--list-models", "--json"])
    assert rc == 71
    assert calls == [
        (
            ["delegated_robust_model"],
            {"json_output": True, "json_payload": {"models": ["delegated_robust_model"]}},
        )
    ]


def test_robust_cli_delegates_run_mode_to_robustness_service(monkeypatch, capsys) -> None:
    import pyimgano.services.robustness_service as robustness_service

    calls = []

    monkeypatch.setattr(
        robustness_service,
        "run_robustness_request",
        lambda request: calls.append(request)
        or {
            "dataset": request.dataset,
            "category": request.category,
            "model": request.model,
            "robustness": {"clean": {}, "corruptions": {}},
        },
    )

    rc = main(
        [
            "--dataset",
            "mvtec",
            "--root",
            "/tmp",
            "--category",
            "bottle",
            "--model",
            "vision_ecod",
        ]
    )
    assert rc == 0
    assert calls[0].model == "vision_ecod"
    assert '"model": "vision_ecod"' in capsys.readouterr().out


def test_robust_cli_jsonable_output_uses_cli_output_helper(monkeypatch) -> None:
    import pyimgano.robust_cli as robust_cli
    import pyimgano.services.robustness_service as robustness_service

    monkeypatch.setattr(
        robustness_service,
        "run_robustness_request",
        lambda request: {
            "dataset": request.dataset,
            "category": request.category,
            "model": request.model,
            "robustness": {"clean": {}, "corruptions": {}},
        },
    )

    calls = []
    monkeypatch.setattr(
        robust_cli,
        "cli_output",
        type(
            "_StubCliOutput",
            (),
            {
                "emit_json": staticmethod(lambda payload, **kwargs: 0),
                "emit_jsonable": staticmethod(
                    lambda payload, **kwargs: calls.append((payload, kwargs)) or 19
                ),
                "print_cli_error": staticmethod(lambda exc, **kwargs: None),
            },
        ),
        raising=False,
    )

    rc = robust_cli.main(
        [
            "--dataset",
            "mvtec",
            "--root",
            "/tmp",
            "--category",
            "bottle",
            "--model",
            "vision_ecod",
        ]
    )
    assert rc == 19
    assert calls and calls[0][0]["model"] == "vision_ecod"


def test_robust_cli_smoke(tmp_path, capsys) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    import cv2

    root = tmp_path / "mvtec"
    cat = "bottle"

    (root / cat / "train" / "good").mkdir(parents=True)
    (root / cat / "test" / "good").mkdir(parents=True)
    (root / cat / "test" / "crack").mkdir(parents=True)
    (root / cat / "ground_truth" / "crack").mkdir(parents=True)

    img = np.ones((32, 32, 3), dtype=np.uint8) * 128
    cv2.imwrite(str(root / cat / "train" / "good" / "train_0.png"), img)
    cv2.imwrite(str(root / cat / "test" / "good" / "good_0.png"), img)

    bad = img.copy()
    bad[8:24, 8:24] = 255
    cv2.imwrite(str(root / cat / "test" / "crack" / "bad_0.png"), bad)

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    cv2.imwrite(str(root / cat / "ground_truth" / "crack" / "bad_0_mask.png"), mask)

    code = main(
        [
            "--dataset",
            "mvtec",
            "--root",
            str(root),
            "--category",
            cat,
            "--model",
            "vision_patchcore",
            "--device",
            "cpu",
            "--no-pretrained",
            "--no-pixel-segf1",
            "--corruptions",
            "lighting",
            "--severities",
            "1",
            "--limit-train",
            "1",
            "--limit-test",
            "2",
        ]
    )
    assert code == 0

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["dataset"] == "mvtec"
    assert payload["category"] == cat
    assert payload["model"] == "vision_patchcore"

    report = payload["robustness"]
    assert "clean" in report
    assert "corruptions" in report
    assert "lighting" in report["corruptions"]
    assert "severity_1" in report["corruptions"]["lighting"]


def test_robust_cli_paths_mode_runs_clean_only_for_path_detectors(tmp_path, capsys) -> None:
    import cv2

    root = tmp_path / "mvtec"
    cat = "bottle"

    (root / cat / "train" / "good").mkdir(parents=True)
    (root / cat / "test" / "good").mkdir(parents=True)
    (root / cat / "test" / "crack").mkdir(parents=True)
    (root / cat / "ground_truth" / "crack").mkdir(parents=True)

    img = np.ones((32, 32, 3), dtype=np.uint8) * 128
    cv2.imwrite(str(root / cat / "train" / "good" / "train_0.png"), img)
    cv2.imwrite(str(root / cat / "test" / "good" / "good_0.png"), img)

    bad = img.copy()
    bad[8:24, 8:24] = 255
    cv2.imwrite(str(root / cat / "test" / "crack" / "bad_0.png"), bad)

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    cv2.imwrite(str(root / cat / "ground_truth" / "crack" / "bad_0_mask.png"), mask)

    code = main(
        [
            "--dataset",
            "mvtec",
            "--root",
            str(root),
            "--category",
            cat,
            "--model",
            "vision_ecod",
            "--input-mode",
            "paths",
            "--limit-train",
            "1",
            "--limit-test",
            "2",
            "--corruptions",
            "lighting",
            "--severities",
            "1",
        ]
    )
    assert code == 0

    out = capsys.readouterr().out
    payload = json.loads(out)
    report = payload["robustness"]
    assert report["input_mode"] == "paths"
    assert report["corruption_mode"] == "clean_only"
    assert report["corruptions"] == {}
    assert "corruptions_skipped_reason" in report
    assert any("pixel_segf1 disabled" in n for n in report.get("notes", []))
