from __future__ import annotations

import json
from pathlib import Path


def test_resolve_category_uses_single_detected_candidate() -> None:
    from pyimgano.datasets.inspection import _resolve_category

    detection = {
        "candidates": [
            {
                "name": "mvtec_ad2",
                "category_candidates": ["bottle"],
            }
        ]
    }

    assert (
        _resolve_category(
            dataset="mvtec_ad2",
            detection=detection,
            category=None,
        )
        == "bottle"
    )


def test_build_profile_sections_counts_manifest_splits(tmp_path: Path) -> None:
    from pyimgano.datasets.inspection import _build_profile_sections

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps({"image_path": "train.png", "category": "demo", "split": "train"}),
                json.dumps(
                    {"image_path": "good.png", "category": "demo", "split": "test", "label": 0}
                ),
                json.dumps(
                    {"image_path": "bad.png", "category": "demo", "split": "test", "label": 1}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (
        dataset_profile,
        task_profile,
        constraints,
        evaluation_readiness,
        stats,
    ) = _build_profile_sections(
        manifest_path=manifest_path,
        root_fallback=tmp_path,
    )

    assert dataset_profile["train_count"] == 1
    assert dataset_profile["test_normal_count"] == 1
    assert dataset_profile["test_anomaly_count"] == 1
    assert task_profile["train_split_present"] is True
    assert constraints["fewshot_risk"] is True
    assert evaluation_readiness["ready_for_image_metrics"] is True
    assert isinstance(stats, dict)


def test_profile_dataset_target_emits_readiness_issue_codes_for_fewshot_custom_layout(
    tmp_path: Path,
) -> None:
    from pyimgano.datasets.inspection import profile_dataset_target

    root = tmp_path / "custom"
    (root / "train" / "normal").mkdir(parents=True, exist_ok=True)
    (root / "test" / "normal").mkdir(parents=True, exist_ok=True)
    (root / "test" / "anomaly").mkdir(parents=True, exist_ok=True)
    (root / "ground_truth" / "anomaly").mkdir(parents=True, exist_ok=True)
    (root / "train" / "normal" / "train_0.png").write_bytes(b"png")
    (root / "test" / "normal" / "good_0.png").write_bytes(b"png")
    (root / "test" / "anomaly" / "bad_0.png").write_bytes(b"png")
    (root / "ground_truth" / "anomaly" / "bad_0_mask.png").write_bytes(b"png")

    payload = profile_dataset_target(target=root)

    readiness = payload["readiness"]
    assert readiness["status"] == "warning"
    assert readiness["issue_codes"] == ["FEWSHOT_TRAIN_SET"]
    assert readiness["issue_details"] == [
        {
            "code": "FEWSHOT_TRAIN_SET",
            "message": "Train split has fewer than 16 normal samples; results may be unstable.",
        }
    ]


def test_lint_dataset_target_emits_error_issue_codes_for_missing_test_anomaly(tmp_path: Path) -> None:
    from pyimgano.datasets.inspection import lint_dataset_target

    root = tmp_path / "custom"
    (root / "train" / "normal").mkdir(parents=True, exist_ok=True)
    (root / "test" / "normal").mkdir(parents=True, exist_ok=True)
    (root / "train" / "normal" / "train_0.png").write_bytes(b"png")
    (root / "test" / "normal" / "good_0.png").write_bytes(b"png")

    payload = lint_dataset_target(target=root, dataset="custom")

    readiness = payload["readiness"]
    assert readiness["status"] == "error"
    assert "MISSING_TEST_ANOMALY" in set(readiness["issue_codes"])
    assert "PIXEL_METRICS_UNAVAILABLE" in set(readiness["issue_codes"])
