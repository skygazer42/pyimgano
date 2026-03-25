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

    assert _resolve_category(
        dataset="mvtec_ad2",
        detection=detection,
        category=None,
    ) == "bottle"


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

    dataset_profile, task_profile, constraints, evaluation_readiness, stats = _build_profile_sections(
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
