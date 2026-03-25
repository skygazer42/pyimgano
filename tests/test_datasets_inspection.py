from __future__ import annotations


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
