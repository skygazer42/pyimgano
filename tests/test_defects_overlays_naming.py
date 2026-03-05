from __future__ import annotations


def test_overlay_filename_includes_region_ids_and_score_stats() -> None:
    from pyimgano.defects.overlays import build_overlay_filename

    regions = [
        {"id": 3, "score_max": 0.9, "score_mean": 0.5, "area": 10},
        {"id": 7, "score_max": 0.8, "score_mean": 0.4, "area": 5},
    ]

    name = build_overlay_filename(index=12, stem="img", regions=regions)
    assert name.startswith("000012_img__")
    assert "r3-7" in name
    assert "smax0.900" in name
    assert "smean0.500" in name
    assert "a10" in name
