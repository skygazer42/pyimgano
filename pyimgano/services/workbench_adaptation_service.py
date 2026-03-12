from __future__ import annotations

from typing import Any, Mapping


def build_postprocess_from_payload(payload: Mapping[str, Any]) -> Any:
    import pyimgano.workbench.adaptation as adaptation

    pr_raw = payload.get("percentile_range", (1.0, 99.0))
    if isinstance(pr_raw, (list, tuple)) and len(pr_raw) == 2:
        percentile_range = (float(pr_raw[0]), float(pr_raw[1]))
    else:
        percentile_range = (1.0, 99.0)

    ct = payload.get("component_threshold", None)
    component_threshold = float(ct) if ct is not None else None

    return adaptation.build_postprocess(
        adaptation.MapPostprocessConfig(
            normalize=bool(payload.get("normalize", True)),
            normalize_method=str(payload.get("normalize_method", "minmax")),
            percentile_range=percentile_range,
            gaussian_sigma=float(payload.get("gaussian_sigma", 0.0)),
            morph_open_ksize=int(payload.get("morph_open_ksize", 0)),
            morph_close_ksize=int(payload.get("morph_close_ksize", 0)),
            component_threshold=component_threshold,
            min_component_area=int(payload.get("min_component_area", 0)),
        )
    )


__all__ = ["build_postprocess_from_payload"]
