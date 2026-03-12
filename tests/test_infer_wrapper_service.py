from __future__ import annotations

from pyimgano.services.infer_wrapper_service import (
    InferDetectorWrapperRequest,
    apply_infer_detector_wrappers,
)


def test_infer_wrapper_service_exports_expected_boundary() -> None:
    import pyimgano.services.infer_wrapper_service as infer_wrapper_service

    assert infer_wrapper_service.__all__ == [
        "InferDetectorWrapperRequest",
        "InferDetectorWrapperResult",
        "apply_infer_detector_wrappers",
    ]


def test_apply_infer_detector_wrappers_applies_defaults_and_preserves_threshold(
    monkeypatch,
) -> None:
    import pyimgano.inference.preprocessing as preprocessing
    import pyimgano.inference.tiling as tiling

    events: list[dict[str, object]] = []

    class _BaseDetector:
        threshold_ = None

    class _FakeTiledDetector:
        def __init__(
            self,
            *,
            detector,
            tile_size: int,
            stride: int | None,
            score_reduce: str,
            score_topk: float,
            map_reduce: str,
            u16_max: int | None,
        ) -> None:
            events.append(
                {
                    "kind": "tiling",
                    "detector": detector,
                    "tile_size": int(tile_size),
                    "stride": (None if stride is None else int(stride)),
                    "score_reduce": str(score_reduce),
                    "score_topk": float(score_topk),
                    "map_reduce": str(map_reduce),
                    "u16_max": (None if u16_max is None else int(u16_max)),
                }
            )
            self.detector = detector

        def __getattr__(self, name: str):
            return getattr(self.detector, name)

    class _FakePreprocessingDetector:
        def __init__(self, *, detector, illumination_contrast, u16_max: int | None) -> None:
            events.append(
                {
                    "kind": "preprocessing",
                    "detector": detector,
                    "illumination_contrast": illumination_contrast,
                    "u16_max": (None if u16_max is None else int(u16_max)),
                }
            )
            self.detector = detector

        def __getattr__(self, name: str):
            return getattr(self.detector, name)

    monkeypatch.setattr(tiling, "TiledDetector", _FakeTiledDetector)
    monkeypatch.setattr(preprocessing, "PreprocessingDetector", _FakePreprocessingDetector)

    base = _BaseDetector()
    result = apply_infer_detector_wrappers(
        InferDetectorWrapperRequest(
            detector=base,
            model_name="vision_patchcore",
            threshold=0.73,
            tiling_payload={
                "tile_size": 4,
                "stride": 3,
                "score_reduce": "topk_mean",
                "score_topk": 0.2,
                "map_reduce": "hann",
            },
            tile_size=None,
            tile_stride=None,
            tile_score_reduce="max",
            tile_score_topk=0.1,
            tile_map_reduce="max",
            illumination_contrast_knobs={"white_balance": "gray_world"},
            u16_max=4095,
        )
    )

    assert events == [
        {
            "kind": "tiling",
            "detector": base,
            "tile_size": 4,
            "stride": 3,
            "score_reduce": "topk_mean",
            "score_topk": 0.2,
            "map_reduce": "hann",
            "u16_max": 4095,
        },
        {
            "kind": "preprocessing",
            "detector": result.detector.detector,
            "illumination_contrast": {"white_balance": "gray_world"},
            "u16_max": 4095,
        },
    ]
    assert result.detector.threshold_ == 0.73
    assert result.detector.detector.threshold_ == 0.73
