from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pyimgano.inference.api import InferenceResult
from pyimgano.services.infer_artifact_service import (
    DefectsArtifactConfig,
    DefectsArtifactConfigBuildRequest,
    InferArtifactOptions,
    InferResultArtifactAssemblyRequest,
    InferResultArtifactBuildRequest,
    InferResultArtifactCliRequest,
    InferResultArtifactRequest,
    build_defects_artifact_config,
    build_infer_result_artifact_build_request_from_cli,
    build_infer_result_artifact_request,
    build_infer_result_artifact_request_from_cli,
    build_infer_result_artifact_request_from_options,
    materialize_infer_result_artifacts,
)


def _write_png(path: Path) -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


def test_materialize_infer_result_artifacts_saves_maps_masks_overlays_and_regions(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "inputs" / "a.png"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    _write_png(input_path)

    maps_dir = tmp_path / "maps"
    masks_dir = tmp_path / "masks"
    overlays_dir = tmp_path / "overlays"

    result = materialize_infer_result_artifacts(
        InferResultArtifactRequest(
            index=0,
            input_path=str(input_path),
            result=InferenceResult(
                score=0.9,
                label=1,
                anomaly_map=np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            ),
            include_status=True,
            include_anomaly_map_values=False,
            maps_dir=str(maps_dir),
            overlays_dir=str(overlays_dir),
            defects_config=DefectsArtifactConfig(
                pixel_threshold_value=0.5,
                pixel_threshold_provenance={"source": "explicit", "method": "fixed"},
                roi_xyxy_norm=None,
                mask_space="full",
                border_ignore_px=0,
                map_smoothing_method="none",
                map_smoothing_ksize=0,
                map_smoothing_sigma=0.0,
                hysteresis_enabled=False,
                hysteresis_low=None,
                hysteresis_high=None,
                open_ksize=0,
                close_ksize=0,
                fill_holes=False,
                mask_dilate_ksize=0,
                min_area=1,
                min_fill_ratio=None,
                max_aspect_ratio=None,
                min_solidity=None,
                min_score_max=None,
                min_score_mean=None,
                merge_nearby_enabled=False,
                merge_nearby_max_gap_px=0,
                max_regions_sort_by="score_max",
                max_regions=None,
                masks_dir=str(masks_dir),
                mask_format="png",
                defects_image_space=True,
            ),
        )
    )

    record = result.record
    assert record["status"] == "ok"
    assert record["index"] == 0
    assert record["input"] == str(input_path)
    assert record["anomaly_map"]["path"].endswith(".npy")

    defects = record["defects"]
    assert defects["pixel_threshold"] == pytest.approx(0.5)
    assert defects["pixel_threshold_provenance"]["source"] == "explicit"
    assert defects["mask"]["path"].endswith(".png")
    assert defects["mask"]["encoding"] == "png"
    assert len(defects["regions"]) == 1
    assert defects["regions"][0]["bbox_xyxy"] == [1, 1, 2, 2]
    assert defects["regions"][0]["bbox_xyxy_image"] == [2, 2, 5, 5]

    assert result.regions_payload is not None
    payload = result.regions_payload
    assert payload["input"] == str(input_path)
    assert payload["defects"]["regions"][0]["bbox_xyxy"] == [1, 1, 2, 2]

    assert sorted(maps_dir.glob("*.npy"))
    assert sorted(masks_dir.glob("*.png"))
    overlay_files = sorted(overlays_dir.glob("*.png"))
    assert len(overlay_files) == 1
    with Image.open(overlay_files[0]) as im:
        assert im.size == (8, 8)


def test_build_defects_artifact_config_populates_fields(tmp_path: Path) -> None:
    masks_dir = tmp_path / "masks"

    config = build_defects_artifact_config(
        DefectsArtifactConfigBuildRequest(
            defects_enabled=True,
            pixel_threshold_value=0.75,
            pixel_threshold_provenance={"source": "explicit"},
            roi_xyxy_norm=[0.1, 0.2, 0.9, 0.8],
            mask_space="full",
            border_ignore_px=2,
            map_smoothing_method="gaussian",
            map_smoothing_ksize=5,
            map_smoothing_sigma=1.25,
            hysteresis_enabled=True,
            hysteresis_low=0.2,
            hysteresis_high=0.8,
            open_ksize=3,
            close_ksize=4,
            fill_holes=True,
            mask_dilate_ksize=1,
            min_area=7,
            min_fill_ratio=0.3,
            max_aspect_ratio=4.5,
            min_solidity=0.6,
            min_score_max=0.7,
            min_score_mean=0.4,
            merge_nearby_enabled=True,
            merge_nearby_max_gap_px=6,
            max_regions_sort_by="area",
            max_regions=9,
            masks_dir=str(masks_dir),
            mask_format="npz",
            defects_image_space=True,
        )
    )

    assert config == DefectsArtifactConfig(
        pixel_threshold_value=0.75,
        pixel_threshold_provenance={"source": "explicit"},
        roi_xyxy_norm=[0.1, 0.2, 0.9, 0.8],
        mask_space="full",
        border_ignore_px=2,
        map_smoothing_method="gaussian",
        map_smoothing_ksize=5,
        map_smoothing_sigma=1.25,
        hysteresis_enabled=True,
        hysteresis_low=0.2,
        hysteresis_high=0.8,
        open_ksize=3,
        close_ksize=4,
        fill_holes=True,
        mask_dilate_ksize=1,
        min_area=7,
        min_fill_ratio=0.3,
        max_aspect_ratio=4.5,
        min_solidity=0.6,
        min_score_max=0.7,
        min_score_mean=0.4,
        merge_nearby_enabled=True,
        merge_nearby_max_gap_px=6,
        max_regions_sort_by="area",
        max_regions=9,
        masks_dir=str(masks_dir),
        mask_format="npz",
        defects_image_space=True,
    )


def test_build_defects_artifact_config_requires_resolved_pixel_threshold() -> None:
    with pytest.raises(RuntimeError, match="pixel threshold was not resolved"):
        build_defects_artifact_config(
            DefectsArtifactConfigBuildRequest(
                defects_enabled=True,
                pixel_threshold_value=None,
                pixel_threshold_provenance=None,
            )
        )


def test_build_infer_result_artifact_request_populates_fields_and_defects(tmp_path: Path) -> None:
    input_path = tmp_path / "inputs" / "a.png"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    _write_png(input_path)

    result = InferenceResult(
        score=0.9,
        label=1,
        anomaly_map=np.ones((4, 4), dtype=np.float32),
    )

    request = build_infer_result_artifact_request(
        InferResultArtifactBuildRequest(
            index=3,
            input_path=str(input_path),
            result=result,
            include_status=True,
            include_anomaly_map_values=True,
            maps_dir=str(tmp_path / "maps"),
            overlays_dir=str(tmp_path / "overlays"),
            defects=DefectsArtifactConfigBuildRequest(
                defects_enabled=True,
                pixel_threshold_value=0.75,
                pixel_threshold_provenance={"source": "explicit"},
                roi_xyxy_norm=[0.1, 0.2, 0.9, 0.8],
                mask_space="full",
                min_area=5,
                masks_dir=str(tmp_path / "masks"),
                mask_format="png",
                defects_image_space=True,
            ),
        )
    )

    assert request == InferResultArtifactRequest(
        index=3,
        input_path=str(input_path),
        result=result,
        include_status=True,
        include_anomaly_map_values=True,
        maps_dir=str(tmp_path / "maps"),
        overlays_dir=str(tmp_path / "overlays"),
        defects_config=DefectsArtifactConfig(
            pixel_threshold_value=0.75,
            pixel_threshold_provenance={"source": "explicit"},
            roi_xyxy_norm=[0.1, 0.2, 0.9, 0.8],
            mask_space="full",
            min_area=5,
            masks_dir=str(tmp_path / "masks"),
            mask_format="png",
            defects_image_space=True,
        ),
    )


def test_build_infer_result_artifact_request_skips_defects_when_disabled() -> None:
    result = InferenceResult(score=0.2, label=0, anomaly_map=None)

    request = build_infer_result_artifact_request(
        InferResultArtifactBuildRequest(
            index=0,
            input_path="a.png",
            result=result,
            include_status=False,
            defects=DefectsArtifactConfigBuildRequest(defects_enabled=False),
        )
    )

    assert request == InferResultArtifactRequest(
        index=0,
        input_path="a.png",
        result=result,
        include_status=False,
        include_anomaly_map_values=False,
        maps_dir=None,
        overlays_dir=None,
        defects_config=None,
    )


def test_build_infer_result_artifact_build_request_from_options_populates_defects_fields(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "inputs" / "a.png"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    _write_png(input_path)

    result = InferenceResult(score=0.9, label=1, anomaly_map=np.ones((4, 4), dtype=np.float32))

    request = build_infer_result_artifact_request_from_options(
        InferResultArtifactAssemblyRequest(
            index=4,
            input_path=str(input_path),
            result=result,
            include_status=True,
            options=InferArtifactOptions(
                include_anomaly_map_values=True,
                maps_dir=str(tmp_path / "maps"),
                overlays_dir=str(tmp_path / "overlays"),
                defects_enabled=True,
                pixel_threshold_value=0.6,
                pixel_threshold_provenance={"source": "explicit"},
                roi_xyxy_norm=[0.1, 0.2, 0.9, 0.8],
                mask_space="full",
                border_ignore_px=2,
                map_smoothing_method="gaussian",
                map_smoothing_ksize=5,
                map_smoothing_sigma=1.5,
                hysteresis_enabled=True,
                hysteresis_low=0.2,
                hysteresis_high=0.8,
                open_ksize=3,
                close_ksize=4,
                fill_holes=True,
                mask_dilate_ksize=1,
                min_area=7,
                min_fill_ratio=0.3,
                max_aspect_ratio=4.5,
                min_solidity=0.6,
                min_score_max=0.7,
                min_score_mean=0.4,
                merge_nearby_enabled=True,
                merge_nearby_max_gap_px=6,
                max_regions_sort_by="area",
                max_regions=9,
                masks_dir=str(tmp_path / "masks"),
                mask_format="npz",
                defects_image_space=True,
            ),
        )
    )

    assert request == InferResultArtifactBuildRequest(
        index=4,
        input_path=str(input_path),
        result=result,
        include_status=True,
        include_anomaly_map_values=True,
        maps_dir=str(tmp_path / "maps"),
        overlays_dir=str(tmp_path / "overlays"),
        defects=DefectsArtifactConfigBuildRequest(
            defects_enabled=True,
            pixel_threshold_value=0.6,
            pixel_threshold_provenance={"source": "explicit"},
            roi_xyxy_norm=[0.1, 0.2, 0.9, 0.8],
            mask_space="full",
            border_ignore_px=2,
            map_smoothing_method="gaussian",
            map_smoothing_ksize=5,
            map_smoothing_sigma=1.5,
            hysteresis_enabled=True,
            hysteresis_low=0.2,
            hysteresis_high=0.8,
            open_ksize=3,
            close_ksize=4,
            fill_holes=True,
            mask_dilate_ksize=1,
            min_area=7,
            min_fill_ratio=0.3,
            max_aspect_ratio=4.5,
            min_solidity=0.6,
            min_score_max=0.7,
            min_score_mean=0.4,
            merge_nearby_enabled=True,
            merge_nearby_max_gap_px=6,
            max_regions_sort_by="area",
            max_regions=9,
            masks_dir=str(tmp_path / "masks"),
            mask_format="npz",
            defects_image_space=True,
        ),
    )


def test_build_infer_result_artifact_build_request_from_options_skips_defects_when_disabled() -> None:
    result = InferenceResult(score=0.2, label=0, anomaly_map=None)

    request = build_infer_result_artifact_request_from_options(
        InferResultArtifactAssemblyRequest(
            index=0,
            input_path="a.png",
            result=result,
            options=InferArtifactOptions(
                include_anomaly_map_values=True,
                maps_dir="maps",
                overlays_dir="overlays",
                defects_enabled=False,
                pixel_threshold_value=0.5,
                pixel_threshold_provenance={"source": "explicit"},
            ),
        )
    )

    assert request == InferResultArtifactBuildRequest(
        index=0,
        input_path="a.png",
        result=result,
        include_status=False,
        include_anomaly_map_values=True,
        maps_dir="maps",
        overlays_dir="overlays",
        defects=DefectsArtifactConfigBuildRequest(
            defects_enabled=False,
            pixel_threshold_value=0.5,
            pixel_threshold_provenance={"source": "explicit"},
        ),
    )


def test_build_infer_result_artifact_build_request_from_cli_maps_cli_fields(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "inputs" / "a.png"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    _write_png(input_path)

    result = InferenceResult(score=0.9, label=1, anomaly_map=np.ones((4, 4), dtype=np.float32))

    cli_args = SimpleNamespace(
        include_anomaly_map_values=True,
        defects=True,
        roi_xyxy_norm=[0.1, 0.2, 0.9, 0.8],
        defects_mask_space="full",
        defect_border_ignore_px=2,
        defect_map_smoothing="gaussian",
        defect_map_smoothing_ksize=5,
        defect_map_smoothing_sigma=1.5,
        defect_hysteresis=True,
        defect_hysteresis_low=0.2,
        defect_hysteresis_high=0.8,
        defect_open_ksize=3,
        defect_close_ksize=4,
        defect_fill_holes=True,
        defects_mask_dilate=1,
        defect_min_area=7,
        defect_min_fill_ratio=0.3,
        defect_max_aspect_ratio=4.5,
        defect_min_solidity=0.6,
        defect_min_score_max=0.7,
        defect_min_score_mean=0.4,
        defect_merge_nearby=True,
        defect_merge_nearby_max_gap_px=6,
        defect_max_regions_sort_by="area",
        defect_max_regions=9,
        mask_format="npz",
        defects_image_space=True,
    )

    request = build_infer_result_artifact_build_request_from_cli(
        InferResultArtifactCliRequest(
            index=4,
            input_path=str(input_path),
            result=result,
            cli_args=cli_args,
            include_status=True,
            maps_dir=str(tmp_path / "maps"),
            overlays_dir=str(tmp_path / "overlays"),
            masks_dir=str(tmp_path / "masks"),
            pixel_threshold_value=0.6,
            pixel_threshold_provenance={"source": "explicit"},
        )
    )

    assert request == InferResultArtifactBuildRequest(
        index=4,
        input_path=str(input_path),
        result=result,
        include_status=True,
        include_anomaly_map_values=True,
        maps_dir=str(tmp_path / "maps"),
        overlays_dir=str(tmp_path / "overlays"),
        defects=DefectsArtifactConfigBuildRequest(
            defects_enabled=True,
            pixel_threshold_value=0.6,
            pixel_threshold_provenance={"source": "explicit"},
            roi_xyxy_norm=[0.1, 0.2, 0.9, 0.8],
            mask_space="full",
            border_ignore_px=2,
            map_smoothing_method="gaussian",
            map_smoothing_ksize=5,
            map_smoothing_sigma=1.5,
            hysteresis_enabled=True,
            hysteresis_low=0.2,
            hysteresis_high=0.8,
            open_ksize=3,
            close_ksize=4,
            fill_holes=True,
            mask_dilate_ksize=1,
            min_area=7,
            min_fill_ratio=0.3,
            max_aspect_ratio=4.5,
            min_solidity=0.6,
            min_score_max=0.7,
            min_score_mean=0.4,
            merge_nearby_enabled=True,
            merge_nearby_max_gap_px=6,
            max_regions_sort_by="area",
            max_regions=9,
            masks_dir=str(tmp_path / "masks"),
            mask_format="npz",
            defects_image_space=True,
        ),
    )


def test_build_infer_result_artifact_request_from_cli_returns_final_request(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "inputs" / "a.png"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    _write_png(input_path)

    result = InferenceResult(score=0.9, label=1, anomaly_map=np.ones((4, 4), dtype=np.float32))

    cli_args = SimpleNamespace(
        include_anomaly_map_values=True,
        defects=True,
        roi_xyxy_norm=[0.1, 0.2, 0.9, 0.8],
        defects_mask_space="full",
        defect_border_ignore_px=2,
        defect_map_smoothing="gaussian",
        defect_map_smoothing_ksize=5,
        defect_map_smoothing_sigma=1.5,
        defect_hysteresis=False,
        defect_hysteresis_low=None,
        defect_hysteresis_high=None,
        defect_open_ksize=0,
        defect_close_ksize=0,
        defect_fill_holes=False,
        defects_mask_dilate=0,
        defect_min_area=7,
        defect_min_fill_ratio=None,
        defect_max_aspect_ratio=None,
        defect_min_solidity=None,
        defect_min_score_max=0.7,
        defect_min_score_mean=None,
        defect_merge_nearby=False,
        defect_merge_nearby_max_gap_px=0,
        defect_max_regions_sort_by="score_max",
        defect_max_regions=None,
        mask_format="png",
        defects_image_space=False,
    )

    request = build_infer_result_artifact_request_from_cli(
        InferResultArtifactCliRequest(
            index=5,
            input_path=str(input_path),
            result=result,
            cli_args=cli_args,
            include_status=True,
            maps_dir=str(tmp_path / "maps"),
            overlays_dir=str(tmp_path / "overlays"),
            masks_dir=str(tmp_path / "masks"),
            pixel_threshold_value=0.55,
            pixel_threshold_provenance={"source": "explicit"},
        )
    )

    assert request == InferResultArtifactRequest(
        index=5,
        input_path=str(input_path),
        result=result,
        include_status=True,
        include_anomaly_map_values=True,
        maps_dir=str(tmp_path / "maps"),
        overlays_dir=str(tmp_path / "overlays"),
        defects_config=DefectsArtifactConfig(
            pixel_threshold_value=0.55,
            pixel_threshold_provenance={"source": "explicit"},
            roi_xyxy_norm=[0.1, 0.2, 0.9, 0.8],
            mask_space="full",
            border_ignore_px=2,
            map_smoothing_method="gaussian",
            map_smoothing_ksize=5,
            map_smoothing_sigma=1.5,
            min_area=7,
            min_score_max=0.7,
            masks_dir=str(tmp_path / "masks"),
            mask_format="png",
        ),
    )
