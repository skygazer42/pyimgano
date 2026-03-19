from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pyimgano.inference.config import INFER_CONFIG_SCHEMA_VERSION
from pyimgano.models.registry import create_model
from pyimgano.reporting.report import stamp_report_payload
from pyimgano.services.model_options import resolve_model_options
from pyimgano.workbench.calibration import (
    calibrate_detector_threshold_with_summary,
    resolve_default_quantile,
)
from pyimgano.workbench.config import WorkbenchConfig


@dataclass(frozen=True)
class WorkbenchThresholdCalibration:
    threshold: float
    quantile: float
    quantile_source: str
    score_summary: dict[str, Any] | None = None


def resolve_preprocessing_preset_payload(name: str) -> dict[str, Any]:
    """Resolve a deployable preprocessing preset into workbench config payload."""

    from pyimgano.preprocessing.catalog import list_preprocessing_schemes

    key = str(name).strip()
    for scheme in list_preprocessing_schemes(deployable_only=True):
        if str(scheme.name) != key:
            continue
        if str(getattr(scheme, "config_key", "")) != "preprocessing.illumination_contrast":
            raise ValueError(
                "Only preprocessing.illumination_contrast presets are currently "
                "supported by pyimgano-train."
            )
        payload = getattr(scheme, "payload", None)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Preprocessing preset does not provide a config payload: {name!r}")
        return dict(payload)

    raise ValueError(f"Unknown preprocessing preset: {name!r}")


def apply_workbench_overrides(
    raw: dict[str, Any],
    *,
    dataset_name: str | None = None,
    root: str | None = None,
    category: str | None = None,
    model_name: str | None = None,
    device: str | None = None,
    preprocessing_preset: str | None = None,
) -> dict[str, Any]:
    out = dict(raw)

    dataset = dict(out.get("dataset", {}) or {})
    model = dict(out.get("model", {}) or {})

    if dataset_name is not None:
        dataset["name"] = str(dataset_name)
    if root is not None:
        dataset["root"] = str(root)
    if category is not None:
        dataset["category"] = str(category)

    if model_name is not None:
        model["name"] = str(model_name)
    if device is not None:
        model["device"] = str(device)

    if preprocessing_preset is not None:
        preprocessing = dict(out.get("preprocessing", {}) or {})
        preprocessing["illumination_contrast"] = resolve_preprocessing_preset_payload(
            str(preprocessing_preset)
        )
        out["preprocessing"] = preprocessing

    if dataset:
        out["dataset"] = dataset
    if model:
        out["model"] = model

    return out


def create_workbench_detector(config: WorkbenchConfig) -> Any:
    import pyimgano.models  # noqa: F401

    user_kwargs = dict(config.model.model_kwargs)

    auto_kwargs: dict[str, Any] = {
        "device": config.model.device,
        "contamination": float(config.model.contamination),
        "pretrained": bool(config.model.pretrained),
    }
    if config.seed is not None:
        auto_kwargs["random_seed"] = int(config.seed)
        auto_kwargs["random_state"] = int(config.seed)

    model_kwargs = resolve_model_options(
        model_name=config.model.name,
        preset=config.model.preset,
        user_kwargs=user_kwargs,
        auto_kwargs=auto_kwargs,
        checkpoint_path=(
            str(config.model.checkpoint_path) if config.model.checkpoint_path is not None else None
        ),
    )
    detector = create_model(config.model.name, **model_kwargs)

    # Threshold calibration helpers expect contamination to be discoverable when applicable.
    try:
        existing = getattr(detector, "contamination", None)
    except Exception:  # noqa: BLE001 - best-effort metadata
        existing = None

    if existing is None:
        try:
            setattr(detector, "contamination", float(config.model.contamination))
        except Exception:  # noqa: BLE001 - best-effort metadata
            pass

    return detector


def calibrate_workbench_threshold(
    *,
    detector: Any,
    calibration_inputs: list[Any],
    input_format: str | None,
) -> WorkbenchThresholdCalibration:
    quantile, quantile_source = resolve_default_quantile(detector)
    threshold, score_summary = calibrate_detector_threshold_with_summary(
        detector,
        calibration_inputs,
        input_format=input_format,
        quantile=float(quantile),
    )
    return WorkbenchThresholdCalibration(
        threshold=float(threshold),
        quantile=float(quantile),
        quantile_source=str(quantile_source),
        score_summary=dict(score_summary),
    )


def build_infer_config_payload(
    *, config: WorkbenchConfig, report: Mapping[str, Any]
) -> dict[str, Any]:
    model_payload: dict[str, Any] = {
        "name": str(config.model.name),
        "preset": config.model.preset,
        "device": str(config.model.device),
        "pretrained": bool(config.model.pretrained),
        "contamination": float(config.model.contamination),
        "model_kwargs": dict(config.model.model_kwargs),
        "checkpoint_path": (
            str(config.model.checkpoint_path) if config.model.checkpoint_path is not None else None
        ),
    }

    adaptation_payload: dict[str, Any] = {
        "tiling": {
            "tile_size": config.adaptation.tiling.tile_size,
            "stride": config.adaptation.tiling.stride,
            "score_reduce": config.adaptation.tiling.score_reduce,
            "score_topk": float(config.adaptation.tiling.score_topk),
            "map_reduce": config.adaptation.tiling.map_reduce,
        },
        "postprocess": (
            config.adaptation.postprocess.__dict__
            if config.adaptation.postprocess is not None
            else None
        ),
        "save_maps": bool(config.adaptation.save_maps),
    }

    defects_payload: dict[str, Any] = {
        "enabled": bool(config.defects.enabled),
        "pixel_threshold": (
            float(config.defects.pixel_threshold)
            if config.defects.pixel_threshold is not None
            else None
        ),
        "pixel_threshold_strategy": str(config.defects.pixel_threshold_strategy),
        "pixel_normal_quantile": float(config.defects.pixel_normal_quantile),
        "mask_format": str(config.defects.mask_format),
        "roi_xyxy_norm": (
            [float(v) for v in config.defects.roi_xyxy_norm]
            if config.defects.roi_xyxy_norm is not None
            else None
        ),
        "border_ignore_px": int(config.defects.border_ignore_px),
        "map_smoothing": {
            "method": str(config.defects.map_smoothing.method),
            "ksize": int(config.defects.map_smoothing.ksize),
            "sigma": float(config.defects.map_smoothing.sigma),
        },
        "hysteresis": {
            "enabled": bool(config.defects.hysteresis.enabled),
            "low": (
                float(config.defects.hysteresis.low)
                if config.defects.hysteresis.low is not None
                else None
            ),
            "high": (
                float(config.defects.hysteresis.high)
                if config.defects.hysteresis.high is not None
                else None
            ),
        },
        "shape_filters": {
            "min_fill_ratio": (
                float(config.defects.shape_filters.min_fill_ratio)
                if config.defects.shape_filters.min_fill_ratio is not None
                else None
            ),
            "max_aspect_ratio": (
                float(config.defects.shape_filters.max_aspect_ratio)
                if config.defects.shape_filters.max_aspect_ratio is not None
                else None
            ),
            "min_solidity": (
                float(config.defects.shape_filters.min_solidity)
                if config.defects.shape_filters.min_solidity is not None
                else None
            ),
        },
        "merge_nearby": {
            "enabled": bool(config.defects.merge_nearby.enabled),
            "max_gap_px": int(config.defects.merge_nearby.max_gap_px),
        },
        "min_area": int(config.defects.min_area),
        "min_score_max": (
            float(config.defects.min_score_max)
            if config.defects.min_score_max is not None
            else None
        ),
        "min_score_mean": (
            float(config.defects.min_score_mean)
            if config.defects.min_score_mean is not None
            else None
        ),
        "open_ksize": int(config.defects.open_ksize),
        "close_ksize": int(config.defects.close_ksize),
        "fill_holes": bool(config.defects.fill_holes),
        "max_regions": (
            int(config.defects.max_regions) if config.defects.max_regions is not None else None
        ),
        "max_regions_sort_by": str(config.defects.max_regions_sort_by),
    }

    prediction_payload: dict[str, Any] | None = None
    if (
        config.prediction.reject_confidence_below is not None
        or config.prediction.reject_label is not None
    ):
        prediction_payload = {
            "reject_confidence_below": (
                float(config.prediction.reject_confidence_below)
                if config.prediction.reject_confidence_below is not None
                else None
            ),
            "reject_label": (
                int(config.prediction.reject_label)
                if config.prediction.reject_label is not None
                else None
            ),
        }

    preprocessing_payload: dict[str, Any] | None = None
    ic = config.preprocessing.illumination_contrast
    if ic is not None:
        preprocessing_payload = {
            "illumination_contrast": {
                "white_balance": str(ic.white_balance),
                "homomorphic": bool(ic.homomorphic),
                "homomorphic_cutoff": float(ic.homomorphic_cutoff),
                "homomorphic_gamma_low": float(ic.homomorphic_gamma_low),
                "homomorphic_gamma_high": float(ic.homomorphic_gamma_high),
                "homomorphic_c": float(ic.homomorphic_c),
                "homomorphic_per_channel": bool(ic.homomorphic_per_channel),
                "clahe": bool(ic.clahe),
                "clahe_clip_limit": float(ic.clahe_clip_limit),
                "clahe_tile_grid_size": [
                    int(ic.clahe_tile_grid_size[0]),
                    int(ic.clahe_tile_grid_size[1]),
                ],
                "gamma": (float(ic.gamma) if ic.gamma is not None else None),
                "contrast_stretch": bool(ic.contrast_stretch),
                "contrast_lower_percentile": float(ic.contrast_lower_percentile),
                "contrast_upper_percentile": float(ic.contrast_upper_percentile),
            }
        }

    operator_contract_payload: dict[str, Any] = {
        "schema_version": 1,
        "review_policy": {
            "review_on": ["anomalous", "rejected_low_confidence"],
            "confidence_gate_enabled": (config.prediction.reject_confidence_below is not None),
            "reject_confidence_below": (
                float(config.prediction.reject_confidence_below)
                if config.prediction.reject_confidence_below is not None
                else None
            ),
            "reject_label": (
                int(config.prediction.reject_label)
                if config.prediction.reject_label is not None
                else None
            ),
        },
        "runtime_policy": {
            "defects_enabled": bool(config.defects.enabled),
            "mask_format": str(config.defects.mask_format),
            "max_regions": (
                int(config.defects.max_regions)
                if config.defects.max_regions is not None
                else None
            ),
            "max_regions_sort_by": str(config.defects.max_regions_sort_by),
        },
        "output_contract": {
            "requires_image_score": True,
            "supports_pixel_outputs": bool(config.defects.enabled),
            "supports_reject_label": (config.prediction.reject_label is not None),
            "score_order": "higher_is_more_anomalous",
            "confidence_semantics": "predicted_label_confidence",
            "confidence_range": [0.0, 1.0],
            "decision_values": [
                *(
                    ["score_only", "normal", "anomalous", "rejected_low_confidence"]
                    if config.prediction.reject_confidence_below is not None
                    else ["score_only", "normal", "anomalous"]
                )
            ],
            "label_encoding": {
                **{
                    "normal": 0,
                    "anomalous": 1,
                },
                **(
                    {"rejected": int(config.prediction.reject_label)}
                    if config.prediction.reject_label is not None
                    else {}
                ),
            },
        },
    }

    out: dict[str, Any] = {
        "schema_version": int(INFER_CONFIG_SCHEMA_VERSION),
        "model": model_payload,
        "adaptation": adaptation_payload,
        "defects": defects_payload,
        "operator_contract": operator_contract_payload,
    }
    if preprocessing_payload is not None:
        out["preprocessing"] = preprocessing_payload
    if prediction_payload is not None:
        out["prediction"] = prediction_payload

    has_threshold_provenance = "threshold_provenance" in report
    threshold_scope = "per_category" if "per_category" in report else "image"
    out["artifact_quality"] = {
        "status": ("audited" if has_threshold_provenance else "reproducible"),
        "threshold_scope": threshold_scope,
        "has_threshold_provenance": bool(has_threshold_provenance),
        "has_split_fingerprint": isinstance(report.get("split_fingerprint", None), Mapping),
        "has_prediction_policy": prediction_payload is not None,
        "has_operator_contract": True,
        "has_deploy_bundle": False,
        "has_bundle_manifest": False,
        "required_bundle_artifacts_present": False,
        "bundle_artifact_roles": {},
        "audit_refs": {
            "calibration_card": "artifacts/calibration_card.json",
            "operator_contract": "artifacts/operator_contract.json",
        },
        "deploy_refs": {},
    }

    run_dir = report.get("run_dir", None)
    if run_dir is not None:
        out["from_run"] = str(run_dir)

    if "threshold" in report:
        out["threshold"] = report.get("threshold")
    if "threshold_provenance" in report:
        out["threshold_provenance"] = report.get("threshold_provenance")
    if "checkpoint" in report:
        out["checkpoint"] = report.get("checkpoint")
    if "split_fingerprint" in report:
        out["split_fingerprint"] = report.get("split_fingerprint")
    if "category" in report:
        out["category"] = report.get("category")
    if "per_category" in report:
        out["per_category"] = report.get("per_category")

    return stamp_report_payload(out)


def run_workbench(*, config: WorkbenchConfig, recipe_name: str) -> dict[str, Any]:
    from pyimgano.workbench.runner import run_workbench as _run_workbench

    return _run_workbench(config=config, recipe_name=recipe_name)


__all__ = [
    "WorkbenchThresholdCalibration",
    "apply_workbench_overrides",
    "build_infer_config_payload",
    "calibrate_workbench_threshold",
    "create_workbench_detector",
    "resolve_preprocessing_preset_payload",
    "run_workbench",
]
