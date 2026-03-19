from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pyimgano.inference.config import (
    INFER_CONFIG_SCHEMA_VERSION,
    load_infer_config,
    normalize_infer_config_schema,
    resolve_infer_checkpoint_path,
    resolve_infer_model_checkpoint_path,
    select_infer_category,
)

_ALLOWED_PIXEL_THRESHOLD_STRATEGIES = {"normal_pixel_quantile", "fixed", "infer_config"}
_ALLOWED_MASK_FORMATS = {"png", "npy", "npz"}
_ALLOWED_MAP_SMOOTHING_METHODS = {"none", "median", "gaussian", "box"}
_ALLOWED_TILING_SCORE_REDUCE = {"max", "mean", "topk_mean"}
_ALLOWED_TILING_MAP_REDUCE = {"max", "mean", "hann", "gaussian"}
_ALLOWED_WHITE_BALANCE = {"none", "gray_world", "max_rgb"}
_ALLOWED_ARTIFACT_QUALITY_STATUS = {"reproducible", "audited", "deployable"}
_ALLOWED_ARTIFACT_QUALITY_SCOPE = {"image", "per_category"}
_ALLOWED_OUTPUT_SCORE_ORDER = {"higher_is_more_anomalous"}
_ALLOWED_OUTPUT_CONFIDENCE_SEMANTICS = {"predicted_label_confidence"}
_ALLOWED_OUTPUT_DECISION_VALUES = {
    "score_only",
    "normal",
    "anomalous",
    "rejected_low_confidence",
}


@dataclass(frozen=True)
class InferConfigValidation:
    payload: dict[str, Any]
    resolved_checkpoint_path: Path | None
    resolved_model_checkpoint_path: Path | None
    trust_summary: dict[str, Any]
    warnings: list[str]


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"infer-config key {name!r} must be a JSON object/dict, got {type(value).__name__}"
        )
    return value


def _optional_mapping(value: Any, *, name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(
            f"infer-config key {name!r} must be a JSON object/dict, got {type(value).__name__}"
        )
    return value


def _coerce_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(int(value))
    raise ValueError(f"infer-config key {name!r} must be a boolean, got {value!r}")


def _coerce_int(value: Any, *, name: str) -> int:
    try:
        return int(value)
    except Exception as exc:  # noqa: BLE001 - validation boundary
        raise ValueError(f"infer-config key {name!r} must be an int, got {value!r}") from exc


def _coerce_float(value: Any, *, name: str) -> float:
    try:
        return float(value)
    except Exception as exc:  # noqa: BLE001 - validation boundary
        raise ValueError(f"infer-config key {name!r} must be a float, got {value!r}") from exc


def _coerce_str(value: Any, *, name: str) -> str:
    if value is None:
        raise ValueError(f"infer-config key {name!r} must be a string, got null")
    return str(value)


def _coerce_optional_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _coerce_float(value, name=name)


def _coerce_optional_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    return _coerce_int(value, name=name)


def _resolve_infer_artifact_ref_path(
    raw: str,
    *,
    config_path: str | Path,
    field_name: str,
) -> Path:
    text = str(raw).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty path.")

    p = Path(text)
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"{field_name} not found: {p}")
        return p

    cfg_path = Path(config_path)
    base = cfg_path.parent
    candidates: list[Path] = [
        (base / p).resolve(),
        (base.parent / p).resolve(),
    ]

    for cand in candidates:
        if cand.exists():
            return cand

    tried = "\n".join(f"- {cand}" for cand in candidates)
    raise FileNotFoundError(
        f"{field_name} not found.\n"
        f"{field_name}={text!r}\n"
        "Tried:\n"
        f"{tried}"
    )


def _build_infer_config_trust_summary(
    payload: Mapping[str, Any],
    *,
    check_files: bool,
) -> dict[str, Any]:
    artifact_quality = payload.get("artifact_quality", None)
    if not isinstance(artifact_quality, Mapping):
        return {
            "status": "partial",
            "trust_signals": {
                "file_refs_checked": bool(check_files),
            },
            "degraded_by": ["missing_artifact_quality"],
            "audit_refs": {},
        }

    audit_refs = artifact_quality.get("audit_refs", None)
    deploy_refs = artifact_quality.get("deploy_refs", None)
    audit_ref_map = dict(audit_refs) if isinstance(audit_refs, Mapping) else {}
    deploy_ref_map = dict(deploy_refs) if isinstance(deploy_refs, Mapping) else {}
    audit_status = str(artifact_quality.get("status", "") or "").strip().lower()

    trust_signals = {
        "file_refs_checked": bool(check_files),
        "has_threshold_provenance": bool(artifact_quality.get("has_threshold_provenance")),
        "has_split_fingerprint": bool(artifact_quality.get("has_split_fingerprint")),
        "has_prediction_policy": bool(artifact_quality.get("has_prediction_policy")),
        "has_operator_contract": bool(artifact_quality.get("has_operator_contract")),
        "has_deploy_bundle": bool(artifact_quality.get("has_deploy_bundle")),
        "has_bundle_manifest": bool(artifact_quality.get("has_bundle_manifest")),
        "has_required_bundle_artifacts": bool(
            artifact_quality.get("required_bundle_artifacts_present")
        ),
        "has_bundle_artifact_roles": bool(
            isinstance(artifact_quality.get("bundle_artifact_roles", None), Mapping)
            and len(dict(artifact_quality.get("bundle_artifact_roles", {}))) > 0
        ),
        "has_calibration_card_ref": bool(
            isinstance(audit_ref_map.get("calibration_card", None), str)
            and str(audit_ref_map.get("calibration_card")).strip()
        ),
        "has_operator_contract_ref": bool(
            isinstance(audit_ref_map.get("operator_contract", None), str)
            and str(audit_ref_map.get("operator_contract")).strip()
        ),
        "has_bundle_manifest_ref": bool(
            isinstance(deploy_ref_map.get("bundle_manifest", None), str)
            and str(deploy_ref_map.get("bundle_manifest")).strip()
        ),
    }
    degraded_by: list[str] = []
    if audit_status in {"audited", "deployable"}:
        if not trust_signals["has_threshold_provenance"]:
            degraded_by.append("missing_threshold_provenance")
        if not trust_signals["has_split_fingerprint"]:
            degraded_by.append("missing_split_fingerprint")
        if not trust_signals["has_calibration_card_ref"]:
            degraded_by.append("missing_calibration_card_ref")
        if (
            trust_signals["has_operator_contract"]
            and not trust_signals["has_operator_contract_ref"]
        ):
            degraded_by.append("missing_operator_contract_ref")
    if audit_status == "deployable":
        if not trust_signals["has_deploy_bundle"]:
            degraded_by.append("missing_deploy_bundle")
        if not trust_signals["has_bundle_manifest"]:
            degraded_by.append("missing_bundle_manifest")
        if not trust_signals["has_bundle_manifest_ref"]:
            degraded_by.append("missing_bundle_manifest_ref")
        if not trust_signals["has_required_bundle_artifacts"]:
            degraded_by.append("missing_required_bundle_artifacts")
        if not trust_signals["has_bundle_artifact_roles"]:
            degraded_by.append("missing_bundle_artifact_roles")
    if not bool(check_files):
        degraded_by.append("file_checks_skipped")

    status = "trust-signaled" if not degraded_by else "partial"
    merged_refs: dict[str, str] = {}
    for mapping in (audit_ref_map, deploy_ref_map):
        for key, value in mapping.items():
            if isinstance(value, str) and value.strip():
                merged_refs[str(key)] = str(value)
    return {
        "status": status,
        "trust_signals": trust_signals,
        "degraded_by": degraded_by,
        "audit_refs": merged_refs,
    }


def validate_infer_config_payload(
    payload: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
    category: str | None = None,
    check_files: bool = True,
) -> InferConfigValidation:
    """Validate and normalize an infer-config payload.

    This function is intentionally permissive: it validates known keys and
    best-effort coerces values into the types expected by `pyimgano-infer`.

    Args:
        payload: Parsed JSON object (dict-like).
        config_path: Path to the infer-config file (used for checkpoint resolution).
        category: Optional category selector for multi-category infer-configs.
        check_files: If True, validate referenced checkpoints exist on disk.

    Returns:
        InferConfigValidation containing the normalized payload, resolved checkpoint
        path (if present), resolved model checkpoint_path (if present), and a list
        of warnings.
    """

    if not isinstance(payload, Mapping):
        raise ValueError(
            f"infer-config payload must be a JSON object/dict, got {type(payload).__name__}"
        )

    warnings: list[str] = []
    normalized: dict[str, Any] = dict(payload)
    normalized, schema_warnings = normalize_infer_config_schema(normalized)
    warnings.extend(schema_warnings)

    normalized = select_infer_category(normalized, category=category)

    normalized["schema_version"] = int(
        _coerce_int(normalized["schema_version"], name="schema_version")
    )
    if int(normalized["schema_version"]) != int(INFER_CONFIG_SCHEMA_VERSION):
        raise ValueError(
            "infer-config schema_version is not compatible with this pyimgano build.\n"
            f"Got schema_version={normalized['schema_version']}, expected={INFER_CONFIG_SCHEMA_VERSION}."
        )

    if "pyimgano_version" in normalized and normalized["pyimgano_version"] is not None:
        normalized["pyimgano_version"] = str(normalized["pyimgano_version"])

    model_raw = normalized.get("model", None)
    model_map = _require_mapping(model_raw, name="model")
    model: dict[str, Any] = dict(model_map)

    model_name = model.get("name", None)
    if model_name is None:
        raise ValueError("infer-config model.name is required.")
    model["name"] = _coerce_str(model_name, name="model.name").strip()
    if not model["name"]:
        raise ValueError("infer-config model.name must be non-empty.")

    if "device" in model and model["device"] is not None:
        model["device"] = str(model["device"])
    if "preset" in model and model["preset"] is not None:
        model["preset"] = str(model["preset"])

    if "contamination" in model and model["contamination"] is not None:
        model["contamination"] = _coerce_float(model["contamination"], name="model.contamination")
    if "pretrained" in model and model["pretrained"] is not None:
        model["pretrained"] = _coerce_bool(model["pretrained"], name="model.pretrained")

    model_kwargs_raw = model.get("model_kwargs", None)
    if model_kwargs_raw is None:
        model["model_kwargs"] = {}
    else:
        model_kwargs_map = _require_mapping(model_kwargs_raw, name="model.model_kwargs")
        model["model_kwargs"] = dict(model_kwargs_map)

    if "checkpoint_path" in model and model["checkpoint_path"] is not None:
        model["checkpoint_path"] = str(model["checkpoint_path"])

    normalized["model"] = model

    preprocessing_raw = normalized.get("preprocessing", None)
    preprocessing_map = _optional_mapping(preprocessing_raw, name="preprocessing")
    if preprocessing_map is not None:
        preprocessing: dict[str, Any] = dict(preprocessing_map)

        ic_raw = preprocessing.get("illumination_contrast", None)
        ic_map = _optional_mapping(ic_raw, name="preprocessing.illumination_contrast")
        if ic_map is not None:
            ic: dict[str, Any] = dict(ic_map)

            if "white_balance" in ic and ic["white_balance"] is not None:
                wb = str(ic["white_balance"]).strip().lower()
                if wb in ("", "none"):
                    wb = "none"
                elif wb in ("gray_world", "gray-world", "grayworld"):
                    wb = "gray_world"
                elif wb in ("max_rgb", "max-rgb", "maxrgb"):
                    wb = "max_rgb"
                if wb not in _ALLOWED_WHITE_BALANCE:
                    raise ValueError(
                        "infer-config preprocessing.illumination_contrast.white_balance must be one of: "
                        f"{sorted(_ALLOWED_WHITE_BALANCE)}"
                    )
                ic["white_balance"] = wb

            for key in (
                "homomorphic",
                "homomorphic_per_channel",
                "clahe",
                "contrast_stretch",
            ):
                if key in ic and ic[key] is not None:
                    ic[key] = _coerce_bool(
                        ic[key], name=f"preprocessing.illumination_contrast.{key}"
                    )

            if "homomorphic_cutoff" in ic and ic["homomorphic_cutoff"] is not None:
                cutoff = _coerce_float(
                    ic["homomorphic_cutoff"],
                    name="preprocessing.illumination_contrast.homomorphic_cutoff",
                )
                if not (0.0 < float(cutoff) <= 1.0):
                    raise ValueError(
                        "infer-config preprocessing.illumination_contrast.homomorphic_cutoff must be in (0,1]."
                    )
                ic["homomorphic_cutoff"] = float(cutoff)

            for key in (
                "homomorphic_gamma_low",
                "homomorphic_gamma_high",
                "homomorphic_c",
                "clahe_clip_limit",
                "contrast_lower_percentile",
                "contrast_upper_percentile",
            ):
                if key in ic and ic[key] is not None:
                    ic[key] = _coerce_float(
                        ic[key], name=f"preprocessing.illumination_contrast.{key}"
                    )

            if "gamma" in ic and ic["gamma"] is not None:
                gamma = _coerce_float(ic["gamma"], name="preprocessing.illumination_contrast.gamma")
                if float(gamma) <= 0.0:
                    raise ValueError(
                        "infer-config preprocessing.illumination_contrast.gamma must be > 0."
                    )
                ic["gamma"] = float(gamma)

            if "clahe_tile_grid_size" in ic and ic["clahe_tile_grid_size"] is not None:
                tgs = ic["clahe_tile_grid_size"]
                if not isinstance(tgs, (list, tuple)) or len(tgs) != 2:
                    raise ValueError(
                        "infer-config preprocessing.illumination_contrast.clahe_tile_grid_size must be a list of length 2."
                    )
                a = _coerce_int(
                    tgs[0],
                    name="preprocessing.illumination_contrast.clahe_tile_grid_size[0]",
                )
                b = _coerce_int(
                    tgs[1],
                    name="preprocessing.illumination_contrast.clahe_tile_grid_size[1]",
                )
                if a <= 0 or b <= 0:
                    raise ValueError(
                        "infer-config preprocessing.illumination_contrast.clahe_tile_grid_size must be positive."
                    )
                ic["clahe_tile_grid_size"] = [int(a), int(b)]

            # Best-effort range validation (optional keys).
            if "clahe_clip_limit" in ic and ic["clahe_clip_limit"] is not None:
                if float(ic["clahe_clip_limit"]) <= 0.0:
                    raise ValueError(
                        "infer-config preprocessing.illumination_contrast.clahe_clip_limit must be > 0."
                    )

            if (
                "contrast_lower_percentile" in ic
                and "contrast_upper_percentile" in ic
                and ic.get("contrast_lower_percentile", None) is not None
                and ic.get("contrast_upper_percentile", None) is not None
            ):
                lo = float(ic["contrast_lower_percentile"])
                hi = float(ic["contrast_upper_percentile"])
                if not (0.0 <= lo <= 100.0 and 0.0 <= hi <= 100.0 and lo < hi):
                    raise ValueError(
                        "infer-config preprocessing.illumination_contrast contrast percentiles must satisfy 0<=lower<upper<=100."
                    )

            preprocessing["illumination_contrast"] = ic

        normalized["preprocessing"] = preprocessing

    adaptation_raw = normalized.get("adaptation", None)
    adaptation_map = _optional_mapping(adaptation_raw, name="adaptation")
    if adaptation_map is not None:
        adaptation: dict[str, Any] = dict(adaptation_map)

        tiling_raw = adaptation.get("tiling", None)
        tiling_map = _optional_mapping(tiling_raw, name="adaptation.tiling")
        if tiling_map is not None:
            tiling: dict[str, Any] = dict(tiling_map)

            if "tile_size" in tiling and tiling["tile_size"] is not None:
                tile_size = _coerce_int(tiling["tile_size"], name="adaptation.tiling.tile_size")
                if tile_size <= 0:
                    raise ValueError("infer-config adaptation.tiling.tile_size must be > 0.")
                tiling["tile_size"] = int(tile_size)

            if "stride" in tiling and tiling["stride"] is not None:
                stride = _coerce_int(tiling["stride"], name="adaptation.tiling.stride")
                if stride <= 0:
                    raise ValueError("infer-config adaptation.tiling.stride must be > 0.")
                tiling["stride"] = int(stride)

            if "score_topk" in tiling and tiling["score_topk"] is not None:
                tiling["score_topk"] = _coerce_float(
                    tiling["score_topk"], name="adaptation.tiling.score_topk"
                )

            if "score_reduce" in tiling and tiling["score_reduce"] is not None:
                mode = str(tiling["score_reduce"]).lower().strip()
                if mode not in _ALLOWED_TILING_SCORE_REDUCE:
                    raise ValueError(
                        "infer-config adaptation.tiling.score_reduce must be one of: "
                        f"{sorted(_ALLOWED_TILING_SCORE_REDUCE)}"
                    )
                tiling["score_reduce"] = mode

            if "map_reduce" in tiling and tiling["map_reduce"] is not None:
                mode = str(tiling["map_reduce"]).lower().strip()
                if mode not in _ALLOWED_TILING_MAP_REDUCE:
                    raise ValueError(
                        "infer-config adaptation.tiling.map_reduce must be one of: "
                        f"{sorted(_ALLOWED_TILING_MAP_REDUCE)}"
                    )
                tiling["map_reduce"] = mode

            adaptation["tiling"] = tiling

        if "save_maps" in adaptation and adaptation["save_maps"] is not None:
            adaptation["save_maps"] = _coerce_bool(
                adaptation["save_maps"], name="adaptation.save_maps"
            )

        post_cfg = adaptation.get("postprocess", None)
        if post_cfg is not None and not isinstance(post_cfg, Mapping):
            raise ValueError(
                f"infer-config key 'adaptation.postprocess' must be a JSON object/dict or null, got {type(post_cfg).__name__}"
            )

        normalized["adaptation"] = adaptation

    defects_raw = normalized.get("defects", None)
    defects_map = _optional_mapping(defects_raw, name="defects")
    if defects_map is not None:
        defects: dict[str, Any] = dict(defects_map)

        if "enabled" in defects and defects["enabled"] is not None:
            defects["enabled"] = _coerce_bool(defects["enabled"], name="defects.enabled")

        if "pixel_threshold" in defects and defects["pixel_threshold"] is not None:
            defects["pixel_threshold"] = _coerce_float(
                defects["pixel_threshold"], name="defects.pixel_threshold"
            )

        if (
            "pixel_threshold_strategy" in defects
            and defects["pixel_threshold_strategy"] is not None
        ):
            strategy = str(defects["pixel_threshold_strategy"]).lower().strip()
            if strategy not in _ALLOWED_PIXEL_THRESHOLD_STRATEGIES:
                raise ValueError(
                    "infer-config defects.pixel_threshold_strategy must be one of: "
                    f"{sorted(_ALLOWED_PIXEL_THRESHOLD_STRATEGIES)}"
                )
            defects["pixel_threshold_strategy"] = strategy

        if "pixel_normal_quantile" in defects and defects["pixel_normal_quantile"] is not None:
            q = _coerce_float(
                defects["pixel_normal_quantile"], name="defects.pixel_normal_quantile"
            )
            if not 0.0 < q < 1.0:
                raise ValueError("infer-config defects.pixel_normal_quantile must be in (0,1).")
            defects["pixel_normal_quantile"] = float(q)

        if "mask_format" in defects and defects["mask_format"] is not None:
            fmt = str(defects["mask_format"]).lower().strip()
            if fmt not in _ALLOWED_MASK_FORMATS:
                raise ValueError(
                    f"infer-config defects.mask_format must be one of: {sorted(_ALLOWED_MASK_FORMATS)}"
                )
            defects["mask_format"] = fmt

        if "roi_xyxy_norm" in defects and defects["roi_xyxy_norm"] is not None:
            roi = defects["roi_xyxy_norm"]
            if not isinstance(roi, (list, tuple)) or len(roi) != 4:
                raise ValueError(
                    "infer-config defects.roi_xyxy_norm must be a list of length 4 or null"
                )
            defects["roi_xyxy_norm"] = [float(v) for v in roi]

        if "border_ignore_px" in defects and defects["border_ignore_px"] is not None:
            v = _coerce_int(defects["border_ignore_px"], name="defects.border_ignore_px")
            if v < 0:
                raise ValueError("infer-config defects.border_ignore_px must be >= 0.")
            defects["border_ignore_px"] = int(v)

        ms_raw = defects.get("map_smoothing", None)
        ms_map = _optional_mapping(ms_raw, name="defects.map_smoothing")
        if ms_map is not None:
            ms: dict[str, Any] = dict(ms_map)
            if "method" in ms and ms["method"] is not None:
                method = str(ms["method"]).lower().strip()
                if method not in _ALLOWED_MAP_SMOOTHING_METHODS:
                    raise ValueError(
                        "infer-config defects.map_smoothing.method must be one of: "
                        f"{sorted(_ALLOWED_MAP_SMOOTHING_METHODS)}"
                    )
                ms["method"] = method
            if "ksize" in ms and ms["ksize"] is not None:
                k = _coerce_int(ms["ksize"], name="defects.map_smoothing.ksize")
                if k < 0:
                    raise ValueError("infer-config defects.map_smoothing.ksize must be >= 0.")
                ms["ksize"] = int(k)
            if "sigma" in ms and ms["sigma"] is not None:
                s = _coerce_float(ms["sigma"], name="defects.map_smoothing.sigma")
                if s < 0.0:
                    raise ValueError("infer-config defects.map_smoothing.sigma must be >= 0.")
                ms["sigma"] = float(s)
            defects["map_smoothing"] = ms

        hyst_raw = defects.get("hysteresis", None)
        hyst_map = _optional_mapping(hyst_raw, name="defects.hysteresis")
        if hyst_map is not None:
            hyst: dict[str, Any] = dict(hyst_map)
            if "enabled" in hyst and hyst["enabled"] is not None:
                hyst["enabled"] = _coerce_bool(hyst["enabled"], name="defects.hysteresis.enabled")
            if "low" in hyst:
                hyst["low"] = _coerce_optional_float(hyst["low"], name="defects.hysteresis.low")
            if "high" in hyst:
                hyst["high"] = _coerce_optional_float(hyst["high"], name="defects.hysteresis.high")
            defects["hysteresis"] = hyst

        normalized["defects"] = defects

    prediction_raw = normalized.get("prediction", None)
    prediction_map = _optional_mapping(prediction_raw, name="prediction")
    if prediction_map is not None:
        prediction: dict[str, Any] = dict(prediction_map)

        if (
            "reject_confidence_below" in prediction
            and prediction["reject_confidence_below"] is not None
        ):
            reject_confidence_below = _coerce_float(
                prediction["reject_confidence_below"],
                name="prediction.reject_confidence_below",
            )
            if not (0.0 < float(reject_confidence_below) <= 1.0):
                raise ValueError(
                    "infer-config prediction.reject_confidence_below must be in (0,1]."
                )
            prediction["reject_confidence_below"] = float(reject_confidence_below)

        if "reject_label" in prediction and prediction["reject_label"] is not None:
            prediction["reject_label"] = _coerce_int(
                prediction["reject_label"],
                name="prediction.reject_label",
            )

        normalized["prediction"] = prediction

    operator_contract_raw = normalized.get("operator_contract", None)
    operator_contract_map = _optional_mapping(operator_contract_raw, name="operator_contract")
    if operator_contract_map is not None:
        operator_contract: dict[str, Any] = dict(operator_contract_map)

        if "schema_version" in operator_contract and operator_contract["schema_version"] is not None:
            schema_version = _coerce_int(
                operator_contract["schema_version"],
                name="operator_contract.schema_version",
            )
            if int(schema_version) != 1:
                raise ValueError("infer-config operator_contract.schema_version must be 1.")
            operator_contract["schema_version"] = int(schema_version)

        review_policy_raw = operator_contract.get("review_policy", None)
        review_policy_map = _optional_mapping(
            review_policy_raw,
            name="operator_contract.review_policy",
        )
        if review_policy_map is not None:
            review_policy: dict[str, Any] = dict(review_policy_map)
            if "review_on" in review_policy and review_policy["review_on"] is not None:
                review_on = review_policy["review_on"]
                if not isinstance(review_on, list):
                    raise ValueError(
                        "infer-config operator_contract.review_policy.review_on must be a list of strings."
                    )
                items: list[str] = []
                for index, value in enumerate(review_on):
                    text = _coerce_str(
                        value,
                        name=f"operator_contract.review_policy.review_on[{index}]",
                    ).strip()
                    if not text:
                        raise ValueError(
                            "infer-config operator_contract.review_policy.review_on"
                            f"[{index}] must be non-empty."
                        )
                    items.append(text)
                review_policy["review_on"] = items

            if (
                "confidence_gate_enabled" in review_policy
                and review_policy["confidence_gate_enabled"] is not None
            ):
                review_policy["confidence_gate_enabled"] = _coerce_bool(
                    review_policy["confidence_gate_enabled"],
                    name="operator_contract.review_policy.confidence_gate_enabled",
                )

            if (
                "reject_confidence_below" in review_policy
                and review_policy["reject_confidence_below"] is not None
            ):
                reject_confidence_below = _coerce_float(
                    review_policy["reject_confidence_below"],
                    name="operator_contract.review_policy.reject_confidence_below",
                )
                if not (0.0 < float(reject_confidence_below) <= 1.0):
                    raise ValueError(
                        "infer-config operator_contract.review_policy.reject_confidence_below "
                        "must be in (0,1]."
                    )
                review_policy["reject_confidence_below"] = float(reject_confidence_below)

            if "reject_label" in review_policy and review_policy["reject_label"] is not None:
                review_policy["reject_label"] = _coerce_int(
                    review_policy["reject_label"],
                    name="operator_contract.review_policy.reject_label",
                )

            if bool(review_policy.get("confidence_gate_enabled", False)) and (
                review_policy.get("reject_confidence_below", None) is None
            ):
                raise ValueError(
                    "infer-config operator_contract.review_policy.confidence_gate_enabled=true "
                    "requires operator_contract.review_policy.reject_confidence_below."
                )

            prediction = normalized.get("prediction", None)
            if isinstance(prediction, Mapping):
                pred_threshold = prediction.get("reject_confidence_below", None)
                review_threshold = review_policy.get("reject_confidence_below", None)
                if (
                    pred_threshold is not None
                    and review_threshold is not None
                    and abs(float(pred_threshold) - float(review_threshold)) > 1e-12
                ):
                    raise ValueError(
                        "infer-config operator_contract.review_policy.reject_confidence_below "
                        "must match prediction.reject_confidence_below when both are set."
                    )

                pred_label = prediction.get("reject_label", None)
                review_label = review_policy.get("reject_label", None)
                if (
                    pred_label is not None
                    and review_label is not None
                    and int(pred_label) != int(review_label)
                ):
                    raise ValueError(
                        "infer-config operator_contract.review_policy.reject_label must match "
                        "prediction.reject_label when both are set."
                    )

            operator_contract["review_policy"] = review_policy

        runtime_policy_raw = operator_contract.get("runtime_policy", None)
        runtime_policy_map = _optional_mapping(
            runtime_policy_raw,
            name="operator_contract.runtime_policy",
        )
        if runtime_policy_map is not None:
            runtime_policy: dict[str, Any] = dict(runtime_policy_map)
            if "defects_enabled" in runtime_policy and runtime_policy["defects_enabled"] is not None:
                runtime_policy["defects_enabled"] = _coerce_bool(
                    runtime_policy["defects_enabled"],
                    name="operator_contract.runtime_policy.defects_enabled",
                )

            defects = normalized.get("defects", None)
            if (
                isinstance(defects, Mapping)
                and "defects_enabled" in runtime_policy
                and defects.get("enabled", None) is not None
                and bool(runtime_policy["defects_enabled"]) != bool(defects.get("enabled"))
            ):
                raise ValueError(
                    "infer-config operator_contract.runtime_policy.defects_enabled must match "
                    "defects.enabled when both are set."
                )

            operator_contract["runtime_policy"] = runtime_policy

        output_contract_raw = operator_contract.get("output_contract", None)
        output_contract_map = _optional_mapping(
            output_contract_raw,
            name="operator_contract.output_contract",
        )
        if output_contract_map is not None:
            output_contract: dict[str, Any] = dict(output_contract_map)
            for key in ("requires_image_score", "supports_pixel_outputs", "supports_reject_label"):
                if key in output_contract and output_contract[key] is not None:
                    output_contract[key] = _coerce_bool(
                        output_contract[key],
                        name=f"operator_contract.output_contract.{key}",
                    )

            if "score_order" in output_contract and output_contract["score_order"] is not None:
                score_order = str(output_contract["score_order"]).strip().lower()
                if score_order not in _ALLOWED_OUTPUT_SCORE_ORDER:
                    raise ValueError(
                        "infer-config operator_contract.output_contract.score_order must be one of: "
                        f"{sorted(_ALLOWED_OUTPUT_SCORE_ORDER)}"
                    )
                output_contract["score_order"] = score_order

            if (
                "confidence_semantics" in output_contract
                and output_contract["confidence_semantics"] is not None
            ):
                confidence_semantics = str(output_contract["confidence_semantics"]).strip().lower()
                if confidence_semantics not in _ALLOWED_OUTPUT_CONFIDENCE_SEMANTICS:
                    raise ValueError(
                        "infer-config operator_contract.output_contract.confidence_semantics "
                        f"must be one of: {sorted(_ALLOWED_OUTPUT_CONFIDENCE_SEMANTICS)}"
                    )
                output_contract["confidence_semantics"] = confidence_semantics

            if "confidence_range" in output_contract and output_contract["confidence_range"] is not None:
                confidence_range = output_contract["confidence_range"]
                if not isinstance(confidence_range, (list, tuple)) or len(confidence_range) != 2:
                    raise ValueError(
                        "infer-config operator_contract.output_contract.confidence_range must be a list of length 2."
                    )
                lo = _coerce_float(
                    confidence_range[0],
                    name="operator_contract.output_contract.confidence_range[0]",
                )
                hi = _coerce_float(
                    confidence_range[1],
                    name="operator_contract.output_contract.confidence_range[1]",
                )
                if abs(float(lo) - 0.0) > 1e-12 or abs(float(hi) - 1.0) > 1e-12:
                    raise ValueError(
                        "infer-config operator_contract.output_contract.confidence_range must be [0.0, 1.0]."
                    )
                output_contract["confidence_range"] = [0.0, 1.0]

            if "decision_values" in output_contract and output_contract["decision_values"] is not None:
                decision_values = output_contract["decision_values"]
                if not isinstance(decision_values, list):
                    raise ValueError(
                        "infer-config operator_contract.output_contract.decision_values must be a list of strings."
                    )
                normalized_decisions: list[str] = []
                seen_decisions: set[str] = set()
                for index, value in enumerate(decision_values):
                    decision = _coerce_str(
                        value,
                        name=f"operator_contract.output_contract.decision_values[{index}]",
                    ).strip()
                    if not decision:
                        raise ValueError(
                            "infer-config operator_contract.output_contract.decision_values"
                            f"[{index}] must be non-empty."
                        )
                    if decision not in _ALLOWED_OUTPUT_DECISION_VALUES:
                        raise ValueError(
                            "infer-config operator_contract.output_contract.decision_values "
                            f"contains unsupported value: {decision!r}"
                        )
                    if decision in seen_decisions:
                        raise ValueError(
                            "infer-config operator_contract.output_contract.decision_values "
                            f"contains duplicate value: {decision!r}"
                        )
                    seen_decisions.add(decision)
                    normalized_decisions.append(decision)
                output_contract["decision_values"] = normalized_decisions

            label_encoding_raw = output_contract.get("label_encoding", None)
            label_encoding_map = _optional_mapping(
                label_encoding_raw,
                name="operator_contract.output_contract.label_encoding",
            )
            if label_encoding_map is not None:
                label_encoding: dict[str, int] = {}
                for label_name, label_value in label_encoding_map.items():
                    key = str(label_name).strip()
                    if not key:
                        raise ValueError(
                            "infer-config operator_contract.output_contract.label_encoding keys must be non-empty."
                        )
                    label_encoding[key] = _coerce_int(
                        label_value,
                        name=f"operator_contract.output_contract.label_encoding.{key}",
                    )

                if label_encoding.get("normal", None) != 0:
                    raise ValueError(
                        "infer-config operator_contract.output_contract.label_encoding.normal must be 0."
                    )
                if label_encoding.get("anomalous", None) != 1:
                    raise ValueError(
                        "infer-config operator_contract.output_contract.label_encoding.anomalous must be 1."
                    )

                review_policy = operator_contract.get("review_policy", None)
                review_reject_label = None
                if isinstance(review_policy, Mapping):
                    review_reject_label = review_policy.get("reject_label", None)

                prediction = normalized.get("prediction", None)
                prediction_reject_label = None
                if isinstance(prediction, Mapping):
                    prediction_reject_label = prediction.get("reject_label", None)

                expected_reject_label = (
                    prediction_reject_label
                    if prediction_reject_label is not None
                    else review_reject_label
                )

                if "rejected" in label_encoding:
                    if not bool(output_contract.get("supports_reject_label", False)):
                        raise ValueError(
                            "infer-config operator_contract.output_contract.label_encoding.rejected "
                            "requires supports_reject_label=true."
                        )
                    if expected_reject_label is None:
                        raise ValueError(
                            "infer-config operator_contract.output_contract.label_encoding.rejected "
                            "requires prediction.reject_label or "
                            "operator_contract.review_policy.reject_label."
                        )
                    if int(label_encoding["rejected"]) != int(expected_reject_label):
                        raise ValueError(
                            "infer-config operator_contract.output_contract.label_encoding.rejected "
                            "must match prediction.reject_label when both are set."
                        )
                elif bool(output_contract.get("supports_reject_label", False)) and (
                    expected_reject_label is not None
                ):
                    raise ValueError(
                        "infer-config operator_contract.output_contract.supports_reject_label=true "
                        "requires operator_contract.output_contract.label_encoding.rejected."
                    )

                output_contract["label_encoding"] = label_encoding

            defects = normalized.get("defects", None)
            if (
                isinstance(defects, Mapping)
                and "supports_pixel_outputs" in output_contract
                and defects.get("enabled", None) is not None
                and bool(output_contract["supports_pixel_outputs"]) != bool(defects.get("enabled"))
            ):
                raise ValueError(
                    "infer-config operator_contract.output_contract.supports_pixel_outputs must "
                    "match defects.enabled when both are set."
                )

            prediction = normalized.get("prediction", None)
            if isinstance(prediction, Mapping) and "supports_reject_label" in output_contract:
                has_reject_label = prediction.get("reject_label", None) is not None
                if bool(output_contract["supports_reject_label"]) != bool(has_reject_label):
                    raise ValueError(
                        "infer-config operator_contract.output_contract.supports_reject_label "
                        "must match whether prediction.reject_label is set."
                    )

            operator_contract["output_contract"] = output_contract

        normalized["operator_contract"] = operator_contract

    # Optional thresholds/checkpoint metadata. Validate types when present.
    if "threshold" in normalized and normalized["threshold"] is not None:
        normalized["threshold"] = _coerce_float(normalized["threshold"], name="threshold")
    if "threshold_provenance" in normalized and normalized["threshold_provenance"] is not None:
        _require_mapping(normalized["threshold_provenance"], name="threshold_provenance")

    if "from_run" in normalized and normalized["from_run"] is not None:
        normalized["from_run"] = str(normalized["from_run"])

    artifact_quality_raw = normalized.get("artifact_quality", None)
    artifact_quality_map = _optional_mapping(artifact_quality_raw, name="artifact_quality")
    if artifact_quality_map is not None:
        artifact_quality: dict[str, Any] = dict(artifact_quality_map)

        if "status" in artifact_quality and artifact_quality["status"] is not None:
            status = str(artifact_quality["status"]).strip().lower()
            if status not in _ALLOWED_ARTIFACT_QUALITY_STATUS:
                raise ValueError(
                    "infer-config artifact_quality.status must be one of: "
                    f"{sorted(_ALLOWED_ARTIFACT_QUALITY_STATUS)}"
                )
            artifact_quality["status"] = status

        if "threshold_scope" in artifact_quality and artifact_quality["threshold_scope"] is not None:
            threshold_scope = str(artifact_quality["threshold_scope"]).strip().lower()
            if threshold_scope not in _ALLOWED_ARTIFACT_QUALITY_SCOPE:
                raise ValueError(
                    "infer-config artifact_quality.threshold_scope must be one of: "
                    f"{sorted(_ALLOWED_ARTIFACT_QUALITY_SCOPE)}"
                )
            artifact_quality["threshold_scope"] = threshold_scope

        for key in (
            "has_threshold_provenance",
            "has_split_fingerprint",
            "has_prediction_policy",
            "has_operator_contract",
            "has_deploy_bundle",
            "has_bundle_manifest",
            "required_bundle_artifacts_present",
        ):
            if key in artifact_quality and artifact_quality[key] is not None:
                artifact_quality[key] = _coerce_bool(
                    artifact_quality[key],
                    name=f"artifact_quality.{key}",
                )

        bundle_artifact_roles_raw = artifact_quality.get("bundle_artifact_roles", None)
        bundle_artifact_roles_map = _optional_mapping(
            bundle_artifact_roles_raw,
            name="artifact_quality.bundle_artifact_roles",
        )
        if bundle_artifact_roles_map is not None:
            bundle_artifact_roles: dict[str, list[str]] = {}
            for role_name, role_paths in bundle_artifact_roles_map.items():
                if not isinstance(role_paths, list):
                    raise ValueError(
                        "infer-config artifact_quality.bundle_artifact_roles."
                        f"{role_name} must be a list of strings."
                    )
                collected: list[str] = []
                for index, role_path in enumerate(role_paths):
                    text = _coerce_str(
                        role_path,
                        name=(
                            "artifact_quality.bundle_artifact_roles."
                            f"{role_name}[{index}]"
                        ),
                    ).strip()
                    if not text:
                        raise ValueError(
                            "infer-config artifact_quality.bundle_artifact_roles."
                            f"{role_name}[{index}] must be non-empty."
                        )
                    collected.append(text)
                bundle_artifact_roles[str(role_name)] = collected
            artifact_quality["bundle_artifact_roles"] = bundle_artifact_roles

        audit_refs_raw = artifact_quality.get("audit_refs", None)
        audit_refs_map = _optional_mapping(audit_refs_raw, name="artifact_quality.audit_refs")
        if audit_refs_map is not None:
            audit_refs: dict[str, str] = {}
            for ref_name, ref_path in audit_refs_map.items():
                text = _coerce_str(
                    ref_path,
                    name=f"artifact_quality.audit_refs.{ref_name}",
                ).strip()
                if not text:
                    raise ValueError(
                        f"infer-config artifact_quality.audit_refs.{ref_name} must be non-empty."
                    )
                audit_refs[str(ref_name)] = text
            artifact_quality["audit_refs"] = audit_refs

        deploy_refs_raw = artifact_quality.get("deploy_refs", None)
        deploy_refs_map = _optional_mapping(deploy_refs_raw, name="artifact_quality.deploy_refs")
        if deploy_refs_map is not None:
            deploy_refs: dict[str, str] = {}
            for ref_name, ref_path in deploy_refs_map.items():
                text = _coerce_str(
                    ref_path,
                    name=f"artifact_quality.deploy_refs.{ref_name}",
                ).strip()
                if not text:
                    raise ValueError(
                        f"infer-config artifact_quality.deploy_refs.{ref_name} must be non-empty."
                    )
                deploy_refs[str(ref_name)] = text
            artifact_quality["deploy_refs"] = deploy_refs

        if bool(artifact_quality.get("has_bundle_manifest", False)):
            deploy_refs = artifact_quality.get("deploy_refs", None)
            if not isinstance(deploy_refs, Mapping) or not str(
                deploy_refs.get("bundle_manifest", "")
            ).strip():
                raise ValueError(
                    "infer-config artifact_quality.has_bundle_manifest=true requires "
                    "artifact_quality.deploy_refs.bundle_manifest."
                )

        if bool(artifact_quality.get("has_operator_contract", False)):
            audit_refs = artifact_quality.get("audit_refs", None)
            if not isinstance(audit_refs, Mapping) or not str(
                audit_refs.get("operator_contract", "")
            ).strip():
                raise ValueError(
                    "infer-config artifact_quality.has_operator_contract=true requires "
                    "artifact_quality.audit_refs.operator_contract."
                )
            if not isinstance(normalized.get("operator_contract", None), Mapping):
                raise ValueError(
                    "infer-config artifact_quality.has_operator_contract=true requires "
                    "operator_contract object."
                )

        normalized["artifact_quality"] = artifact_quality

    resolved_ckpt: Path | None = None
    resolved_model_ckpt: Path | None = None
    if bool(check_files):
        if config_path is None:
            warnings.append(
                "check_files=True but config_path is missing; checkpoint_path existence was not checked."
            )
            if isinstance(normalized.get("artifact_quality", None), Mapping):
                audit_refs = normalized["artifact_quality"].get("audit_refs", None)
                if isinstance(audit_refs, Mapping):
                    warnings.append(
                        "check_files=True but config_path is missing; artifact_quality audit_refs existence was not checked."
                    )
                deploy_refs = normalized["artifact_quality"].get("deploy_refs", None)
                if isinstance(deploy_refs, Mapping):
                    warnings.append(
                        "check_files=True but config_path is missing; artifact_quality deploy_refs existence was not checked."
                    )
        else:
            resolved_ckpt = resolve_infer_checkpoint_path(normalized, config_path=config_path)
            resolved_model_ckpt = resolve_infer_model_checkpoint_path(
                normalized, config_path=config_path
            )
            if isinstance(normalized.get("artifact_quality", None), Mapping):
                audit_refs = normalized["artifact_quality"].get("audit_refs", None)
                if isinstance(audit_refs, Mapping):
                    for ref_name, ref_path in audit_refs.items():
                        _resolve_infer_artifact_ref_path(
                            str(ref_path),
                            config_path=config_path,
                            field_name=f"artifact_quality.audit_refs.{ref_name}",
                        )
                deploy_refs = normalized["artifact_quality"].get("deploy_refs", None)
                if isinstance(deploy_refs, Mapping):
                    for ref_name, ref_path in deploy_refs.items():
                        _resolve_infer_artifact_ref_path(
                            str(ref_path),
                            config_path=config_path,
                            field_name=f"artifact_quality.deploy_refs.{ref_name}",
                        )

    return InferConfigValidation(
        payload=normalized,
        resolved_checkpoint_path=resolved_ckpt,
        resolved_model_checkpoint_path=resolved_model_ckpt,
        trust_summary=_build_infer_config_trust_summary(
            normalized,
            check_files=bool(check_files),
        ),
        warnings=warnings,
    )


def validate_infer_config_file(
    path: str | Path,
    *,
    category: str | None = None,
    check_files: bool = True,
) -> InferConfigValidation:
    """Load + validate an infer-config JSON file from disk."""

    p = Path(path)
    payload = load_infer_config(p)
    return validate_infer_config_payload(
        payload, config_path=p, category=category, check_files=check_files
    )
