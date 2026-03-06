from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from pyimgano.inference.api import (
    InferenceTiming,
    calibrate_threshold,
    infer_iter,
    result_to_jsonable,
)
from pyimgano.models.registry import create_model
from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _parse_json_mapping_arg(text: str, *, arg_name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be valid JSON. Original error: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"{arg_name} must be a JSON object (e.g. '{{\"k\": 1}}').")

    return dict(parsed)


def _parse_csv_ints_arg(text: str, *, arg_name: str) -> list[int]:
    raw = [t.strip() for t in str(text).split(",")]
    out: list[int] = []
    for item in raw:
        if not item:
            continue
        try:
            out.append(int(item))
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            raise ValueError(f"{arg_name} must be a comma-separated list of ints, got {text!r}") from exc
    return out


def _parse_csv_strs_arg(text: str, *, arg_name: str) -> list[str]:
    raw = [t.strip() for t in str(text).split(",")]
    return [t for t in raw if t]


def _looks_like_onnx_route(model_name: str, user_kwargs: dict[str, Any]) -> bool:
    name = str(model_name)
    if "onnx" in name:
        return True
    if str(user_kwargs.get("embedding_extractor", "")).strip() == "onnx_embed":
        return True
    fx = user_kwargs.get("feature_extractor", None)
    if fx == "onnx_embed":
        return True
    if isinstance(fx, dict) and str(fx.get("name", "")).strip() == "onnx_embed":
        return True
    return False


def _apply_onnx_session_options_shorthand(
    *,
    model_name: str,
    user_kwargs: dict[str, Any],
    session_options: dict[str, Any] | None,
) -> dict[str, Any]:
    """Best-effort injection of session_options into model kwargs.

    Supported targets:
    - `vision_onnx_*` wrappers: top-level `session_options`
    - embedding wrappers that accept `embedding_kwargs`: nested `embedding_kwargs.session_options`
    - feature-pipeline models that accept `feature_extractor`: nested `feature_extractor.kwargs.session_options`
    """

    if not session_options:
        return dict(user_kwargs)

    if not _looks_like_onnx_route(model_name, user_kwargs):
        raise ValueError(
            "--onnx-session-options is only supported for ONNX-based routes "
            "(e.g. vision_onnx_* or embedding_extractor='onnx_embed')."
        )

    from pyimgano.models.registry import MODEL_REGISTRY

    entry = MODEL_REGISTRY.info(str(model_name))
    sig = inspect.signature(entry.constructor)
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    accepted = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }

    out = dict(user_kwargs)

    # Preferred: first-class kwarg on industrial ONNX wrappers.
    if accepts_var_kwargs or ("session_options" in accepted):
        base = out.get("session_options", None)
        merged = dict(base) if isinstance(base, dict) else {}
        merged.update(dict(session_options))
        out["session_options"] = merged
        return out

    # Next-best: embedding-core style models.
    if "embedding_kwargs" in accepted:
        embedder = out.get("embedding_extractor", None)
        if embedder is not None and str(embedder) != "onnx_embed":
            raise ValueError(
                "--onnx-session-options requires embedding_extractor='onnx_embed' when targeting embedding_kwargs."
            )
        ek = out.get("embedding_kwargs", None)
        ek_dict = dict(ek) if isinstance(ek, dict) else {}
        base = ek_dict.get("session_options", None)
        merged = dict(base) if isinstance(base, dict) else {}
        merged.update(dict(session_options))
        ek_dict["session_options"] = merged
        out["embedding_kwargs"] = ek_dict
        return out

    # Feature-pipeline models may pass a JSON-friendly feature_extractor spec.
    if "feature_extractor" in accepted:
        fx = out.get("feature_extractor", None)
        if fx == "onnx_embed":
            out["feature_extractor"] = {
                "name": "onnx_embed",
                "kwargs": {"session_options": dict(session_options)},
            }
            return out
        if isinstance(fx, dict) and str(fx.get("name", "")).strip() == "onnx_embed":
            fx_kwargs = fx.get("kwargs", None)
            fx_kwargs_dict = dict(fx_kwargs) if isinstance(fx_kwargs, dict) else {}
            base = fx_kwargs_dict.get("session_options", None)
            merged = dict(base) if isinstance(base, dict) else {}
            merged.update(dict(session_options))
            fx_kwargs_dict["session_options"] = merged
            out["feature_extractor"] = {"name": "onnx_embed", "kwargs": fx_kwargs_dict}
            return out

    raise ValueError(
        "--onnx-session-options could not be applied to this model. "
        "Use --model-kwargs to pass session_options directly."
    )


def _default_onnx_sweep_intra_values() -> list[int]:
    import os

    n = int(os.cpu_count() or 8)
    cap = max(1, min(n, 16))
    vals = [1, 2, 4, 8, 16]
    out = [v for v in vals if v <= cap]
    return out or [1]


def _run_onnx_session_options_sweep(
    *,
    checkpoint_path: str,
    device: str,
    image_size: int,
    batch_size: int,
    inputs: list[str],
    base_session_options: dict[str, Any],
    intra_values: list[int],
    opt_levels: list[str],
    repeats: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run a small timing+stability sweep for onnx_embed SessionOptions."""

    import statistics
    import time

    if str(device).strip().lower() != "cpu":
        raise ValueError("--onnx-sweep currently supports device='cpu' only.")
    if repeats <= 0:
        raise ValueError("--onnx-sweep-repeats must be > 0")
    if not inputs:
        raise ValueError("--onnx-sweep requires at least one input image")

    from pyimgano.features.onnx_embed import ONNXEmbedExtractor

    sample_inputs = list(inputs)

    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for intra in intra_values:
        for opt in opt_levels:
            so = dict(base_session_options)
            so["intra_op_num_threads"] = int(intra)
            so["graph_optimization_level"] = str(opt)

            row: dict[str, Any] = {"session_options": dict(so)}
            try:
                extractor = ONNXEmbedExtractor(
                    checkpoint_path=str(checkpoint_path),
                    device="cpu",
                    batch_size=int(batch_size),
                    image_size=int(image_size),
                    session_options=dict(so),
                )

                # Warm up to materialize the ORT session and stabilize timings.
                extractor.extract(sample_inputs[:1])

                timings: list[float] = []
                first_out = None
                second_out = None
                for i in range(int(repeats)):
                    t0 = time.perf_counter()
                    out = extractor.extract(sample_inputs)
                    t1 = time.perf_counter()
                    timings.append(float(t1 - t0))
                    if i == 0:
                        first_out = np.asarray(out)
                    elif i == 1:
                        second_out = np.asarray(out)

                stable = True
                if first_out is None:
                    stable = False
                else:
                    stable = bool(np.all(np.isfinite(first_out)))
                if stable and (first_out is not None) and (second_out is not None):
                    stable = bool(np.allclose(first_out, second_out, rtol=1e-5, atol=1e-6))

                row.update(
                    {
                        "ok": True,
                        "stable": bool(stable),
                        "timing_seconds": list(timings),
                        "median_seconds": float(statistics.median(timings)) if timings else None,
                        "stdev_seconds": float(statistics.pstdev(timings))
                        if len(timings) >= 2
                        else 0.0,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - CLI boundary
                row.update(
                    {
                        "ok": False,
                        "stable": False,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                )

            candidates.append(row)

            if bool(row.get("ok")) and bool(row.get("stable")):
                if best is None:
                    best = dict(row)
                else:
                    # Rank by median latency; tie-break by lower stdev.
                    a = float(row.get("median_seconds") or 1e99)
                    b = float(best.get("median_seconds") or 1e99)
                    if a < b:
                        best = dict(row)
                    elif a == b:
                        sa = float(row.get("stdev_seconds") or 1e99)
                        sb = float(best.get("stdev_seconds") or 1e99)
                        if sa < sb:
                            best = dict(row)

    payload = {
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "image_size": int(image_size),
        "batch_size": int(batch_size),
        "samples": int(len(sample_inputs)),
        "repeats": int(repeats),
        "grid": {
            "intra_op_num_threads": [int(v) for v in intra_values],
            "graph_optimization_level": [str(v) for v in opt_levels],
        },
        "base_session_options": dict(base_session_options),
        "candidates": list(candidates),
        "best": None if best is None else dict(best),
    }

    if best is None:
        raise RuntimeError(
            "ONNX sweep failed: no stable candidates. "
            "Try reducing the grid (threads/opt-levels) or inspect --onnx-sweep-json."
        )

    best_opts = dict(best.get("session_options") or {})
    return best_opts, payload


def _extract_onnx_checkpoint_path_for_sweep(user_kwargs: dict[str, Any]) -> str | None:
    ckpt = user_kwargs.get("checkpoint_path", None)
    if ckpt is not None and str(ckpt).strip():
        return str(ckpt)

    ek = user_kwargs.get("embedding_kwargs", None)
    if isinstance(ek, dict):
        ckpt2 = ek.get("checkpoint_path", None)
        if ckpt2 is not None and str(ckpt2).strip():
            return str(ckpt2)

    fx = user_kwargs.get("feature_extractor", None)
    if isinstance(fx, dict) and str(fx.get("name", "")).strip() == "onnx_embed":
        kw = fx.get("kwargs", None)
        if isinstance(kw, dict):
            ckpt3 = kw.get("checkpoint_path", None)
            if ckpt3 is not None and str(ckpt3).strip():
                return str(ckpt3)

    return None


def _extract_session_options_for_sweep(user_kwargs: dict[str, Any]) -> dict[str, Any]:
    so = user_kwargs.get("session_options", None)
    if isinstance(so, dict):
        return dict(so)
    ek = user_kwargs.get("embedding_kwargs", None)
    if isinstance(ek, dict):
        so2 = ek.get("session_options", None)
        if isinstance(so2, dict):
            return dict(so2)
    fx = user_kwargs.get("feature_extractor", None)
    if isinstance(fx, dict) and str(fx.get("name", "")).strip() == "onnx_embed":
        kw = fx.get("kwargs", None)
        if isinstance(kw, dict):
            so3 = kw.get("session_options", None)
            if isinstance(so3, dict):
                return dict(so3)
    return {}


def _maybe_apply_onnx_session_options_and_sweep(
    *,
    args: argparse.Namespace,
    model_name: str,
    device: str,
    user_kwargs: dict[str, Any],
    inputs: list[str],
    onnx_session_options_cli: dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply --onnx-session-options and/or --onnx-sweep to model kwargs (best-effort)."""

    if not bool(getattr(args, "onnx_sweep", False)) and not onnx_session_options_cli:
        return dict(user_kwargs)

    if bool(getattr(args, "onnx_sweep", False)):
        sweep_inputs = inputs[: int(getattr(args, "onnx_sweep_samples", 32) or 32)]
        intra_values = (
            _parse_csv_ints_arg(str(args.onnx_sweep_intra), arg_name="--onnx-sweep-intra")
            if getattr(args, "onnx_sweep_intra", None)
            else _default_onnx_sweep_intra_values()
        )
        opt_levels = (
            _parse_csv_strs_arg(
                str(args.onnx_sweep_opt_levels), arg_name="--onnx-sweep-opt-levels"
            )
            if getattr(args, "onnx_sweep_opt_levels", None)
            else ["all", "extended"]
        )

        base_so = _extract_session_options_for_sweep(user_kwargs)
        if onnx_session_options_cli:
            base_so.update(dict(onnx_session_options_cli))

        ckpt_for_sweep = _extract_onnx_checkpoint_path_for_sweep(user_kwargs)
        if ckpt_for_sweep is None:
            raise ValueError(
                "--onnx-sweep requires checkpoint_path for ONNX models. "
                "Provide --checkpoint-path (or set checkpoint_path in --model-kwargs)."
            )

        best_so, sweep_payload = _run_onnx_session_options_sweep(
            checkpoint_path=str(ckpt_for_sweep),
            device=str(device),
            image_size=int(user_kwargs.get("image_size", 224)),
            batch_size=int(user_kwargs.get("batch_size", 16)),
            inputs=list(sweep_inputs),
            base_session_options=dict(base_so),
            intra_values=[int(v) for v in intra_values],
            opt_levels=[str(v) for v in opt_levels],
            repeats=int(getattr(args, "onnx_sweep_repeats", 3)),
        )

        if getattr(args, "onnx_sweep_json", None) is not None:
            sweep_path = Path(str(args.onnx_sweep_json))
            sweep_path.parent.mkdir(parents=True, exist_ok=True)
            sweep_path.write_text(
                json.dumps({"tool": "pyimgano-infer", "onnx_sweep": sweep_payload}, indent=2),
                encoding="utf-8",
            )

        return _apply_onnx_session_options_shorthand(
            model_name=model_name,
            user_kwargs=dict(user_kwargs),
            session_options=dict(best_so),
        )

    # No sweep: just apply the shorthand.
    return _apply_onnx_session_options_shorthand(
        model_name=model_name,
        user_kwargs=dict(user_kwargs),
        session_options=dict(onnx_session_options_cli or {}),
    )


def _require_numpy_model_for_preprocessing(model_name: str) -> None:
    from pyimgano.models.capabilities import compute_model_capabilities
    from pyimgano.models.registry import MODEL_REGISTRY

    entry = MODEL_REGISTRY.info(str(model_name))
    caps = compute_model_capabilities(entry)
    supported_input_modes = tuple(str(m) for m in caps.input_modes)
    if "numpy" in supported_input_modes:
        return

    raise ValueError(
        "PREPROCESSING_REQUIRES_NUMPY_MODEL: preprocessing.illumination_contrast requires a model that supports numpy inputs. "
        f"model={model_name!r} supported_input_modes={supported_input_modes!r}. "
        "Choose a model with tag 'numpy' (e.g. vision_patchcore) or remove preprocessing from infer-config."
    )


def _enforce_checkpoint_requirement(
    *,
    model_name: str,
    model_kwargs: dict[str, Any],
    trained_checkpoint_path: str | None,
) -> None:
    """Fail fast when a model is marked as checkpoint-required.

    This keeps CLI UX predictable and avoids confusing downstream ImportErrors
    when optional backends are involved.
    """

    from pyimgano.models.registry import MODEL_REGISTRY

    entry = MODEL_REGISTRY.info(str(model_name))
    requires = bool(entry.metadata.get("requires_checkpoint", False))
    if not requires:
        return

    has_kwarg = model_kwargs.get("checkpoint_path", None) is not None
    has_trained = trained_checkpoint_path is not None
    if has_kwarg or has_trained:
        return

    raise ValueError(
        f"Model {model_name!r} requires a checkpoint. "
        "Provide --checkpoint-path (or set checkpoint_path in --model-kwargs), "
        "or load via --from-run/--infer-config that includes a checkpoint."
    )


def _apply_defects_defaults_from_payload(
    args: argparse.Namespace,
    defects_payload: dict[str, Any],
) -> None:
    """Apply defaults from an infer-config/run 'defects' payload.

    This keeps `infer_config.json` deploy-friendly: defects settings exported by
    workbench can travel with the model and be picked up by `pyimgano-infer`,
    while still allowing explicit CLI flags to override.
    """

    if not defects_payload:
        return

    def _coerce_roi(value: Any) -> list[float] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError(
                "infer-config defects.roi_xyxy_norm must be a list of length 4 or null"
            )
        try:
            return [float(v) for v in value]
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            raise ValueError(
                f"infer-config defects.roi_xyxy_norm must contain floats, got {value!r}"
            ) from exc

    def _coerce_bool(value: Any, *, name: str) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return bool(value)
        raise ValueError(f"infer-config defects.{name} must be a boolean, got {value!r}")

    def _coerce_int(value: Any, *, name: str) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            raise ValueError(f"infer-config defects.{name} must be an int, got {value!r}") from exc

    def _coerce_float(value: Any, *, name: str) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            raise ValueError(f"infer-config defects.{name} must be a float, got {value!r}") from exc

    def _coerce_mask_format(value: Any) -> str | None:
        if value is None:
            return None
        fmt = str(value)
        if fmt not in ("png", "npy"):
            raise ValueError("infer-config defects.mask_format must be 'png' or 'npy'")
        return fmt

    def _coerce_smoothing_method(value: Any) -> str | None:
        if value is None:
            return None
        method = str(value).lower().strip()
        if method in ("none", "median", "gaussian", "box"):
            return method
        raise ValueError(
            "infer-config defects.map_smoothing.method must be one of: none|median|gaussian|box"
        )

    # Apply defaults only when the CLI did not explicitly set a value.
    if args.roi_xyxy_norm is None:
        roi = _coerce_roi(defects_payload.get("roi_xyxy_norm", None))
        if roi is not None:
            args.roi_xyxy_norm = roi

    # Numeric knobs (only apply if still default).
    if int(getattr(args, "defect_min_area", 0)) == 0:
        v = _coerce_int(defects_payload.get("min_area", None), name="min_area")
        if v is not None:
            args.defect_min_area = int(v)

    if int(getattr(args, "defect_border_ignore_px", 0)) == 0:
        v = _coerce_int(defects_payload.get("border_ignore_px", None), name="border_ignore_px")
        if v is not None:
            args.defect_border_ignore_px = int(v)

    # Optional map smoothing block.
    ms_raw = defects_payload.get("map_smoothing", None)
    if ms_raw is not None:
        if not isinstance(ms_raw, dict):
            raise ValueError("infer-config defects.map_smoothing must be a JSON object/dict.")
        if str(getattr(args, "defect_map_smoothing", "none")) == "none":
            v = _coerce_smoothing_method(ms_raw.get("method", None))
            if v is not None:
                args.defect_map_smoothing = str(v)
        if int(getattr(args, "defect_map_smoothing_ksize", 0)) == 0:
            v = _coerce_int(ms_raw.get("ksize", None), name="map_smoothing.ksize")
            if v is not None:
                args.defect_map_smoothing_ksize = int(v)
        if float(getattr(args, "defect_map_smoothing_sigma", 0.0)) == 0.0:
            v = _coerce_float(ms_raw.get("sigma", None), name="map_smoothing.sigma")
            if v is not None:
                args.defect_map_smoothing_sigma = float(v)

    # Optional hysteresis thresholding block.
    hyst_raw = defects_payload.get("hysteresis", None)
    if hyst_raw is not None:
        if not isinstance(hyst_raw, dict):
            raise ValueError("infer-config defects.hysteresis must be a JSON object/dict.")

        enabled = hyst_raw.get("enabled", None)
        if enabled is True and not bool(getattr(args, "defect_hysteresis", False)):
            args.defect_hysteresis = True

        if getattr(args, "defect_hysteresis_low", None) is None:
            v = _coerce_float(hyst_raw.get("low", None), name="hysteresis.low")
            if v is not None:
                args.defect_hysteresis_low = float(v)

        if getattr(args, "defect_hysteresis_high", None) is None:
            v = _coerce_float(hyst_raw.get("high", None), name="hysteresis.high")
            if v is not None:
                args.defect_hysteresis_high = float(v)

    # Optional shape filters block.
    shape_raw = defects_payload.get("shape_filters", None)
    if shape_raw is not None:
        if not isinstance(shape_raw, dict):
            raise ValueError("infer-config defects.shape_filters must be a JSON object/dict.")

        if getattr(args, "defect_min_fill_ratio", None) is None:
            v = _coerce_float(
                shape_raw.get("min_fill_ratio", None), name="shape_filters.min_fill_ratio"
            )
            if v is not None:
                args.defect_min_fill_ratio = float(v)

        if getattr(args, "defect_max_aspect_ratio", None) is None:
            v = _coerce_float(
                shape_raw.get("max_aspect_ratio", None), name="shape_filters.max_aspect_ratio"
            )
            if v is not None:
                args.defect_max_aspect_ratio = float(v)

        if getattr(args, "defect_min_solidity", None) is None:
            v = _coerce_float(
                shape_raw.get("min_solidity", None), name="shape_filters.min_solidity"
            )
            if v is not None:
                args.defect_min_solidity = float(v)

    # Optional merge-nearby block (regions output only; mask unchanged).
    merge_raw = defects_payload.get("merge_nearby", None)
    if merge_raw is not None:
        if not isinstance(merge_raw, dict):
            raise ValueError("infer-config defects.merge_nearby must be a JSON object/dict.")

        enabled = merge_raw.get("enabled", None)
        if enabled is True and not bool(getattr(args, "defect_merge_nearby", False)):
            args.defect_merge_nearby = True

        if int(getattr(args, "defect_merge_nearby_max_gap_px", 0)) == 0:
            v = _coerce_int(merge_raw.get("max_gap_px", None), name="merge_nearby.max_gap_px")
            if v is not None:
                args.defect_merge_nearby_max_gap_px = int(v)

    if getattr(args, "defect_min_score_max", None) is None:
        v = _coerce_float(defects_payload.get("min_score_max", None), name="min_score_max")
        if v is not None:
            args.defect_min_score_max = float(v)

    if getattr(args, "defect_min_score_mean", None) is None:
        v = _coerce_float(defects_payload.get("min_score_mean", None), name="min_score_mean")
        if v is not None:
            args.defect_min_score_mean = float(v)

    if int(getattr(args, "defect_open_ksize", 0)) == 0:
        v = _coerce_int(defects_payload.get("open_ksize", None), name="open_ksize")
        if v is not None:
            args.defect_open_ksize = int(v)

    if int(getattr(args, "defect_close_ksize", 0)) == 0:
        v = _coerce_int(defects_payload.get("close_ksize", None), name="close_ksize")
        if v is not None:
            args.defect_close_ksize = int(v)

    if getattr(args, "defect_max_regions", None) is None:
        v = defects_payload.get("max_regions", None)
        if v is not None:
            args.defect_max_regions = int(v)

    if str(getattr(args, "defect_max_regions_sort_by", "score_max")) == "score_max":
        v = defects_payload.get("max_regions_sort_by", None)
        if v is not None:
            vv = str(v).lower().strip()
            if vv not in ("score_max", "score_mean", "area"):
                raise ValueError(
                    "infer-config defects.max_regions_sort_by must be one of: score_max|score_mean|area"
                )
            args.defect_max_regions_sort_by = vv

    if float(getattr(args, "pixel_normal_quantile", 0.999)) == 0.999:
        v = _coerce_float(
            defects_payload.get("pixel_normal_quantile", None), name="pixel_normal_quantile"
        )
        if v is not None:
            args.pixel_normal_quantile = float(v)

    if (
        str(getattr(args, "pixel_threshold_strategy", "normal_pixel_quantile"))
        == "normal_pixel_quantile"
    ):
        v = defects_payload.get("pixel_threshold_strategy", None)
        if v is not None:
            args.pixel_threshold_strategy = str(v)

    if str(getattr(args, "mask_format", "png")) == "png":
        v = _coerce_mask_format(defects_payload.get("mask_format", None))
        if v is not None:
            args.mask_format = str(v)

    if not bool(getattr(args, "defect_fill_holes", False)):
        v = _coerce_bool(defects_payload.get("fill_holes", None), name="fill_holes")
        if v is True:
            args.defect_fill_holes = True


def _apply_defects_preset_if_requested(args: argparse.Namespace) -> None:
    preset_name = getattr(args, "defects_preset", None)
    if preset_name is None:
        return

    from pyimgano.cli_presets import resolve_defects_preset

    preset = resolve_defects_preset(str(preset_name))
    if preset is None:
        raise ValueError(f"Unknown defects preset: {preset_name!r}")

    # Presets are intended to be a one-flag industrial on-ramp.
    args.defects = True
    _apply_defects_defaults_from_payload(args, dict(preset.payload))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyimgano-infer")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", default=None, help="Registered model name")
    source.add_argument(
        "--model-preset",
        default=None,
        help=(
            "Model preset name (shortcut to a registered model + kwargs). "
            "Use --list-model-presets to see options."
        ),
    )
    source.add_argument(
        "--from-run",
        default=None,
        help="Load model + threshold + (optional) checkpoint from a workbench run directory",
    )
    source.add_argument(
        "--infer-config",
        default=None,
        help="Load model + threshold + (optional) checkpoint from an exported infer_config.json file",
    )
    source.add_argument(
        "--list-models",
        action="store_true",
        help="List available model names and exit (default output: text, one per line)",
    )
    source.add_argument(
        "--model-info",
        default=None,
        help="Show tags/metadata/signature/accepted kwargs for a model name and exit",
    )
    source.add_argument(
        "--list-model-presets",
        action="store_true",
        help="List available model preset names and exit (default output: text, one per line)",
    )
    source.add_argument(
        "--model-preset-info",
        default=None,
        help="Show model preset details (model/kwargs/description) and exit",
    )
    parser.add_argument(
        "--from-run-category",
        default=None,
        help="When --from-run has multiple categories, select one category name",
    )
    parser.add_argument(
        "--infer-category",
        default=None,
        help="When --infer-config has multiple categories, select one category name",
    )
    parser.add_argument(
        "--preset",
        default=None,
        choices=["industrial-fast", "industrial-balanced", "industrial-accurate"],
        help="Optional model preset (applied before --model-kwargs). Default: none",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "When used with discovery flags (--list-models/--model-info/"
            "--list-model-presets/--model-preset-info), output JSON instead of text"
        ),
    )
    parser.add_argument(
        "--tags",
        action="append",
        default=None,
        help=(
            "Filter --list-models by required tags (comma-separated or repeatable). "
            "Example: --tags vision,classical"
        ),
    )
    parser.add_argument("--device", default=None, help="cpu|cuda (model dependent)")
    parser.add_argument("--contamination", type=float, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for reproducibility (best-effort; passed as "
            "random_seed/random_state when supported)"
        ),
    )
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--model-kwargs",
        default=None,
        help="JSON object of extra model constructor kwargs, e.g. '{\"k\": 1}' (advanced)",
    )
    parser.add_argument(
        "--onnx-session-options",
        default=None,
        help=(
            "Optional JSON object of onnxruntime SessionOptions to pass to ONNX-based routes "
            "(e.g. vision_onnx_* wrappers). This avoids nested --model-kwargs. "
            "Example: '{\"intra_op_num_threads\":8,\"graph_optimization_level\":\"all\"}'"
        ),
    )
    parser.add_argument(
        "--onnx-sweep",
        action="store_true",
        help=(
            "Run a small onnxruntime CPU SessionOptions sweep (threads + graph optimization level) "
            "and apply the best session_options before inference. Requires an ONNX route and --device cpu."
        ),
    )
    parser.add_argument(
        "--onnx-sweep-intra",
        default=None,
        help="Comma-separated intra_op_num_threads values for --onnx-sweep (default: 1,2,4,8 capped at 16).",
    )
    parser.add_argument(
        "--onnx-sweep-opt-levels",
        default=None,
        help="Comma-separated graph_optimization_level values for --onnx-sweep (default: all,extended).",
    )
    parser.add_argument(
        "--onnx-sweep-repeats",
        type=int,
        default=3,
        help="Timed repeats per candidate for --onnx-sweep (default: 3).",
    )
    parser.add_argument(
        "--onnx-sweep-samples",
        type=int,
        default=32,
        help="Max number of input images used for --onnx-sweep timing (default: 32).",
    )
    parser.add_argument(
        "--onnx-sweep-json",
        default=None,
        help="Optional path to write ONNX sweep results JSON (grid + timings + selected best).",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional checkpoint path for checkpoint-backed models; sets model kwarg checkpoint_path",
    )
    parser.add_argument(
        "--train-dir",
        default=None,
        help="Optional directory of normal images used to `fit()` and calibrate threshold",
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        help=(
            "Optional directory of golden reference images for reference-based detectors. "
            "Matched by basename (filename)."
        ),
    )
    parser.add_argument(
        "--calibration-quantile",
        type=float,
        default=None,
        help=(
            "Optional score quantile used to set threshold_ from train scores (e.g. 0.995). "
            "Requires --train-dir. When omitted and the detector does not set threshold_ during fit(), "
            "defaults to 1-contamination when available, else 0.995."
        ),
    )
    parser.add_argument(
        "--input",
        action="append",
        required=False,
        help="Input image path or directory (repeatable). Directories are scanned recursively.",
    )
    parser.add_argument(
        "--include-maps",
        action="store_true",
        help="Request anomaly maps if detector supports them",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Optional tile size for high-resolution inference (requires a numpy-capable detector)",
    )
    parser.add_argument(
        "--tile-stride",
        type=int,
        default=None,
        help="Optional tile stride (defaults to tile-size)",
    )
    parser.add_argument(
        "--tile-score-reduce",
        default="max",
        choices=["max", "mean", "topk_mean"],
        help="How to aggregate tile scores into an image score",
    )
    parser.add_argument(
        "--tile-map-reduce",
        default="max",
        choices=["max", "mean", "hann", "gaussian"],
        help="How to blend overlapping tile maps",
    )
    parser.add_argument(
        "--tile-score-topk",
        type=float,
        default=0.1,
        help="Top-k fraction used for tile-score reduce mode 'topk_mean'",
    )
    parser.add_argument("--save-jsonl", default=None, help="Optional JSONL output path")
    parser.add_argument(
        "--save-maps",
        default=None,
        help="Optional directory to save anomaly maps as .npy (requires --include-maps)",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Apply standard postprocess to anomaly maps (only if --include-maps)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Optional chunk size for inference (default: 0, disabled)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print stage timing summary to stderr",
    )
    parser.add_argument(
        "--profile-json",
        default=None,
        help="Optional path to write a JSON profile payload (stable, machine-friendly).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Best-effort AMP/autocast for torch-backed models (requires torch + CUDA)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Best-effort production mode: record per-input errors and keep going. "
            "Exits with code 1 if any errors occurred."
        ),
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=0,
        help=(
            "Stop early after N errors when --continue-on-error is set. " "Default: 0 (no limit)."
        ),
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=0,
        help=(
            "Flush JSONL output files every N records (stability vs performance). "
            "Default: 0 (no periodic flush)."
        ),
    )
    parser.add_argument(
        "--include-anomaly-map-values",
        action="store_true",
        help="Include raw anomaly map values in JSONL (debug only; very large output)",
    )

    # Industrial defects export (mask + regions). Opt-in.
    parser.add_argument(
        "--defects",
        action="store_true",
        help="Enable defects export: binary mask + connected-component regions (requires anomaly maps)",
    )
    from pyimgano.cli_presets import list_defects_presets

    parser.add_argument(
        "--defects-preset",
        default=None,
        choices=list_defects_presets(),
        help=(
            "Optional defects postprocess preset (implies --defects). "
            "Applies default ROI/border/smoothing/hysteresis/shape filters for industrial FP reduction."
        ),
    )
    parser.add_argument(
        "--defects-image-space",
        action="store_true",
        help="Add bbox_xyxy_image to defects regions (best-effort; requires image size)",
    )
    parser.add_argument(
        "--save-masks",
        default=None,
        help="Optional directory to save binary defect masks (requires --defects)",
    )
    parser.add_argument(
        "--defects-regions-jsonl",
        default=None,
        help="Optional JSONL path to write per-image defect regions payloads (requires --defects).",
    )
    parser.add_argument(
        "--save-overlays",
        default=None,
        help="Optional directory to save FP debugging overlays (original + heatmap + mask outline/fill)",
    )
    parser.add_argument(
        "--mask-format",
        default="png",
        choices=["png", "npy", "npz"],
        help="Mask artifact format when saving masks (default: png)",
    )
    parser.add_argument(
        "--defects-mask-space",
        default="roi",
        choices=["roi", "full"],
        help=(
            "Which defect mask to export when ROI is set. "
            "roi=mask is limited to ROI (default); full=mask includes full-map pixels. "
            "Regions are always extracted within ROI when ROI is set."
        ),
    )
    parser.add_argument(
        "--pixel-threshold",
        type=float,
        default=None,
        help="Optional fixed pixel threshold used to binarize anomaly maps into defect masks",
    )
    parser.add_argument(
        "--pixel-threshold-strategy",
        default="normal_pixel_quantile",
        choices=["normal_pixel_quantile", "fixed", "infer_config"],
        help="How to resolve pixel threshold for defects (default: normal_pixel_quantile)",
    )
    parser.add_argument(
        "--pixel-normal-quantile",
        type=float,
        default=0.999,
        help="Quantile used for pixel threshold calibration on normal pixels (default: 0.999)",
    )
    parser.add_argument(
        "--defect-min-area",
        type=int,
        default=0,
        help="Remove connected components smaller than this area (default: 0)",
    )
    parser.add_argument(
        "--defect-border-ignore-px",
        type=int,
        default=0,
        help="Ignore N pixels at the anomaly-map border for defects extraction (default: 0)",
    )
    parser.add_argument(
        "--defect-map-smoothing",
        default="none",
        choices=["none", "median", "gaussian", "box"],
        help="Optional anomaly-map smoothing method for defects extraction (default: none)",
    )
    parser.add_argument(
        "--defect-map-smoothing-ksize",
        type=int,
        default=0,
        help="Optional smoothing kernel size (method dependent; default: 0)",
    )
    parser.add_argument(
        "--defect-map-smoothing-sigma",
        type=float,
        default=0.0,
        help="Optional gaussian sigma for map smoothing (default: 0.0)",
    )
    parser.add_argument(
        "--defect-hysteresis",
        action="store_true",
        help="Enable hysteresis thresholding (keeps low regions connected to high seeds)",
    )
    parser.add_argument(
        "--defect-hysteresis-low",
        type=float,
        default=None,
        help="Low threshold for hysteresis (default: derived from high threshold)",
    )
    parser.add_argument(
        "--defect-hysteresis-high",
        type=float,
        default=None,
        help="High threshold for hysteresis (default: pixel threshold)",
    )
    parser.add_argument(
        "--defect-merge-nearby",
        action="store_true",
        help="Merge nearby defect regions in JSONL output (mask unchanged; default: off)",
    )
    parser.add_argument(
        "--defect-merge-nearby-max-gap-px",
        type=int,
        default=0,
        help="Max bbox gap (px) for merging nearby regions (default: 0, disabled)",
    )
    parser.add_argument(
        "--defect-min-fill-ratio",
        type=float,
        default=None,
        help="Optional minimum fill ratio (area / bbox_area) for components (default: none)",
    )
    parser.add_argument(
        "--defect-max-aspect-ratio",
        type=float,
        default=None,
        help="Optional maximum aspect ratio for components (default: none)",
    )
    parser.add_argument(
        "--defect-min-solidity",
        type=float,
        default=None,
        help="Optional minimum solidity for components (default: none)",
    )
    parser.add_argument(
        "--defect-min-score-max",
        type=float,
        default=None,
        help="Remove components whose max anomaly score is below this (default: none)",
    )
    parser.add_argument(
        "--defect-min-score-mean",
        type=float,
        default=None,
        help="Remove components whose mean anomaly score is below this (default: none)",
    )
    parser.add_argument(
        "--defect-open-ksize",
        type=int,
        default=0,
        help="Morphology open kernel size applied to defect masks (default: 0, disabled)",
    )
    parser.add_argument(
        "--defect-close-ksize",
        type=int,
        default=0,
        help="Morphology close kernel size applied to defect masks (default: 0, disabled)",
    )
    parser.add_argument(
        "--defect-fill-holes",
        action="store_true",
        help="Fill internal holes in defect masks",
    )
    parser.add_argument(
        "--defects-mask-dilate",
        type=int,
        default=0,
        help="Optional dilation kernel size applied to exported defect masks (default: 0, disabled)",
    )
    parser.add_argument(
        "--defect-max-regions",
        type=int,
        default=None,
        help="Optional maximum number of regions to emit per image (after sorting)",
    )
    parser.add_argument(
        "--defect-max-regions-sort-by",
        default="score_max",
        choices=["score_max", "score_mean", "area"],
        help="Sort key for max-regions selection (default: score_max)",
    )
    parser.add_argument(
        "--roi-xyxy-norm",
        type=float,
        nargs=4,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help=(
            "Optional normalized ROI rectangle (x1 y1 x2 y2 in [0,1]) applied to defects only. "
            "When using --infer-config/--from-run, this defaults from the exported defects.roi_xyxy_norm."
        ),
    )
    return parser


def _collect_image_paths(raw: str | Path) -> list[str]:
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.is_file():
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {path}")
        return [str(path)]

    out: list[str] = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
            out.append(str(p))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _apply_defects_preset_if_requested(args)

    try:
        # Import implementations for side effects (registry population).
        import time

        import pyimgano.models  # noqa: F401
        from pyimgano.cli import _resolve_preset_kwargs
        from pyimgano.cli_common import (
            build_model_kwargs,
            merge_checkpoint_path,
            parse_model_kwargs,
        )
        from pyimgano.models.registry import MODEL_REGISTRY, materialize_model_constructor

        discovery_flags = [
            bool(getattr(args, "list_models", False)),
            getattr(args, "model_info", None) is not None,
            bool(getattr(args, "list_model_presets", False)),
            getattr(args, "model_preset_info", None) is not None,
        ]
        if sum(1 for f in discovery_flags if f) > 1:
            raise ValueError(
                "--list-models, --model-info, --list-model-presets, and --model-preset-info are mutually exclusive."
            )

        if bool(getattr(args, "list_models", False)):
            tags: list[str] = []
            tags_raw = getattr(args, "tags", None)
            if tags_raw:
                for item in tags_raw:
                    for tag in str(item).split(","):
                        tag = tag.strip()
                        if tag:
                            tags.append(tag)

            names = MODEL_REGISTRY.available(tags=tags or None)
            if bool(getattr(args, "json", False)):
                print(json.dumps(names, indent=2))
            else:
                for name in names:
                    print(name)
            return 0

        if getattr(args, "model_info", None) is not None:
            from pyimgano.utils.jsonable import to_jsonable

            model_name = str(getattr(args, "model_info"))
            try:
                materialize_model_constructor(model_name)
                entry = MODEL_REGISTRY.info(model_name)
            except KeyError as exc:
                raise ValueError(f"Unknown model: {model_name!r}") from exc

            signature = inspect.signature(entry.constructor)
            accepts_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
            )
            accepted = {
                name
                for name, p in signature.parameters.items()
                if p.kind
                in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }

            payload = {
                "name": entry.name,
                "tags": list(entry.tags),
                "metadata": dict(entry.metadata),
                "signature": str(signature),
                "accepted_kwargs": sorted(accepted),
                "accepts_var_kwargs": bool(accepts_var_kwargs),
                "constructor": {
                    "module": getattr(entry.constructor, "__module__", "<unknown>"),
                    "qualname": getattr(entry.constructor, "__qualname__", "<unknown>"),
                },
            }

            if bool(getattr(args, "json", False)):
                print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
            else:
                print(f"Name: {payload['name']}")
                tags_out = payload["tags"]
                print(f"Tags: {', '.join(tags_out) if tags_out else '<none>'}")
                print("Metadata:")
                metadata = payload["metadata"]
                if metadata:
                    for key in sorted(metadata):
                        print(f"  {key}: {metadata[key]}")
                else:
                    print("  <none>")
                print("Signature:")
                print(f"  {payload['signature']}")
                print(f"Accepts **kwargs: {'yes' if payload['accepts_var_kwargs'] else 'no'}")
                print("Accepted kwargs:")
                for key in payload["accepted_kwargs"]:
                    print(f"  - {key}")
            return 0

        if bool(getattr(args, "list_model_presets", False)):
            from pyimgano.cli_presets import list_model_presets

            names = list_model_presets()
            if bool(getattr(args, "json", False)):
                print(json.dumps(names, indent=2))
            else:
                for name in names:
                    print(name)
            return 0

        if getattr(args, "model_preset_info", None) is not None:
            from pyimgano.cli_presets import resolve_model_preset
            from pyimgano.utils.jsonable import to_jsonable

            preset_name = str(getattr(args, "model_preset_info"))
            preset = resolve_model_preset(preset_name)
            if preset is None:
                raise ValueError(f"Unknown model preset: {preset_name!r}")

            payload = {
                "name": preset.name,
                "model": preset.model,
                "kwargs": dict(preset.kwargs),
                "description": preset.description,
                "optional": bool(preset.optional),
            }
            if bool(getattr(args, "json", False)):
                print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
            else:
                print(f"Name: {payload['name']}")
                print(f"Model: {payload['model']}")
                print("Kwargs:")
                kwargs = payload["kwargs"]
                if kwargs:
                    for key in sorted(kwargs):
                        print(f"  {key}: {kwargs[key]}")
                else:
                    print("  <none>")
                print(f"Optional: {'yes' if payload['optional'] else 'no'}")
                print(f"Description: {payload['description']}")
            return 0

        t_total_start = time.perf_counter()
        t_load_model = 0.0
        t_fit_calibrate = 0.0
        t_infer = 0.0
        t_artifacts = 0.0

        if not getattr(args, "input", None):
            raise ValueError(
                "--input is required for inference modes. "
                "Use --list-models/--model-info/--list-model-presets for discovery."
            )

        onnx_session_options_cli: dict[str, Any] | None = None
        if getattr(args, "onnx_session_options", None) is not None:
            onnx_session_options_cli = _parse_json_mapping_arg(
                str(args.onnx_session_options), arg_name="--onnx-session-options"
            )

        seed = int(args.seed) if args.seed is not None else None
        if seed is not None:
            from pyimgano.utils.seeding import seed_everything

            seed_everything(int(seed))

        inputs: list[str] = []
        for raw in args.input:
            inputs.extend(_collect_image_paths(raw))
        if not inputs:
            raise ValueError("No input images found.")

        from_run = args.from_run is not None
        infer_config_mode = args.infer_config is not None
        trained_checkpoint_path = None
        threshold_from_run = None
        infer_config_postprocess = None
        defects_payload: dict[str, Any] | None = None
        defects_payload_source: str | None = None
        illumination_contrast_knobs = None
        tiling_payload: dict[str, Any] | None = None

        t_load_start = time.perf_counter()
        if from_run:
            from pyimgano.workbench.load_run import (
                extract_threshold,
                load_checkpoint_into_detector,
                load_report_from_run,
                load_workbench_config_from_run,
                resolve_checkpoint_path,
                select_category_report,
            )

            cfg = load_workbench_config_from_run(args.from_run)
            report = load_report_from_run(args.from_run)
            _cat_name, cat_report = select_category_report(
                report,
                category=(
                    str(args.from_run_category) if args.from_run_category is not None else None
                ),
            )

            threshold_from_run = extract_threshold(cat_report)
            trained_checkpoint_path = resolve_checkpoint_path(args.from_run, cat_report)

            model_name = str(cfg.model.name)
            preset = cfg.model.preset
            device = str(cfg.model.device)
            contamination = float(cfg.model.contamination)
            pretrained = bool(cfg.model.pretrained)

            if args.preset is not None:
                preset = str(args.preset)
            if args.device is not None:
                device = str(args.device)
            if args.contamination is not None:
                contamination = float(args.contamination)
            if args.pretrained is not None:
                pretrained = bool(args.pretrained)

            base_user_kwargs = dict(cfg.model.model_kwargs)
            if args.model_kwargs is not None:
                base_user_kwargs.update(parse_model_kwargs(args.model_kwargs))

            checkpoint_path: str | None = None
            if args.checkpoint_path is not None:
                checkpoint_path = str(args.checkpoint_path)
            elif cfg.model.checkpoint_path is not None:
                raw = str(cfg.model.checkpoint_path).strip()
                if raw:
                    p = Path(raw)
                    if p.is_absolute():
                        if not p.exists():
                            raise FileNotFoundError(f"Model checkpoint_path not found: {p}")
                        checkpoint_path = str(p)
                    else:
                        # Best-effort: interpret relative paths as relative to the run directory.
                        run_dir = Path(str(args.from_run))
                        candidates = [
                            (run_dir / p).resolve(),
                            (run_dir / "artifacts" / p).resolve(),
                            (run_dir / "checkpoints" / p).resolve(),
                        ]
                        resolved: Path | None = None
                        for cand in candidates:
                            if cand.exists():
                                resolved = cand
                                break

                        if resolved is None:
                            # Backwards-compat fallback: allow relative to the current working dir.
                            cwd_cand = p.resolve()
                            if cwd_cand.exists():
                                import sys

                                print(
                                    "warning: model.checkpoint_path resolved relative to CWD; "
                                    "consider making it relative to --from-run for portability.",
                                    file=sys.stderr,
                                )
                                resolved = cwd_cand
                            else:
                                tried = "\n".join(f"- {c}" for c in (candidates + [cwd_cand]))
                                raise FileNotFoundError(
                                    "Model checkpoint_path not found for --from-run.\n"
                                    f"model.checkpoint_path={raw!r}\n"
                                    f"from_run={run_dir}\n"
                                    "Tried:\n"
                                    f"{tried}"
                                )

                        checkpoint_path = str(resolved)
            user_kwargs = merge_checkpoint_path(base_user_kwargs, checkpoint_path=checkpoint_path)

            user_kwargs = _maybe_apply_onnx_session_options_and_sweep(
                args=args,
                model_name=model_name,
                device=str(device),
                user_kwargs=dict(user_kwargs),
                inputs=list(inputs),
                onnx_session_options_cli=onnx_session_options_cli,
            )

            preset_kwargs = _resolve_preset_kwargs(preset, model_name)
            model_kwargs = build_model_kwargs(
                model_name,
                user_kwargs=user_kwargs,
                preset_kwargs=preset_kwargs,
                auto_kwargs={
                    "device": device,
                    "contamination": contamination,
                    "pretrained": pretrained,
                    **(
                        {"random_seed": int(seed), "random_state": int(seed)}
                        if seed is not None
                        else {}
                    ),
                },
            )
            _enforce_checkpoint_requirement(
                model_name=model_name,
                model_kwargs=dict(model_kwargs),
                trained_checkpoint_path=trained_checkpoint_path,
            )
            detector = create_model(model_name, **model_kwargs)

            if trained_checkpoint_path is not None:
                load_checkpoint_into_detector(detector, trained_checkpoint_path)
            if threshold_from_run is not None:
                setattr(detector, "threshold_", float(threshold_from_run))

            # Use workbench defects config as deploy defaults for `pyimgano-infer`.
            defects_payload = asdict(cfg.defects)
            defects_payload_source = "from_run"

            # Optional preprocessing defaults from workbench runs.
            try:
                ic = cfg.preprocessing.illumination_contrast
            except Exception:
                ic = None
            if ic is not None:
                illumination_contrast_knobs = ic
            try:
                tiling = cfg.adaptation.tiling
            except Exception:
                tiling = None
            if tiling is not None and getattr(tiling, "tile_size", None) is not None:
                tiling_payload = {
                    "tile_size": int(tiling.tile_size),
                    "stride": (
                        int(tiling.stride) if getattr(tiling, "stride", None) is not None else None
                    ),
                    "score_reduce": str(tiling.score_reduce),
                    "score_topk": float(tiling.score_topk),
                    "map_reduce": str(tiling.map_reduce),
                }
        elif infer_config_mode:
            from pyimgano.inference.config import (
                load_infer_config,
                resolve_infer_checkpoint_path,
                resolve_infer_model_checkpoint_path,
                select_infer_category,
            )
            from pyimgano.workbench.load_run import extract_threshold, load_checkpoint_into_detector

            cfg_path = Path(str(args.infer_config))
            payload = load_infer_config(cfg_path)
            payload = select_infer_category(
                payload,
                category=(str(args.infer_category) if args.infer_category is not None else None),
            )
            defects_map = payload.get("defects", None)
            if defects_map is not None:
                if not isinstance(defects_map, dict):
                    raise ValueError("infer-config key 'defects' must be a JSON object/dict.")
                defects_payload = dict(defects_map)
                defects_payload_source = "infer_config"

            preprocessing_map = payload.get("preprocessing", None)
            if preprocessing_map is not None:
                if not isinstance(preprocessing_map, dict):
                    raise ValueError("infer-config key 'preprocessing' must be a JSON object/dict.")
                ic_map = preprocessing_map.get("illumination_contrast", None)
                if ic_map is not None:
                    if not isinstance(ic_map, dict):
                        raise ValueError(
                            "infer-config key 'preprocessing.illumination_contrast' must be a JSON object/dict."
                        )
                    from pyimgano.inference.preprocessing import parse_illumination_contrast_knobs

                    illumination_contrast_knobs = parse_illumination_contrast_knobs(ic_map)

            model_payload = payload.get("model", None)
            if not isinstance(model_payload, dict):
                raise ValueError("infer-config must contain a JSON object at key 'model'.")
            model_name = model_payload.get("name", None)
            if model_name is None:
                raise ValueError("infer-config model.name is required.")
            model_name = str(model_name)

            adaptation_payload = payload.get("adaptation", None)
            if adaptation_payload is None:
                adaptation_payload = {}
            if not isinstance(adaptation_payload, dict):
                raise ValueError("infer-config key 'adaptation' must be a JSON object/dict.")
            tiling_raw = adaptation_payload.get("tiling", None)
            if isinstance(tiling_raw, dict):
                tiling_payload = dict(tiling_raw)

            threshold_from_run = extract_threshold(payload)
            trained_checkpoint_path = resolve_infer_checkpoint_path(payload, config_path=cfg_path)

            preset = model_payload.get("preset", None)
            device = model_payload.get("device", "cpu")
            contamination = model_payload.get("contamination", 0.1)
            pretrained = model_payload.get("pretrained", True)

            if args.preset is not None:
                preset = str(args.preset)
            if args.device is not None:
                device = str(args.device)
            if args.contamination is not None:
                contamination = float(args.contamination)
            if args.pretrained is not None:
                pretrained = bool(args.pretrained)

            base_user_kwargs = dict(model_payload.get("model_kwargs", {}) or {})
            if args.model_kwargs is not None:
                base_user_kwargs.update(parse_model_kwargs(args.model_kwargs))

            checkpoint_path: str | None = None
            if args.checkpoint_path is not None:
                checkpoint_path = str(args.checkpoint_path)
            elif model_payload.get("checkpoint_path", None) is not None:
                resolved_model_ckpt = resolve_infer_model_checkpoint_path(
                    payload, config_path=cfg_path
                )
                checkpoint_path = (
                    str(resolved_model_ckpt) if resolved_model_ckpt is not None else None
                )
            user_kwargs = merge_checkpoint_path(base_user_kwargs, checkpoint_path=checkpoint_path)
            user_kwargs = _maybe_apply_onnx_session_options_and_sweep(
                args=args,
                model_name=model_name,
                device=str(device),
                user_kwargs=dict(user_kwargs),
                inputs=list(inputs),
                onnx_session_options_cli=onnx_session_options_cli,
            )

            preset_kwargs = _resolve_preset_kwargs(preset, model_name)
            model_kwargs = build_model_kwargs(
                model_name,
                user_kwargs=user_kwargs,
                preset_kwargs=preset_kwargs,
                auto_kwargs={
                    "device": str(device),
                    "contamination": float(contamination),
                    "pretrained": bool(pretrained),
                    **(
                        {"random_seed": int(seed), "random_state": int(seed)}
                        if seed is not None
                        else {}
                    ),
                },
            )
            _enforce_checkpoint_requirement(
                model_name=model_name,
                model_kwargs=dict(model_kwargs),
                trained_checkpoint_path=trained_checkpoint_path,
            )
            detector = create_model(model_name, **model_kwargs)

            if trained_checkpoint_path is not None:
                load_checkpoint_into_detector(detector, trained_checkpoint_path)
            if threshold_from_run is not None:
                setattr(detector, "threshold_", float(threshold_from_run))

            # Infer-config may request maps/postprocess by default.
            post_cfg = adaptation_payload.get("postprocess", None)
            if isinstance(post_cfg, dict):
                infer_config_postprocess = dict(post_cfg)

            if not bool(args.include_maps):
                if (
                    bool(adaptation_payload.get("save_maps", False))
                    or infer_config_postprocess is not None
                ):
                    args.include_maps = True
        else:
            from pyimgano.cli_presets import resolve_model_preset

            requested_model: str | None = None
            preset_model_auto_kwargs: dict[str, Any] = {}

            if getattr(args, "model_preset", None) is not None:
                requested_model = str(args.model_preset)
                preset = resolve_model_preset(requested_model)
                if preset is None:
                    raise ValueError(f"Unknown model preset: {requested_model!r}")
                model_name = str(preset.model)
                preset_model_auto_kwargs = dict(preset.kwargs)
                try:
                    MODEL_REGISTRY.info(model_name)
                except KeyError as exc:
                    raise ValueError(
                        f"Model preset {requested_model!r} refers to unknown model: {model_name!r}"
                    ) from exc
            else:
                if args.model is None:
                    raise ValueError(
                        "--model or --model-preset is required when --from-run/--infer-config are not provided"
                    )

                requested_model = str(args.model)
                model_name = requested_model
                try:
                    MODEL_REGISTRY.info(model_name)
                except KeyError as exc:
                    # Allow preset names (JSON-ready configs) for industrial workflows.
                    preset = resolve_model_preset(model_name)
                    if preset is None:
                        raise ValueError(
                            f"Unknown model or model preset: {requested_model!r}. "
                            "Use --list-models/--list-model-presets for discovery."
                        ) from exc

                    model_name = str(preset.model)
                    preset_model_auto_kwargs = dict(preset.kwargs)
                    try:
                        MODEL_REGISTRY.info(model_name)
                    except KeyError as exc2:
                        raise ValueError(
                            f"Model preset {requested_model!r} refers to unknown model: {model_name!r}"
                        ) from exc2

            preset_kwargs = _resolve_preset_kwargs(args.preset, model_name)

            device = str(args.device) if args.device is not None else "cpu"
            contamination = float(args.contamination) if args.contamination is not None else 0.1
            # Industrial default: keep direct CLI usage offline-safe. Users can opt in
            # via `--pretrained` (may download weights).
            pretrained = bool(args.pretrained) if args.pretrained is not None else False

            user_kwargs = parse_model_kwargs(args.model_kwargs)
            user_kwargs = merge_checkpoint_path(user_kwargs, checkpoint_path=args.checkpoint_path)
            user_kwargs = _maybe_apply_onnx_session_options_and_sweep(
                args=args,
                model_name=model_name,
                device=str(device),
                user_kwargs=dict(user_kwargs),
                inputs=list(inputs),
                onnx_session_options_cli=onnx_session_options_cli,
            )

            auto_kwargs: dict[str, Any] = dict(preset_model_auto_kwargs)
            auto_kwargs.update(
                {
                    "device": device,
                    "contamination": contamination,
                    "pretrained": pretrained,
                    **(
                        {"random_seed": int(seed), "random_state": int(seed)}
                        if seed is not None
                        else {}
                    ),
                }
            )
            model_kwargs = build_model_kwargs(
                model_name,
                user_kwargs=user_kwargs,
                preset_kwargs=preset_kwargs,
                auto_kwargs=auto_kwargs,
            )

            _enforce_checkpoint_requirement(
                model_name=model_name,
                model_kwargs=dict(model_kwargs),
                trained_checkpoint_path=trained_checkpoint_path,
            )
            detector = create_model(model_name, **model_kwargs)

        if defects_payload is not None:
            _apply_defects_defaults_from_payload(args, defects_payload)

        if (
            args.tile_size is None
            and isinstance(tiling_payload, dict)
            and tiling_payload.get("tile_size", None) is not None
        ):
            args.tile_size = int(tiling_payload.get("tile_size"))
            if tiling_payload.get("stride", None) is not None:
                args.tile_stride = int(tiling_payload.get("stride"))
            if tiling_payload.get("score_reduce", None) is not None:
                args.tile_score_reduce = str(tiling_payload.get("score_reduce"))
            if tiling_payload.get("map_reduce", None) is not None:
                args.tile_map_reduce = str(tiling_payload.get("map_reduce"))
            if tiling_payload.get("score_topk", None) is not None:
                args.tile_score_topk = float(tiling_payload.get("score_topk"))

        if args.tile_size is not None:
            from pyimgano.inference.tiling import TiledDetector

            detector = TiledDetector(
                detector=detector,
                tile_size=int(args.tile_size),
                stride=(int(args.tile_stride) if args.tile_stride is not None else None),
                score_reduce=str(args.tile_score_reduce),
                score_topk=float(args.tile_score_topk),
                map_reduce=str(args.tile_map_reduce),
            )
            # Ensure a threshold loaded from --from-run/--infer-config remains visible after wrapping.
            if threshold_from_run is not None:
                setattr(detector, "threshold_", float(threshold_from_run))

        if args.reference_dir is not None:
            setter = getattr(detector, "set_reference_dir", None)
            if callable(setter):
                setter(str(args.reference_dir))
            else:
                raise ValueError(
                    "--reference-dir is only supported for reference-based detectors "
                    "(detectors implementing set_reference_dir())."
                )

        # Apply preprocessing outside tiling so illumination/contrast normalization happens
        # on the full image before tiling.
        if illumination_contrast_knobs is not None:
            _require_numpy_model_for_preprocessing(model_name)
            from pyimgano.inference.preprocessing import PreprocessingDetector

            detector = PreprocessingDetector(
                detector=detector,
                illumination_contrast=illumination_contrast_knobs,
            )
            if threshold_from_run is not None:
                setattr(detector, "threshold_", float(threshold_from_run))

        t_load_model = time.perf_counter() - t_load_start

        t_fit_start = time.perf_counter()
        train_paths: list[str] = []
        if args.train_dir is not None:
            train_paths = _collect_image_paths(args.train_dir)
            if not train_paths:
                raise ValueError(f"No images found in --train-dir={args.train_dir!r}")
            detector.fit(train_paths)

        if args.calibration_quantile is not None and not train_paths:
            raise ValueError("--calibration-quantile requires --train-dir")

        # Industrial-friendly default: when a train set is provided but the detector does
        # not set a threshold during fit(), auto-calibrate by a quantile on train scores.
        # This aligns `pyimgano-infer` with workbench/benchmark behavior while preserving
        # detectors that already provide `threshold_`.
        if train_paths:
            threshold_before = getattr(detector, "threshold_", None)
            should_calibrate = args.calibration_quantile is not None or threshold_before is None
            if should_calibrate:
                from pyimgano.calibration.score_threshold import resolve_calibration_quantile

                q, _src = resolve_calibration_quantile(
                    detector,
                    calibration_quantile=(
                        float(args.calibration_quantile)
                        if args.calibration_quantile is not None
                        else None
                    ),
                )
                batch_size = int(args.batch_size) if int(args.batch_size) > 0 else None
                calibrate_threshold(
                    detector,
                    train_paths,
                    quantile=float(q),
                    batch_size=batch_size,
                    amp=bool(args.amp),
                )

        if bool(args.defects) and not bool(args.include_maps):
            args.include_maps = True

        postprocess: AnomalyMapPostprocess | None = None
        if bool(args.include_maps):
            if bool(args.postprocess):
                postprocess = AnomalyMapPostprocess()
            elif infer_config_postprocess is not None:
                postprocess = _build_postprocess_from_payload(infer_config_postprocess)

        pixel_threshold_value: float | None = None
        pixel_threshold_provenance: dict[str, Any] | None = None
        if bool(args.defects):
            from pyimgano.defects.pixel_threshold import resolve_pixel_threshold

            infer_cfg_source = defects_payload_source or "infer_config"
            strategy = str(args.pixel_threshold_strategy)

            infer_cfg_thr = None
            if defects_payload is not None:
                raw_thr = defects_payload.get("pixel_threshold", None)
                if raw_thr is not None:
                    infer_cfg_thr = float(raw_thr)

            calibration_maps = None
            infer_cfg_thr_for_resolve = infer_cfg_thr

            if args.pixel_threshold is None and strategy == "normal_pixel_quantile":
                if not train_paths:
                    if infer_cfg_thr is None:
                        raise ValueError(
                            "--defects requires a pixel threshold.\n"
                            "Provide --pixel-threshold, set defects.pixel_threshold in infer_config.json, "
                            "or provide --train-dir for normal-pixel quantile calibration."
                        )
                else:
                    # Prefer re-calibration from normal pixels when train data is available,
                    # even if the infer-config/run contains a pre-set pixel threshold.
                    infer_cfg_thr_for_resolve = None

                    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else None
                    calibration_maps = []
                    for r in infer_iter(
                        detector,
                        train_paths,
                        include_maps=True,
                        postprocess=postprocess,
                        batch_size=batch_size,
                        amp=bool(args.amp),
                    ):
                        if r.anomaly_map is not None:
                            calibration_maps.append(r.anomaly_map)

            pixel_threshold_value, pixel_threshold_provenance = resolve_pixel_threshold(
                pixel_threshold=(
                    float(args.pixel_threshold) if args.pixel_threshold is not None else None
                ),
                pixel_threshold_strategy=str(args.pixel_threshold_strategy),
                infer_config_pixel_threshold=infer_cfg_thr_for_resolve,
                calibration_maps=calibration_maps,
                pixel_normal_quantile=float(args.pixel_normal_quantile),
                infer_config_source=str(infer_cfg_source),
                roi_xyxy_norm=(
                    list(args.roi_xyxy_norm) if args.roi_xyxy_norm is not None else None
                ),
            )

        t_fit_calibrate = time.perf_counter() - t_fit_start

        infer_timing = InferenceTiming()
        batch_size = int(args.batch_size) if int(args.batch_size) > 0 else None
        continue_on_error = bool(getattr(args, "continue_on_error", False))
        max_errors = int(getattr(args, "max_errors", 0))
        flush_every = int(getattr(args, "flush_every", 0))

        results_iter = None
        if not bool(continue_on_error):
            results_iter = infer_iter(
                detector,
                inputs,
                include_maps=bool(args.include_maps),
                postprocess=postprocess,
                batch_size=batch_size,
                amp=bool(args.amp),
                timing=infer_timing,
            )

        maps_dir: Path | None = None
        if args.save_maps is not None:
            if not bool(args.include_maps):
                raise ValueError("--save-maps requires --include-maps")
            maps_dir = Path(args.save_maps)
            maps_dir.mkdir(parents=True, exist_ok=True)

        masks_dir: Path | None = None
        if args.save_masks is not None:
            masks_dir = Path(args.save_masks)
            masks_dir.mkdir(parents=True, exist_ok=True)

        overlays_dir: Path | None = None
        if args.save_overlays is not None:
            overlays_dir = Path(args.save_overlays)
            overlays_dir.mkdir(parents=True, exist_ok=True)

        out_f = None
        if args.save_jsonl is not None:
            out_path = Path(args.save_jsonl)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_f = out_path.open("w", encoding="utf-8")

        regions_f = None
        if args.defects_regions_jsonl is not None:
            if not bool(args.defects):
                raise ValueError("--defects-regions-jsonl requires --defects")
            regions_path = Path(args.defects_regions_jsonl)
            regions_path.parent.mkdir(parents=True, exist_ok=True)
            regions_f = regions_path.open("w", encoding="utf-8")

        try:
            t_loop_start = time.perf_counter()
            out_written = 0
            regions_written = 0
            errors = 0
            processed = 0
            stop_early = False

            def _write_out_record(record: dict[str, Any]) -> None:
                nonlocal out_written
                line = json.dumps(record, sort_keys=True)
                if out_f is not None:
                    out_f.write(line)
                    out_f.write("\n")
                    out_written += 1
                    if int(flush_every) > 0 and (out_written % int(flush_every) == 0):
                        out_f.flush()
                else:
                    print(line)

            def _write_regions_payload(payload: dict[str, Any]) -> None:
                nonlocal regions_written
                if regions_f is None:
                    return
                regions_f.write(json.dumps(payload, sort_keys=True))
                regions_f.write("\n")
                regions_written += 1
                if int(flush_every) > 0 and (regions_written % int(flush_every) == 0):
                    regions_f.flush()

            def _make_error_record(
                *, index: int, input_path: str, exc: Exception, stage: str
            ) -> dict[str, Any]:
                return {
                    "status": "error",
                    "index": int(index),
                    "input": str(input_path),
                    "error": {
                        "type": str(type(exc).__name__),
                        "message": str(exc),
                        "stage": str(stage),
                    },
                }

            def _process_ok_result(
                *, index: int, input_path: str, result: Any, include_status: bool
            ) -> None:
                anomaly_map_path: str | None = None
                if maps_dir is not None:
                    if result.anomaly_map is None:
                        anomaly_map_path = None
                    else:
                        stem = Path(input_path).stem
                        out_path = maps_dir / f"{index:06d}_{stem}.npy"
                        np.save(out_path, np.asarray(result.anomaly_map, dtype=np.float32))
                        anomaly_map_path = str(out_path)

                record = result_to_jsonable(
                    result,
                    anomaly_map_path=anomaly_map_path,
                    include_anomaly_map_values=bool(args.include_anomaly_map_values),
                )
                record["index"] = int(index)
                record["input"] = str(input_path)
                if include_status:
                    record["status"] = "ok"

                saved_defects_mask: np.ndarray | None = None
                if bool(args.defects):
                    if result.anomaly_map is None:
                        raise ValueError(
                            "Defects export requires anomaly maps, but no anomaly_map was returned.\n"
                            "Try a detector that supports get_anomaly_map/predict_anomaly_map, and "
                            "ensure --include-maps (or --defects) is enabled."
                        )
                    if pixel_threshold_value is None or pixel_threshold_provenance is None:
                        raise RuntimeError(
                            "Internal error: pixel threshold was not resolved for --defects."
                        )

                    from pyimgano.defects.extract import extract_defects_from_anomaly_map
                    from pyimgano.defects.io import save_binary_mask

                    defects = extract_defects_from_anomaly_map(
                        np.asarray(result.anomaly_map, dtype=np.float32),
                        pixel_threshold=float(pixel_threshold_value),
                        roi_xyxy_norm=(
                            list(args.roi_xyxy_norm) if args.roi_xyxy_norm is not None else None
                        ),
                        mask_space=str(args.defects_mask_space),
                        border_ignore_px=int(args.defect_border_ignore_px),
                        map_smoothing_method=str(args.defect_map_smoothing),
                        map_smoothing_ksize=int(args.defect_map_smoothing_ksize),
                        map_smoothing_sigma=float(args.defect_map_smoothing_sigma),
                        hysteresis_enabled=bool(args.defect_hysteresis),
                        hysteresis_low=(
                            float(args.defect_hysteresis_low)
                            if args.defect_hysteresis_low is not None
                            else None
                        ),
                        hysteresis_high=(
                            float(args.defect_hysteresis_high)
                            if args.defect_hysteresis_high is not None
                            else None
                        ),
                        open_ksize=int(args.defect_open_ksize),
                        close_ksize=int(args.defect_close_ksize),
                        fill_holes=bool(args.defect_fill_holes),
                        mask_dilate_ksize=int(args.defects_mask_dilate),
                        min_area=int(args.defect_min_area),
                        min_fill_ratio=(
                            float(args.defect_min_fill_ratio)
                            if args.defect_min_fill_ratio is not None
                            else None
                        ),
                        max_aspect_ratio=(
                            float(args.defect_max_aspect_ratio)
                            if args.defect_max_aspect_ratio is not None
                            else None
                        ),
                        min_solidity=(
                            float(args.defect_min_solidity)
                            if args.defect_min_solidity is not None
                            else None
                        ),
                        min_score_max=(
                            float(args.defect_min_score_max)
                            if args.defect_min_score_max is not None
                            else None
                        ),
                        min_score_mean=(
                            float(args.defect_min_score_mean)
                            if args.defect_min_score_mean is not None
                            else None
                        ),
                        merge_nearby_enabled=bool(args.defect_merge_nearby),
                        merge_nearby_max_gap_px=int(args.defect_merge_nearby_max_gap_px),
                        max_regions_sort_by=str(args.defect_max_regions_sort_by),
                        max_regions=(
                            int(args.defect_max_regions)
                            if args.defect_max_regions is not None
                            else None
                        ),
                    )
                    saved_defects_mask = defects["mask"]

                    mask_meta: dict[str, Any] = {
                        "shape": [int(d) for d in defects["mask"].shape],
                        "dtype": str(defects["mask"].dtype),
                    }
                    if masks_dir is not None:
                        stem = Path(input_path).stem
                        fmt = str(args.mask_format)
                        if fmt == "png":
                            ext = ".png"
                        elif fmt == "npy":
                            ext = ".npy"
                        elif fmt == "npz":
                            ext = ".npz"
                        else:  # pragma: no cover - guarded by argparse choices
                            raise ValueError(f"Unknown mask_format: {fmt!r}")
                        out_path = masks_dir / f"{index:06d}_{stem}{ext}"
                        written = save_binary_mask(
                            defects["mask"], out_path, format=str(args.mask_format)
                        )
                        mask_meta.update(
                            {
                                "path": str(written),
                                "encoding": str(args.mask_format),
                            }
                        )
                    else:
                        mask_meta["encoding"] = str(args.mask_format)

                    record["defects"] = {
                        "space": defects["space"],
                        "pixel_threshold": float(pixel_threshold_value),
                        "pixel_threshold_provenance": dict(pixel_threshold_provenance),
                        "mask": mask_meta,
                        "regions": defects["regions"],
                        "map_stats_roi": defects.get("map_stats_roi", None),
                    }

                    _write_regions_payload(
                        {
                            "index": int(index),
                            "input": str(input_path),
                            "defects": {
                                "space": defects["space"],
                                "pixel_threshold": float(pixel_threshold_value),
                                "pixel_threshold_provenance": dict(pixel_threshold_provenance),
                                "regions": defects["regions"],
                                "map_stats_roi": defects.get("map_stats_roi", None),
                            },
                        }
                    )

                    if bool(args.defects_image_space):
                        try:
                            from PIL import Image

                            with Image.open(input_path) as im:
                                w_img, h_img = im.size

                            from pyimgano.defects.space import scale_bbox_xyxy_inclusive

                            src_hw = (int(defects["mask"].shape[0]), int(defects["mask"].shape[1]))
                            dst_hw = (int(h_img), int(w_img))
                            for region in record["defects"]["regions"]:
                                bbox = region.get("bbox_xyxy", None)
                                if bbox is None:
                                    continue
                                region["bbox_xyxy_image"] = scale_bbox_xyxy_inclusive(
                                    bbox,
                                    src_hw=src_hw,
                                    dst_hw=dst_hw,
                                )
                        except Exception:
                            # Best-effort: avoid failing the whole run for a debugging-only knob.
                            pass

                if overlays_dir is not None:
                    from pyimgano.defects.overlays import save_overlay_image

                    stem = Path(input_path).stem
                    out_path = overlays_dir / f"{index:06d}_{stem}.png"
                    save_overlay_image(
                        input_path,
                        anomaly_map=(
                            result.anomaly_map if result.anomaly_map is not None else None
                        ),
                        defect_mask=saved_defects_mask,
                        out_path=out_path,
                    )

                _write_out_record(record)

            if bool(continue_on_error):
                chunk_size = int(batch_size) if batch_size is not None else 1
                for start in range(0, len(inputs), int(chunk_size)):
                    chunk = inputs[start : start + int(chunk_size)]
                    try:
                        chunk_results = list(
                            infer_iter(
                                detector,
                                chunk,
                                include_maps=bool(args.include_maps),
                                postprocess=postprocess,
                                batch_size=None,
                                amp=bool(args.amp),
                                timing=infer_timing,
                            )
                        )
                        for j, r in enumerate(chunk_results):
                            idx = int(start + j)
                            try:
                                _process_ok_result(
                                    index=idx,
                                    input_path=str(chunk[j]),
                                    result=r,
                                    include_status=True,
                                )
                            except Exception as exc:  # noqa: BLE001 - best-effort mode
                                errors += 1
                                _write_out_record(
                                    _make_error_record(
                                        index=idx,
                                        input_path=str(chunk[j]),
                                        exc=exc,
                                        stage="artifacts",
                                    )
                                )
                            processed += 1
                    except Exception:
                        # Fallback: isolate per-input failures (batch failed).
                        for j, p in enumerate(chunk):
                            idx = int(start + j)
                            try:
                                one = list(
                                    infer_iter(
                                        detector,
                                        [p],
                                        include_maps=bool(args.include_maps),
                                        postprocess=postprocess,
                                        batch_size=None,
                                        amp=bool(args.amp),
                                        timing=infer_timing,
                                    )
                                )
                                if len(one) != 1:
                                    raise RuntimeError(
                                        "Internal error: expected 1 result for 1 input"
                                    )
                                _process_ok_result(
                                    index=idx,
                                    input_path=str(p),
                                    result=one[0],
                                    include_status=True,
                                )
                            except Exception as exc:  # noqa: BLE001 - best-effort mode
                                errors += 1
                                _write_out_record(
                                    _make_error_record(
                                        index=idx,
                                        input_path=str(p),
                                        exc=exc,
                                        stage="infer",
                                    )
                                )
                            processed += 1

                            if int(max_errors) > 0 and int(errors) >= int(max_errors):
                                stop_early = True
                                break
                    if bool(stop_early):
                        break
            else:
                if results_iter is None:  # pragma: no cover - defensive
                    raise RuntimeError("Internal error: results_iter was not initialized")

                count = 0
                for i, (input_path, result) in enumerate(zip(inputs, results_iter)):
                    _process_ok_result(
                        index=int(i),
                        input_path=str(input_path),
                        result=result,
                        include_status=False,
                    )
                    count += 1

                if count != len(inputs):
                    raise RuntimeError(
                        "Internal error: inference iterator produced fewer results than inputs "
                        f"({count} vs {len(inputs)})."
                    )

            t_loop = time.perf_counter() - t_loop_start
            t_infer = float(infer_timing.seconds)
            t_artifacts = max(0.0, float(t_loop) - float(t_infer))
        finally:
            if out_f is not None:
                out_f.close()
            if regions_f is not None:
                regions_f.close()

        if bool(args.profile):
            import sys

            total = time.perf_counter() - t_total_start
            print(
                "profile: "
                + " ".join(
                    [
                        f"load_model={t_load_model:.3f}s",
                        f"fit_calibrate={t_fit_calibrate:.3f}s",
                        f"infer={t_infer:.3f}s",
                        f"artifacts={t_artifacts:.3f}s",
                        f"total={total:.3f}s",
                    ]
                ),
                file=sys.stderr,
            )

        if args.profile_json is not None:
            profile_path = Path(str(args.profile_json))
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            total = time.perf_counter() - t_total_start
            profile_payload = {
                "tool": "pyimgano-infer",
                "counts": {
                    "inputs": int(len(inputs)),
                    "processed": int(len(inputs) if not bool(continue_on_error) else processed),
                    "errors": int(errors),
                },
                "timing_seconds": {
                    "load_model": float(t_load_model),
                    "fit_calibrate": float(t_fit_calibrate),
                    "infer": float(t_infer),
                    "artifacts": float(t_artifacts),
                    "total": float(total),
                },
            }
            profile_path.write_text(
                json.dumps(profile_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        if bool(continue_on_error) and int(errors) > 0:
            return 1
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        import sys

        print(f"error: {exc}", file=sys.stderr)
        from_run = getattr(args, "from_run", None)
        if from_run:
            print(f"context: from_run={from_run!r}", file=sys.stderr)
            cat = getattr(args, "from_run_category", None)
            if cat:
                print(f"context: from_run_category={cat!r}", file=sys.stderr)
        if isinstance(exc, ImportError):
            model_name = getattr(args, "model", None)
            if model_name:
                print(f"context: model={model_name!r}", file=sys.stderr)
            model_preset = getattr(args, "model_preset", None)
            if model_preset:
                print(f"context: model_preset={model_preset!r}", file=sys.stderr)
        return 2


def _build_postprocess_from_payload(payload: dict[str, Any]) -> AnomalyMapPostprocess:
    pr_raw = payload.get("percentile_range", (1.0, 99.0))
    if isinstance(pr_raw, (list, tuple)) and len(pr_raw) == 2:
        pr = (float(pr_raw[0]), float(pr_raw[1]))
    else:
        pr = (1.0, 99.0)

    ct = payload.get("component_threshold", None)
    component_threshold = float(ct) if ct is not None else None

    return AnomalyMapPostprocess(
        normalize=bool(payload.get("normalize", True)),
        normalize_method=str(payload.get("normalize_method", "minmax")),
        percentile_range=pr,
        gaussian_sigma=float(payload.get("gaussian_sigma", 0.0)),
        morph_open_ksize=int(payload.get("morph_open_ksize", 0)),
        morph_close_ksize=int(payload.get("morph_close_ksize", 0)),
        component_threshold=component_threshold,
        min_component_area=int(payload.get("min_component_area", 0)),
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
