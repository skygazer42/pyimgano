from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np


def apply_onnx_session_options_shorthand(
    *,
    model_name: str,
    user_kwargs: dict[str, Any],
    session_options: dict[str, Any] | None,
) -> dict[str, Any]:
    import pyimgano.services.model_options as model_options

    return model_options.apply_onnx_session_options_shorthand(
        model_name=model_name,
        user_kwargs=user_kwargs,
        session_options=session_options,
    )


def default_onnx_sweep_intra_values() -> list[int]:
    n = int(os.cpu_count() or 8)
    cap = max(1, min(n, 16))
    vals = [1, 2, 4, 8, 16]
    out = [v for v in vals if v <= cap]
    return out or [1]


def run_onnx_session_options_sweep(
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
            session_options = dict(base_session_options)
            session_options["intra_op_num_threads"] = int(intra)
            session_options["graph_optimization_level"] = str(opt)

            row: dict[str, Any] = {"session_options": dict(session_options)}
            try:
                extractor = ONNXEmbedExtractor(
                    checkpoint_path=str(checkpoint_path),
                    device="cpu",
                    batch_size=int(batch_size),
                    image_size=int(image_size),
                    session_options=dict(session_options),
                )
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

                stable = first_out is not None and bool(np.all(np.isfinite(first_out)))
                if stable and first_out is not None and second_out is not None:
                    stable = bool(np.allclose(first_out, second_out, rtol=1e-5, atol=1e-6))

                row.update(
                    {
                        "ok": True,
                        "stable": bool(stable),
                        "timing_seconds": list(timings),
                        "median_seconds": float(statistics.median(timings)) if timings else None,
                        "stdev_seconds": (
                            float(statistics.pstdev(timings)) if len(timings) >= 2 else 0.0
                        ),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - CLI boundary
                row.update(
                    {
                        "ok": False,
                        "stable": False,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                )

            candidates.append(row)
            if bool(row.get("ok")) and bool(row.get("stable")):
                if best is None:
                    best = dict(row)
                else:
                    row_median = float(row.get("median_seconds") or 1e99)
                    best_median = float(best.get("median_seconds") or 1e99)
                    if row_median < best_median:
                        best = dict(row)
                    elif row_median == best_median:
                        row_stdev = float(row.get("stdev_seconds") or 1e99)
                        best_stdev = float(best.get("stdev_seconds") or 1e99)
                        if row_stdev < best_stdev:
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

    return dict(best.get("session_options") or {}), payload


def extract_onnx_checkpoint_path_for_sweep(user_kwargs: dict[str, Any]) -> str | None:
    checkpoint_path = user_kwargs.get("checkpoint_path", None)
    if checkpoint_path is not None and str(checkpoint_path).strip():
        return str(checkpoint_path)

    embedding_kwargs = user_kwargs.get("embedding_kwargs", None)
    if isinstance(embedding_kwargs, dict):
        checkpoint_path = embedding_kwargs.get("checkpoint_path", None)
        if checkpoint_path is not None and str(checkpoint_path).strip():
            return str(checkpoint_path)

    feature_extractor = user_kwargs.get("feature_extractor", None)
    if isinstance(feature_extractor, dict) and str(feature_extractor.get("name", "")).strip() == "onnx_embed":
        kwargs = feature_extractor.get("kwargs", None)
        if isinstance(kwargs, dict):
            checkpoint_path = kwargs.get("checkpoint_path", None)
            if checkpoint_path is not None and str(checkpoint_path).strip():
                return str(checkpoint_path)

    return None


def extract_session_options_for_sweep(user_kwargs: dict[str, Any]) -> dict[str, Any]:
    session_options = user_kwargs.get("session_options", None)
    if isinstance(session_options, dict):
        return dict(session_options)

    embedding_kwargs = user_kwargs.get("embedding_kwargs", None)
    if isinstance(embedding_kwargs, dict):
        session_options = embedding_kwargs.get("session_options", None)
        if isinstance(session_options, dict):
            return dict(session_options)

    feature_extractor = user_kwargs.get("feature_extractor", None)
    if isinstance(feature_extractor, dict) and str(feature_extractor.get("name", "")).strip() == "onnx_embed":
        kwargs = feature_extractor.get("kwargs", None)
        if isinstance(kwargs, dict):
            session_options = kwargs.get("session_options", None)
            if isinstance(session_options, dict):
                return dict(session_options)
    return {}


def maybe_apply_onnx_session_options_and_sweep(
    *,
    args: argparse.Namespace,
    model_name: str,
    device: str,
    user_kwargs: dict[str, Any],
    inputs: list[str],
    onnx_session_options_cli: dict[str, Any] | None,
    parse_csv_ints_arg=None,
    parse_csv_strs_arg=None,
) -> dict[str, Any]:
    if parse_csv_ints_arg is None or parse_csv_strs_arg is None:
        from pyimgano.infer_cli_inputs import parse_csv_ints_arg as default_parse_csv_ints_arg
        from pyimgano.infer_cli_inputs import parse_csv_strs_arg as default_parse_csv_strs_arg

        if parse_csv_ints_arg is None:
            parse_csv_ints_arg = default_parse_csv_ints_arg
        if parse_csv_strs_arg is None:
            parse_csv_strs_arg = default_parse_csv_strs_arg

    if not bool(getattr(args, "onnx_sweep", False)) and not onnx_session_options_cli:
        return dict(user_kwargs)

    if bool(getattr(args, "onnx_sweep", False)):
        sweep_inputs = inputs[: int(getattr(args, "onnx_sweep_samples", 32) or 32)]
        intra_values = (
            parse_csv_ints_arg(str(args.onnx_sweep_intra), arg_name="--onnx-sweep-intra")
            if getattr(args, "onnx_sweep_intra", None)
            else default_onnx_sweep_intra_values()
        )
        opt_levels = (
            parse_csv_strs_arg(
                str(args.onnx_sweep_opt_levels), arg_name="--onnx-sweep-opt-levels"
            )
            if getattr(args, "onnx_sweep_opt_levels", None)
            else ["all", "extended"]
        )

        base_session_options = extract_session_options_for_sweep(user_kwargs)
        if onnx_session_options_cli:
            base_session_options.update(dict(onnx_session_options_cli))

        checkpoint_path = extract_onnx_checkpoint_path_for_sweep(user_kwargs)
        if checkpoint_path is None:
            raise ValueError(
                "--onnx-sweep requires checkpoint_path for ONNX models. "
                "Provide --checkpoint-path (or set checkpoint_path in --model-kwargs)."
            )

        best_session_options, sweep_payload = run_onnx_session_options_sweep(
            checkpoint_path=str(checkpoint_path),
            device=str(device),
            image_size=int(user_kwargs.get("image_size", 224)),
            batch_size=int(user_kwargs.get("batch_size", 16)),
            inputs=list(sweep_inputs),
            base_session_options=dict(base_session_options),
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

        return apply_onnx_session_options_shorthand(
            model_name=model_name,
            user_kwargs=dict(user_kwargs),
            session_options=dict(best_session_options),
        )

    return apply_onnx_session_options_shorthand(
        model_name=model_name,
        user_kwargs=dict(user_kwargs),
        session_options=dict(onnx_session_options_cli or {}),
    )


__all__ = [
    "apply_onnx_session_options_shorthand",
    "default_onnx_sweep_intra_values",
    "extract_onnx_checkpoint_path_for_sweep",
    "extract_session_options_for_sweep",
    "maybe_apply_onnx_session_options_and_sweep",
    "run_onnx_session_options_sweep",
]
