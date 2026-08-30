from __future__ import annotations

"""Official-weight GPU end-to-end smoke for PatchCore and OpenCLIP."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyimgano.models import create_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run-official-gpu-e2e")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Acknowledge that official model weights may be downloaded.",
    )
    return parser


def _images() -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    train = [np.full((224, 224, 3), value, dtype=np.uint8) for value in (78, 80, 82, 84)]
    normal = np.full((224, 224, 3), 81, dtype=np.uint8)
    defect = normal.copy()
    defect[72:152, 80:160, :] = 245
    return train, normal, defect


def _validate_result(name: str, scores: np.ndarray, maps: np.ndarray) -> dict[str, Any]:
    if scores.shape != (2,) or not np.all(np.isfinite(scores)):
        raise RuntimeError(f"{name} returned invalid scores: shape={scores.shape}, values={scores}")
    if maps.shape != (2, 224, 224) or not np.all(np.isfinite(maps)):
        raise RuntimeError(f"{name} returned invalid maps: shape={maps.shape}")
    if float(scores[1]) <= float(scores[0]):
        raise RuntimeError(f"{name} score direction failed: normal={scores[0]}, defect={scores[1]}")
    return {
        "normal_score": float(scores[0]),
        "defect_score": float(scores[1]),
        "map_shape": list(maps.shape),
        "finite": True,
        "score_direction": "pass",
    }


def _predict_maps(detector: Any, inputs: list[np.ndarray]) -> np.ndarray:
    predict_many = getattr(detector, "predict_anomaly_map", None)
    if callable(predict_many):
        return np.asarray(predict_many(inputs), dtype=np.float32)

    predict_one = getattr(detector, "get_anomaly_map", None)
    if callable(predict_one):
        return np.stack(
            [np.asarray(predict_one(image), dtype=np.float32) for image in inputs],
            axis=0,
        )
    raise TypeError(
        f"{type(detector).__name__} exposes neither predict_anomaly_map nor get_anomaly_map."
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.allow_download:
        raise ValueError(
            "Official-weight E2E may download model artifacts. Pass --allow-download "
            "after verifying the source and accepting its terms."
        )

    import torch

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    train, normal, defect = _images()
    inputs = [normal, defect]
    report: dict[str, Any] = {
        "device": str(args.device),
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "torch": str(torch.__version__),
    }

    openclip = create_model(
        "vision_openclip_promptscore",
        class_name="industrial object",
        openclip_model_name="ViT-B-32",
        openclip_pretrained="laion2b_s34b_b79k",
        allow_download=True,
        device=str(args.device),
        contamination=0.1,
    )
    openclip.fit(train)
    report["openclip"] = _validate_result(
        "OpenCLIP",
        np.asarray(openclip.decision_function(inputs), dtype=np.float64),
        _predict_maps(openclip, inputs),
    )

    patchcore = create_model(
        "vision_patchcore",
        backbone="wide_resnet50_2",
        pretrained=True,
        device=str(args.device),
        contamination=0.1,
        coreset_sampling_ratio=0.01,
        random_seed=42,
        n_neighbors=1,
        knn_backend="sklearn",
    )
    patchcore.fit(train)
    patchcore_result = _validate_result(
        "PatchCore",
        np.asarray(patchcore.decision_function(inputs), dtype=np.float64),
        _predict_maps(patchcore, inputs),
    )
    memory_bank = getattr(patchcore, "memory_bank", None)
    if memory_bank is None:
        memory_bank = getattr(patchcore, "memory_bank_", None)
    if memory_bank is None:
        memory_bank = getattr(patchcore, "_memory_bank", None)
    if memory_bank is not None:
        patchcore_result["memory_bank_shape"] = list(np.asarray(memory_bank).shape)
    report["patchcore"] = patchcore_result

    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
