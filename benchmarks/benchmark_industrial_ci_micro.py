"""
Industrial anomaly detection CI microbenchmark (synthetic images).

Goal
----
Provide a tiny, deterministic benchmark that:
- runs fast on CPU (no dataset downloads)
- produces a compact Markdown table for docs/CI notes
- supports CI-style regression thresholds (via unit tests)

This script is intentionally small and dependency-light.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

# Add repo root to import path (run-friendly).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyimgano.pipelines.run_benchmark import run_benchmark  # noqa: E402


@dataclass(frozen=True)
class _Row:
    model: str
    auroc: float
    ap: float
    fit_s: float
    score_s: float
    total_s: float


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def _make_custom_dataset(
    root: Path,
    *,
    kind: str,
    h: int,
    w: int,
    n_train: int,
    n_test_normal: int,
    n_test_anomaly: int,
    noise_sigma: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(int(seed))

    kind_key = str(kind).strip().lower()
    if kind_key not in {"noise_patch", "template_patch"}:
        raise ValueError(f"Unknown dataset kind: {kind!r}. Choose: noise_patch, template_patch")

    if kind_key == "template_patch":
        # Deterministic "template-style" image: stable pattern + small noise.
        hh, ww = int(h), int(w)
        yy, xx = np.mgrid[0:hh, 0:ww]
        yy = yy.astype(np.float32, copy=False)
        xx = xx.astype(np.float32, copy=False)

        period_x = max(8.0, float(ww) / 4.0)
        period_y = max(8.0, float(hh) / 5.0)
        bg = 128.0
        bg += 25.0 * np.sin(2.0 * np.pi * xx / float(period_x))
        bg += 18.0 * np.cos(2.0 * np.pi * yy / float(period_y))
        bg = np.clip(bg, 0.0, 255.0).astype(np.float32, copy=False)
        base = np.stack([bg, bg, bg], axis=2)

        def _rect(y0: float, y1: float, x0: float, x1: float, *, v: float) -> None:
            ya = int(max(0, min(hh, round(y0 * hh))))
            yb = int(max(0, min(hh, round(y1 * hh))))
            xa = int(max(0, min(ww, round(x0 * ww))))
            xb = int(max(0, min(ww, round(x1 * ww))))
            if yb > ya and xb > xa:
                base[ya:yb, xa:xb, :] = float(v)

        # A few stable blobs/lines to create correspondence.
        _rect(0.12, 0.30, 0.18, 0.34, v=200.0)
        _rect(0.62, 0.82, 0.54, 0.74, v=120.0)
        _rect(0.40, 0.44, 0.00, 1.00, v=80.0)

        cy, cx = hh // 2, max(0, ww // 4)
        r = max(1.0, 0.10 * float(min(hh, ww)))
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r * r)
        base[mask, :] = 160.0

        def _make_normal() -> np.ndarray:
            noise = rng.normal(0.0, float(noise_sigma), size=(hh, ww, 3))
            return np.clip(base + noise, 0.0, 255.0)

        patch = max(4, int(round(0.18 * float(min(hh, ww)))))
        margin = max(1, patch // 2)
        xs = np.linspace(margin, max(margin, ww - patch - margin), num=3).astype(int)
        ys = np.linspace(margin, max(margin, hh - patch - margin), num=3).astype(int)

        for i in range(int(n_train)):
            _write_png(root / "train" / "normal" / f"train_{i}.png", _make_normal())

        for i in range(int(n_test_normal)):
            _write_png(root / "test" / "normal" / f"normal_{i}.png", _make_normal())

        for i in range(int(n_test_anomaly)):
            img = _make_normal()
            x0 = int(xs[int(i) % int(xs.size)])
            y0 = int(ys[(int(i) // int(xs.size)) % int(ys.size)])
            img[y0 : y0 + patch, x0 : x0 + patch, :] = 255.0 - img[y0 : y0 + patch, x0 : x0 + patch, :]
            img = np.clip(img, 0.0, 255.0)
            _write_png(root / "test" / "anomaly" / f"anomaly_{i}.png", img)
        return

    for i in range(int(n_train)):
        noise = rng.normal(0.0, float(noise_sigma), size=(h, w, 3))
        img = np.clip(128.0 + noise, 0, 255)
        _write_png(root / "train" / "normal" / f"train_{i}.png", img)

    for i in range(int(n_test_normal)):
        noise = rng.normal(0.0, float(noise_sigma), size=(h, w, 3))
        img = np.clip(128.0 + noise, 0, 255)
        _write_png(root / "test" / "normal" / f"normal_{i}.png", img)

    for i in range(int(n_test_anomaly)):
        noise = rng.normal(0.0, float(noise_sigma), size=(h, w, 3))
        img = 128.0 + noise
        # Deterministic patch placement: vary position in a small grid.
        x1 = 8 + (i % 4) * 8
        y1 = 8 + (i // 4) * 8
        img[y1 : y1 + 16, x1 : x1 + 16] = 240.0
        img = np.clip(img, 0, 255)
        _write_png(root / "test" / "anomaly" / f"anomaly_{i}.png", img)


def _format_markdown_table(rows: Iterable[_Row]) -> str:
    lines = [
        "| Model | AUROC | AP | fit_s | score_test_s | total_s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.model),
                    f"{float(r.auroc):.4f}",
                    f"{float(r.ap):.4f}",
                    f"{float(r.fit_s):.3f}",
                    f"{float(r.score_s):.3f}",
                    f"{float(r.total_s):.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmark_industrial_ci_micro")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names to benchmark.",
    )
    parser.add_argument(
        "--dataset-kind",
        default="noise_patch",
        choices=["noise_patch", "template_patch"],
        help=(
            "Synthetic dataset kind:\n"
            "- noise_patch: gray+noise normals; anomalies add a bright square (good for structural/pixel-stats baselines)\n"
            "- template_patch: stable pattern+noise; anomalies invert a patch (good for template/SSIM/NCC baselines)"
        ),
    )
    parser.add_argument("--h", type=int, default=64)
    parser.add_argument("--w", type=int, default=64)
    parser.add_argument("--train", type=int, default=16, help="Number of training normals.")
    parser.add_argument("--test-normal", type=int, default=8, help="Number of normal test images.")
    parser.add_argument("--test-anomaly", type=int, default=8, help="Number of anomaly test images.")
    parser.add_argument("--noise-sigma", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0, help="Dataset generation seed.")
    args = parser.parse_args(argv)

    dataset_kind = str(getattr(args, "dataset_kind", "noise_patch"))

    models_str = args.models
    if models_str is None:
        if dataset_kind == "template_patch":
            models_str = ",".join(
                [
                    "vision_ecod",
                    "vision_copod",
                    "vision_structural_ecod",
                    "vision_structural_knn",
                    "vision_structural_pca_md",
                    "vision_pixel_mean_absdiff_map",
                    "vision_pixel_gaussian_map",
                    "vision_pixel_mad_map",
                    "ssim_template_map",
                    "vision_template_ncc_map",
                    "vision_phase_correlation_map",
                ]
            )
        else:
            models_str = ",".join(
                [
                    "vision_ecod",
                    "vision_copod",
                    "vision_structural_ecod",
                    "vision_structural_knn",
                    "vision_structural_pca_md",
                    "vision_pixel_mean_absdiff_map",
                    "vision_pixel_gaussian_map",
                    "vision_pixel_mad_map",
                    "ssim_template_map",
                    "vision_phase_correlation_map",
                ]
            )

    models = [m.strip() for m in str(models_str).split(",") if m.strip()]
    if not models:
        raise ValueError("--models must contain at least one model name.")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "custom"
        _make_custom_dataset(
            root,
            kind=str(dataset_kind),
            h=int(args.h),
            w=int(args.w),
            n_train=int(args.train),
            n_test_normal=int(args.test_normal),
            n_test_anomaly=int(args.test_anomaly),
            noise_sigma=float(args.noise_sigma),
            seed=int(args.seed),
        )

        rows: list[_Row] = []
        for model in models:
            model_kwargs = None
            if model in {
                "vision_pixel_mean_absdiff_map",
                "vision_pixel_gaussian_map",
                "vision_pixel_mad_map",
                "ssim_template_map",
                "vision_template_ncc_map",
                "vision_phase_correlation_map",
            }:
                # Keep microbenchmark fast by matching the synthetic image size.
                model_kwargs = {"resize_hw": (int(args.h), int(args.w))}

            payload = run_benchmark(
                dataset="custom",
                root=str(root),
                category="custom",
                model=str(model),
                seed=None,  # keep this microbenchmark torch-free unless the model needs it
                device="cpu",
                pretrained=False,
                contamination=0.1,
                resize=(int(args.h), int(args.w)),
                model_kwargs=model_kwargs,
                save_run=False,
                per_image_jsonl=False,
            )
            res = payload["results"]
            timing = payload["timing"]
            rows.append(
                _Row(
                    model=str(model),
                    auroc=float(res["auroc"]),
                    ap=float(res["average_precision"]),
                    fit_s=float(timing["fit_s"]),
                    score_s=float(timing["score_test_s"]),
                    total_s=float(timing["total_s"]),
                )
            )

        print(_format_markdown_table(rows))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
