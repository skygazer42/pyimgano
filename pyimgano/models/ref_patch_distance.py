# -*- coding: utf-8 -*-
"""Reference-based patch distance anomaly map (query vs golden reference).

This detector compares a query image against a per-image "golden reference"
image from `reference_dir` (matched by basename by default).

Algorithm (v1, simple + industrial-friendly):
1) Extract a conv feature map from a torchvision backbone at `node` (default: layer4).
2) Compute a per-spatial-location distance between query and reference features.
3) Upsample the distance map back to the original image resolution.
4) Reduce the map to an image-level score (max/mean/topk_mean).

This is intentionally dependency-stable:
- uses only torch/torchvision (already optional in this repo)
- no implicit weight downloads by default (`pretrained=False`)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pyimgano.pipelines.reference_map_pipeline import ReferenceMapPipeline

from .registry import register_model


_Metric = Literal["l2", "cosine"]


def _as_rgb_u8_hwc(path: str) -> np.ndarray:
    from PIL import Image

    img = Image.open(str(path)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


@dataclass
@register_model(
    "vision_ref_patch_distance_map",
    tags=("vision", "deep", "torch", "torchvision", "pixel_map", "reference"),
    metadata={
        "description": "Reference-based patch distance anomaly map (torchvision feature map)",
        "input": "paths",
    },
)
class VisionRefPatchDistanceMapDetector(ReferenceMapPipeline):
    """Query-vs-reference patch distance anomaly maps."""

    # Base (ReferenceMapPipeline) fields
    contamination: float = 0.1
    reference_dir: str | Path | None = None
    match_mode: Literal["basename"] = "basename"
    reduction: Literal["max", "mean", "topk_mean"] = "max"
    topk: float = 0.1

    # Feature extractor config
    backbone: str = "resnet18"
    pretrained: bool = False
    node: str = "layer4"
    image_size: int = 224
    device: str = "cpu"

    # Distance config
    metric: _Metric = "l2"
    eps: float = 1e-12

    # Optional tiling (for high-res)
    tile_size: int | None = None
    tile_stride: int | None = None
    tile_map_reduce: str = "max"  # max|mean|hann|gaussian (see inference.tiling)
    tile_pad_mode: str = "reflect"
    tile_pad_value: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()

        self.backbone = str(self.backbone)
        self.node = str(self.node)
        self.image_size = int(self.image_size)
        self.device = str(self.device)
        self.metric = str(self.metric).strip().lower()  # type: ignore[assignment]
        if self.metric not in ("l2", "cosine"):
            raise ValueError("metric must be one of: l2|cosine")

        if self.image_size <= 0:
            raise ValueError("image_size must be positive")

        if self.tile_size is not None:
            ts = int(self.tile_size)
            if ts <= 0:
                raise ValueError("tile_size must be positive")
            self.tile_size = ts
            if self.tile_stride is not None:
                st = int(self.tile_stride)
                if st <= 0:
                    raise ValueError("tile_stride must be positive")
                self.tile_stride = st

        # Lazy init: don't import torch/torchvision at registry import time.
        self._model = None
        self._transform = None
        self._device_obj = None
        self._torch = None
        self._F = None

    # ------------------------------------------------------------------
    def _ensure_ready(self) -> None:
        if self._model is not None:
            return

        import torch
        import torch.nn.functional as F
        import torchvision.transforms as T
        from torchvision.models.feature_extraction import create_feature_extractor

        from pyimgano.utils.torchvision_safe import load_torchvision_backbone

        model, weight_transform = load_torchvision_backbone(str(self.backbone), pretrained=bool(self.pretrained))
        # Extract a conv feature map at `node`.
        model = create_feature_extractor(model, return_nodes={str(self.node): "feat"})

        dev = torch.device(str(self.device))
        model.to(dev)
        model.eval()

        if weight_transform is not None:
            transform = weight_transform
        else:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            transform = T.Compose(
                [
                    T.Resize((int(self.image_size), int(self.image_size))),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                ]
            )

        self._model = model
        self._transform = transform
        self._device_obj = dev
        self._torch = torch
        self._F = F

    def _extract_feat_map(self, rgb_u8_hwc: np.ndarray):  # noqa: ANN001, ANN201 - torch tensor
        self._ensure_ready()
        assert self._torch is not None
        assert self._model is not None
        assert self._transform is not None
        assert self._device_obj is not None

        from PIL import Image

        pil = Image.fromarray(np.asarray(rgb_u8_hwc, dtype=np.uint8), mode="RGB")
        x = self._transform(pil)
        x = x.unsqueeze(0).to(self._device_obj)
        with self._torch.inference_mode():
            out = self._model(x)
        feat = out["feat"]
        return feat

    def _distance_map(self, q_feat, r_feat, *, out_hw: tuple[int, int]):  # noqa: ANN001, ANN201
        assert self._torch is not None
        assert self._F is not None

        q = q_feat
        r = r_feat
        if q.ndim != 4 or r.ndim != 4:
            raise ValueError("feature maps must be 4D (N,C,H,W)")
        if q.shape != r.shape:
            raise ValueError(f"query/reference feature map shape mismatch: {tuple(q.shape)} vs {tuple(r.shape)}")

        if self.metric == "l2":
            diff = q - r
            # Mean squared distance across channels -> sqrt for l2-ish units.
            dist = (diff * diff).mean(dim=1, keepdim=True).sqrt()
        else:
            # Cosine distance on channel vectors at each spatial location.
            eps = float(self.eps)
            qn = q / (q.norm(dim=1, keepdim=True) + eps)
            rn = r / (r.norm(dim=1, keepdim=True) + eps)
            cos = (qn * rn).sum(dim=1, keepdim=True)
            dist = 1.0 - cos

        dist = dist.to(dtype=self._torch.float32)
        up = self._F.interpolate(dist, size=(int(out_hw[0]), int(out_hw[1])), mode="bilinear", align_corners=False)
        return up[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    def _compute_pair_map(self, *, query_rgb: np.ndarray, reference_rgb: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
        q_feat = self._extract_feat_map(query_rgb)
        r_feat = self._extract_feat_map(reference_rgb)
        return self._distance_map(q_feat, r_feat, out_hw=out_hw)

    def _compute_anomaly_map(self, *, query_path: str, reference_path: str) -> np.ndarray:
        q_rgb = _as_rgb_u8_hwc(str(query_path))
        r_rgb = _as_rgb_u8_hwc(str(reference_path))

        if q_rgb.shape[:2] != r_rgb.shape[:2]:
            raise ValueError(
                "Query/reference image size mismatch. "
                f"query_shape={tuple(q_rgb.shape)} reference_shape={tuple(r_rgb.shape)} "
                f"query_path={query_path!r} reference_path={reference_path!r}"
            )

        h, w = int(q_rgb.shape[0]), int(q_rgb.shape[1])

        if self.tile_size is None:
            return self._compute_pair_map(query_rgb=q_rgb, reference_rgb=r_rgb, out_hw=(h, w))

        # Tile both query and reference in sync and stitch maps.
        from pyimgano.inference.tiling import extract_tile, iter_tile_coords, stitch_maps

        tile = int(self.tile_size)
        stride = int(self.tile_stride) if self.tile_stride is not None else tile

        coords = iter_tile_coords(h, w, tile_size=tile, stride=stride)
        tiles = []
        maps = []
        for y0, x0 in coords:
            q_tile, t = extract_tile(
                q_rgb,
                y0=int(y0),
                x0=int(x0),
                tile_size=tile,
                pad_mode=str(self.tile_pad_mode),
                pad_value=int(self.tile_pad_value),
            )
            r_tile, _t2 = extract_tile(
                r_rgb,
                y0=int(y0),
                x0=int(x0),
                tile_size=tile,
                pad_mode=str(self.tile_pad_mode),
                pad_value=int(self.tile_pad_value),
            )
            tiles.append(t)
            maps.append(self._compute_pair_map(query_rgb=q_tile, reference_rgb=r_tile, out_hw=(tile, tile)))

        full = stitch_maps(tiles, maps, out_shape=(h, w), reduce=str(self.tile_map_reduce))
        return np.asarray(full, dtype=np.float32)


__all__ = ["VisionRefPatchDistanceMapDetector"]

