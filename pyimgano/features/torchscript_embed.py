from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from pyimgano.features.base import BaseFeatureExtractor
from pyimgano.features.registry import register_feature_extractor

_InputColor = Literal["rgb", "bgr"]


def _looks_like_torch_tensor(x: Any) -> bool:
    """Best-effort torch.Tensor detection without importing torch."""

    mod = getattr(getattr(x, "__class__", None), "__module__", "")
    if not isinstance(mod, str) or not mod.startswith("torch"):
        return False
    return bool(hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"))


def _as_pil_rgb(item: Any, *, input_color: _InputColor):  # noqa: ANN001, ANN201
    # Lazy import: keep registry discovery light.
    from PIL import Image

    if isinstance(item, (str, Path)):
        img = Image.open(str(item)).convert("RGB")
        return img

    if isinstance(item, Image.Image):
        return item.convert("RGB")

    if isinstance(item, np.ndarray):
        arr = np.asarray(item)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[2] not in (1, 3) and arr.shape[0] in (1, 3):
            # Accept channels-first CHW numpy images (common in torch pipelines).
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim != 3 or arr.shape[2] not in (1, 3):
            raise ValueError(f"Unsupported numpy image shape: {arr.shape}")

        if arr.dtype != np.uint8:
            arr_f = arr.astype(np.float32, copy=False)
            if float(np.nanmax(arr_f)) <= 1.0:
                arr_f = arr_f * 255.0
            arr_f = np.clip(arr_f, 0.0, 255.0)
            arr = arr_f.astype(np.uint8)

        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)

        if input_color == "bgr":
            arr = arr[..., ::-1]  # BGR -> RGB

        return Image.fromarray(arr, mode="RGB")

    # Optional: torch tensor inputs (without importing torch).
    if _looks_like_torch_tensor(item):  # pragma: no cover - depends on torch being installed
        t = item.detach().cpu()
        if t.ndim == 2:
            return _as_pil_rgb(t.numpy(), input_color=input_color)
        if t.ndim == 3:
            # Accept both CHW and HWC (best-effort).
            if int(t.shape[0]) in (1, 3):  # CHW
                arr = t.numpy()
                arr_hwc = np.transpose(arr, (1, 2, 0))
                return _as_pil_rgb(arr_hwc, input_color=input_color)
            if int(t.shape[2]) in (1, 3):  # HWC
                return _as_pil_rgb(t.numpy(), input_color=input_color)
            raise ValueError(
                f"Unsupported torch image shape: {tuple(t.shape)} (expected CHW or HWC)"
            )

        raise ValueError(f"Unsupported torch image ndim={int(t.ndim)} (expected 2 or 3)")

    raise TypeError(
        "TorchscriptEmbedExtractor expects inputs of type str|Path|np.ndarray|PIL.Image|torch.Tensor, "
        f"got {type(item)}"
    )


def _normalize_rgb_chw_f32(
    chw: np.ndarray,
    *,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    if chw.shape[0] != 3:
        raise ValueError(f"Expected CHW with 3 channels, got shape {chw.shape}")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean/std must be sequences of length 3")

    out = np.asarray(chw, dtype=np.float32)
    m = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    s = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    return (out - m) / s


def _make_device(torch, device: str):  # noqa: ANN001, ANN201 - torch is dynamic
    dev = str(device).strip().lower()
    if dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    return torch.device(dev)


def _select_output_tensor(
    output: Any,
    *,
    output_key: str | None,
    output_index: int,
):
    # torchscript models can return Tensor, tuple/list, or dict-like objects.
    if hasattr(output, "detach"):
        return output

    if isinstance(output, (tuple, list)):
        idx = int(output_index)
        if idx < 0 or idx >= len(output):
            raise IndexError(f"output_index out of range: {idx} for output of len {len(output)}")
        return output[idx]

    if isinstance(output, Mapping):
        if output_key is None:
            if len(output) != 1:
                raise ValueError(
                    "Model output is a mapping with multiple keys. "
                    "Provide output_key to select which tensor to use."
                )
            return next(iter(output.values()))
        if output_key not in output:
            raise KeyError(
                f"output_key {output_key!r} not found in model output keys {sorted(output)}"
            )
        return output[output_key]

    raise TypeError(
        "Unsupported model output type. Expected Tensor, tuple/list, or mapping. "
        f"Got {type(output)}"
    )


def _as_2d_embedding(torch, out_tensor):  # noqa: ANN001, ANN201 - torch is dynamic
    del torch
    t = out_tensor
    if not hasattr(t, "ndim"):
        raise TypeError("Selected model output is not a tensor")

    if int(t.ndim) == 0:
        raise ValueError("Model output must have a batch dimension; got scalar tensor")

    if int(t.ndim) == 1:
        t = t.unsqueeze(0)

    if int(t.ndim) == 2:
        return t

    # Common cases:
    # - (B, C, H, W): conv features -> pool spatial
    # - (B, N, C): token/features -> pool over tokens
    if int(t.ndim) == 3:
        return t.mean(dim=1)
    if int(t.ndim) == 4:
        return t.mean(dim=(2, 3))

    b = int(t.shape[0])
    return t.reshape(b, -1)


@dataclass
@register_feature_extractor(
    "torchscript_embed",
    tags=("embeddings", "torch", "torchscript", "deep-features"),
    metadata={
        "description": "TorchScript model embeddings (checkpoint path required; no downloads)",
    },
)
class TorchscriptEmbedExtractor(BaseFeatureExtractor):
    """Extract embeddings from a TorchScript checkpoint.

    Industrial intent
    -----------------
    - Stable deployment path: export a model to TorchScript once, then reuse it
      for embedding extraction without relying on upstream model registries.
    - Offline-safe: **no implicit weight downloads**.
    """

    checkpoint: str | None = None
    checkpoint_path: str | None = None
    device: str = "cpu"
    batch_size: int = 16
    input_color: _InputColor = "rgb"

    image_size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    output_key: str | None = None
    output_index: int = 0

    # Optional disk cache for path inputs (embedding rows keyed by path+mtime+extractor config).
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        self._model = None
        self._device = None
        self._torch = None
        self._cache = None
        self.last_cache_stats_ = {"hits": 0, "misses": 0, "enabled": False}

        cp = None
        if self.checkpoint is not None and str(self.checkpoint).strip():
            cp = str(self.checkpoint)
        if self.checkpoint_path is not None and str(self.checkpoint_path).strip():
            if cp is None:
                cp = str(self.checkpoint_path)
            elif str(self.checkpoint_path) != cp:
                raise ValueError("checkpoint and checkpoint_path must match when both are provided")
        if cp is None:
            raise ValueError(
                "checkpoint is required for torchscript_embed (provide checkpoint or checkpoint_path)"
            )

        # Canonicalize to keep serialization/caching stable.
        self.checkpoint = str(cp)
        self.checkpoint_path = str(cp)

        if int(self.image_size) <= 0:
            raise ValueError(f"image_size must be > 0, got {self.image_size}")
        if int(self.batch_size) <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")

        ic = str(self.input_color).strip().lower()
        if ic not in ("rgb", "bgr"):
            raise ValueError("input_color must be one of: rgb|bgr")
        self.input_color = ic  # type: ignore[assignment]

        if self.cache_dir is not None:
            from pyimgano.cache.embeddings import EmbeddingCache, fingerprint_payload

            payload = {
                "type": "torchscript_embed",
                "checkpoint_path": str(self.checkpoint_path),
                "image_size": int(self.image_size),
                "mean": tuple(float(x) for x in self.mean),
                "std": tuple(float(x) for x in self.std),
                "input_color": str(self.input_color),
                "output_key": None if self.output_key is None else str(self.output_key),
                "output_index": int(self.output_index),
            }
            fp = fingerprint_payload(payload)
            self._cache = EmbeddingCache(
                cache_dir=Path(str(self.cache_dir)), extractor_fingerprint=fp
            )

    def _ensure_ready(self) -> None:
        if self._model is not None:
            return

        from pyimgano.utils.optional_deps import require

        torch = require("torch", extra="torch", purpose="TorchscriptEmbedExtractor")

        ckpt = Path(str(self.checkpoint)).expanduser()
        if not ckpt.exists():
            raise FileNotFoundError(f"TorchScript checkpoint not found: {ckpt}")
        dev = _make_device(torch, str(self.device))

        model = torch.jit.load(str(ckpt), map_location=dev)
        # `.eval()` exists for ScriptModule.
        model.eval()
        self._model = model
        self._device = dev
        self._torch = torch

    def _preprocess_one(self, item: Any) -> np.ndarray:
        from PIL import Image

        img = _as_pil_rgb(item, input_color=self.input_color)
        # PIL resize is available without torchvision.
        img = img.resize((int(self.image_size), int(self.image_size)), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected RGB image array, got shape {arr.shape}")
        # (H,W,3) uint8 -> (3,H,W) float32 in [0,1]
        chw = np.transpose(arr, (2, 0, 1)).astype(np.float32) / 255.0
        chw = _normalize_rgb_chw_f32(chw, mean=self.mean, std=self.std)
        return np.asarray(chw, dtype=np.float32)

    def _extract_no_cache(self, items: list[Any]) -> np.ndarray:
        self._ensure_ready()
        torch = self._torch
        model = self._model
        dev = self._device
        if torch is None or model is None or dev is None:
            raise RuntimeError("Extractor not initialized; this is a bug.")

        bs = int(self.batch_size)
        rows: list[np.ndarray] = []
        for i in range(0, len(items), bs):
            batch_items = items[i : i + bs]
            batch_np = np.stack(
                [self._preprocess_one(it) for it in batch_items], axis=0
            )  # (B,3,H,W)
            batch = torch.from_numpy(batch_np).to(dev)

            with torch.no_grad():
                out = model(batch)

            t = _select_output_tensor(
                out, output_key=self.output_key, output_index=int(self.output_index)
            )
            t2 = _as_2d_embedding(torch, t)
            emb = t2.detach().to("cpu").numpy()
            rows.append(np.asarray(emb, dtype=np.float64))

        out = np.concatenate(rows, axis=0)
        if out.shape[0] != len(items):
            raise ValueError("Extractor must return one row per input")
        return np.asarray(out, dtype=np.float64)

    def extract(self, inputs: Iterable[Any]) -> np.ndarray:
        items = list(inputs)
        if not items:
            return np.zeros((0, 1), dtype=np.float64)

        if self._cache is not None and all(isinstance(it, (str, Path)) for it in items):
            paths = [str(p) for p in items]
            cached_rows: dict[str, np.ndarray] = {}
            missing: list[str] = []
            for p in paths:
                row = self._cache.load(p)
                if row is None:
                    missing.append(p)
                else:
                    cached_rows[p] = np.asarray(row)

            self.last_cache_stats_ = {
                "hits": int(len(cached_rows)),
                "misses": int(len(missing)),
                "enabled": True,
            }

            if not missing:
                return np.stack(
                    [np.asarray(cached_rows[p], dtype=np.float64).reshape(-1) for p in paths]
                )

            computed = self._extract_no_cache(list(missing))
            if computed.shape[0] != len(missing):
                raise RuntimeError("Internal error: computed embeddings batch mismatch")
            # zip(strict=...) is Python 3.10+. We already validated lengths above.
            for p, row in zip(missing, computed):
                self._cache.save(p, np.asarray(row, dtype=np.float64).reshape(-1))
                cached_rows[p] = np.asarray(row, dtype=np.float64).reshape(-1)

            return np.stack(
                [np.asarray(cached_rows[p], dtype=np.float64).reshape(-1) for p in paths]
            )

        self.last_cache_stats_ = {"hits": 0, "misses": 0, "enabled": False}
        return self._extract_no_cache(items)


__all__ = ["TorchscriptEmbedExtractor"]
