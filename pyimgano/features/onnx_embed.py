from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence

import numpy as np

from pyimgano.features.base import BaseFeatureExtractor
from pyimgano.features.registry import register_feature_extractor
from pyimgano.utils.optional_deps import require

from .torchvision_backbone import _as_pil_rgb

_InputColor = Literal["rgb", "bgr"]


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


def _as_2d_embedding(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 0:
        raise ValueError("Model output must have a batch dimension; got scalar")
    if x.ndim == 1:
        return x.reshape(1, -1)
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        # (B, N, C) -> mean over tokens
        return np.mean(x, axis=1)
    if x.ndim == 4:
        # (B, C, H, W) -> global average pool
        return np.mean(x, axis=(2, 3))
    b = int(x.shape[0])
    return x.reshape(b, -1)


def _default_providers_for_device(ort, device: str) -> list[str]:  # noqa: ANN001 - ort is dynamic
    dev = str(device).strip().lower()
    available = set(str(p) for p in ort.get_available_providers())
    if dev == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "device='cuda' requested but onnxruntime CUDAExecutionProvider is not available. "
                f"Available providers: {sorted(available)}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


@dataclass
@register_feature_extractor(
    "onnx_embed",
    tags=("embeddings", "onnx", "onnxruntime", "deep-features"),
    metadata={
        "description": "ONNX Runtime embedding extractor (loads a .onnx backbone and returns global embeddings).",
    },
)
class ONNXEmbedExtractor(BaseFeatureExtractor):
    """Extract global image embeddings using onnxruntime.

    This is intended as a deploy-friendly alternative to `torchvision_backbone`
    and `torchscript_embed` for industrial workflows.
    """

    checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None
    device: str = "cpu"
    batch_size: int = 16
    image_size: int = 224
    input_color: _InputColor = "rgb"
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    input_name: Optional[str] = None
    output_name: Optional[str] = None
    output_index: int = 0
    providers: Optional[Sequence[str]] = None
    session_options: Optional[Mapping[str, Any]] = None
    cache_dir: Optional[str] = None

    def __post_init__(self) -> None:
        self._sess = None
        self._ort = None
        self._input_name = None
        self._output_name = None
        self.last_cache_stats_ = None
        self._cache = None

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
                "checkpoint is required for onnx_embed (provide checkpoint or checkpoint_path)"
            )

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

        # Optional: onnxruntime SessionOptions knobs (production tuning).
        if self.session_options is not None:
            if not isinstance(self.session_options, Mapping):
                raise TypeError("session_options must be a mapping/dict when provided.")

            allowed = {
                "intra_op_num_threads",
                "inter_op_num_threads",
                "execution_mode",
                "graph_optimization_level",
                "enable_mem_pattern",
                "enable_cpu_mem_arena",
                "log_severity_level",
                "log_verbosity_level",
                "session_config_entries",
            }
            opts = dict(self.session_options)
            unknown = sorted(set(opts) - set(allowed))
            if unknown:
                raise ValueError(
                    "Unknown session_options key(s): "
                    f"{unknown}. Allowed: {sorted(allowed)}"
                )

            v = opts.get("intra_op_num_threads", None)
            if v is not None:
                vv = int(v)
                if vv < 0:
                    raise ValueError("session_options.intra_op_num_threads must be >= 0")

            v = opts.get("inter_op_num_threads", None)
            if v is not None:
                vv = int(v)
                if vv < 0:
                    raise ValueError("session_options.inter_op_num_threads must be >= 0")

            v = opts.get("execution_mode", None)
            if v is not None:
                mode = str(v).strip().lower()
                if mode not in ("sequential", "parallel"):
                    raise ValueError(
                        "session_options.execution_mode must be 'sequential' or 'parallel'"
                    )

            v = opts.get("graph_optimization_level", None)
            if v is not None:
                lvl = str(v).strip().lower()
                if lvl not in ("disable", "basic", "extended", "all"):
                    raise ValueError(
                        "session_options.graph_optimization_level must be one of: "
                        "disable|basic|extended|all"
                    )

            v = opts.get("enable_mem_pattern", None)
            if v is not None and not isinstance(v, bool):
                raise ValueError("session_options.enable_mem_pattern must be a boolean")

            v = opts.get("enable_cpu_mem_arena", None)
            if v is not None and not isinstance(v, bool):
                raise ValueError("session_options.enable_cpu_mem_arena must be a boolean")

            v = opts.get("log_severity_level", None)
            if v is not None:
                _ = int(v)

            v = opts.get("log_verbosity_level", None)
            if v is not None:
                _ = int(v)

            v = opts.get("session_config_entries", None)
            if v is not None and not isinstance(v, Mapping):
                raise ValueError("session_options.session_config_entries must be a mapping/dict")

        if self.cache_dir is not None:
            from pyimgano.cache.embeddings import EmbeddingCache, fingerprint_payload

            payload = {
                "type": "onnx_embed",
                "checkpoint_path": str(self.checkpoint_path),
                "image_size": int(self.image_size),
                "mean": tuple(float(x) for x in self.mean),
                "std": tuple(float(x) for x in self.std),
                "input_color": str(self.input_color),
                "input_name": None if self.input_name is None else str(self.input_name),
                "output_name": None if self.output_name is None else str(self.output_name),
                "output_index": int(self.output_index),
                "providers": None if self.providers is None else list(self.providers),
                "session_options": (
                    None if self.session_options is None else dict(self.session_options)
                ),
            }
            fp = fingerprint_payload(payload)
            self._cache = EmbeddingCache(
                cache_dir=Path(str(self.cache_dir)), extractor_fingerprint=fp
            )

    def _ensure_ready(self) -> None:
        if self._sess is not None:
            return

        ort = require("onnxruntime", extra="onnx", purpose="ONNXEmbedExtractor")

        ckpt = Path(str(self.checkpoint)).expanduser()
        if not ckpt.exists():
            raise FileNotFoundError(f"ONNX checkpoint not found: {ckpt}")

        if self.providers is None:
            providers = _default_providers_for_device(ort, str(self.device))
        else:
            providers = [str(p) for p in self.providers]

        sess_options = None
        if self.session_options is not None:
            opts = dict(self.session_options)
            sess_options = ort.SessionOptions()

            v = opts.get("intra_op_num_threads", None)
            if v is not None:
                sess_options.intra_op_num_threads = int(v)

            v = opts.get("inter_op_num_threads", None)
            if v is not None:
                sess_options.inter_op_num_threads = int(v)

            v = opts.get("execution_mode", None)
            if v is not None:
                mode = str(v).strip().lower()
                enum = getattr(ort, "ExecutionMode", None)
                if enum is None:
                    raise ValueError("onnxruntime.ExecutionMode is not available in this version.")
                if mode == "sequential":
                    sess_options.execution_mode = getattr(enum, "ORT_SEQUENTIAL")
                elif mode == "parallel":
                    sess_options.execution_mode = getattr(enum, "ORT_PARALLEL")
                else:  # pragma: no cover - validated in __post_init__
                    raise ValueError(f"Unknown execution_mode: {mode!r}")

            v = opts.get("graph_optimization_level", None)
            if v is not None:
                lvl = str(v).strip().lower()
                enum = getattr(ort, "GraphOptimizationLevel", None)
                if enum is None:
                    raise ValueError(
                        "onnxruntime.GraphOptimizationLevel is not available in this version."
                    )
                if lvl == "disable":
                    sess_options.graph_optimization_level = getattr(enum, "ORT_DISABLE_ALL")
                elif lvl == "basic":
                    sess_options.graph_optimization_level = getattr(enum, "ORT_ENABLE_BASIC")
                elif lvl == "extended":
                    sess_options.graph_optimization_level = getattr(enum, "ORT_ENABLE_EXTENDED")
                elif lvl == "all":
                    sess_options.graph_optimization_level = getattr(enum, "ORT_ENABLE_ALL")
                else:  # pragma: no cover - validated in __post_init__
                    raise ValueError(f"Unknown graph_optimization_level: {lvl!r}")

            v = opts.get("enable_mem_pattern", None)
            if v is not None and hasattr(sess_options, "enable_mem_pattern"):
                sess_options.enable_mem_pattern = bool(v)

            v = opts.get("enable_cpu_mem_arena", None)
            if v is not None and hasattr(sess_options, "enable_cpu_mem_arena"):
                sess_options.enable_cpu_mem_arena = bool(v)

            v = opts.get("log_severity_level", None)
            if v is not None and hasattr(sess_options, "log_severity_level"):
                sess_options.log_severity_level = int(v)

            v = opts.get("log_verbosity_level", None)
            if v is not None and hasattr(sess_options, "log_verbosity_level"):
                sess_options.log_verbosity_level = int(v)

            entries = opts.get("session_config_entries", None)
            if entries is not None:
                for k, vv in dict(entries).items():
                    sess_options.add_session_config_entry(str(k), str(vv))

        if sess_options is None:
            sess = ort.InferenceSession(str(ckpt), providers=providers)
        else:
            sess = ort.InferenceSession(str(ckpt), providers=providers, sess_options=sess_options)

        inputs = list(sess.get_inputs())
        if not inputs:
            raise RuntimeError("ONNX model has no inputs.")

        in_name = str(inputs[0].name) if self.input_name is None else str(self.input_name)

        outputs = list(sess.get_outputs())
        if not outputs:
            raise RuntimeError("ONNX model has no outputs.")

        out_name: str
        if self.output_name is not None:
            out_name = str(self.output_name)
            available = [str(o.name) for o in outputs]
            if out_name not in available:
                raise ValueError(f"output_name {out_name!r} not found in model outputs {available}")
        else:
            idx = int(self.output_index)
            if idx < 0 or idx >= len(outputs):
                raise IndexError(
                    f"output_index out of range: {idx} for outputs of len {len(outputs)}"
                )
            out_name = str(outputs[idx].name)

        self._sess = sess
        self._ort = ort
        self._input_name = in_name
        self._output_name = out_name

    def _preprocess_one(self, item: Any) -> np.ndarray:
        from PIL import Image

        img = _as_pil_rgb(item, input_color=self.input_color)
        img = img.resize((int(self.image_size), int(self.image_size)), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected RGB image array, got shape {arr.shape}")
        chw = np.transpose(arr, (2, 0, 1)).astype(np.float32) / 255.0
        chw = _normalize_rgb_chw_f32(chw, mean=self.mean, std=self.std)
        return np.asarray(chw, dtype=np.float32)

    def _extract_no_cache(self, items: list[Any]) -> np.ndarray:
        self._ensure_ready()
        sess = self._sess
        in_name = self._input_name
        out_name = self._output_name
        if sess is None or in_name is None or out_name is None:
            raise RuntimeError("Extractor not initialized; this is a bug.")

        bs = int(self.batch_size)
        rows: list[np.ndarray] = []
        for i in range(0, len(items), bs):
            batch_items = items[i : i + bs]
            batch_np = np.stack(
                [self._preprocess_one(it) for it in batch_items], axis=0
            )  # (B,3,H,W)
            out_list = sess.run([out_name], {in_name: batch_np})
            if not out_list:
                raise RuntimeError("onnxruntime returned no outputs")
            emb = _as_2d_embedding(np.asarray(out_list[0]))
            if int(emb.shape[0]) != len(batch_items):
                raise ValueError(
                    "onnx_embed must return one row per input. "
                    f"Got {int(emb.shape[0])} rows for batch of {len(batch_items)}."
                )
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
            for p, row in zip(missing, computed, strict=True):
                self._cache.save(p, np.asarray(row, dtype=np.float64).reshape(-1))
                cached_rows[p] = np.asarray(row, dtype=np.float64).reshape(-1)

            return np.stack(
                [np.asarray(cached_rows[p], dtype=np.float64).reshape(-1) for p in paths]
            )

        self.last_cache_stats_ = {"hits": 0, "misses": 0, "enabled": False}
        return self._extract_no_cache(items)


__all__ = ["ONNXEmbedExtractor"]
