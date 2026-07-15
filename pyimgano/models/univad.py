"""Official-source UniVAD inference adapter (CVPR 2025).

UniVAD depends on the authors' C³ component masks and their full
CLIP/DINOv2/CAPM/GECM runtime.  This module delegates to that source instead
of approximating it with a local feature prototype.
"""

from __future__ import annotations

import importlib
import io
import os
import sys
from contextlib import contextmanager, nullcontext, redirect_stdout
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .base_detector import BaseDetector
from .registry import register_model

OFFICIAL_REPOSITORY = "https://github.com/FantasticGNU/UniVAD"
OFFICIAL_COMMIT = "64d32873dda44fad69786834ea5ee1394ef81975"
PAPER_IMAGE_SIZE = 448
PAPER_CLIP_BACKBONE = "ViT-L-14-336"
PAPER_DINOV2_BACKBONE = "dinov2_vitg14"
PAPER_DINO_BACKBONE = "vit_small_patch8"
PAPER_CLIP_LAYERS = (6, 12, 18, 24)
PAPER_STRUCTURAL_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)
PAPER_LOGICAL_WEIGHTS = (0.5, 0.5)
PAPER_FINAL_WEIGHTS = (0.5, 0.5)


def _as_image_paths(value: Any) -> list[Path]:
    raw = [value] if isinstance(value, (str, Path)) else list(value)
    if not raw:
        raise ValueError("UniVAD requires at least one image path.")
    paths: list[Path] = []
    for item in raw:
        if not isinstance(item, (str, Path)):
            raise TypeError(
                "The official UniVAD runtime requires image paths because C³ masks are "
                "stored beside the author dataset layout."
            )
        path = Path(item).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"UniVAD image not found: {path}")
        paths.append(path)
    return paths


def _paper_image_tensor(path: Path) -> torch.Tensor:
    """Apply the authors' PIL resize and ToTensor preprocessing."""

    resampling = getattr(Image, "Resampling", Image)
    with Image.open(path) as source:
        image = source.convert("RGB").resize(
            (PAPER_IMAGE_SIZE, PAPER_IMAGE_SIZE), resample=resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def _author_mask_path(repository_path: Path, image_path: Path) -> Path:
    """Resolve the exact mask location used by the released setup path."""

    suffix = str(image_path).split("/data/")[-1].lstrip("/")
    for extension in (".png", ".JPG"):
        if suffix.endswith(extension):
            suffix = suffix[: -len(extension)] + "/grounding_mask.png"
            return repository_path / "masks" / suffix
    raise ValueError(
        "The released UniVAD setup supports .png and .JPG inputs; its .jpeg mask lookup "
        "is inconsistent with segment_components.py."
    )


def _require_component_masks(repository_path: Path, paths: Sequence[Path]) -> None:
    missing = [
        path
        for path in (_author_mask_path(repository_path, item) for item in paths)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "UniVAD C³ component mask not found. Run the official segment_components.py; "
            f"missing: {missing[0]}"
        )


@contextmanager
def _author_runtime(repository_path: Path) -> Iterator[None]:
    paths = [str(repository_path), str(repository_path / "models" / "GroundingDINO")]
    previous_cwd = Path.cwd()
    previous_backend = os.environ.get("MPLBACKEND")
    os.environ.setdefault("MPLBACKEND", "Agg")
    for path in reversed(paths):
        sys.path.insert(0, path)
    # ponytail: upstream hard-codes cwd-relative masks; use a subprocess if concurrent adapters matter.
    os.chdir(repository_path)
    try:
        importlib.invalidate_caches()
        yield
    finally:
        os.chdir(previous_cwd)
        for path in paths:
            try:
                sys.path.remove(path)
            except ValueError:
                pass
        if previous_backend is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = previous_backend


def _load_author_model(*, repository_path: Path, device: torch.device, allow_download: bool) -> Any:
    source = repository_path / "UniVAD.py"
    dinov2_hub = repository_path / "models" / "dinov2" / "hubconf.py"
    if not source.is_file() or not dinov2_hub.is_file():
        raise FileNotFoundError(
            f"Not a complete official UniVAD source tree: {repository_path}. Clone "
            f"{OFFICIAL_REPOSITORY} with --recurse-submodules."
        )
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The released UniVAD runtime requires a CUDA device.")
    if not allow_download:
        raise ValueError(
            "UniVAD loads public CLIP, DINOv2, and DINO weights on first use; pass "
            "allow_download=True after reviewing the upstream licenses."
        )

    try:
        with _author_runtime(repository_path), redirect_stdout(io.StringIO()):
            module = importlib.import_module("UniVAD")
            loaded_source = Path(module.__file__).resolve()
            if loaded_source != source.resolve():
                raise RuntimeError(
                    "A different top-level UniVAD module is already imported; start a clean "
                    "process before loading the author repository."
                )
            model = module.UniVAD(image_size=PAPER_IMAGE_SIZE)
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "The UniVAD author runtime dependencies are unavailable. Install its "
            "requirements.txt and GroundingDINO package in an isolated environment."
        ) from exc
    return model.to(device).eval()


class AuthorUniVADBackend:
    """Execute the authors' training-free few-shot setup and inference path."""

    def __init__(
        self,
        *,
        repository_path: str | Path | None,
        device: str,
        allow_download: bool = False,
        model: Any = None,
    ) -> None:
        self.repository_path = None if repository_path is None else Path(repository_path).resolve()
        self.device = torch.device(device)
        self.allow_download = bool(allow_download)
        self.model = model
        self._loaded = False
        self._setup = False

    def _runtime(self):  # noqa: ANN202 - context manager varies only for injected tests
        return (
            nullcontext() if self.repository_path is None else _author_runtime(self.repository_path)
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.model is None:
            if self.repository_path is None:
                raise ValueError("vision_univad requires repository_path to the official source.")
            self.model = _load_author_model(
                repository_path=self.repository_path,
                device=self.device,
                allow_download=self.allow_download,
            )
        else:
            self.model = self.model.to(self.device).eval()

        if int(getattr(self.model, "image_size", -1)) != PAPER_IMAGE_SIZE:
            raise ValueError(f"UniVAD requires image_size={PAPER_IMAGE_SIZE}.")
        if tuple(getattr(self.model, "out_layers", ())) != PAPER_CLIP_LAYERS:
            raise ValueError(f"UniVAD requires CLIP layers {PAPER_CLIP_LAYERS}.")
        if not callable(getattr(self.model, "setup", None)) or not callable(self.model):
            raise TypeError("Expected the author UniVAD model with setup() and forward().")
        self._loaded = True

    @torch.inference_mode()
    def setup_support(
        self,
        paths: Sequence[Path],
        *,
        class_name: str,
        resegment_components: bool,
    ) -> None:
        self._ensure_loaded()
        if self.repository_path is not None:
            _require_component_masks(self.repository_path, paths)
        images = torch.stack([_paper_image_tensor(path) for path in paths]).to(self.device)
        payload = {
            "few_shot_samples": images,
            "dataset_category": class_name,
            "image_path": [str(path) for path in paths],
        }
        with self._runtime():
            self.model.setup(payload, re_seg=bool(resegment_components))
        self._setup = True

    @torch.inference_mode()
    def score_paths(self, paths: Sequence[Path]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        self._ensure_loaded()
        if not self._setup:
            raise RuntimeError("UniVAD support is not initialized; call fit() first.")
        if self.repository_path is not None:
            _require_component_masks(self.repository_path, paths)

        scores: list[float] = []
        maps: list[NDArray[np.float32]] = []
        with self._runtime():
            for path in paths:
                output = self.model(
                    _paper_image_tensor(path).unsqueeze(0).to(self.device), str(path)
                )
                if not isinstance(output, Mapping) or not {"pred_score", "pred_mask"}.issubset(
                    output
                ):
                    raise TypeError("Author UniVAD returned an invalid inference payload.")
                score = torch.as_tensor(output["pred_score"]).detach().cpu().reshape(-1)
                anomaly_map = torch.as_tensor(output["pred_mask"]).detach().float().cpu().squeeze()
                if score.numel() != 1 or anomaly_map.ndim != 2:
                    raise ValueError("Author UniVAD returned invalid score or anomaly-map shapes.")
                scores.append(float(score.item()))
                maps.append(anomaly_map.numpy().astype(np.float32, copy=False))

        score_array = np.asarray(scores, dtype=np.float32)
        map_array = np.stack(maps).astype(np.float32, copy=False)
        if not np.isfinite(score_array).all() or not np.isfinite(map_array).all():
            raise ValueError("Author UniVAD returned non-finite scores or maps.")
        return score_array, map_array


@register_model(
    "vision_univad",
    tags=(
        "vision",
        "deep",
        "neighbors",
        "few-shot",
        "pixel_map",
        "univad",
        "cvpr2025",
        "external-backend",
    ),
    metadata={
        "description": "Official-source UniVAD C³/CAPM/GECM inference adapter",
        "paper": "UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2025/html/Gu_UniVAD_A_Training-free_Unified_Model_for_Few-shot_Visual_Anomaly_Detection_CVPR_2025_paper.html",
        "official_repository": "https://github.com/FantasticGNU/UniVAD",
        "official_commit": "64d32873dda44fad69786834ea5ee1394ef81975",
        "year": 2025,
        "conference": "CVPR",
        "implementation_status": "official-source-inference-adapter",
        "paper_fidelity": "external-backend",
        "backend": "official-univad",
        "type": "training-free-few-shot",
        "supervision": "few-shot",
        "requires_checkpoint": False,
        "supports_pixel_map": True,
        "weights_source": "official public CLIP/DINOv2/DINO pretrained weights",
        "upstream_license": "CC BY-NC-SA 4.0",
    },
)
class VisionUniVAD(BaseDetector):
    """UniVAD author inference; ``fit`` installs normal references and calibrates."""

    input_mode = "images"

    def __init__(
        self,
        *,
        repository_path: str | Path | None = None,
        class_name: str | None = None,
        device: str = "cuda",
        allow_download: bool = False,
        resegment_components: bool = True,
        backend: Any = None,
        contamination: float = 0.1,
    ) -> None:
        super().__init__(contamination=contamination)
        self._set_n_classes(None)
        self.repository_path = repository_path
        self.class_name = class_name
        self.device = str(device)
        self.allow_download = bool(allow_download)
        self.resegment_components = bool(resegment_components)
        self.backend = (
            backend
            if backend is not None
            else AuthorUniVADBackend(
                repository_path=repository_path,
                device=self.device,
                allow_download=self.allow_download,
            )
        )
        self.is_fitted_ = False

    def _score_paths(
        self, paths: Sequence[Path]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if not callable(getattr(self.backend, "score_paths", None)):
            raise TypeError("UniVAD backend must implement score_paths(paths).")
        result = self.backend.score_paths(paths)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("UniVAD backend must return (scores, anomaly_maps).")
        scores = np.asarray(result[0], dtype=np.float32).reshape(-1)
        maps = np.asarray(result[1], dtype=np.float32)
        if scores.shape != (len(paths),) or maps.ndim != 3 or maps.shape[0] != len(paths):
            raise ValueError("UniVAD backend returned shapes inconsistent with the inputs.")
        if not np.isfinite(scores).all() or not np.isfinite(maps).all():
            raise ValueError("UniVAD backend returned non-finite scores or maps.")
        return scores, maps

    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray[Any]] = None,
        **kwargs: object,
    ) -> "VisionUniVAD":
        del y
        values = resolve_legacy_x_keyword(x, kwargs, method_name="fit")
        paths = _as_image_paths(values)
        class_name = "" if self.class_name is None else str(self.class_name).strip()
        if not class_name:
            raise ValueError("vision_univad requires class_name for the support category.")
        if not callable(getattr(self.backend, "setup_support", None)):
            raise TypeError("UniVAD backend must implement setup_support(paths, ...).")
        self.backend.setup_support(
            paths,
            class_name=class_name,
            resegment_components=self.resegment_components,
        )
        self.decision_scores_ = self._score_paths(paths)[0]
        self._process_decision_scores()
        self.is_fitted_ = True
        return self

    def decision_function(self, x: object = MISSING, **kwargs: object) -> NDArray[np.float32]:
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() with normal reference paths first.")
        values = resolve_legacy_x_keyword(x, kwargs, method_name="decision_function")
        return self._score_paths(_as_image_paths(values))[0]

    def predict(self, x: object = MISSING, **kwargs: object) -> NDArray[np.int64]:
        values = resolve_legacy_x_keyword(x, kwargs, method_name="predict")
        return (self.decision_function(values) > float(self.threshold_)).astype(np.int64)

    def predict_anomaly_map(self, x: object = MISSING, **kwargs: object) -> NDArray[np.float32]:
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call fit() with normal reference paths first.")
        values = resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map")
        return self._score_paths(_as_image_paths(values))[1]

    def get_anomaly_map(self, image: str | Path) -> NDArray[np.float32]:
        return self.predict_anomaly_map([image])[0]


__all__ = [
    "AuthorUniVADBackend",
    "OFFICIAL_COMMIT",
    "PAPER_CLIP_BACKBONE",
    "PAPER_CLIP_LAYERS",
    "PAPER_DINO_BACKBONE",
    "PAPER_DINOV2_BACKBONE",
    "PAPER_IMAGE_SIZE",
    "VisionUniVAD",
]
