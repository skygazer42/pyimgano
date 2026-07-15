"""Official-source LogSAD few-shot inference adapter (CVPR 2025)."""

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

OFFICIAL_REPOSITORY = "https://github.com/zhang0jhon/LogSAD"
OFFICIAL_COMMIT = "06aed1a8d4181ce08ffa91f9e5f8733c27833b55"
PAPER_IMAGE_SIZE = 448
PAPER_FEATURE_SIZE = 64
PAPER_CLIP_BACKBONE = "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
PAPER_DINOV2_BACKBONE = "dinov2_vitl14"
PAPER_SAM_BACKBONE = "vit_h"
PAPER_FEATURE_LAYERS = (6, 12, 18, 24)
PAPER_MEMORY_SIZE = 2048
PAPER_N_NEIGHBORS = 2
SUPPORTED_CATEGORIES = (
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
)


def _as_image_paths(value: Any) -> list[Path]:
    raw = [value] if isinstance(value, (str, Path)) else list(value)
    if not raw:
        raise ValueError("LogSAD requires at least one image path.")
    paths: list[Path] = []
    for item in raw:
        if not isinstance(item, (str, Path)):
            raise TypeError(
                "The official LogSAD runtime requires image paths for SAM and "
                "category-specific composition matching."
            )
        path = Path(item).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"LogSAD image not found: {path}")
        paths.append(path)
    return paths


def _paper_image_tensor(path: Path) -> torch.Tensor:
    """Match the author dataloader's RGB float tensor and 448px resize."""

    resampling = getattr(Image, "Resampling", Image)
    with Image.open(path) as source:
        image = source.convert("RGB").resize(
            (PAPER_IMAGE_SIZE, PAPER_IMAGE_SIZE), resample=resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


@contextmanager
def _author_runtime(repository_path: Path) -> Iterator[None]:
    root = str(repository_path)
    previous_cwd = Path.cwd()
    previous_backend = os.environ.get("MPLBACKEND")
    added_numpy_aliases: list[str] = []
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, root)
    for name, value in (("bool", np.bool_), ("int", int)):
        if name not in np.__dict__:
            setattr(np, name, value)
            added_numpy_aliases.append(name)
    # ponytail: author paths are cwd-relative; isolate in a subprocess if concurrent use matters.
    os.chdir(repository_path)
    try:
        importlib.invalidate_caches()
        yield
    finally:
        os.chdir(previous_cwd)
        try:
            sys.path.remove(root)
        except ValueError:
            pass
        for name in added_numpy_aliases:
            delattr(np, name)
        if previous_backend is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = previous_backend


def _load_author_model(*, repository_path: Path, device: torch.device, allow_download: bool) -> Any:
    source = repository_path / "model_ensemble_few_shot.py"
    required = (
        source,
        repository_path / "open_clip_local" / "__init__.py",
        repository_path / "dinov2" / "dinov2" / "hub" / "backbones.py",
        repository_path / "memory_bank" / "statistic_scores_model_ensemble_few_shot_val.pkl",
        repository_path / "checkpoint" / "sam_vit_h_4b8939.pth",
    )
    missing = next((path for path in required if not path.is_file()), None)
    if missing is not None:
        raise FileNotFoundError(
            f"Incomplete official LogSAD runtime; missing: {missing}. Follow {OFFICIAL_REPOSITORY}."
        )
    author_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != author_device.type or (device.index not in (None, 0)):
        raise RuntimeError(
            f"The released LogSAD source selects {author_device}; requested device was {device}."
        )
    if not allow_download:
        raise ValueError(
            "LogSAD loads public CLIP and DINOv2 weights on first use; pass "
            "allow_download=True after reviewing the upstream terms."
        )

    try:
        with _author_runtime(repository_path), redirect_stdout(io.StringIO()):
            module = importlib.import_module("model_ensemble_few_shot")
            if Path(module.__file__).resolve() != source.resolve():
                raise RuntimeError(
                    "A different model_ensemble_few_shot module is already imported; "
                    "start a clean process."
                )
            model = module.MyModel()
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "The LogSAD author dependencies are unavailable. Install the pinned upstream "
            "requirements in an isolated environment."
        ) from exc
    return model.to(device).eval()


class AuthorLogSADBackend:
    """Run the authors' CLIP/DINOv2/SAM multi-granularity inference graph."""

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

    def _runtime(self):  # noqa: ANN202 - context manager differs only for injected tests
        return (
            nullcontext() if self.repository_path is None else _author_runtime(self.repository_path)
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.model is None:
            if self.repository_path is None:
                raise ValueError("vision_logsad requires repository_path to the official source.")
            self.model = _load_author_model(
                repository_path=self.repository_path,
                device=self.device,
                allow_download=self.allow_download,
            )
        else:
            self.model = self.model.to(self.device).eval()

        expected = {
            "feature_list": PAPER_FEATURE_LAYERS,
            "feature_list_dinov2": PAPER_FEATURE_LAYERS,
            "feat_size": PAPER_FEATURE_SIZE,
            "ori_feat_size": PAPER_IMAGE_SIZE // 14,
            "memory_size": PAPER_MEMORY_SIZE,
            "n_neighbors": PAPER_N_NEIGHBORS,
        }
        for name, value in expected.items():
            actual = getattr(self.model, name, None)
            matches = tuple(actual or ()) == value if isinstance(value, tuple) else actual == value
            if not matches:
                raise ValueError(
                    f"LogSAD author parameter mismatch: {name}={actual!r}, expected {value!r}."
                )
        if not callable(getattr(self.model, "setup", None)) or not callable(self.model):
            raise TypeError("Expected the author LogSAD MyModel with setup() and forward().")
        self._loaded = True

    @torch.inference_mode()
    def setup_support(self, paths: Sequence[Path], *, class_name: str) -> None:
        self._ensure_loaded()
        images = torch.stack([_paper_image_tensor(path) for path in paths]).to(self.device)
        payload = {
            "few_shot_samples": images,
            "few_shot_samples_path": [str(path) for path in paths],
            "dataset_category": class_name,
        }
        with self._runtime():
            self.model.setup(payload)
        self._setup = True

    @torch.inference_mode()
    def score_paths(self, paths: Sequence[Path]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        self._ensure_loaded()
        if not self._setup:
            raise RuntimeError("LogSAD support is not initialized; call fit() first.")

        scores: list[float] = []
        maps: list[NDArray[np.float32]] = []
        with self._runtime():
            for path in paths:
                output = self.model(
                    _paper_image_tensor(path).unsqueeze(0).to(self.device), [str(path)]
                )
                if not isinstance(output, Mapping) or not {"pred_score", "anomaly_map"}.issubset(
                    output
                ):
                    raise TypeError("Author LogSAD returned an invalid inference payload.")
                score = torch.as_tensor(output["pred_score"]).detach().cpu().reshape(-1)
                anomaly_map = (
                    torch.as_tensor(output["anomaly_map"]).detach().float().cpu().squeeze()
                )
                if score.numel() != 1 or anomaly_map.shape != (
                    PAPER_FEATURE_SIZE,
                    PAPER_FEATURE_SIZE,
                ):
                    raise ValueError("Author LogSAD returned invalid score or anomaly-map shapes.")
                scores.append(float(score.item()))
                maps.append(anomaly_map.numpy().astype(np.float32, copy=False))

        score_array = np.asarray(scores, dtype=np.float32)
        map_array = np.stack(maps).astype(np.float32, copy=False)
        if not np.isfinite(score_array).all() or not np.isfinite(map_array).all():
            raise ValueError("Author LogSAD returned non-finite scores or maps.")
        return score_array, map_array


@register_model(
    "vision_logsad",
    tags=(
        "vision",
        "deep",
        "few-shot",
        "logical",
        "structural",
        "pixel_map",
        "logsad",
        "cvpr2025",
        "external-backend",
    ),
    metadata={
        "description": "Official-source LogSAD CLIP/DINOv2/SAM inference adapter",
        "paper": "Towards Training-free Anomaly Detection with Vision and Language Foundation Models",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Towards_Training-free_Anomaly_Detection_with_Vision_and_Language_Foundation_Models_CVPR_2025_paper.html",
        "official_repository": "https://github.com/zhang0jhon/LogSAD",
        "official_commit": "06aed1a8d4181ce08ffa91f9e5f8733c27833b55",
        "year": 2025,
        "conference": "CVPR",
        "implementation_status": "official-source-inference-adapter",
        "paper_fidelity": "external-backend",
        "backend": "official-logsad",
        "type": "training-free-few-shot-logical-structural",
        "supervision": "few-shot",
        "requires_checkpoint": True,
        "supports_pixel_map": True,
        "weights_source": "official public CLIP, DINOv2, and SAM pretrained weights",
        "upstream_license": "not declared at pinned repository commit",
    },
)
class VisionLogSAD(BaseDetector):
    """LogSAD author inference; ``fit`` installs normal references and calibrates."""

    input_mode = "images"

    def __init__(
        self,
        *,
        repository_path: str | Path | None = None,
        class_name: str | None = None,
        device: str = "cuda",
        allow_download: bool = False,
        backend: Any = None,
        contamination: float = 0.1,
    ) -> None:
        super().__init__(contamination=contamination)
        self._set_n_classes(None)
        self.repository_path = repository_path
        self.class_name = class_name
        self.device = str(device)
        self.allow_download = bool(allow_download)
        self.backend = (
            backend
            if backend is not None
            else AuthorLogSADBackend(
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
            raise TypeError("LogSAD backend must implement score_paths(paths).")
        result = self.backend.score_paths(paths)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("LogSAD backend must return (scores, anomaly_maps).")
        scores = np.asarray(result[0], dtype=np.float32).reshape(-1)
        maps = np.asarray(result[1], dtype=np.float32)
        if scores.shape != (len(paths),) or maps.ndim != 3 or maps.shape[0] != len(paths):
            raise ValueError("LogSAD backend returned shapes inconsistent with the inputs.")
        if not np.isfinite(scores).all() or not np.isfinite(maps).all():
            raise ValueError("LogSAD backend returned non-finite scores or maps.")
        return scores, maps

    def fit(
        self,
        x: object = MISSING,
        _y: Optional[NDArray[Any]] = None,
        **kwargs: object,
    ) -> "VisionLogSAD":
        del _y
        paths = _as_image_paths(resolve_legacy_x_keyword(x, kwargs, method_name="fit"))
        class_name = "" if self.class_name is None else str(self.class_name).strip()
        if class_name not in SUPPORTED_CATEGORIES:
            raise ValueError(
                f"The released LogSAD code supports categories {SUPPORTED_CATEGORIES}; got {class_name!r}."
            )
        if not callable(getattr(self.backend, "setup_support", None)):
            raise TypeError("LogSAD backend must implement setup_support(paths, ...).")
        self.backend.setup_support(paths, class_name=class_name)
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
    "AuthorLogSADBackend",
    "OFFICIAL_COMMIT",
    "PAPER_CLIP_BACKBONE",
    "PAPER_DINOV2_BACKBONE",
    "PAPER_FEATURE_LAYERS",
    "PAPER_FEATURE_SIZE",
    "PAPER_IMAGE_SIZE",
    "PAPER_SAM_BACKBONE",
    "SUPPORTED_CATEGORIES",
    "VisionLogSAD",
]
