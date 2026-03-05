# -*- coding: utf-8 -*-
"""EfficientAD-lite (teacher/student distillation, contract-aligned).

This is a simplified, production-friendly variant inspired by EfficientAD:
- freeze a teacher backbone
- train a lightweight student to match teacher embeddings on normal data
- score images by teacher/student embedding MSE (higher = more anomalous)

Design constraints:
- No implicit weight downloads by default (`teacher_pretrained=False`).
- Contract-aligned with `BaseVisionDeepDetector` (`fit`, `decision_function`).
- Keep imports lazy (avoid heavy imports at module import time).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from pyimgano.utils.torchvision_safe import load_torchvision_backbone

from .baseCv import BaseVisionDeepDetector
from .registry import register_model


def _to_hw(image_size: int | Tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_size, tuple):
        h, w = int(image_size[0]), int(image_size[1])
    else:
        h = w = int(image_size)
    if h <= 0 or w <= 0:
        raise ValueError(f"image_size must be positive, got {image_size!r}")
    return h, w


def _make_transforms(image_size: tuple[int, int]):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((int(image_size[0]), int(image_size[1]))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


@dataclass(frozen=True)
class EfficientADConfig:
    image_size: tuple[int, int]
    teacher_net: str
    teacher_pretrained: bool
    student_width: int


def _infer_teacher_dim(teacher, *, hw: tuple[int, int], device) -> int:
    import torch

    with torch.no_grad():
        dummy = torch.zeros((1, 3, int(hw[0]), int(hw[1])), dtype=torch.float32, device=device)
        out = teacher(dummy)
        out_t = torch.as_tensor(out)
        if out_t.ndim > 2:
            out_t = torch.flatten(out_t, start_dim=1)
        return int(out_t.shape[1])


def _build_student(*, out_dim: int, width: int):
    import torch.nn as nn

    w = int(width)
    d = int(out_dim)
    return nn.Sequential(
        nn.Conv2d(3, w, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(w, w * 2, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(w * 2, w * 4, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(w * 4, d),
    )


@register_model(
    "efficient_ad",
    tags=("vision", "deep", "distillation"),
    metadata={
        "description": "EfficientAD-lite: teacher/student embedding distillation (contract-aligned)",
    },
    overwrite=True,
)
class EfficientADDetector(BaseVisionDeepDetector):
    def __init__(
        self,
        *,
        contamination: float = 0.1,
        epochs: int = 10,
        batch_size: int = 16,
        lr: float = 1e-3,
        device: str | None = None,
        random_state: int = 0,
        verbose: int = 0,
        tiny: bool = False,
        image_size: int | Tuple[int, int] = 256,
        teacher_net: str = "resnet18",
        teacher_pretrained: bool = False,
        student_width: int = 16,
    ) -> None:
        self.tiny = bool(tiny)
        hw = _to_hw(image_size)

        if self.tiny:
            hw = (min(hw[0], 128), min(hw[1], 128))
            epochs = min(int(epochs), 1)
            batch_size = min(int(batch_size), 4)
            student_width = min(int(student_width), 8)

        self.cfg = EfficientADConfig(
            image_size=hw,
            teacher_net=str(teacher_net),
            teacher_pretrained=bool(teacher_pretrained),
            student_width=int(student_width),
        )

        self.teacher = None
        self.student = None
        self._teacher_dim = None

        train_transform = _make_transforms(hw)
        eval_transform = _make_transforms(hw)

        super().__init__(
            contamination=float(contamination),
            preprocessing=True,
            lr=float(lr),
            epoch_num=int(epochs),
            batch_size=int(batch_size),
            optimizer_name="adam",
            criterion_name="mse",
            device=device,
            random_state=int(random_state),
            verbose=int(verbose),
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

    def build_model(self):
        import torch

        # Teacher backbone (frozen). Note: pretrained=False by default (no downloads).
        teacher, _t = load_torchvision_backbone(
            str(self.cfg.teacher_net), pretrained=bool(self.cfg.teacher_pretrained)
        )
        teacher.to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        d = _infer_teacher_dim(teacher, hw=self.cfg.image_size, device=self.device)
        student = _build_student(out_dim=int(d), width=int(self.cfg.student_width))
        student.to(self.device)
        student.train()

        self.teacher = teacher
        self.student = student
        self._teacher_dim = int(d)

        # For BaseDeepLearningDetector, `.model` is the trainable module.
        return student

    def _teacher_features(self, images):  # noqa: ANN001
        import torch

        assert self.teacher is not None
        with torch.no_grad():
            out = self.teacher(images)
            out_t = torch.as_tensor(out)
            if out_t.ndim > 2:
                out_t = torch.flatten(out_t, start_dim=1)
            return out_t

    def training_forward(self, batch) -> float:  # noqa: ANN001
        import torch

        images, _targets = batch
        images = images.to(self.device)

        assert self.optimizer is not None
        self.optimizer.zero_grad(set_to_none=True)

        t_feat = self._teacher_features(images)
        s_feat = self.model(images)  # type: ignore[operator]
        s_feat = torch.as_tensor(s_feat)
        if s_feat.ndim > 2:
            s_feat = torch.flatten(s_feat, start_dim=1)

        loss = self.criterion(s_feat, t_feat)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def evaluating_forward(self, batch):  # noqa: ANN001
        import torch

        images, _targets = batch
        images = images.to(self.device)

        with torch.no_grad():
            t_feat = self._teacher_features(images)
            s_feat = self.model(images)  # type: ignore[operator]
            s_feat = torch.as_tensor(s_feat)
            if s_feat.ndim > 2:
                s_feat = torch.flatten(s_feat, start_dim=1)

            diff = (s_feat - t_feat) ** 2
            score = diff.mean(dim=1)
            return score.detach().cpu().numpy().astype(np.float32, copy=False)

    # Backward-compatibility aliases (old API).
    def train_fast(self, train_folder: str, epochs: int = 10):  # noqa: D401
        """Alias for legacy code paths (expects a folder of images).

        Prefer calling `fit(list_of_paths)` directly in new code.
        """

        from pathlib import Path

        exts = (".png", ".jpg", ".jpeg", ".bmp")
        paths = [
            str(p)
            for p in sorted(Path(train_folder).iterdir())
            if p.is_file() and p.suffix.lower() in exts
        ]
        # Best-effort: set epoch_num for this call only.
        self.epoch_num = int(epochs)
        return self.fit(paths)

    def predict_fast(self, img_path: str):  # noqa: D401
        """Legacy single-image scoring helper."""

        score = float(
            np.asarray(self.decision_function([str(img_path)]), dtype=np.float64).reshape(-1)[0]
        )
        return {"image": str(img_path), "anomaly_score": score}


__all__ = ["EfficientADDetector"]
