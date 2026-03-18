# -*- coding: utf-8 -*-
from __future__ import annotations

"""Vision deep detector base class (import-light).

This module intentionally avoids importing heavy optional dependencies like
`torch`, `torchvision`, or `cv2` at import time.

Why?
- `pyimgano.models` re-exports `BaseVisionDeepDetector`.
- Industrial discovery flows (CLI `--list-models`, workbench schema checks)
  should not implicitly import deep runtimes.

Deep dependencies are required only when instantiating / running detectors.
"""

from pathlib import Path

from pyimgano.utils.optional_deps import require

from .base_deep import BaseDeepLearningDetector


class BaseVisionDeepDetector(BaseDeepLearningDetector):
    """
    所有基于深度学习的端到端视觉异常检测算法的基类。
    本类继承自 `pyimgano` 的 BaseDeepLearningDetector，复用了其训练框架，
    """

    def __init__(
        self,
        contamination=0.1,
        preprocessing=True,
        lr=1e-3,
        epoch_num=10,
        batch_size=32,
        optimizer_name="adam",
        criterion_name="mse",
        device=None,
        random_state=42,
        verbose=1,
        train_transform=None,
        eval_transform=None,
        **kwargs,
    ):
        # 调用父类 (pyimgano.models.base_deep.BaseDeepLearningDetector) 的构造函数
        super().__init__(
            contamination=contamination,
            preprocessing=preprocessing,
            lr=lr,
            epoch_num=epoch_num,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            criterion_name=criterion_name,
            device=device,
            random_state=random_state,
            verbose=verbose,
            **kwargs,
        )
        # Compatibility: many utilities (e.g. `predict_proba`) expect
        # `_classes` to exist. In unsupervised detection this is always binary.
        self._set_n_classes(None)

        if train_transform is not None:
            self.train_transform = train_transform
        else:
            if self.preprocessing:
                # 默认训练时使用 datasets 模块提供的预设
                from pyimgano.datasets import default_train_transforms

                self.train_transform = default_train_transforms()
            else:
                T = require(
                    "torchvision.transforms",
                    purpose="BaseVisionDeepDetector default train transform",
                )
                self.train_transform = T.ToTensor()

        if eval_transform is not None:
            self.eval_transform = eval_transform
        else:
            if self.preprocessing:
                # 评估时不应有数据增强，保证结果一致性
                from pyimgano.datasets import default_eval_transforms

                self.eval_transform = default_eval_transforms()
            else:
                T = require(
                    "torchvision.transforms",
                    purpose="BaseVisionDeepDetector default eval transform",
                )
                self.eval_transform = T.ToTensor()

        # Optional eval-time tensor cache (best-effort).
        self._eval_tensor_cache = None

    # ------------------------------------------------------------------
    # Deep learning interface (required abstract methods)
    #
    # Many `pyimgano` vision detectors implement their own `fit` /
    # `decision_function` without using the shared training loop, but still inherit
    # from `BaseDeepLearningDetector` for shared thresholding semantics.
    #
    # To keep those detectors instantiable, we provide default implementations
    # that raise clear errors if a subclass actually relies on the training loop.
    def build_model(self, *args, **kwargs):  # pragma: no cover
        if getattr(self, "model", None) is not None:
            return self.model
        raise NotImplementedError(
            "Subclasses using BaseVisionDeepDetector.fit must implement build_model()."
        )

    def training_forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(
            "Subclasses using BaseVisionDeepDetector.fit must implement training_forward()."
        )

    def evaluating_forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(
            "Subclasses using BaseVisionDeepDetector.decision_function must implement evaluating_forward()."
        )

    def fit(self, X, y=None):
        """
        【特色功能 3: 重写 fit 方法以处理图像路径】
        使用正常的、无缺陷的图像数据来拟合检测器。

        Parameters
        ----------
        X : list of str
            输入的训练样本，必须是图像文件路径的列表。
        """
        import numpy as np

        from pyimgano.datasets import VisionArrayDataset, VisionImageDataset

        data_loader_cls = require("torch.utils.data", purpose="deep vision training").DataLoader

        # 1. 构建模型
        self.model = self.build_model()

        x_list = list(X)
        if x_list and isinstance(x_list[0], np.ndarray):
            # Numpy-first industrial workflows: images already decoded in memory.
            train_dataset = VisionArrayDataset(images=x_list, transform=self.train_transform)
        else:
            # Default: list of file paths.
            train_dataset = VisionImageDataset(image_paths=x_list, transform=self.train_transform)

        train_loader = data_loader_cls(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=bool(self.shuffle_train),
            num_workers=int(self.num_workers),
            drop_last=bool(self.drop_last),
            pin_memory=bool(self.pin_memory),
            persistent_workers=bool(self.persistent_workers) and int(self.num_workers) > 0,
        )

        # 3. 准备训练 (来自父类的方法)
        self.training_prepare()

        # 4. 执行训练循环 (来自父类的方法，它会调用我们子类实现的 training_forward)
        if self.verbose:
            import logging

            logging.getLogger(__name__).info("开始在 %s 设备上进行训练...", self.device)
        self.train(train_loader)
        if self.verbose:
            import logging

            logging.getLogger(__name__).info("训练完成。")

        # 5. 计算训练集上的异常分数
        if self.verbose:
            import logging

            logging.getLogger(__name__).info("正在计算训练集上的异常分数...")
        self.decision_scores_ = self.decision_function(X)

        # 6. 调用基类的方法来计算阈值和标签
        self._process_decision_scores()
        # Compatibility: enable `predict_proba()` by initializing `_classes`.
        self._set_n_classes(y)
        return self

    def decision_function(self, X, batch_size=None):
        import numpy as np

        from pyimgano.datasets import VisionArrayDataset, VisionImageDataset

        data_loader_cls = require("torch.utils.data", purpose="deep vision evaluation").DataLoader

        current_batch_size = batch_size if batch_size is not None else self.batch_size

        x_list = list(X)
        if x_list and isinstance(x_list[0], np.ndarray):
            eval_dataset = VisionArrayDataset(images=x_list, transform=self.eval_transform)
        else:
            if getattr(self, "_eval_tensor_cache", None) is not None:
                from pyimgano.cache.deep_embeddings import CachedVisionImageDataset

                eval_dataset = CachedVisionImageDataset(
                    image_paths=[str(p) for p in x_list],
                    transform=self.eval_transform,
                    cache=self._eval_tensor_cache,  # type: ignore[arg-type]
                )
            else:
                eval_dataset = VisionImageDataset(
                    image_paths=[str(p) for p in x_list],
                    transform=self.eval_transform,
                )
        eval_loader = data_loader_cls(eval_dataset, batch_size=current_batch_size, shuffle=False)

        # 调用父类的评估方法 evaluating_forward
        scores = self.evaluate(eval_loader)
        return scores

    def set_eval_cache(self, cache_dir: str | Path | None) -> None:
        """Enable/disable eval-time tensor caching for path inputs.

        Notes
        -----
        - Intended for repeated scoring runs on the same image paths.
        - Only applied for path inputs (not for numpy inputs).
        - Uses a fingerprint of the eval_transform to reduce cache collisions.
        """

        if cache_dir is None:
            self._eval_tensor_cache = None
            return

        from pyimgano.cache.deep_embeddings import TensorCache, fingerprint_transform

        fp = fingerprint_transform(self.eval_transform)
        self._eval_tensor_cache = TensorCache(
            cache_dir=Path(cache_dir),
            transform_fingerprint=str(fp),
        )
