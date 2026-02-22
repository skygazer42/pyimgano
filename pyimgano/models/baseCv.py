# -*- coding: utf-8 -*-
from __future__ import annotations

from pyimgano.utils.optional_deps import optional_import

_torch, _torch_error = optional_import("torch")
_torchvision, _torchvision_error = optional_import("torchvision")
_pyod_base_dl, _pyod_error = optional_import("pyod.models.base_dl")


if _torch is not None and _torchvision is not None and _pyod_base_dl is not None:
    import torch  # type: ignore
    from pyod.models.base_dl import BaseDeepLearningDetector  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore
    from torchvision import transforms  # type: ignore

    # --- 核心基类 ---
    class BaseVisionDeepDetector(BaseDeepLearningDetector):
        """
        所有基于深度学习的端到端视觉异常检测算法的基类。
        本类继承自 PyOD 的 BaseDeepLearningDetector，复用了其完整的训练框架，
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
            # 调用父类 (pyod.BaseDeepLearningDetector) 的构造函数
            super(BaseVisionDeepDetector, self).__init__(
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

            if train_transform is not None:
                self.train_transform = train_transform
            else:
                if self.preprocessing:
                    # 默认训练时使用 datasets 模块提供的预设
                    from pyimgano.datasets import default_train_transforms

                    self.train_transform = default_train_transforms()
                else:
                    self.train_transform = transforms.ToTensor()
            # 为评估过程设置 transform
            if eval_transform is not None:
                self.eval_transform = eval_transform
            else:
                if self.preprocessing:
                    # 评估时不应有数据增强，保证结果一致性
                    from pyimgano.datasets import default_eval_transforms

                    self.eval_transform = default_eval_transforms()
                else:
                    self.eval_transform = transforms.ToTensor()

        # ------------------------------------------------------------------
        # PyOD deep learning interface (required abstract methods)
        #
        # Many `pyimgano` vision detectors implement their own `fit` /
        # `decision_function` without using PyOD's training loop, but still inherit
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

            # 1. 构建模型
            self.model = self.build_model()

            X_list = list(X)
            if X_list and isinstance(X_list[0], np.ndarray):
                # Numpy-first industrial workflows: images already decoded in memory.
                train_dataset = VisionArrayDataset(images=X_list, transform=self.train_transform)
            else:
                # Default: list of file paths.
                train_dataset = VisionImageDataset(
                    image_paths=X_list, transform=self.train_transform
                )

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            # 3. 准备训练 (来自父类的方法)
            self.training_prepare()

            # 4. 执行训练循环 (来自父类的方法，它会调用我们子类实现的 training_forward)
            if self.verbose:
                print(f"开始在 {self.device} 设备上进行训练...")
            self.train(train_loader)
            if self.verbose:
                print("训练完成。")

            # 5. 计算训练集上的异常分数
            if self.verbose:
                print("正在计算训练集上的异常分数...")
            self.decision_scores_ = self.decision_function(X)

            # 6. 调用 PyOD 的方法来计算阈值和标签 (继承来的免费功能)
            self._process_decision_scores()
            return self

        def decision_function(self, X, batch_size=None):
            import numpy as np

            from pyimgano.datasets import VisionArrayDataset, VisionImageDataset

            current_batch_size = batch_size if batch_size is not None else self.batch_size

            X_list = list(X)
            if X_list and isinstance(X_list[0], np.ndarray):
                eval_dataset = VisionArrayDataset(images=X_list, transform=self.eval_transform)
            else:
                eval_dataset = VisionImageDataset(image_paths=X_list, transform=self.eval_transform)
            eval_loader = DataLoader(eval_dataset, batch_size=current_batch_size, shuffle=False)

            # 调用父类的评估方法 evaluating_forward
            scores = self.evaluate(eval_loader)
            return scores

else:

    class BaseVisionDeepDetector:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            missing: list[str] = []
            if _torch is None:
                missing.append("torch")
            if _torchvision is None:
                missing.append("torchvision")
            if _pyod_base_dl is None:
                missing.append("pyod")

            missing_text = ", ".join(missing) if missing else "<unknown>"
            raise ImportError(
                "Deep-learning detectors require extra dependencies.\n"
                f"Missing: {missing_text}\n"
                "Install them via (example):\n"
                "  pip install 'torch' 'torchvision' 'pyod'\n"
                f"Original errors:\n  torch: {_torch_error}\n  torchvision: {_torchvision_error}\n  pyod: {_pyod_error}"
            )
