# -*- coding: utf-8 -*-
"""Kernel PCA anomaly detector implementations."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.utils import check_array, check_random_state

from ..utils.param_check import check_parameter
from ._legacy_x import MISSING
from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model


def _resolve_legacy_copy_x_keyword(
    copy_x: object,
    kwargs: dict[str, object],
    *,
    default: bool,
    method_name: str,
) -> bool:
    legacy_copy_x = kwargs.pop("copy_X", MISSING)
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(f"{method_name}() got an unexpected keyword argument {unexpected!r}")
    if copy_x is MISSING:
        return bool(default if legacy_copy_x is MISSING else legacy_copy_x)
    if legacy_copy_x is not MISSING:
        raise TypeError(f"{method_name}() got multiple values for argument 'copy_x'")
    return bool(copy_x)


class _PyODKernelPCA(KernelPCA):
    """Light wrapper around sklearn KernelPCA exposing internal helpers."""

    def __init__(
        self,
        *,
        n_components=None,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=1.0,
        fit_inverse_transform=False,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        remove_zero_eig=False,
        copy_x: object = MISSING,
        n_jobs=None,
        random_state=None,
        **kwargs,
    ) -> None:
        copy_x_value = _resolve_legacy_copy_x_keyword(
            copy_x,
            kwargs,
            default=True,
            method_name="_PyODKernelPCA",
        )
        self.copy_x = copy_x_value
        super().__init__(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            alpha=alpha,
            fit_inverse_transform=fit_inverse_transform,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            remove_zero_eig=remove_zero_eig,
            copy_X=copy_x_value,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    @property
    def centerer(self):
        return self._centerer

    @property
    def kernel_callable(self):
        return self._get_kernel


@register_model(
    "core_kpca",
    tags=("classical", "core", "features", "kernel", "projection"),
    metadata={"description": "Kernel PCA anomaly detector"},
)
class CoreKPCA(BaseDetector):
    """Kernel PCA anomaly detector."""

    def __init__(
        self,
        contamination=0.1,
        *,
        n_components=None,
        n_selected_components=None,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=1.0,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        remove_zero_eig=False,
        copy_x: object = MISSING,
        n_jobs=None,
        sampling=False,
        subset_size=20,
        random_state=None,
        **kwargs,
    ) -> None:
        copy_x_value = _resolve_legacy_copy_x_keyword(
            copy_x,
            kwargs,
            default=True,
            method_name="CoreKPCA",
        )
        super().__init__(contamination=contamination)
        self.n_components = n_components
        self.n_selected_components = n_selected_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.alpha = alpha
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.remove_zero_eig = remove_zero_eig
        self.copy_x = copy_x_value
        self.n_jobs = n_jobs
        self.sampling = sampling
        self.subset_size = subset_size
        self.random_state = check_random_state(random_state)
        self.n_selected_components_ = None
        self.kpca = None

    def _check_subset_size(self, array: np.ndarray) -> int:
        n_samples = array.shape[0]

        if isinstance(self.subset_size, int):
            if 0 < self.subset_size <= n_samples:
                return int(self.subset_size)
            raise ValueError(f"subset_size={self.subset_size} must be in (0, {n_samples}]")

        if isinstance(self.subset_size, float):
            if 0.0 < self.subset_size <= 1.0:
                return max(1, int(self.subset_size * n_samples))
            raise ValueError("subset_size as a float must be in (0.0, 1.0]")

        raise TypeError("subset_size only supports int or float")

    def fit(self, x, y=None):
        x = check_array(x, copy=self.copy_x)
        self._set_n_classes(y)

        if self.sampling:
            subset_size = self._check_subset_size(x)
            indices = self.random_state.choice(x.shape[0], size=subset_size, replace=False)
            x = x[indices, :]

        if self.n_components is None:
            n_components = x.shape[0]
        else:
            if self.n_components < 1:
                raise ValueError("n_components must be >= 1")
            n_components = min(x.shape[0], self.n_components)

        if self.n_selected_components is None:
            self.n_selected_components_ = n_components
        else:
            self.n_selected_components_ = self.n_selected_components

        check_parameter(
            self.n_selected_components_,
            low=1,
            high=n_components,
            include_left=True,
            include_right=True,
            param_name="n_selected_components",
        )

        self.kpca = _PyODKernelPCA(
            n_components=self.n_components,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            kernel_params=self.kernel_params,
            alpha=self.alpha,
            fit_inverse_transform=False,
            eigen_solver=self.eigen_solver,
            tol=self.tol,
            max_iter=self.max_iter,
            remove_zero_eig=self.remove_zero_eig,
            copy_x=self.copy_x,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        transformed = self.kpca.fit_transform(x)
        transformed = transformed[:, : self.n_selected_components_]

        centerer = self.kpca.centerer
        kernel = self.kpca.kernel_callable

        potential = []
        for sample in x:
            potential.append(kernel(sample.reshape(1, -1)))
        potential = np.asarray(potential).squeeze()
        potential = potential - 2 * centerer.K_fit_rows_ + centerer.K_fit_all_

        self.decision_scores_ = potential - np.sum(np.square(transformed), axis=1)
        self._process_decision_scores()
        return self

    def decision_function(self, x):
        if self.kpca is None or self.n_selected_components_ is None:
            raise RuntimeError("Detector must be fitted before calling decision_function")
        x = check_array(x)

        kernel = self.kpca.kernel_callable
        centerer = self.kpca.centerer

        gram_matrix = kernel(x, self.kpca.X_fit_)
        transformed = self.kpca.transform(x)
        transformed = transformed[:, : self.n_selected_components_]

        potential = []
        for sample in x:
            potential.append(kernel(sample.reshape(1, -1)))
        potential = np.asarray(potential).squeeze()

        gram_fit_rows = np.sum(gram_matrix, axis=1) / gram_matrix.shape[1]
        potential = potential - 2 * gram_fit_rows + centerer.K_fit_all_

        return potential - np.sum(np.square(transformed), axis=1)


@register_model(
    "vision_kpca",
    tags=("vision", "classical", "kernel"),
    metadata={"description": "Vision wrapper for Kernel PCA anomaly detection"},
)
class VisionKPCA(BaseVisionDetector):
    """Vision wrapper combining feature extraction with Kernel PCA."""

    def __init__(
        self,
        contamination=0.1,
        feature_extractor=None,
        *,
        n_components=None,
        n_selected_components=None,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=1.0,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        remove_zero_eig=False,
        copy_x: object = MISSING,
        n_jobs=None,
        sampling=False,
        subset_size=20,
        random_state=None,
        **kwargs,
    ) -> None:
        copy_x_value = _resolve_legacy_copy_x_keyword(
            copy_x,
            kwargs,
            default=True,
            method_name="VisionKPCA",
        )
        self.detector_params = {
            "contamination": contamination,
            "n_components": n_components,
            "n_selected_components": n_selected_components,
            "kernel": kernel,
            "gamma": gamma,
            "degree": degree,
            "coef0": coef0,
            "kernel_params": kernel_params,
            "alpha": alpha,
            "eigen_solver": eigen_solver,
            "tol": tol,
            "max_iter": max_iter,
            "remove_zero_eig": remove_zero_eig,
            "copy_x": copy_x_value,
            "n_jobs": n_jobs,
            "sampling": sampling,
            "subset_size": subset_size,
            "random_state": random_state,
        }

        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreKPCA(**self.detector_params)
