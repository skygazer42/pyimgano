from __future__ import annotations

from pyimgano.models.kpca import _PyODKernelPCA


def test_pyod_kernel_pca_get_params_uses_wrapper_constructor_names() -> None:
    estimator = _PyODKernelPCA(copy_x=False, random_state=42)

    params = estimator.get_params(deep=False)

    assert params["copy_x"] is False
    assert "copy_X" not in params
    assert params["random_state"] == 42
    assert estimator.copy_X is False
