from __future__ import annotations

import numpy as np


def _make_rgb_batch(*, count: int = 4, size: int = 16) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((count, size, size, 3), dtype=np.float32)


def _make_uint8_rgb_batch(*, count: int = 4, size: int = 16) -> np.ndarray:
    rng = np.random.default_rng(43)
    return rng.integers(0, 255, size=(count, size, size, 3), dtype=np.uint8)


def _make_uint8_gray_batch(*, count: int = 4, size: int = 16) -> np.ndarray:
    rng = np.random.default_rng(44)
    return rng.integers(0, 255, size=(count, size, size), dtype=np.uint8)


def test_histogram_comparison_fit_does_not_print_progress(capsys) -> None:
    from pyimgano.models.histogram_comparison import HistogramComparison

    model = HistogramComparison(n_bins=8, color_space="RGB", spatial=False)
    model.fit(_make_rgb_batch())

    out = capsys.readouterr().out
    assert out == ""


def test_template_matching_fit_does_not_print_progress(capsys) -> None:
    from pyimgano.models.template_matching import TemplateMatching

    model = TemplateMatching(
        method="ncc",
        use_multiple_templates=True,
        max_templates=2,
        color_space="GRAY",
        resize_shape=(16, 16),
        random_state=0,
    )
    model.fit(_make_rgb_batch())

    out = capsys.readouterr().out
    assert out == ""


def test_spc_fit_does_not_print_progress(capsys) -> None:
    from pyimgano.models.spc import SPC

    model = SPC(
        chart_type="shewhart",
        feature_extraction="mean_std",
        resize_shape=(16, 16),
    )
    model.fit(_make_rgb_batch())

    out = capsys.readouterr().out
    assert out == ""


def test_lbp_fit_does_not_print_progress(capsys) -> None:
    from pyimgano.models.lbp import LBP

    model = LBP(
        n_points=8,
        radius=1,
        detector="isolation_forest",
        n_bins=16,
        grid_size=(2, 2),
        resize_shape=None,
    )
    model.fit(_make_uint8_gray_batch())

    out = capsys.readouterr().out
    assert out == ""


def test_hog_svm_fit_does_not_print_progress(capsys) -> None:
    from pyimgano.models.hog_svm import HOG_SVM

    model = HOG_SVM(
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(1, 1),
        resize_shape=(16, 16),
    )
    model.fit(_make_rgb_batch())

    out = capsys.readouterr().out
    assert out == ""
