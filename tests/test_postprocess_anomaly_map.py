import numpy as np

from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess


def test_postprocess_preserves_shape():
    pp = AnomalyMapPostprocess(normalize=True, gaussian_sigma=1.0)
    m = np.random.rand(32, 32).astype("float32")
    out = pp(m)
    assert out.shape == m.shape


def test_postprocess_requires_2d():
    pp = AnomalyMapPostprocess()
    try:
        pp(np.zeros((1, 8, 8), dtype=np.float32))
    except ValueError as exc:
        assert "Expected 2D" in str(exc)
    else:
        raise AssertionError("Expected ValueError to be raised")


def test_postprocess_percentile_normalization_is_robust_to_outliers():
    # Mix a "normal" bulk of values and a single extreme outlier so that:
    # - min-max normalization is dominated by the outlier
    # - percentile normalization keeps the bulk informative
    m = np.zeros((10, 20), dtype=np.float32)
    m[:, :15] = 1.0  # 150 ones, 50 zeros
    m[0, 0] = 100.0  # single extreme outlier

    out_minmax = AnomalyMapPostprocess(normalize=True, normalize_method="minmax")(m)
    out_pct = AnomalyMapPostprocess(
        normalize=True,
        normalize_method="percentile",
        percentile_range=(1.0, 99.0),
    )(m)

    # With a single huge outlier, min-max squashes the bulk of values near 0.
    assert float(out_pct.mean()) > float(out_minmax.mean())
