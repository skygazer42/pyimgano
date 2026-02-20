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

