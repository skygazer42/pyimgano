import numpy as np
import pytest

pytest.importorskip("skimage")


def test_ssim_template_fit_and_scores_on_numpy_images() -> None:
    import pyimgano.models  # noqa: F401
    from pyimgano.models import create_model

    base = np.zeros((128, 128, 3), dtype=np.uint8)
    base[40:88, 50:78, :] = 200
    changed = base.copy()
    changed[10:20, 10:20, :] = 255

    train = [base for _ in range(6)]
    det = create_model("ssim_template", contamination=0.2, n_templates=1, resize_hw=(96, 96))
    det.fit(train)

    s_base = float(det.decision_function([base])[0])
    s_changed = float(det.decision_function([changed])[0])
    assert np.isfinite(s_base)
    assert np.isfinite(s_changed)
    assert s_changed >= s_base

    preds = det.predict([base, changed])
    assert preds.shape == (2,)
    assert set(np.unique(preds)).issubset({0, 1})
