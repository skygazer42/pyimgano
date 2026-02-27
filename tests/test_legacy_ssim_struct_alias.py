import numpy as np


def test_ssim_struct_fit_and_scores_on_numpy_images() -> None:
    import pyimgano.models  # noqa: F401

    from pyimgano.models import create_model

    base = np.zeros((128, 128, 3), dtype=np.uint8)
    base[32:96, 48:80, :] = 180
    changed = base.copy()
    changed[60:90, 10:40, :] = 255

    train = [base for _ in range(6)]
    det = create_model(
        "ssim_struct",
        contamination=0.2,
        n_templates=1,
        resize_hw=(96, 96),
        canny_threshold1=50,
        canny_threshold2=150,
    )
    det.fit(train)

    s_base = float(det.decision_function([base])[0])
    s_changed = float(det.decision_function([changed])[0])
    assert np.isfinite(s_base)
    assert np.isfinite(s_changed)
    assert s_changed >= s_base

    preds = det.predict([base, changed])
    assert preds.shape == (2,)
    assert set(np.unique(preds)).issubset({0, 1})

