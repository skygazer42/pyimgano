import numpy as np

from pyimgano.models.openclip_backend import _prompt_patch_scores


def test_prompt_patch_scores_diff_mode():
    patches = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    text_normal = np.array([1.0, 0.0], dtype=np.float32)
    text_anom = np.array([0.0, 1.0], dtype=np.float32)
    scores = _prompt_patch_scores(
        patches,
        text_features_normal=text_normal,
        text_features_anomaly=text_anom,
        mode="diff",
    )
    assert scores.shape == (2,)
    assert float(scores[1]) > float(scores[0])

