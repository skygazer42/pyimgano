from __future__ import annotations

import numpy as np


def test_image_preprocessor_cv2_backend_processes_paths(tmp_path) -> None:
    import cv2

    from pyimgano.utils.image_ops import ImagePreprocessor

    img = np.ones((20, 30, 3), dtype=np.uint8) * 127
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), img)

    pre = ImagePreprocessor(resize=(16, 16), output_tensor=False, backend="cv2")
    out = pre.process(str(p))
    assert out.shape == (16, 16, 3)
    assert out.dtype == np.float32


def test_image_preprocessor_cv2_backend_supports_output_tensor(tmp_path) -> None:
    import cv2

    from pyimgano.utils.image_ops import ImagePreprocessor

    img = np.ones((20, 30, 3), dtype=np.uint8) * 200
    p = tmp_path / "x.png"
    cv2.imwrite(str(p), img)

    pre = ImagePreprocessor(resize=(16, 16), output_tensor=True, backend="cv2")
    out = pre.process(str(p))
    assert out.shape == (3, 16, 16)
    assert out.dtype == np.float32

