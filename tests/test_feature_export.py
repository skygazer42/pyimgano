from __future__ import annotations

import numpy as np


def test_extract_features_with_ids_preserves_order_and_shape(tmp_path) -> None:  # noqa: ANN001
    from pyimgano.features import create_feature_extractor
    from pyimgano.features.export import (
        extract_features_with_ids,
        load_feature_export,
        save_feature_export,
    )

    ext = create_feature_extractor("identity")
    x = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]
    ids = ["a", "b"]

    export = extract_features_with_ids(ext, x, ids=ids)
    assert export.ids == ids
    assert export.features.shape == (2, 2)
    assert np.allclose(export.features, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    p = tmp_path / "feats.npz"
    save_feature_export(p, export)
    export2 = load_feature_export(p)
    assert export2.ids == ids
    assert export2.features.shape == (2, 2)
    assert np.allclose(export2.features, export.features)
