import numpy as np
import pytest


def _write_png(path, *, value: int = 128) -> None:
    from PIL import Image

    img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(str(path))


@pytest.mark.parametrize("model_name", ["vision_ecod", "vision_copod", "vision_knn"])
def test_detector_contract_fit_score_predict(tmp_path, model_name: str) -> None:
    # Import model implementations for side effects (registry population).
    import pyimgano.models  # noqa: F401

    from pyimgano.models.registry import create_model

    root = tmp_path / "ds"
    test0 = root / "test_0.png"
    test1 = root / "test_1.png"
    train_paths: list[str] = []
    for i in range(8):
        p = root / f"train_{i}.png"
        _write_png(p, value=110 + i)
        train_paths.append(str(p))
    _write_png(test0, value=120)
    _write_png(test1, value=240)

    detector = create_model(model_name, contamination=0.1)

    fitted = detector.fit(train_paths)
    assert fitted is detector

    scores = np.asarray(detector.decision_function([str(test0), str(test1)]), dtype=np.float64)
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()

    preds = np.asarray(detector.predict([str(test0), str(test1)]), dtype=int)
    assert preds.shape == (2,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_detector_contract_pixel_map_smoke_softpatch_stub_embedder(tmp_path) -> None:
    # Contract coverage for "deep + pixel_map" detectors without requiring torch downloads.
    pytest.importorskip("cv2")

    import zlib

    import pyimgano.models  # noqa: F401
    from pyimgano.models.registry import create_model

    class DummyEmbedder:
        def embed(self, image):
            key = str(image)
            seed = int(zlib.adler32(key.encode("utf-8")) & 0xFFFFFFFF)
            rng = np.random.default_rng(seed)
            patch_embeddings = rng.standard_normal(size=(4, 8)).astype(np.float32)
            grid_shape = (2, 2)
            original_size = (16, 16)
            return patch_embeddings, grid_shape, original_size

    root = tmp_path / "ds"
    test0 = root / "test_0.png"
    test1 = root / "test_1.png"
    train_paths: list[str] = []
    for i in range(6):
        p = root / f"train_{i}.png"
        _write_png(p, value=110 + i)
        train_paths.append(str(p))
    _write_png(test0, value=120)
    _write_png(test1, value=240)

    detector = create_model(
        "vision_softpatch",
        embedder=DummyEmbedder(),
        contamination=0.1,
        n_neighbors=1,
        knn_backend="sklearn",
        coreset_sampling_ratio=1.0,
    )
    detector.fit(train_paths)

    scores = np.asarray(detector.decision_function([str(test0), str(test1)]), dtype=np.float64)
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()

    preds = np.asarray(detector.predict([str(test0), str(test1)]), dtype=int)
    assert preds.shape == (2,)
    assert set(np.unique(preds)).issubset({0, 1})

    maps = np.asarray(detector.predict_anomaly_map([str(test0), str(test1)]), dtype=np.float32)
    assert maps.shape == (2, 16, 16)
    assert np.isfinite(maps).all()
