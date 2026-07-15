import inspect

import numpy as np
import pytest

from pyimgano.models import create_model

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_patchcore_accepts_numpy_images(monkeypatch):
    det = create_model(
        "vision_patchcore",
        coreset_sampling_ratio=1.0,
        n_neighbors=1,
        pretrained=False,
        device="cpu",
    )

    assert [type(item).__name__ for item in det.transform.transforms] == [
        "ToPILImage",
        "Resize",
        "CenterCrop",
        "ToTensor",
        "Normalize",
    ]
    assert det.backbone_name == "wide_resnet50_2"
    assert det.pretrain_embed_dimension == 1024
    assert det.target_embed_dimension == 1024
    assert det.patch_size == 3
    assert det.patch_stride == 1
    assert det.coreset_projection_dim == 128
    assert det.coreset_starting_points == 10
    signature = inspect.signature(type(det).__init__)
    assert signature.parameters["n_neighbors"].default == 1
    assert signature.parameters["pretrain_embed_dimension"].default == 1024
    assert signature.parameters["target_embed_dimension"].default == 1024

    def fake_extract(image):
        assert isinstance(image, np.ndarray)
        features = np.zeros((4, 2), dtype=np.float32)
        return features, (2, 2)

    monkeypatch.setattr(det, "_extract_patch_features", fake_extract)

    imgs = [np.zeros((10, 20, 3), dtype=np.uint8) for _ in range(2)]
    det.fit(imgs)

    scores = det.decision_function(imgs)
    assert scores.shape == (2,)

    def _imread_should_not_be_called(*_a, **_k):  # noqa: ANN002, ANN003, ANN202 - test stub
        raise AssertionError("cv2.imread called")

    monkeypatch.setattr(
        det._cv2,
        "imread",
        _imread_should_not_be_called,
    )
    anomaly_map = det.get_anomaly_map(imgs[0])
    assert anomaly_map.shape == (10, 20)
    assert anomaly_map.dtype == np.float32


def test_patchcore_uses_reference_unfold_and_1024_style_embedding():
    torch = pytest.importorskip("torch")
    det = create_model(
        "vision_patchcore",
        backbone="resnet50",
        pretrained=False,
        device="cpu",
        pretrain_embed_dimension=5,
        target_embed_dimension=7,
    )

    feature = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
    patches, patch_shape = det._patchify_feature_map(feature)
    assert patch_shape == (3, 3)
    assert tuple(patches.shape) == (1, 9, 1, 3, 3)
    torch.testing.assert_close(patches[0, 4, 0], feature[0, 0])
    torch.testing.assert_close(
        patches[0, 0, 0],
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 3.0, 4.0]]),
    )

    embedded, reference_shape = det._embed_feature_maps(
        [feature, torch.ones((1, 2, 2, 2), dtype=torch.float32)]
    )
    assert reference_shape == (3, 3)
    assert tuple(embedded.shape) == (9, 7)
    assert bool(torch.isfinite(embedded).all())


def test_patchcore_coreset_projection_only_selects_original_features():
    det = create_model(
        "vision_patchcore",
        backbone="resnet50",
        coreset_sampling_ratio=0.5,
        coreset_projection_dim=1,
        coreset_starting_points=2,
        pretrained=False,
        device="cpu",
    )
    features = np.arange(18, dtype=np.float32).reshape(6, 3)

    sampled = det._coreset_sampling(features)

    assert sampled.shape == (3, 3)
    assert all(any(np.array_equal(row, original) for original in features) for row in sampled)


def test_patchcore_feature_projection_reduces_feature_dim(monkeypatch):
    det = create_model(
        "vision_patchcore",
        coreset_sampling_ratio=1.0,
        n_neighbors=1,
        pretrained=False,
        device="cpu",
        feature_projection_dim=1,
        projection_fit_samples=1,
        random_seed=0,
    )

    def fake_extract(_image):
        features = np.arange(8, dtype=np.float32).reshape(4, 2)
        return features, (2, 2)

    monkeypatch.setattr(det, "_extract_patch_features", fake_extract)

    imgs = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
    det.fit(imgs)

    assert det.memory_bank is not None
    assert det.memory_bank.ndim == 2
    assert det.memory_bank.shape[1] == 1


def test_patchcore_memory_bank_dtype_float16(monkeypatch):
    det = create_model(
        "vision_patchcore",
        coreset_sampling_ratio=1.0,
        n_neighbors=1,
        pretrained=False,
        device="cpu",
        memory_bank_dtype="float16",
    )

    def fake_extract(_image):
        features = np.random.default_rng(0).standard_normal((4, 2)).astype(np.float32)
        return features, (2, 2)

    monkeypatch.setattr(det, "_extract_patch_features", fake_extract)

    imgs = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
    det.fit(imgs)

    assert det.memory_bank is not None
    assert det.memory_bank.dtype == np.float16


def test_patchcore_scores_with_nearest_patch_then_paper_reweighting(monkeypatch):
    det = create_model(
        "vision_patchcore",
        coreset_sampling_ratio=1.0,
        n_neighbors=2,
        pretrained=False,
        device="cpu",
    )

    train_features = np.asarray([[0.0], [2.0]], dtype=np.float32)
    query_features = np.asarray([[1.0], [4.0]], dtype=np.float32)
    monkeypatch.setattr(
        det,
        "_extract_patch_features",
        lambda _image: (train_features, (1, 2)),
    )

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    det.fit([image])
    monkeypatch.setattr(
        det,
        "_extract_patch_features",
        lambda _image: (query_features, (1, 2)),
    )

    score = float(det.decision_function([image])[0])

    # Worst query patch is 4; its nearest memory patch is 2 (distance 2).
    # The two support distances are 2 and 4.
    expected_weight = 1.0 - np.exp(2.0) / (np.exp(2.0) + np.exp(4.0))
    assert score == pytest.approx(expected_weight * 2.0)
